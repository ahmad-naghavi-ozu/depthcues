import os
from functools import partial
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomApply, ColorJitter, RandomResizedCrop, InterpolationMode
import submodules.depth_anything.depth_anything.util.transform as depth_anything_transform
from models.depth_anything import TensorToNumpy, DictToTensor
import cv2
import torchvision.transforms.functional as TF
import numpy as np
import random
import scipy.io
from PIL import Image
import pickle


def shared_random_resized_crop(image, depth, size, scale=(0.5, 1.0), ratio=(1.0, 1.0), p=0.5):
    if random.random() < p:
        return image, depth
    i, j, h, w = RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
    image = TF.crop(image, i, j, h, w)
    depth = TF.crop(depth, i, j, h, w)
    image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
    depth = TF.resize(depth, size, interpolation=InterpolationMode.NEAREST)
    
    return image, depth

class NYU_geonet(Dataset):
    def __init__(
        self,
        data_path,
        transform=None,
        split='train',
        return_path=False,
        center_crop=True,
        augment_train=True,
    ):
        super().__init__()
        self.center_crop = center_crop
        self.max_depth = 10.0

        # parse dataset
        self.root_dir = data_path
        insts = os.listdir(data_path)
        insts.sort()

        # remove bad indices
        del insts[21181]
        del insts[6919]

        if split == "train":
            self.instances = [x for i, x in enumerate(insts) if i % 20 != 0]
        elif split == "val":
            self.instances = [x for i, x in enumerate(insts) if i % 20 == 0]
        else:
            raise ValueError()

        self.return_path = return_path

        self.resize = False
        self.img_size = (448,448) # close to 480 in Probe3D paper, and divisible by both 14 and 16
        depthanything_flag = False
        if transform is None:
            self.image_transform = Compose([
                Resize(size=self.img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
                ToTensor()
            ])
            self.resize = True
        else:
            self.image_transform = []
            for t in transform.transforms:
                if not isinstance(t, (Resize, depth_anything_transform.Resize)):
                    self.image_transform.append(t)
                elif isinstance(t, (Resize)):
                    self.image_transform.append(
                        Resize(size=self.img_size, interpolation=t.interpolation, antialias=t.antialias)
                    )
                    self.resize = True
                elif isinstance(t, depth_anything_transform.Resize):
                    depthanything_flag = True
                    self.image_transform.append(
                        depth_anything_transform.Resize(
                            width=self.img_size[1],
                            height=self.img_size[0],
                            resize_target=False,
                            keep_aspect_ratio=False,
                            ensure_multiple_of=14,
                            resize_method='lower_bound',
                            image_interpolation_method=cv2.INTER_CUBIC,
                        )
                    )
                    self.resize = True
            self.image_transform = Compose(self.image_transform)
        assert self.resize, "Resize transform not found"
        print('Resizing images and masks to', self.img_size)

        # get augmentation transforms
        self.augment = augment_train and "train" in split
        if self.augment:
            # get index to insert
            for insert_idx, tr in enumerate(self.image_transform.transforms):
                if 'normalize' in str(tr).lower():
                    break
            if not depthanything_flag:
                self.image_transform.transforms.insert(
                    insert_idx,
                    RandomApply([ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.8) 
                )
            else:
                self.image_transform.transforms.insert(
                    insert_idx,
                    RandomApply(torch.nn.ModuleList([
                        DictToTensor(),
                        ColorJitter(0.2, 0.2, 0.2, 0.2),
                        TensorToNumpy()
                    ]), p=0.8)
                )
            self.shared_transform = partial(shared_random_resized_crop, size=self.img_size, scale=(0.5, 1.0), ratio=(1.0, 1.0), p=0.5)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        file_name = self.instances[index]
        room = "_".join(file_name.split("-")[0].split("_")[:-2])

        # extract elements from the matlab thing
        instance = scipy.io.loadmat(os.path.join(self.root_dir, file_name))
        image = instance["img"][:480, :640]
        depth = instance["depth"][:480, :640]
        # center crop
        if self.center_crop:
            image = image[:, 80:-80]
            depth = depth[:, 80:-80]
        # process image
        image[:, :, 0] = image[:, :, 0] + 2 * 122.175
        image[:, :, 1] = image[:, :, 1] + 2 * 116.169
        image[:, :, 2] = image[:, :, 2] + 2 * 103.508
        image = Image.fromarray(image.astype(np.uint8))

        image = self.image_transform(image)
        depth = torch.Tensor(depth).unsqueeze(0)
        if self.resize:
            _, H, W = image.shape
            depth = TF.resize(depth, size=(H,W), interpolation=InterpolationMode.NEAREST)

        # set max depth to 10
        depth[depth > self.max_depth] = 0

        if self.augment:
            image, depth = self.shared_transform(image=image, depth=depth)

        ret = {
            'image': image,
            'label': depth,
        }
        if self.return_path:
            ret['img_path'] = room

        return ret
    
class NYU_test(Dataset):
    """
    Dataset loader based on Ishan Misra's SSL benchmark
    """

    def __init__(self,
                 data_path='nyuv2_test.pkl',
                 transform=None,
                 split='test',
                 return_path=False,
                 center_crop=True):
        super().__init__()
        self.center_crop = center_crop
        self.max_depth = 10.0

        # get transforms
        self.return_path = return_path

        self.resize = False
        self.img_size = (448,448) # close to 480 in Probe3D paper, and divisible by both 14 and 16
        if transform is None:
            self.image_transform = Compose([
                Resize(size=self.img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
                ToTensor()
            ])
            self.resize = True
        else:
            self.image_transform = []
            for t in transform.transforms:
                if not isinstance(t, (Resize, depth_anything_transform.Resize)):
                    self.image_transform.append(t)
                elif isinstance(t, (Resize)):
                    self.image_transform.append(
                        Resize(size=self.img_size, interpolation=t.interpolation, antialias=t.antialias)
                    )
                    self.resize = True
                elif isinstance(t, depth_anything_transform.Resize):
                    self.image_transform.append(
                        depth_anything_transform.Resize(
                            width=self.img_size[1],
                            height=self.img_size[0],
                            resize_target=False,
                            keep_aspect_ratio=False,
                            ensure_multiple_of=14,
                            resize_method='lower_bound',
                            image_interpolation_method=cv2.INTER_CUBIC,
                        )
                    )
                    self.resize = True
            self.image_transform = Compose(self.image_transform)
        assert self.resize, "Resize transform not found"
        print('Resizing images and masks to', self.img_size)

        # parse data
        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)

        self.indices = data_dict["test_indices"]
        self.depths = [data_dict["depths"][_i] for _i in self.indices]
        self.images = [data_dict["images"][_i] for _i in self.indices]
        self.scenes = [data_dict["scene_types"][_i][0] for _i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image = self.images[index]
        depth = self.depths[index]
        room = self.scenes[index]

        if self.center_crop:
            image = image[..., 80:-80]
            depth = depth[..., 80:-80]

        # transform image
        image = Image.fromarray(np.transpose(image, (1, 2, 0)))
        image = self.image_transform(image)
        depth = torch.Tensor(depth).unsqueeze(0)
        if self.resize:
            _, H, W = image.shape
            depth = TF.resize(depth, size=(H,W), interpolation=InterpolationMode.NEAREST)

        depth[depth > self.max_depth] = 0

        ret = {
            'image': image,
            'label': depth,
        }
        if self.return_path:
            ret['img_path'] = room

        return ret