import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, InterpolationMode, Compose
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
import random, json
from pycocotools import mask as coco_mask
import submodules.depth_anything.depth_anything.util.transform as depth_anything_transform


def intersect(a0, a1, b0, b1):
    c0 = ccw(a0, a1, b0)
    c1 = ccw(a0, a1, b1)
    d0 = ccw(b0, b1, a0)
    d1 = ccw(b0, b1, a1)
    if abs(d1 - d0) > abs(c1 - c0):
        return (a0 * d1 - a1 * d0) / (d1 - d0)
    else:
        return (b0 * c1 - b1 * c0) / (c1 - c0)


def ccw(c, a, b):
    a0 = a - c
    b0 = b - c
    return a0[0] * b0[1] - b0[0] * a0[1]


class PerspectiveV1Dataset(Dataset):

    def __init__(self, data_path, transform=None, split='train', return_path=False):
        assert split in ['train', 'val',  'test']
        self.split = split

        self.datapath = data_path
        with open(os.path.join(self.datapath, 'train_val_test_split.json'), 'r') as f:
            self.img_list = json.load(f)[self.split]
        
        self.transform = transform
        self.return_path = return_path

        self.resize = False
        if self.transform is None:
            self.transform = Compose([
                Resize(size=(518,518), interpolation=InterpolationMode.BICUBIC, antialias=True),
                ToTensor()
            ])
        for t in self.transform.transforms:
            if isinstance(t, (Resize, depth_anything_transform.Resize)):
                self.resize = True
                input_size = t.size if hasattr(t, 'size') else t.get_size(1,1) # assume square input
                # self.resize_ratio = input_size / 256
                # self.resize_module = Resize(size=input_size, interpolation=InterpolationMode.NEAREST) # nearest neighbour for the mask
                print('Resizing images and masks to', input_size)

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.datapath, 'images', self.img_list[idx])
        tname = img_path.replace(".jpg", ".txt")
        axy, bxy = np.genfromtxt(tname, skip_header=1)
        a0, a1 = np.array(axy[:2]), np.array(axy[2:])
        b0, b1 = np.array(bxy[:2]), np.array(bxy[2:])
        xy = intersect(a0, a1, b0, b1) - 0.5 # (2,) float ndarray

        src_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = src_img.size
        if self.transform:
            src_img = self.transform(src_img)
            h, w = src_img.shape[1:]
            xy[1] *= h / orig_h
            xy[0] *= w / orig_w
        
        # augment
        if self.split == 'train':
            if random.random() > 0.5:
                src_img = TF.hflip(src_img)
                xy[0] = w - xy[0]
            if random.random() > 0.5:
                src_img = TF.vflip(src_img)
                xy[1] = h - xy[1]

        # normalise coordinates
        xy[0] /= w
        xy[1] /= h

        # shift the origin to center of image
        xy -= 0.5
        xy = torch.Tensor(xy)

        ret = {
            'image': src_img,
            'label': xy,
        }

        if self.return_path:
            ret['img_path'] = img_path

        return ret