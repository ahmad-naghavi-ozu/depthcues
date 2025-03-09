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


class DIW(Dataset):
    def __init__(self, data_path, transform=None, split='train', return_path=False):
        assert split in ['train', 'val',  'test']
        self.split = split

        if split == 'test':
            self.datapath = os.path.join(data_path, 'DIW_test')
        else:
            self.datapath = os.path.join(data_path, 'DIW_train_val')

        with open(os.path.join(data_path, f'{split}_data.pkl'), 'rb') as f:
            self.data = pickle.load(f)

        self.return_path = return_path

        self.resize = False
        self.img_size = (448,448) # close to 480 in Probe3D paper, and divisible by both 14 and 16
        if transform is None:
            self.transform = Compose([
                Resize(size=self.img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
                ToTensor()
            ])
            self.resize = True
        else:
            self.transform = []
            for t in transform.transforms:
                if not isinstance(t, (Resize, depth_anything_transform.Resize)):
                    self.transform.append(t)
                elif isinstance(t, (Resize)):
                    self.transform.append(
                        Resize(size=self.img_size, interpolation=t.interpolation, antialias=t.antialias)
                    )
                    self.resize = True
                elif isinstance(t, depth_anything_transform.Resize):
                    self.transform.append(
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
            self.transform = Compose(self.transform)
        assert self.resize, "Resize transform not found"
        print('Resizing images and masks to', self.img_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        cur_data = self.data[idx]
        src_img = Image.open(os.path.join(self.datapath, cur_data['img_path'])).convert("RGB")
        orig_w, orig_h = src_img.size
        
        x_A = cur_data['x_A'] - 1
        x_B = cur_data['x_B'] - 1
        y_A = cur_data['y_A'] - 1
        y_B = cur_data['y_B'] - 1

        if self.transform:
            src_img = self.transform(src_img)
            h, w = src_img.shape[1:]
            x_A *= w / orig_w
            x_B *= w / orig_w
            y_A *= h / orig_h
            y_B *= h / orig_h


        label = torch.tensor(cur_data["label"]).float()

        ret = {
            'image': src_img,
            'label': label,
            'red_dot_loc': torch.round(torch.tensor([y_A, x_A])).long(), # (2,) torch.int64
            'green_dot_loc': torch.round(torch.tensor([y_B, x_B])).long()
        }
        if self.return_path:
            ret['img_path'] = cur_data['img_path']

        return ret