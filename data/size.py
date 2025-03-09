import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, InterpolationMode, Compose
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
import pickle, json
from pycocotools import mask as coco_mask
import submodules.depth_anything.depth_anything.util.transform as depth_anything_transform
import random


class SizeV1Dataset(Dataset):
    def __init__(self, data_path, transform=None, split='train', return_mask=True, return_path=False):
        assert split in ['train', 'val',  'test']
        self.split = split

        self.datapath = data_path

        with open(os.path.join(data_path, f'{split}_data_indoor.pkl'), 'rb') as f:
            data_indoor = pickle.load(f)
            for d in data_indoor:
                d['fname'] = os.path.join('images_indoor', d['fname'])
        with open(os.path.join(data_path, f'{split}_data_outdoor.pkl'), 'rb') as f:
            data_outdoor = pickle.load(f)
            for d in data_outdoor:
                d['fname'] = os.path.join('images_outdoor', d['fname'])
        self.data = data_indoor + data_outdoor

        self.transform = transform
        self.return_path = return_path
        self.return_mask = return_mask

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
        return len(self.data)

    def __getitem__(self, idx):
        
        cur_data = self.data[idx]
        src_img = Image.open(os.path.join(self.datapath, cur_data['fname'])).convert("RGB")
        label = torch.tensor(cur_data["label"]).float()
        if self.transform:
            src_img = self.transform(src_img)

        ret = {
            'image': src_img,
            'label': label,
        }
        if self.return_path:
            ret['img_path'] = cur_data['fname']

        if self.return_mask:
            red_mask = torch.Tensor(coco_mask.decode(cur_data['red_obj_mask'])).unsqueeze(0)
            green_mask = torch.Tensor(coco_mask.decode(cur_data['green_obj_mask'])).unsqueeze(0)
            if self.resize:
                _, H, W = src_img.shape
                ret['red_obj_mask'] = TF.resize(red_mask, size=(H,W), interpolation=InterpolationMode.NEAREST) #(1,H,W) torch.float32
                ret['green_obj_mask'] = TF.resize(green_mask, size=(H,W), interpolation=InterpolationMode.NEAREST) #(1,H,W) torch.float32
            else:
                ret['red_obj_mask'] = red_mask #(1,H,W) torch.float32
                ret['green_obj_mask'] = green_mask

        return ret

class SizeV2Dataset(SizeV1Dataset):
    def __init__(self, data_path, transform=None, split='train', return_mask=True, return_path=False):
        super().__init__(data_path, transform, split, return_mask, return_path)