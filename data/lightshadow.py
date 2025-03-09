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


class LightshadowV1Dataset(Dataset):
    def __init__(self, data_path, transform=None, split='train', return_dot=False, return_mask=True, return_path=False):
        assert split in ['train', 'val',  'test']
        self.split = split

        self.datapath = data_path
        with open(os.path.join(self.datapath, f"{split}_annotations.pkl"), "rb") as f:
            self.data = pickle.load(f)
        self.anno_keys = list(self.data.keys())

        self.transform = transform
        self.return_path = return_path
        self.return_dot = return_dot
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
        return len(self.anno_keys)
    
    def __getitem__(self, idx):
        
        anno_key = self.anno_keys[idx]
        img_path = os.path.split(self.data[anno_key]['fname'])[-1]

        data = self.data[anno_key]
        src_img = Image.open(os.path.join(self.datapath, "images", self.split, img_path)).convert("RGB")
        orig_W, orig_H = src_img.size
        label = torch.tensor(data["class"]).float()
        if self.transform:
            src_img = self.transform(src_img)

        ret = {
            'image': src_img,
            'label': label,
        }
        if self.return_path:
            ret['img_path'] = img_path
        
        if self.return_dot:
            # check whether image is resized
            if self.resize:
                _, H, W = src_img.shape
                resize_ratio = torch.tensor([H/orig_H, W/orig_W]) #(2,)
                ret['red_dot_loc'] = torch.round(torch.tensor(data['red_dot_loc']) * resize_ratio).long() # (2,) torch.int64
                ret['green_dot_loc'] = torch.round(torch.tensor(data['green_dot_loc']) * resize_ratio).long()
            else:
                ret['red_dot_loc'] = torch.tensor(data['red_dot_loc']).long() # (2,) torch.int64
                ret['green_dot_loc'] = torch.tensor(data['green_dot_loc']).long() 

        if self.return_mask:
            red_mask = torch.Tensor(coco_mask.decode(data['red_obj_mask'])).unsqueeze(0)
            green_mask = torch.Tensor(coco_mask.decode(data['green_obj_mask'])).unsqueeze(0)
            if self.resize:
                _, H, W = src_img.shape
                ret['red_obj_mask'] = TF.resize(red_mask, size=(H,W), interpolation=InterpolationMode.NEAREST) #(1,H,W) torch.float32
                ret['green_obj_mask'] = TF.resize(green_mask, size=(H,W), interpolation=InterpolationMode.NEAREST) #(1,H,W) torch.float32
            else:
                ret['red_obj_mask'] = red_mask #(1,H,W) torch.float32
                ret['green_obj_mask'] = green_mask

        return ret