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



class OcclusionV4Dataset(Dataset):
    def __init__(self, data_path, transform=None, split='train', return_mask=True, return_path=False):
        assert split in ['train', 'val',  'test']
        self.split = split

        self.datapath = data_path
        with open(os.path.join(self.datapath, f'{split}_data.pkl'), "rb") as f:
            self.data = pickle.load(f)
        
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
        
        anno = self.data[idx]
        source = anno['source']
        img_fname = anno['img_fname']

        img_path = os.path.join(self.datapath, f'images_{source}', self.split, img_fname)
        src_img = Image.open(img_path).convert("RGB")
        if self.transform:
            src_img = self.transform(src_img)

        label = torch.tensor(anno['label']).float()

        ret = {
            'image': src_img,
            'label': label,
            'name': anno['name']
        }

        if self.return_path:
            ret['img_path'] = idx

        if self.return_mask:
            mask = torch.Tensor(coco_mask.decode(anno['red_obj_mask'])).unsqueeze(0) #(1,H,W) torch.float32
            if self.resize:
                _, H, W = src_img.shape
                ret['red_obj_mask'] = TF.resize(mask, size=(H,W), interpolation=InterpolationMode.NEAREST)
            else:
                ret['red_obj_mask'] = mask

        return ret