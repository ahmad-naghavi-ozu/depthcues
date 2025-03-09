import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, InterpolationMode, Compose
from PIL import Image
import os
import numpy as np
import pickle, json
from pycocotools import mask as coco_mask
import submodules.depth_anything.depth_anything.util.transform as depth_anything_transform
import random

class ElevationV1Dataset(Dataset):
    def __init__(self, data_path, transform=None, split='train', return_path=False):
        '''
        The labels are in the form of [offset, slope];
        offsets satisfy -1 <= offset-0.5 <= 1
        slopes satisfy -0.625 <= slope <= 0.625
        '''
        assert split in ['train', 'val',  'test']
        self.split = split

        self.datapath = data_path

        with open(os.path.join(data_path, f'{split}_data.pkl'), 'rb') as f:
            self.data = pickle.load(f)

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
        return len(self.data)

    def __getitem__(self, idx):
        
        cur_data = self.data[idx]
        src_img = Image.open(os.path.join(self.datapath, 'images', cur_data['fname'])).convert("RGB")
        orig_w, orig_h = src_img.size

        label = torch.Tensor(cur_data["label"]) # (2,), offset, slope
        if self.transform:
            src_img = self.transform(src_img)
            # offset is the real offset coordinate normalised by image height
            # slope is the real slope normalised by (image height / image width)
            # i.e. they are not absolute, but relative to the image size
            # e.g slope is defined as percentage change in height per percentage change in width
            # so no need to scale them when image is resized
            # h, w = src_img.shape[1:]
            # label[1] *= (h / orig_h) / (w / orig_w)
        
        # move origin to image center, this only affects offset
        label[0] -= 0.5

        ret = {
            'image': src_img,
            'label': label,
        }
        if self.return_path:
            ret['img_path'] = cur_data['fname']

        return ret