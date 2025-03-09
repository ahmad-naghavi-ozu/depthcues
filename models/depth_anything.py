import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.functional as F
from submodules.depth_anything.depth_anything.dpt import DepthAnything
from submodules.depth_anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
import numpy as np


class DepthAnythingBackbone(nn.Module):
    def __init__(self,):
        super().__init__()
        encoder = 'vitb'
        self.model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))


    def forward_intermediates(self, x, *args, **kwargs):
        features = self.model.pretrained.get_intermediate_layers(x, reshape=True)[0]
        return [(features, None)]


def build_depth_anything(arch):
    '''
    arch: one of ['vits', 'vitb', 'vitl']
    '''
    depth_anything = DepthAnythingBackbone.from_pretrained('LiheYoung/depth_anything_{:}14'.format(arch)).eval()
    
    return depth_anything

class PillowToNumpy(object):
    """Prepare sample for usage as network input.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.array(sample) / 255.
        return {'image': image}

class GetImageFromDict(object):
    """Prepare sample for usage as network input.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample['image']

class ConvertToTensor(object):
    """Prepare sample for usage as network input.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        return torch.from_numpy(sample)

class DictToTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, sample):
        return F.to_tensor(sample['image']).clamp(0,1) # (H,W,3) ndarray, need to clamp for colorjitter

class TensorToNumpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, sample):
        return {'image': sample.permute(1,2,0).numpy()} # (H,W,3) ndarray

def get_depth_anything_transform():
    transform = Compose([
        PillowToNumpy(),
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        GetImageFromDict(),
        ConvertToTensor()
    ])
    return transform