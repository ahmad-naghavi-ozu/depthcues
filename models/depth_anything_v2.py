import torch
from torchvision.transforms import Compose, ToTensor
import cv2
import numpy as np
from submodules.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
from models.depth_anything import get_depth_anything_transform


class DepthAnythingV2Backbone(DepthAnythingV2):
    def __init__(self, arch='vitb'):
        '''
        layer in [0,1,2,...,11]
        feat dim is [768]
        '''

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        super().__init__(**model_configs[arch])

        self.load_state_dict(torch.load(f'submodules/depth_anything_v2/checkpoints/depth_anything_v2_{arch}.pth', map_location='cpu'))

    def forward_intermediates(self, x, layer, norm=False, reshape=True):
        if self.pretrained.chunked_blocks:
            n = [len(self.pretrained.blocks[-1])] if layer == -1 else [layer]
            outputs = self.pretrained._get_intermediate_layers_chunked(x, n)
        else:
            n = [len(self.pretrained.blocks)] if layer == -1 else [layer]
            outputs = self.pretrained._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.pretrained.norm(out) for out in outputs]
        # class_tokens = [out[:, 0] for out in outputs] # (B,C)
        outputs = [out[:, 1 + self.pretrained.num_register_tokens:] for out in outputs] # (B,L,C)
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.pretrained.patch_size, h // self.pretrained.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ] # (B,C,H,W)

        return list(zip(outputs, [None]))


class DepthAnythingV2MetricIndoorBackbone(DepthAnythingV2Backbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_state_dict(torch.load(f'submodules/depth_anything_v2/checkpoints/depth_anything_v2_metric_hypersim_vitb.pth', map_location='cpu'))


class DepthAnythingV2MetricOutdoorBackbone(DepthAnythingV2Backbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_state_dict(torch.load(f'submodules/depth_anything_v2/checkpoints/depth_anything_v2_metric_vkitti_vitb.pth', map_location='cpu'))