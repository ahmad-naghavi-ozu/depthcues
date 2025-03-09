import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode, Normalize
from submodules.moge.moge.model import MoGeModel

class MoGeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")

    def forward_intermediates(self, x, *args, **kwargs):   
        image = x                       
        raw_img_h, raw_img_w = image.shape[-2:]
        patch_h, patch_w = raw_img_h // 14, raw_img_w // 14

        image = (image - self.model.image_mean) / self.model.image_std

        # Apply image transformation for DINOv2
        image_14 = F.interpolate(image, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True)

        # Get intermediate layers from the backbone
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            features = self.model.backbone.get_intermediate_layers(image_14, reshape=True)[0]
        
        return [(features, None)]
    
def get_moge_transform():
    return Compose([
        Resize(size=(518, 518), interpolation=InterpolationMode.BICUBIC, antialias=True),
        ToTensor(),
    ])