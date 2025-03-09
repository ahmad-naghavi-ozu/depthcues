import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode, Normalize
from submodules.probe3d.evals.models.clip import resize_pos_embed
import timm

class TimmViTBackbone(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k"):
        '''
        ViT Base: feat_dim = 768, probe layers [3,6,9,12]
        '''
        super().__init__()
        self.vit = timm.create_model(model_name=model_name, pretrained=True, num_classes=0)
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.embed_size = self.vit.patch_embed.grid_size
        self.num_prefix_tokens = self.vit.num_prefix_tokens
        self.vit.patch_embed.strict_img_size = False

    def forward_intermediates(self, images, indices, return_prefix_tokens, output_fmt, intermediates_only, **kwargs):
        _, _, img_h, img_w = images.shape
        # get embed h, w
        out_h, out_w = img_h // self.patch_size, img_w // self.patch_size

        if (out_h, out_w) != self.embed_size:
            self.embed_size = (out_h, out_w)
            self.vit.pos_embed.data = resize_pos_embed(
                self.vit.pos_embed[0], self.embed_size, self.num_prefix_tokens>0
            )[None, :, :]

        return self.vit.forward_intermediates(images, indices=indices, return_prefix_tokens=return_prefix_tokens, output_fmt=output_fmt, intermediates_only=intermediates_only, **kwargs)


def get_vit_transform():
    return Compose(
        [
            Resize(size=(512,512), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

def get_dinov2_transform():
    return Compose([
        Resize(size=(518, 518), interpolation=InterpolationMode.BICUBIC, antialias=True),
        ToTensor(),
        Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
    ])