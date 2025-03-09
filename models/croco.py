from submodules.dust3r.croco.models.croco_downstream import CroCoNet
from submodules.dust3r.croco.models.pos_embed import interpolate_pos_embed
import torch
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode, Normalize
from pathlib import Path
from urllib.request import urlretrieve
import os


class CroCoV2Backbone(CroCoNet):
    def __init__(self, arch='vit_b', **kwargs):
        '''
        layer in [0,11], test [2, 5, 8, 11]
        feat dim = 768
        '''

        ckpt_paths = {
            "vit_b": "CroCo_V2_ViTBase_BaseDecoder.pth",
            "vit_l": "CroCo_V2_ViTLarge_BaseDecoder.pth"
        }
        ckpt_file = ckpt_paths[arch]
        torch_cache_dir = os.environ['TORCH_HOME']
        ckpt_path = os.path.join(torch_cache_dir, ckpt_file)
        if not os.path.exists(ckpt_path):
            download_path = (
                f"https://download.europe.naverlabs.com/ComputerVision/CroCo/{ckpt_file}"
            )
            print('Downloading CroCo checkpoints to', ckpt_path)
            urlretrieve(download_path, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        super(CroCoV2Backbone, self).__init__(**ckpt.get('croco_kwargs',{}))
        missing_keys, unexpected_keys = self.load_state_dict(ckpt['model'], strict=False)
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        print(f'Successfully loaded CroCoV2 model from {ckpt_path} with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys!')

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def _set_mask_token(self, *args, **kwargs):
        """ No mask token """
        self.mask_token = None
        return

    def _set_decoder(self, *args, **kwargs):
        """ No decoder """
        return

    def _set_prediction_head(self, *args, **kwargs):
        """ No 'prediction head' for downstream tasks."""
        return

    def forward_intermediates(self, x, layer):
        B, C, H, W = x.size()

        x, pos = self.patch_embed(x)              
        # add positional embedding without cls token  
        if self.enc_pos_embed is not None: 
            x = x + self.enc_pos_embed[None,...]
        # apply masking 
        posvis = pos

        # now apply the transformer encoder and normalization        
        blocks = self.enc_blocks[:layer+1]
        for blk in blocks:
            x = blk(x, posvis)

        dense_tokens = x.reshape(B,H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[0], -1).permute(0, 3, 1, 2).contiguous()

        return [(dense_tokens, None)]


def get_crocov2_transform():
    return Compose(
        [
            Resize(size=(224,224), interpolation=InterpolationMode.LANCZOS, antialias=True),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )


class CroCoBackbone(CroCoNet):
    def __init__(self, img_size=512, **kwargs):
        '''
        layer in [0,11], test [2, 5, 8, 11]
        feat dim = 768
        '''

        torch_cache_dir = os.environ['TORCH_HOME']
        ckpt_path = os.path.join(torch_cache_dir, 'CroCo.pth')
        if not os.path.exists(ckpt_path):
            download_path = (
                f"https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth"
            )
            print('Downloading CroCo checkpoints to', ckpt_path)
            urlretrieve(download_path, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location='cpu')

        super(CroCoBackbone, self).__init__(img_size=img_size, **ckpt.get('croco_kwargs',{}))

        # interpolate pos embed to 512 then load ckpt
        # remove dec_pos_embed otherwise non-interpolated pos embed will be loaded and cause error
        delattr(self, 'dec_pos_embed')
        interpolate_pos_embed(self, ckpt['model'])
        missing_keys, unexpected_keys = self.load_state_dict(ckpt['model'], strict=False)
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        print(f'Successfully loaded CroCo model from {ckpt_path} with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys!')

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def _set_mask_token(self, *args, **kwargs):
        """ No mask token """
        self.mask_token = None
        return

    def _set_decoder(self, *args, **kwargs):
        """ No decoder """
        return

    def _set_prediction_head(self, *args, **kwargs):
        """ No 'prediction head' for downstream tasks."""
        return

    def forward_intermediates(self, x, layer):
        B, C, H, W = x.size()

        x, pos = self.patch_embed(x)              
        # add positional embedding without cls token  
        if self.enc_pos_embed is not None: 
            x = x + self.enc_pos_embed[None,...]
        # apply masking 
        posvis = pos

        # now apply the transformer encoder and normalization        
        blocks = self.enc_blocks[:layer+1]
        for blk in blocks:
            x = blk(x, posvis)

        dense_tokens = x.reshape(B,H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[0], -1).permute(0, 3, 1, 2).contiguous()

        return [(dense_tokens, None)]


def get_croco_transform():
    return Compose(
        [
            Resize(size=(512,512), interpolation=InterpolationMode.LANCZOS, antialias=True),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )