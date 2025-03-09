import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from models.probe3d_backbones import *


class TrivialFeatureExtractor(nn.Module):

    def __init__(self, probe_type, feat_type='dot'):
        '''
        Trivial feature extractor that prepares inputs (dot coords or mask) to trivial baselines.

        args:
        model      -- an instance of timm.models.vision_transformer.VisionTransformer
        feat_type  -- one of ['dot', 'mask']
            'dot': patch token at dot location
            'mask': avg pool of patch tokens at masked locations
        probe_type -- one of ['mlp', 'attn'], either MLP or attentive probing
        '''
        super().__init__()
        assert probe_type in ['mlp', 'attn', 'depth'], f'Unrecognised probe_type {probe_type}!'
        assert feat_type in ['dot', 'mask', 'patch', 'cls-patch'], f'Unrecognised feat_type {feat_type}!'
        self.probe_type = probe_type
        self.feat_type = feat_type

    def forward(self, imgs, batch):
        '''
        If self.probe_type is 'mlp', then should return tensor of shape (B,C)
        If self.probe_type is 'attn', then should return tensor of shape (B,L,C)
        '''
        B, _, height, width = imgs.shape
        device = imgs.device

        if self.feat_type == 'cls-patch' or self.feat_type == 'patch':
            h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            coords = torch.stack((h_indices, w_indices), dim=0).to(device).unsqueeze(0).expand(B,-1,-1,-1)  # (B,2,H,W)
            if self.probe_type == 'mlp':
                ret = coords.mean(-1).mean(-1) # (B,2)
            else:
                ret = coords.reshape(B, 2, height*width).permute(0,2,1).contiguous() # (B,L,C=2)        

        elif self.feat_type == 'dot':
            assert 'red_dot_loc' in batch or 'green_dot_loc' in batch, 'Cannot use patch token at dot location when dot is not specified!'
            # reshape patch tokens to BCHW output format

            if 'red_dot_loc' in batch:
                red_dot_loc = batch['red_dot_loc'].float().to(device) # (B,2) torch.float32
                if self.probe_type == 'mlp':
                    ret = red_dot_loc
                else:
                    ret = red_dot_loc.unsqueeze(1) # (B,L=1,2)
            if 'green_dot_loc' in batch:
                green_dot_loc = batch['green_dot_loc'].float().to(device) # (B,2) torch.float32
                if self.probe_type == 'mlp':
                    ret -= green_dot_loc # (B,C=2)
                else:
                    ret = torch.cat([ret, green_dot_loc.unsqueeze(1)], dim=1) # (B,L=2,C=2)

        elif self.feat_type == 'mask':
            assert 'red_obj_mask' in batch or 'green_obj_mask' in batch, 'Cannot use patch token at mask locations when mask is not specified!'
            h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            coords = torch.stack((h_indices, w_indices), dim=0).to(device).unsqueeze(0).expand(B,-1,-1,-1)  # (B,2,H,W)
            if 'red_obj_mask' in batch:
                red_mask = batch['red_obj_mask'].to(device) # (B,1,H,W), float32
                if self.probe_type == 'mlp': 
                    # average mask coordinate
                    x_red = (coords * red_mask).sum(-1).sum(-1) / red_mask.sum() # (B,2)
                else:
                    # mask coordinate map
                    x_red = (coords * red_mask).reshape(B, 2, height*width).permute(0,2,1).contiguous() # (B,L,C=2)
                x_mask = x_red
            if 'green_obj_mask' in batch:
                green_mask = batch['green_obj_mask'].to(device) # (B,1,H,W), float32
                if self.probe_type == 'mlp': 
                    # average mask coordinate
                    x_green = (coords * green_mask).sum(-1).sum(-1) / green_mask.sum() # (B,2)
                    x_mask -= x_green # (B,2)
                else:
                    x_green = (coords * green_mask).reshape(B, 2, height*width).permute(0,2,1).contiguous() # (B,L,C=2)
                    x_mask = torch.cat([x_mask, x_green], dim=1) # (B,2L,C=2)
            ret = x_mask
        
        # ret always contains coords or difference of coords, so normalise
        ret = ret / (max(height, width) - 1.)
        return ret


class TimmViTFeatureExtractor(nn.Module):

    def __init__(self, model, probe_type, feat_type='cls', layer=-1, use_cls=True):
        '''
        Feature extractor that wraps a backbone ViT and returs features acc. to kwargs.

        args:
        model      -- an instance of timm.models.vision_transformer.VisionTransformer
        feat_type  -- one of ['cls', 'patch', 'dot', 'mask']
            'cls': the class token
            'patch': avg pool of patch tokens
            'dot': patch token at dot location
            'mask': avg pool of patch tokens at masked locations
        probe_type -- one of ['mlp', 'attn'], either MLP or attentive probing
        '''
        super().__init__()
        assert probe_type in ['mlp', 'attn', 'depth', 'linear'], f'Unrecognised probe_type {probe_type}!'
        assert feat_type in ['cls', 'patch', 'cls-patch', 'dot', 'mask'], f'Unrecognised feat_type {feat_type}!'
        self.model = model
        self.probe_type = probe_type
        self.feat_type = feat_type
        self.layer = layer
        self.use_cls = use_cls

    def forward(self, imgs, batch):
        '''
        If self.probe_type is 'mlp', then should return tensor of shape (B,C)
        If self.probe_type is 'attn', then should return tensor of shape (B,L,C)
        '''
        B, _, height, width = imgs.shape
        device = imgs.device
        if self.model.num_prefix_tokens:
            intermediates = self.model.forward_intermediates(imgs, indices=[self.layer], return_prefix_tokens=True, output_fmt='NCHW', intermediates_only=True)
            x, prefix_tokens = intermediates[0] # (B,C,H,W), (B,1,C)
            prefix_tokens = prefix_tokens[:,0] # (B,C)
        else:
            intermediates = self.model.forward_intermediates(imgs, indices=[self.layer], return_prefix_tokens=False, output_fmt='NCHW', intermediates_only=True)
            x = intermediates[0] # (B,C,H,W)
            prefix_tokens = None
        _, _, H, W = x.shape

        if prefix_tokens is not None and not self.use_cls:
            prefix_tokens = None

        if self.feat_type == 'cls':
            ret = prefix_tokens  # (B,C) in this case x is directly output as class token

        elif self.feat_type == 'patch':
            if self.probe_type == 'mlp' or self.probe_type == 'linear':
                ret = torch.mean(x, (2, 3)) # (B,C) avg of patch tokens
            elif self.probe_type == 'depth':
                ret = x
            else:
                ret = x.reshape(B, -1, H*W).permute(0,2,1).contiguous() # (B,L,C) all patch tokens

        elif self.feat_type == 'cls-patch':
            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens[:, :, None, None].repeat(1, 1, H, W)
                x = torch.cat((x, prefix_tokens), dim=1) # (B,2C,H,W)
            if self.probe_type == 'mlp' or self.probe_type == 'linear':
                ret = torch.mean(x, (2, 3)) # (B,C) avg of patch tokens
            elif self.probe_type == 'depth':
                ret = x
            else:
                ret = x.reshape(B, -1, H*W).permute(0,2,1).contiguous() # (B,L,C) all patch tokens

        elif self.feat_type == 'dot':
            assert 'red_dot_loc' in batch or 'green_dot_loc' in batch, 'Cannot use patch token at dot location when dot is not specified!'
            # concat cls tokens to patch tokens
            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens[:, :, None, None].repeat(1, 1, H, W)
                x = torch.cat((x, prefix_tokens), dim=1) # (B,2C,H,W)
            if 'red_dot_loc' in batch:
                red_dot_loc = batch['red_dot_loc'] # (B,2)
                x = nn.Upsample(size=(height, width), mode='bilinear')(x) # (B,3,H,W)
                x_dot = x[torch.arange(B), :, red_dot_loc[:,0], red_dot_loc[:,1]] # (B,C)
            if 'green_dot_loc' in batch:
                green_dot_loc = batch['green_dot_loc'] # (B,2)
                if self.probe_type == 'mlp' or self.probe_type == 'linear':
                    x_dot -= x[torch.arange(B), :, green_dot_loc[:,0], green_dot_loc[:,1]] # (B,C)
                else:
                    x_dot = torch.stack([x_dot, x[torch.arange(B), :, green_dot_loc[:,0], green_dot_loc[:,1]]], dim=1) # (B,L=2,C)
            ret = x_dot

        elif self.feat_type == 'mask':
            assert 'red_obj_mask' in batch or 'green_obj_mask' in batch, 'Cannot use patch token at mask locations when mask is not specified!'
            # concat cls tokens to patch tokens
            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens[:, :, None, None].repeat(1, 1, H, W)
                x = torch.cat((x, prefix_tokens), dim=1) # (B,2C,H,W)
            if 'red_obj_mask' in batch:
                red_mask = batch['red_obj_mask'].to(device) # (B,1,H,W), float32
                if self.probe_type == 'mlp' or self.probe_type == 'linear': 
                    # Upsample feature map as done in https://github.com/Championchess/phy-sd/blob/0f694922feb78f0ce1ff2d2f7beb05ed56fa0e62/SVM/depth_train_test_svm.py#L64
                    x = nn.Upsample(size=(height, width), mode='bilinear')(x) # (B,3,H,W)
                    # take mean vector of pos entries if using mlp
                    x_red = (x * red_mask).sum(-1).sum(-1) / red_mask.sum() # (B,C)
                else:
                    # Downsample the mask
                    red_mask = TF.resize(red_mask, (H,W), interpolation=InterpolationMode.NEAREST) # (B,1,28,28)
                    # masked patch tokens
                    x_red = (x * red_mask).reshape(B, -1, H*W).permute(0,2,1).contiguous() # (B,L,C)
                x_mask = x_red
            if 'green_obj_mask' in batch:
                green_mask = batch['green_obj_mask'].to(device) # (B,1,H,W), float32
                if self.probe_type == 'mlp' or self.probe_type == 'linear': 
                    # take mean vector of pos entries if using mlp
                    x_green = (x * green_mask).sum(-1).sum(-1) / green_mask.sum() # (B,C)
                    x_mask -= x_green # (B,C)
                else:
                    green_mask = TF.resize(green_mask, (H,W), interpolation=InterpolationMode.NEAREST) # (B,1,28,28)
                    x_green = (x * green_mask).reshape(B, -1, H*W).permute(0,2,1).contiguous() # (B,L,C)
                    x_mask = torch.cat([x_mask, x_green], dim=1) # (B,2L,C)
            ret = x_mask

        return ret
    

class Probe3DViTFeatureExtractor(nn.Module):

    def __init__(self, model, probe_type, feat_type='cls', layer=-1, use_cls=True):
        '''
        Feature extractor that wraps a backbone ViT and returns features acc. to kwargs.

        args:
        model      -- an instance of a model in models.probe3d_backbones
        feat_type  -- one of ['cls', 'patch', 'dot', 'mask']
            'cls': the class token
            'patch': avg pool of patch tokens
            'dot': patch token at dot location
            'mask': avg pool of patch tokens at masked locations
        probe_type -- one of ['mlp', 'attn'], either MLP or attentive probing
        '''
        super().__init__()
        assert probe_type in ['mlp', 'attn', 'depth', 'linear'], f'Unrecognised probe_type {probe_type}!'
        assert feat_type in ['cls', 'patch', 'cls-patch', 'dot', 'mask'], f'Unrecognised feat_type {feat_type}!'
        self.model = model
        self.probe_type = probe_type
        self.feat_type = feat_type
        self.layer = layer
        if feat_type == 'cls':
            assert not isinstance(model, SAM), "SAM ViT implemented doens't return cls token!"
        self.use_cls = use_cls

    def forward(self, imgs, batch):
        '''
        If self.probe_type is 'mlp', then should return tensor of shape (B,C)
        If self.probe_type is 'attn', then should return tensor of shape (B,L,C)
        '''
        B, _, height, width = imgs.shape
        device = imgs.device

        intermediates = self.model.forward_intermediates(imgs, layer=self.layer)
        x, prefix_tokens = intermediates[0] # (B,C,H,W), (B,C)
        _, _, H, W = x.shape

        if prefix_tokens is not None and not self.use_cls:
            prefix_tokens = None

        if self.feat_type == 'cls':
            ret = prefix_tokens  # (B,C) in this case x is directly output as class token

        elif self.feat_type == 'patch':
            if self.probe_type == 'mlp' or self.probe_type == 'linear':
                ret = torch.mean(x, (2, 3)) # (B,C) avg of patch tokens
            elif self.probe_type == 'depth':
                ret = x
            else:
                ret = x.reshape(B, -1, H*W).permute(0,2,1).contiguous() # (B,L,C) all patch tokens

        elif self.feat_type == 'cls-patch':
            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens[:, :, None, None].repeat(1, 1, H, W)
                x = torch.cat((x, prefix_tokens), dim=1) # (B,2C,H,W)
            if self.probe_type == 'mlp' or self.probe_type == 'linear':
                ret = torch.mean(x, (2, 3)) # (B,C) avg of patch tokens
            elif self.probe_type == 'depth':
                ret = x
            else:
                ret = x.reshape(B, -1, H*W).permute(0,2,1).contiguous() # (B,L,C) all patch tokens

        elif self.feat_type == 'dot':
            assert 'red_dot_loc' in batch or 'green_dot_loc' in batch, 'Cannot use patch token at dot location when dot is not specified!'
            # concat cls tokens to patch tokens
            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens[:, :, None, None].repeat(1, 1, H, W)
                x = torch.cat((x, prefix_tokens), dim=1) # (B,2C,H,W)
            if 'red_dot_loc' in batch:
                red_dot_loc = batch['red_dot_loc'] # (B,2)
                x = nn.Upsample(size=(height, width), mode='bilinear')(x) # (B,3,H,W)
                x_dot = x[torch.arange(B), :, red_dot_loc[:,0], red_dot_loc[:,1]] # (B,C)
            if 'green_dot_loc' in batch:
                green_dot_loc = batch['green_dot_loc'] # (B,2)
                if self.probe_type == 'mlp' or self.probe_type == 'linear':
                    x_dot -= x[torch.arange(B), :, green_dot_loc[:,0], green_dot_loc[:,1]] # (B,C)
                else:
                    x_dot = torch.stack([x_dot, x[torch.arange(B), :, green_dot_loc[:,0], green_dot_loc[:,1]]], dim=1) # (B,L=2,C)
            ret = x_dot

        elif self.feat_type == 'mask':
            assert 'red_obj_mask' in batch or 'green_obj_mask' in batch, 'Cannot use patch token at mask locations when mask is not specified!'
            # concat cls tokens to patch tokens
            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens[:, :, None, None].repeat(1, 1, H, W)
                x = torch.cat((x, prefix_tokens), dim=1) # (B,2C,H,W)
            if 'red_obj_mask' in batch:
                red_mask = batch['red_obj_mask'].to(device) # (B,1,H,W), float32
                if self.probe_type == 'mlp' or self.probe_type == 'linear': 
                    # Upsample feature map as done in https://github.com/Championchess/phy-sd/blob/0f694922feb78f0ce1ff2d2f7beb05ed56fa0e62/SVM/depth_train_test_svm.py#L64
                    x = nn.Upsample(size=(height, width), mode='bilinear')(x) # (B,3,H,W)
                    # take mean vector of pos entries if using mlp
                    x_red = (x * red_mask).sum(-1).sum(-1) / red_mask.sum() # (B,C)
                else:
                    # Downsample the mask
                    red_mask = TF.resize(red_mask, (H,W), interpolation=InterpolationMode.NEAREST) # (B,1,28,28)
                    # masked patch tokens
                    x_red = (x * red_mask).reshape(B, -1, H*W).permute(0,2,1).contiguous() # (B,L,C)
                x_mask = x_red
            if 'green_obj_mask' in batch:
                green_mask = batch['green_obj_mask'].to(device) # (B,1,H,W), float32
                if self.probe_type == 'mlp' or self.probe_type == 'linear': 
                    # take mean vector of pos entries if using mlp
                    x_green = (x * green_mask).sum(-1).sum(-1) / green_mask.sum() # (B,C)
                    x_mask -= x_green # (B,C)
                else:
                    green_mask = TF.resize(green_mask, (H,W), interpolation=InterpolationMode.NEAREST) # (B,1,28,28)
                    x_green = (x * green_mask).reshape(B, -1, H*W).permute(0,2,1).contiguous() # (B,L,C)
                    x_mask = torch.cat([x_mask, x_green], dim=1) # (B,2L,C)
            ret = x_mask

        return ret
    

class DinoViTFeatureExtractor(Probe3DViTFeatureExtractor):

    def __init__(self, model, probe_type, feat_type='cls', layer=-1, use_cls=True):
        '''
        Feature extractor that wraps a backbone ViT and returns features acc. to kwargs.

        args:
        model      -- an instance of dinov2.models.vision_transformer.DinoVisionTransformer
        feat_type  -- one of ['cls', 'patch', 'dot', 'mask']
            'cls': the class token
            'patch': avg pool of patch tokens
            'dot': patch token at dot location
            'mask': avg pool of patch tokens at masked locations
        probe_type -- one of ['mlp', 'attn'], either MLP or attentive probing
        '''
        super().__init__(model, probe_type, feat_type, layer, use_cls)