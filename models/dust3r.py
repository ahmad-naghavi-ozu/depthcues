from submodules.dust3r.dust3r.model import AsymmetricCroCo3DStereo
import torch
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode, Normalize

class Dust3rBackbone(torch.nn.Module):
    def __init__(self,) -> None:
        '''
        layer in [0,35], test [17, 23, 32, 35]
        feat dim = 768 if 1024 layer < 24 else 
        '''
        super().__init__()

        self.model = AsymmetricCroCo3DStereo.from_pretrained('naver/DUSt3R_ViTLarge_BaseDecoder_512_linear')
        self.feat_dim = [1024]*len(self.model.enc_blocks) + [768]*len(self.model.dec_blocks)
    
    def forward_intermediates(self, x, layer):

        b, c, h, w = x.shape

        view1 = {'img': x, 
                 'true_shape': torch.tensor([h, w], dtype=torch.int32, device=x.device).repeat(b, 1)}

        return self.model.forward_intermediates(view1, layer)


def get_dust3r_transform():
    return Compose(
        [
            Resize(size=(512,512), interpolation=InterpolationMode.LANCZOS, antialias=True),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )