from submodules.openlrm.openlrm.utils.hf_hub import wrap_model_hub
from submodules.openlrm.openlrm.models import ModelLRM
from submodules.openlrm.openlrm.datasets.cam_utils import build_camera_principle, create_intrinsics
import torch
import transformers
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode


class LRMBackbone(torch.nn.Module):

    def __init__(self, arch='vitb', output='dense-cls') -> None:
        '''
        layer in [0,1,...,23]
        feat dim = 1536 if layer < 12 else 2304
        '''
        super().__init__()

        model_class = wrap_model_hub(ModelLRM)
        model_dict = {
            'vits': "zxhezexin/openlrm-mix-small-1.1",
            'vitb': "zxhezexin/openlrm-mix-base-1.1",
            'vitl': "zxhezexin/openlrm-mix-large-1.1",
        }
        self.model = model_class.from_pretrained(model_dict[arch])

        # define default source camera for all inputs
        source_camera_dist = 2.0
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -source_camera_dist],
            [0, 1, 0, 0],
        ]], dtype=torch.float32)
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
        ).unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        self.source_camera = torch.nn.Parameter(source_camera, requires_grad=False)

        cfg = transformers.PretrainedConfig.from_pretrained(model_dict[arch])
        if output == 'dense-cls':
            self.feat_dim = [cfg.encoder_feat_dim*2]*len(self.model.encoder.model.blocks) + [cfg.transformer_dim*3]*cfg.transformer_layers
        else:
            self.feat_dim = [cfg.encoder_feat_dim]*len(self.model.encoder.model.blocks) + [cfg.transformer_dim*3]*cfg.transformer_layers

    def forward_intermediates(self, x, layer):
        # define camera: [N, D_cam_raw]
        camera = self.source_camera.repeat(x.shape[0], 1)

        return self.model.forward_intermediates(x, camera, layer)
    

def get_lrm_transform():
    return Compose(
        [
            Resize(size=(518,518), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToTensor()
        ]
    )