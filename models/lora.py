import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, Compose, Resize, ToTensor, Normalize
from timm.models.vision_transformer import VisionTransformer
from models.vit import TimmViTBackbone
from models.probe3d_backbones import CLIPBackbone
from utils import instantiate_from_config

class _LoRA_qkv(nn.Module):
    """In Dinov2 it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


class TimmViTLoRA(nn.Module):
    """Applies low-rank adaptation to Dinov2 model's image encoder.
    adapted from https://github.com/BeileiCui/SurgicalDINO/blob/main/surgicaldino.py
    """
    def __init__(
            self, 
            backbone: VisionTransformer = None,
            r=4,
            num_classes: int = None,
            train_with_cls=False,
            train_upsample_factor=2,
            task_head='linear',
            lora_layer=None,
            ckpt=None,
            use_orig_model=False,
            assemble_mode = 'concat'
    ):
        super(TimmViTLoRA, self).__init__()

        assert r > 0

        self.assemble_mode = assemble_mode
        if assemble_mode == 'noise':
            self.noise_model = nn.Conv2d(3, 768, kernel_size=1, bias=False)
            nn.init.normal_(self.noise_model.weight)
            for p in self.noise_model.parameters():
                p.requires_grad = False
        if use_orig_model:
            assert 'target' in backbone, 'backbone must be a VisionTransformer/TimmViTBackbone instance or a dict with target, params keys'
            self.orig_model = instantiate_from_config(backbone)
            for param in self.orig_model.parameters():
                param.requires_grad = False

        if type(backbone) not in [VisionTransformer, TimmViTBackbone]:
            assert 'target' in backbone, 'backbone must be a VisionTransformer/TimmViTBackbone instance or a dict with target, params keys'
            backbone = instantiate_from_config(backbone)

        self.train_with_cls = train_with_cls
        self.num_classes = num_classes
        self.train_upsample_factor = train_upsample_factor

        self.embed_dim = backbone.embed_dim if hasattr(backbone, 'embed_dim') else backbone.vit.embed_dim

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # freeze first
        for param in backbone.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        blocks = backbone.blocks if hasattr(backbone, 'blocks') else backbone.vit.blocks

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(blocks)))  # Only apply lora to the image encoder by default

        for t_layer_i, blk in enumerate(blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.backbone = backbone

        # Task head
        dim_mult = 2 if self.train_with_cls else 1
        task_head_input_dim = self.embed_dim * dim_mult
        if task_head == 'linear':
            self.task_head = nn.Linear(task_head_input_dim, num_classes)
        elif task_head == 'mlp':
            self.task_head = nn.Sequential(
                nn.Linear(task_head_input_dim, task_head_input_dim),
                nn.GELU(),
                nn.Linear(task_head_input_dim, num_classes)
            )
        else:
            print('Task head type', task_head, 'not recognized')
        
        if ckpt:
            state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_A_linear.weight = nn.Parameter(saved_tensor)
            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_B_linear.weight = nn.Parameter(saved_tensor)
            print('Loaded LoRA weights from checkpoint:', ckpt)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        merged_dict = {**a_tensors, **b_tensors}
        return merged_dict

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x, batch):
        intermediates = self.backbone.forward_intermediates(x, indices=self.lora_layer, return_prefix_tokens=True, output_fmt='NCHW', intermediates_only=True, stop_early=True)
        x, prefix_tokens = intermediates[0] # (B,C,H,W), (B,1,C)
        prefix_tokens = prefix_tokens[:,0] # (B,C)
        _, _, H, W = x.shape

        # Extract task specific features
        if 'red_obj_mask' in batch or 'green_obj_mask' in batch:
            new_feat_HW = (H*self.train_upsample_factor, W*self.train_upsample_factor)
            x = TF.resize(x, new_feat_HW, interpolation=InterpolationMode.BILINEAR) # (B,C,H,W)
            if 'red_obj_mask' in batch:
                red_mask = batch['red_obj_mask'].to(x.device) # (B,1,height, width)
                red_mask = TF.resize(red_mask, new_feat_HW, interpolation=InterpolationMode.NEAREST) # (B,1,H,W)
                x_out = (x * red_mask).sum(-1).sum(-1) / red_mask.sum() # (B,C)
            if 'green_obj_mask' in batch:
                green_mask = batch['green_obj_mask'].to(x.device) # (B,1,height, width)
                green_mask = TF.resize(green_mask, new_feat_HW, interpolation=InterpolationMode.NEAREST) # (B,1,H,W)
                x_out -= (x * green_mask).sum(-1).sum(-1) / green_mask.sum() # (B,C)
        else:
            x_out = x.mean(-1).mean(-1) # (B,C)
        
        if self.train_with_cls:
            x_out = torch.cat([x_out, prefix_tokens], dim=1) # (B,2C+c)

        return self.task_head(x_out)
    
    def forward_intermediates(self, x, *args, **kwargs):
        if self.assemble_mode != 'noise':
            intermediates = self.backbone.forward_intermediates(x, indices=self.lora_layer, return_prefix_tokens=True, output_fmt='NCHW', intermediates_only=True, stop_early=True)
        if hasattr(self, 'orig_model'):
            orig_intermediates = self.orig_model.forward_intermediates(x, indices=self.lora_layer, return_prefix_tokens=True, output_fmt='NCHW', intermediates_only=True, stop_early=True)
            if self.assemble_mode == 'concat':
                return [(torch.cat([intermediates[0][0], orig_intermediates[0][0]], dim=1), 
                         torch.cat([intermediates[0][1], orig_intermediates[0][1]], dim=2))]
            elif self.assemble_mode == 'add':
                return [(intermediates[0][0] + orig_intermediates[0][0], 
                         intermediates[0][1] + orig_intermediates[0][1])]
            elif self.assemble_mode == 'noise':
                return [(torch.cat([self.noise_model(nn.functional.interpolate(x, size=orig_intermediates[0][0].shape[-2:])), orig_intermediates[0][0]], dim=1), 
                         torch.cat([torch.randn_like(orig_intermediates[0][1], device=orig_intermediates[0][1].device), orig_intermediates[0][1]], dim=2))]
        return intermediates


def get_dinov2adapter_transform():
    return Compose([
        Resize(size=(518, 518), interpolation=InterpolationMode.BICUBIC, antialias=True),
        ToTensor(),
        Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
    ])


class _LoRA_mlp(nn.Module):
    def __init__(
            self,
            mlp: nn.Module,
            linear_a_1: nn.Module,
            linear_b_1: nn.Module,
            linear_a_2: nn.Module,
            linear_b_2: nn.Module,
    ):
        super().__init__()
        self.mlp = mlp
        self.linear_a_1 = linear_a_1
        self.linear_b_1 = linear_b_1
        self.linear_a_2 = linear_a_2
        self.linear_b_2 = linear_b_2

    def forward(self, x):
        res_1 = self.mlp.c_fc(x)
        new_res_1 = self.linear_b_1(self.linear_a_1(x))
        res_1 += new_res_1

        act = self.mlp.gelu(res_1)

        res_2 = self.mlp.c_proj(act)
        new_res_2 = self.linear_b_2(self.linear_a_2(act))
        res_2 += new_res_2
        return res_2

class CLIPLoRA(nn.Module):
    """Applies low-rank adaptation to Dinov2 model's image encoder.
    adapted from https://github.com/BeileiCui/SurgicalDINO/blob/main/surgicaldino.py
    """
    def __init__(
            self, 
            backbone: CLIPBackbone = None,
            r=4,
            num_classes: int = None,
            train_with_cls=False,
            train_upsample_factor=7,
            task_head='mlp',
            lora_layer=[11,],
            train_conv=False,
            ckpt=None,
            use_orig_model=False,
            assemble_mode = 'concat'
    ):
        super(CLIPLoRA, self).__init__()

        assert r > 0

        self.assemble_mode = assemble_mode
        if assemble_mode == 'noise':
            self.noise_model = nn.Conv2d(3, 768, kernel_size=1, bias=False)
            nn.init.normal_(self.noise_model.weight)
            for p in self.noise_model.parameters():
                p.requires_grad = False
        if use_orig_model:
            assert 'target' in backbone, 'backbone must be a VisionTransformer/TimmViTBackbone instance or a dict with target, params keys'
            self.orig_model = instantiate_from_config(backbone)
            for param in self.orig_model.parameters():
                param.requires_grad = False

        if type(backbone) not in [CLIPBackbone]:
            assert 'target' in backbone, 'backbone must be a VisionTransformer/TimmViTBackbone instance or a dict with target, params keys'
            backbone = instantiate_from_config(backbone)

        self.train_with_cls = train_with_cls
        self.num_classes = num_classes
        self.train_upsample_factor = train_upsample_factor
        self.train_conv = train_conv

        self.embed_dim = backbone.visual.transformer.width

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # freeze first
        for param in backbone.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        blocks = backbone.visual.transformer.resblocks

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(blocks)))  # Only apply lora to the image encoder by default

        for t_layer_i, blk in enumerate(blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_linear = blk.mlp.c_fc
            self.dim = w_linear.in_features
            self.mid_dim = w_linear.out_features
            w_a_linear_1 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_1 = nn.Linear(r, self.mid_dim, bias=False)
            w_a_linear_2 = nn.Linear(self.mid_dim, r, bias=False)
            w_b_linear_2 = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_1)
            self.w_Bs.append(w_b_linear_1)
            self.w_As.append(w_a_linear_2)
            self.w_Bs.append(w_b_linear_2)
            blk.mlp = _LoRA_mlp(
                blk.mlp,
                w_a_linear_1,
                w_b_linear_1,
                w_a_linear_2,
                w_b_linear_2,
            )
        self.reset_parameters()
        self.backbone = backbone
        if self.train_conv:
            # unfreeze conv weights
            for param in self.backbone.visual.conv1.parameters():
                param.requires_grad = True

        # Task head
        dim_mult = 2 if self.train_with_cls else 1
        task_head_input_dim = self.embed_dim * dim_mult
        if task_head == 'linear':
            self.task_head = nn.Linear(task_head_input_dim, num_classes)
        elif task_head == 'mlp':
            self.task_head = nn.Sequential(
                nn.Linear(task_head_input_dim, task_head_input_dim),
                nn.GELU(),
                nn.Linear(task_head_input_dim, num_classes)
            )
        else:
            print('Task head type', task_head, 'not recognized')
        
        if ckpt:
            state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_A_linear.weight = nn.Parameter(saved_tensor)
            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_B_linear.weight = nn.Parameter(saved_tensor)
            print('Loaded LoRA weights from checkpoint:', ckpt)
            if self.train_conv:
                self.backbone.visual.conv1.load_state_dict(state_dict['conv1'], strict=True)
                print('Loaded Conv1 weights from checkpoint:', ckpt)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        merged_dict = {**a_tensors, **b_tensors}
        if self.train_conv:
            merged_dict.update({'conv1': self.backbone.visual.conv1.state_dict()})
        return merged_dict

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x, batch):
        intermediates = self.backbone.forward_intermediates(x, layer=self.lora_layer[-1])
        x, prefix_tokens = intermediates[0] # (B,C,H,W), (B,1,C)
        prefix_tokens = prefix_tokens[:,0] # (B,C)
        _, _, H, W = x.shape

        # Extract task specific features
        if 'red_obj_mask' in batch or 'green_obj_mask' in batch:
            new_feat_HW = (H*self.train_upsample_factor, W*self.train_upsample_factor)
            x = TF.resize(x, new_feat_HW, interpolation=InterpolationMode.BILINEAR) # (B,C,H,W)
            if 'red_obj_mask' in batch:
                red_mask = batch['red_obj_mask'].to(x.device) # (B,1,height, width)
                red_mask = TF.resize(red_mask, new_feat_HW, interpolation=InterpolationMode.NEAREST) # (B,1,H,W)
                x_out = (x * red_mask).sum(-1).sum(-1) / red_mask.sum() # (B,C)
            if 'green_obj_mask' in batch:
                green_mask = batch['green_obj_mask'].to(x.device) # (B,1,height, width)
                green_mask = TF.resize(green_mask, new_feat_HW, interpolation=InterpolationMode.NEAREST) # (B,1,H,W)
                x_out -= (x * green_mask).sum(-1).sum(-1) / green_mask.sum() # (B,C)
        else:
            x_out = x.mean(-1).mean(-1) # (B,C)
        
        if self.train_with_cls:
            x_out = torch.cat([x_out, prefix_tokens], dim=1) # (B,2C+c)

        return self.task_head(x_out)
    
    def forward_intermediates(self, x, *args, **kwargs):
        if self.assemble_mode != 'noise':
            intermediates = self.backbone.forward_intermediates(x, layer=self.lora_layer[-1])
        if hasattr(self, 'orig_model'):
            orig_intermediates = self.orig_model.forward_intermediates(x, layer=self.lora_layer[-1])
            if self.assemble_mode == 'concat':
                return [(torch.cat([intermediates[0][0], orig_intermediates[0][0]], dim=1), 
                         torch.cat([intermediates[0][1], orig_intermediates[0][1]], dim=1))]
            elif self.assemble_mode == 'add':
                return [(intermediates[0][0] + orig_intermediates[0][0], 
                         intermediates[0][1] + orig_intermediates[0][1])]
            elif self.assemble_mode == 'noise':
                return [(torch.cat([self.noise_model(nn.functional.interpolate(x, size=orig_intermediates[0][0].shape[-2:])), orig_intermediates[0][0]], dim=1), 
                         torch.cat([torch.randn_like(orig_intermediates[0][1], device=orig_intermediates[0][1].device), orig_intermediates[0][1]], dim=1))]
        return intermediates
    
def get_cliplora_transform():
    return Compose(
        [
            Resize(size=(512,512), interpolation=InterpolationMode.LANCZOS, antialias=True),
            ToTensor(),
            Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ]
    )