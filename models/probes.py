import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from timm.layers.attention_pool import trunc_normal_tf_
from submodules.probe3d.evals.models.probes import DepthBinPrediction, Linear

class AttentiveProbeModel(timm.layers.attention_pool.AttentionPoolLatent):
    def __init__(self, in_features: int, 
                 out_features: int = None, 
                 embed_dim: int = None, 
                 num_heads: int = 8, 
                 mlp_ratio: float = 4, 
                 qkv_bias: bool = True, 
                 qk_norm: bool = False, 
                 latent_len: int = 1, 
                 latent_dim: int = None, 
                 pos_embed: str = '', 
                 pool_type: str = 'token', 
                 norm_layer: Module | None = None, 
                 drop: float = 0,
                 num_classes=1,
                 dropout_rate=0.3):
        super().__init__(in_features, out_features, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_norm, latent_len, latent_dim, pos_embed, pool_type, norm_layer, drop)
        self.linear = torch.nn.Linear(self.latent_dim, num_classes)
        self.batchnorm = torch.nn.BatchNorm1d(self.latent_dim, affine=False, eps=1e-6)
    
    def forward(self, x):
        x = super().forward(x)
        x = self.batchnorm(x)
        x = self.linear(x)
        return x


class DoubleAttentiveProbeModel(timm.layers.attention_pool.AttentionPoolLatent):
    def __init__(self, in_features: int, 
                 out_features: int = None, 
                 embed_dim: int = None, 
                 num_heads: int = 8, 
                 mlp_ratio: float = 4, 
                 qkv_bias: bool = True, 
                 qk_norm: bool = False, 
                 latent_len: int = 1, 
                 latent_dim: int = None, 
                 pos_embed: str = '', 
                 pool_type: str = 'token', 
                 norm_layer: Module | None = None, 
                 drop: float = 0,
                 num_classes=1,
                 dropout_rate=0.3):
        super().__init__(in_features, out_features, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_norm, latent_len, latent_dim, pos_embed, pool_type, norm_layer, drop)
        embed_dim = embed_dim or in_features
        self.latent2 = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))
        self.init_weights2() # initialize weights for latent2
        self.linear = torch.nn.Linear(self.latent_dim, num_classes)

    def init_weights2(self):
        trunc_normal_tf_(self.latent2, std=self.latent_dim ** -0.5)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        B, N, C = x1.shape

        if self.pos_embed is not None:
            # this is None because input features already have pos embeds.
            x1 = x1 + self.pos_embed.unsqueeze(0).to(x1.dtype)
            x2 = x2 + self.pos_embed.unsqueeze(0).to(x2.dtype)

        q1_latent = self.latent.expand(B, -1, -1)
        q2_latent = self.latent2.expand(B, -1, -1)
        # q1 = self.q(q1_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q2 = self.q(q2_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        q1q2 = self.q(torch.cat([q1_latent, q2_latent], dim=0)).reshape(2*B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        # kv1 = self.kv(x1).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # k1, v1 = kv1.unbind(0)
        # kv2 = self.kv(x2).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # k2, v2 = kv2.unbind(0)
        kv1kv2 = self.kv(torch.cat([x1, x2], dim=0)).reshape(2*B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k1k2, v1v2 = kv1kv2.unbind(0)

        # q1, k1 = self.q_norm(q1), self.k_norm(k1)
        # q2, k2 = self.q_norm(q2), self.k_norm(k2)
        q1q2, k1k2 = self.q_norm(q1q2), self.k_norm(k1k2)

        if self.fused_attn:
            # x1 = F.scaled_dot_product_attention(q1, k1, v1)
            # x2 = F.scaled_dot_product_attention(q2, k2, v2)
            x1x2 = F.scaled_dot_product_attention(q1q2, k1k2, v1v2)
        else:
            # q1 = q1 * self.scale
            # attn1 = q1 @ k1.transpose(-2, -1)
            # attn1 = attn1.softmax(dim=-1)
            # x1 = attn1 @ v1
            # q2 = q2 * self.scale
            # attn2 = q2 @ k2.transpose(-2, -1)
            # attn2 = attn2.softmax(dim=-1)
            # x2 = attn2 @ v2
            q1q2 = q1q2 * self.scale
            attn = q1q2 @ k1k2.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x1x2 = attn @ v1v2

        # x1 = x1.transpose(1, 2).reshape(B, self.latent_len, C)
        # x2 = x2.transpose(1, 2).reshape(B, self.latent_len, C)
        # x1 = self.proj(x1)
        # x2 = self.proj(x2)
        # x = self.proj_drop(x)
        x1x2 = x1x2.transpose(1, 2).reshape(2*B, self.latent_len, C)
        x1x2 = self.proj(x1x2)
        x1x2 = self.proj_drop(x1x2)

        x1x2 = x1x2 + self.mlp(self.norm(x1x2))

        # optional pool if latent seq_len > 1 and pooled output is desired
        x1x2 = x1x2[:, 0]
        x1, x2 = x1x2.chunk(2, dim=0)

        x = self.linear(x1-x2)
        return x


class LinearProbeModel(torch.nn.Module):
    def __init__(self, in_features, num_classes=1, dropout_rate=0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, num_classes)
    def forward(self, x):
        x = self.linear(x)
        return x
    

class MlpProbeModel(torch.nn.Module):
    def __init__(self, in_features, num_classes=1, dropout_rate=0.3, **kwargs):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, in_features)
        self.act = torch.nn.GELU()
        self.drop1 = torch.nn.Dropout(p=dropout_rate)
        self.fc2 = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x


class DepthProbeModel(nn.Module):
    def __init__(
        self,
        in_features,
        dropout_rate=0.3,
        min_depth=0.001,
        max_depth=10,
    ):
        super().__init__()

        output_dim = 256
        self.conv = nn.Conv2d(in_features, output_dim, kernel_size=1)
        self.predict = DepthBinPrediction(min_depth, max_depth, n_bins=output_dim)

    def forward(self, x):
        """Prediction each pixel."""
        x = F.interpolate(x, scale_factor=4, mode="bilinear")
        x = self.conv(x)
        depth = self.predict(x)
        return depth


class LinearModelMulti(torch.nn.Module):
    def __init__(self, in_features, num_classes=1, dropout_rate=0.3):
        super().__init__()
        print(in_features)
        self.linear = torch.nn.Sequential(torch.nn.BatchNorm1d(in_features[0], affine=False, eps=1e-6), 
                                            torch.nn.Dropout(p=dropout_rate),
                                            torch.nn.Linear(in_features[0], in_features[1]),
                                            torch.nn.BatchNorm1d(in_features[1], affine=False, eps=1e-6),
                                            torch.nn.Dropout(p=dropout_rate),
                                            torch.nn.Linear(in_features[1], num_classes))
    def forward(self, x):
        return self.linear(x)