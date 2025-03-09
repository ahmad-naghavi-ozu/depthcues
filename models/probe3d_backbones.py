import torch
import einops as E
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from submodules.probe3d.evals.models.sam import SAM
from submodules.probe3d.evals.models.dino import DINO
from submodules.probe3d.evals.models.ibot import iBOT, center_padding
from submodules.probe3d.evals.models.stablediffusion import DIFT, interpolate
from submodules.probe3d.evals.models.mae import MAE
from submodules.probe3d.evals.models.clip import CLIP, resize_pos_embed
from submodules.probe3d.evals.models.convnext import ConvNext, F
from submodules.probe3d.evals.models.siglip import SigLIP, resample_abs_pos_embed
from submodules.probe3d.evals.models.midas_final import make_beit_backbone
from submodules.probe3d.evals.models.deit import DeIT

class SAMBackbone(SAM):
    def __init__(self, arch, output="dense", layer=-1, return_multilayer=False):
        super().__init__(arch, output, layer, return_multilayer)

    def forward_intermediates(self, x, layer):
        _, _, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, f"{h}, {w}"

        if h != self.image_size[0] or w != self.image_size[1]:
            self.resize_pos_embed(image_size=(h, w))

        # run vit
        x = self.vit.patch_embed(x)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed

        embeds = []
        max_index = len(self.vit.blocks)-1 if layer == -1 else layer
        take_indices = {max_index}
        blocks = self.vit.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                embeds.append(x)

        outputs = []
        for i, x_i in enumerate(embeds):
            dense_tokens = x_i.permute(0, 3, 1, 2).contiguous()
            outputs.append((dense_tokens, None))

        return outputs


class iBOTBackbone(iBOT):
    def __init__(self, model_type="base_in22k", output="dense-cls", layer=-1, return_multilayer=False):
        '''
        layer in [0,1,...,11]
        feat dim is [768*2]
        '''
        super().__init__(model_type, output, layer, return_multilayer)
        self.output = output

    def forward_intermediates(self, images, layer):
        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size
        self.h = h

        x = self.vit.prepare_tokens(images)

        blocks = self.vit.blocks[:layer + 1]
        for i, blk in enumerate(blocks):
            x = blk(x)

        cls_tok = x[:, 0] # (B,C)
        spatial = x[:, 1:] # (B,L,C)
        dense_tokens = E.rearrange(spatial, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        if self.output == 'dense-cls':
            return [(dense_tokens, cls_tok)]
        else:
            return [(dense_tokens, None)]


class DINOBackbone(DINO):
    def __init__(self, dino_name="dino", model_name="vitb16", output="dense-cls", layer=-1, return_multilayer=False):
        '''
        layer in [0,1,...,11]
        feat dim is [768*2]
        '''
        super().__init__(dino_name, model_name, output, layer, return_multilayer)
    
    def forward_intermediates(self, images, layer):
        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        blocks = self.vit.blocks[:layer + 1]
        for i, blk in enumerate(blocks):
            x = blk(x)

        num_spatial = h * w
        cls_tok = x[:, 0]
        spatial = x[:, -1 * num_spatial :]
        dense_tokens = E.rearrange(spatial, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        return [(dense_tokens, cls_tok)]
    

class MidasBackbone(torch.nn.Module):
    def __init__(self, layer=-1, output="dense", midas=True, return_multilayer=False):
        '''
        layer in [0,1,...,23]
        feat dim is [1024]
        '''
        super().__init__()
        self.model = make_beit_backbone(layer, output, midas, return_multilayer)
    
    def forward_intermediates(self, x, layer):
        # update shapes
        h, w = x.shape[2:]
        emb_hw = (h // self.model.patch_size, w // self.model.patch_size)
        # assert h == w, f"BeIT can only handle square images, not ({h}, {w})."
        if (h, w) != self.model.image_size:
            self.model.image_size = (h, w)
            self.model.patch_embed.img_size = (h, w)
            self.model.pos_embed.data = resize_pos_embed(self.model.pos_embed[0], emb_hw, True)[None]

        # actual forward from beit
        x = self.model.patch_embed(x)
        x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.model.pos_embed

        x = self.model.norm_pre(x)

        blocks = self.model.blocks[:layer+1]
        for i, blk in enumerate(blocks):
            x = blk(x)

        dense_tokens = E.rearrange(x[:, 1:], "b (h w) c -> b c h w", h=emb_hw[0], w=emb_hw[1]).contiguous()

        return [(dense_tokens, None)]


class DeITBackbone(DeIT):
    def __init__(self, model_size="base", img_size=384, patch_size=16, output="dense", layer=-1, return_multilayer=False):
        '''
        layer in [0,1,...,11]
        feat dim is [768]
        '''
        super().__init__(model_size, img_size, patch_size, output, layer, return_multilayer)

    def forward_intermediates(self, images, layer):
        B, _, h, w = images.shape
        h, w = h // self.patch_size, w // self.patch_size

        if (h, w) != self.embed_size:
            self.embed_size = (h, w)
            self.vit.pos_embed.data = resize_pos_embed(
                self.vit.pos_embed[0], self.embed_size, False
            )[None, :, :]

        x = self.vit.patch_embed(images)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = x + self.vit.pos_embed
        x = torch.cat((cls_tokens, x), dim=1)

        blocks = self.vit.blocks[:layer+1]
        for i, blk in enumerate(blocks):
            x = blk(x)

        dense_tokens = E.rearrange(x[:, 1:], "b (h w) c -> b c h w", h=h, w=w).contiguous()

        return [(dense_tokens, None)]


class StableDiffusionBackbone(DIFT):
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", time_step=1, output="dense", layer=1, return_multilayer=True):
        '''
        layer in [0,1,2,3]
        feat dim is [1280, 1280, 640, 320]
        '''
        super().__init__(model_id, time_step, output, layer, return_multilayer)

    def forward_intermediates(self, images, layer):

        batch_size = images.shape[0]

        # handle prompts
        prompts = ["" for _ in range(batch_size)]

        assert len(prompts) == batch_size

        spatial = self.dift.forward(
            images, prompts=prompts, t=self.time_step, up_ft_index=self.up_ft_index
        )
        h, w = images.shape[2] // self.patch_size, images.shape[3] // self.patch_size
        spatial = [spatial[layer]]

        spatial = [(interpolate(x.contiguous(), (h, w)), None) for x in spatial]

        return spatial


class MAEBackbone(MAE):
    def __init__(self, checkpoint="facebook/vit-mae-base", output="dense", layer=-1, return_multilayer=False):
        '''
        layer in [0,1,...,11]
        feat dim = 768
        '''
        super().__init__(checkpoint, output, layer, return_multilayer)

    def forward_intermediates(self, images, layer):
        # check if positional embeddings are correct
        if self.image_size != images.shape[-2:]:
            self.resize_pos_embed(images.shape[-2:])

        # from MAE implementation
        head_mask = self.vit.get_head_mask(None, self.vit.config.num_hidden_layers)

        # ---- hidden ----
        embedding_output = self.embed_forward(self.vit.embeddings, images)
        encoder_outputs = self.vit.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=self.vit.config.output_attentions,
            output_hidden_states=True,
            return_dict=self.vit.config.return_dict,
        )

        outputs = []
        x_i = encoder_outputs.hidden_states[layer]
        dense_tokens = E.rearrange(x_i[:, 1:], "b (h w) c -> b c h w", h=self.feat_h, w=self.feat_w).contiguous()
        outputs.append((dense_tokens, None))

        return outputs


class CLIPBackbone(CLIP):
    def __init__(self, arch="ViT-B-16", checkpoint="laion2b_s34b_b88k", output="dense-cls", layer=-1, return_multilayer=False):
        super().__init__(arch, checkpoint, output, layer, return_multilayer)

    def forward_intermediates(self, images, layer):
        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        # clip stuff
        x = self.visual.conv1(images)
        x_hw = x.shape[-2:]
        x = E.rearrange(x, "b c h w -> b (h w) c")

        # concat cls token
        _cls_embed = E.repeat(self.visual.class_embedding, "c -> b 1 c", b=x.shape[0])
        x = torch.cat([_cls_embed.to(x.dtype), x], dim=1)

        # add pos embed
        pos_embed = resize_pos_embed(self.visual.positional_embedding, x_hw)
        x = self.visual.ln_pre(x + pos_embed.to(x.dtype))

        blocks = self.visual.transformer.resblocks[:layer+1]
        for i, blk in enumerate(blocks):
            x = blk(x)
        
        class_tokens = x[:, 0]
        dense_tokens = E.rearrange(x[:, 1:], "b (h w) c -> b c h w", h=out_hw[0], w=out_hw[1]).contiguous()

        return [(dense_tokens, class_tokens)]

def get_clip_transform():
    return Compose(
        [
            Resize(size=(512,512), interpolation=InterpolationMode.LANCZOS, antialias=True),
            ToTensor(),
            Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ]
    )


class ConvNextBackbone(ConvNext):
    def __init__(self, arch="convnext_base", checkpoint="in22k", output="dense", layer=-1, return_multilayer=True):
        '''
        layer in [0,1,2,3]
        feat dim = check model
        '''
        super().__init__(arch, checkpoint, output, layer, return_multilayer)

    def forward_intermediates(self, images, layer):
        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        # clip stuff
        x = self.visual.stem(images)

        stages = self.visual.stages[:layer+1]
        for i, stage in enumerate(stages):
            x = stage(x)

        dense_tokens = F.interpolate(x, out_hw, mode="bilinear").contiguous()

        return [(dense_tokens, None)]


class SigLIPBackbone(SigLIP):
    def __init__(
        self,
        checkpoint="vit_base_patch16_siglip_384.webli",
        output="dense",
        layer=-1,
        resize_pos_embeds=True,
        pretrained=True,
        return_multilayer=False,
    ):
        '''
        layer in [0,1,...,11]
        feat dim = 768
        '''
        super().__init__(checkpoint, output, layer, resize_pos_embeds, pretrained, return_multilayer)

    def forward_intermediates(self, images, layer):
        images = center_padding(images, self.patch_size)
        _, _, img_h, img_w = images.shape

        # get embed h, w
        assert img_h % self.patch_size == 0
        assert img_w % self.patch_size == 0
        out_h, out_w = img_h // self.patch_size, img_w // self.patch_size

        if self.resize_pos_embeds and (out_h, out_w) != self.embed_size:
            self.embed_size = (out_h, out_w)
            self.vit.pos_embed.data = resample_abs_pos_embed(
                self.vit.pos_embed, (out_h, out_w), num_prefix_tokens=0
            )

        x = self.vit.patch_embed(images)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        blocks = self.vit.blocks[:layer+1]
        for i, blk in enumerate(blocks):
            x = blk(x)

        dense_tokens = E.rearrange(x, "b (h w) c -> b c h w", h=out_h, w=out_w).contiguous()

        return [(dense_tokens, None)]


def get_siglip_transform():
    return Compose(
        [
            Resize(size=(512,512), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )


if __name__ == '__main__':
    print(SAM(arch='vit_l', output='dense', layer=-1))
    print(DINO(dino_name='dinov2', model_name='vitl14', output='dense-cls', layer=-1))