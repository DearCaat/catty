  # --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial
import re
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import Block as ViT_Block
from timm.models import create_model
from timm.models.registry import register_model

# from .swin_mim import *
from .vit_mim import *

# sys.path.append("..")
from ..utils import patchify,unpatchify


class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class MIM(nn.Module):
    def __init__(self, encoder, encoder_stride,use_mae=False,norm_pix_loss=False,decoder=None):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        # we take the mask_token out of encoder and decoder since the the mask_token will not be fed into the encoder in mae 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim))

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

        self.use_mae = use_mae
        self.norm_pix_loss = norm_pix_loss

        if self.use_mae:
            self.decoder = nn.ModuleList([
                ViT_Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(decoder_depth)])
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.encoder.num_features,
                    out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
                nn.PixelShuffle(self.encoder_stride),
            )

    def forward(self, x, return_img_rec=False,keep_mask=False,unmask=None,require_unmask=False):
        z,logits,mask,unmask = self.encoder(x,keep_mask,unmask=unmask,require_unmask=require_unmask)
        z = z.transpose(1, 2)
        B, C, L = z.shape
        H = W = int(L ** 0.5)
        z = z.reshape(B, C, H, W)
        x_rec = self.decoder(z)
        # print(x_rec.size())
        #x_rec = self.decoder(unpatchify(z,self.encoder.patch_size))
        if self.training or keep_mask:
            target = patchify(x,self.encoder_stride)
            
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

            if self.use_mae:
                loss_mim = (z - target) ** 2
                loss_mim = loss_mim.mean(dim=-1)  # [N, L], mean loss per patch

                loss_mim = (loss_mim * mask).sum() / mask.sum()  # mean loss on removed patches
            else:
                mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
                
                target = unpatchify(target,self.encoder_stride)
                loss_mim = F.l1_loss(target, x_rec, reduction='none')

                loss_mim = (loss_mim * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        else:
            loss_mim = 0.
        if return_img_rec:
            return logits,loss_mim,x_rec,unmask
        else:
            return logits,loss_mim,unmask

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

@register_model
def mim_swin_base_patch4_window12_384_in22k(pretrained=False, **kwargs):
    encoder = create_model(
        model_name='swin_mim_base_patch4_window12_384_in22k',
        pretrained=pretrained,
        **kwargs
    )
    return MIM(encoder,32,use_mae=kwargs['use_mae'],norm_pix_loss=kwargs['norm_pix_loss'])

def __teacher_init(config,teacher):
    try:
        cpt = torch.load(config.MIM.TEACHER_INIT, map_location = 'cpu')
    except:
        cpt = torch.load(os.path.join(config.OUTPUT,'model',config.MIM.TEACHER_INIT), map_location = 'cpu')

    std = cpt['state_dict']
    teacher.load_state_dict(std, strict=False)

@register_model
def mim_vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    norm_pix_loss = kwargs.pop('norm_pix_loss')
    config = kwargs.pop('config')
    encoder = create_model(
        model_name='vit_mim_base_patch16_224_in21k',
        pretrained=pretrained,
        mask_ratio=config.MIM.MASK_RATIO,
        mask_patch_size=config.MIM.MASK_PATCH_SIZE,
        topk=config.MIM.TOPK,
        select_fn=config.MIM.SELECT_FN,
        **kwargs
    )
    return MIM(encoder,16,use_mae=kwargs['use_mae'],norm_pix_loss=norm_pix_loss)


@register_model
def clmim(**kwargs):
    config = kwargs['config']
    assert config.MODEL.BACKBONE != ''

    student = kwargs['student']
    teacher = kwargs['teacher']

    if config.MIM.TEACHER_INIT is not None:
        __teacher_init(config,teacher)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    
    return {'main':student,'teacher':teacher}

def build_mim(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        encoder_stride = 32
    elif model_type == 'vit':
        encoder = VisionTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = MIM(encoder=encoder, encoder_stride=encoder_stride)

    return model