  # --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from copy import deepcopy
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

from .swin_mim import *
from .vit_mim import *

# sys.path.append("..")
from ..utils import patchify,unpatchify,DINOHead

sys.path.append("...")
from utils import ModelEmaV3,cosine_scheduler

class MIM(nn.Module):
    def __init__(self, encoder, encoder_stride,use_mae=False,norm_pix_loss=False,decoder=None,head_out_dim=None,decoder_cl=False,decoder_cl_nlayer=3,decoder_cl_use_bn=False):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        # we take the mask_token out of encoder and decoder since the the mask_token will not be fed into the encoder in mae 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim))

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

        self.use_mae = use_mae
        self.norm_pix_loss = norm_pix_loss

        self.decoder_mim = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )
        if decoder_cl:
            self.decoder_cl = DINOHead(
                in_dim=self.encoder.num_features,
                out_dim=head_out_dim,
                nlayers=decoder_cl_nlayer,
                use_bn=decoder_cl_use_bn,
            )
        else:
            self.decoder_cl = nn.Identity()

    def forward(self, x, return_img_rec=False,keep_mask=False,unmask=None,require_unmask=False,need_mask=False,need_rec=False,need_cl=False):

        global_feat,local_feats,logits_cls,mask,unmask = self.encoder(x,keep_mask,unmask=unmask,require_unmask=require_unmask,no_mask=not need_mask)

        if need_rec:
            z = local_feats
            z = z.transpose(1, 2)
            B, C, L = z.shape
            H = W = int(L ** 0.5)
            z = z.reshape(B, C, H, W)
            x_rec = self.decoder_mim(z)
            # x_rec = self.decoder_cl(local_feats)
        else:
            x_rec = None
            pass
            #x_rec = self.decoder_cl(local_feats)

        if need_cl:
            logits_cl = self.decoder_cl(global_feat)
        else:
            logits_cl = None
        # print(x_rec.size())
        # x_rec = self.decoder(unpatchify(z,self.encoder.patch_size))
        if (self.training or keep_mask) and need_rec:
            target = patchify(x,self.encoder_stride)
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

            mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
                
            target = unpatchify(target,self.encoder_stride)
            loss_mim_pixel = F.l1_loss(target, x_rec, reduction='none')

            loss_mim_pixel = (loss_mim_pixel * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        else:
            loss_mim_pixel = 0.

        return logits_cls,logits_cl,x_rec,unmask,mask,loss_mim_pixel

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

    res = teacher.encoder.load_state_dict(std, strict=False)
    print("model init")

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
        drop_mask_token_in_encoder=config.MIM.DROP_MASK,
        **kwargs
    )
    return MIM(encoder,16,use_mae=kwargs['use_mae'],norm_pix_loss=norm_pix_loss,head_out_dim = config.MIM.HEAD_OUT_DIM,decoder_cl=config.MIM.DECODER_CL,decoder_cl_nlayer = config.MIM.DECODER_CL_NLAYERS,decoder_cl_use_bn=config.MIM.DECODER_CL_USE_BN)


@register_model
def clmim(**kwargs):
    config = kwargs['config']
    assert config.MODEL.BACKBONE != ''

    student = kwargs['backbone']
    teacher = deepcopy(student)

    if config.MIM.TEACHER_INIT is not None:
        __teacher_init(config,teacher)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # all ssl method impl doesn't set the teacher eval mode in training
    # teacher.eval()
    
    return {'main':student,'teacher':teacher}
