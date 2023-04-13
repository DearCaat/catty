import logging
import re

from timm.models.vision_transformer import checkpoint_filter_fn,resolve_pretrained_cfg
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg

from .vit import VisionTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import MaskGenerator

_logger = logging.getLogger(__name__)



class ViTforMiM(VisionTransformer):
    def __init__(self, use_mae=False,mask_ratio=0.6,mask_patch_size=16,topk=1,select_fn='last_layer',drop_mask_token_in_encoder=True,mask_selected_token=True,**kwargs):
        super().__init__(**kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.topk = topk
        self.select_fn = select_fn

        # for mim
        self.use_mae = use_mae
        self.drop_mask_token_in_encoder = drop_mask_token_in_encoder
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if not use_mae else None
        self.mask_patch_size = mask_patch_size
        self.img_size = kwargs['img_size']
        self.patch_size = kwargs['patch_size']
        self.in_chans = kwargs['in_chans']
        self.random_masking = MaskGenerator(self.img_size,self.mask_patch_size,self.patch_size,self.mask_ratio,self.use_mae,mask_some_tokens=mask_selected_token)
        # timm 0.6.7 remove the dist_token
        self.dist_token = None
        self.pre_logits = nn.Identity()
        self.head_dist = None

        if self.dist_token is not None:
            self.patch_index = 2
        else:
            self.patch_index = 1
        if not use_mae:
            self._trunc_normal_(self.mask_token, std=.02)
        else:
            torch.nn.init.normal_(self.mask_token, std=.02)

        self.unmask = None
    # simmim
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward_features(self, x,keep_mask=False,unmask=None,no_mask=False,require_unmask=False):
        attn_weights = []
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        if not no_mask:
            # the mae first add the pos_embed and then masked the patches, simmim is the opposite
            if self.drop_mask_token_in_encoder:
                x = self.pos_drop(x + self.pos_embed)
                if self.training or keep_mask:
                    _tmp = x[:,:self.patch_index]
                    x, mask = self.random_masking(x[:,self.patch_index:],unmask=unmask)
                    x = torch.cat((_tmp,x),dim=1)

                else:
                    mask = None
            else:
                if self.training or keep_mask:
                    x[:,self.patch_index:], mask = self.random_masking(x[:,self.patch_index:],self.mask_token,unmask=self.unmask)
                else:
                    mask = None
                x = self.pos_drop(x + self.pos_embed)
        else:
            x = self.pos_drop(x + self.pos_embed)
            mask = None

        for i,block in enumerate(self.blocks.children()):
                x,attn = block(x)
                if require_unmask:
                    attn_weights.append(attn)
        if require_unmask:
            attn_weights = torch.stack(attn_weights)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x),mask,attn_weights
        else:
            return x,mask,attn_weights

    def _get_unmask(self,attn_weights):

        if self.select_fn == 'last_layer':
            attn_weights = attn_weights[-1,:,:,:]
        # transfg
        elif self.select_fn == 'psm':
            cls_attn_cum = attn_weights[0]
            for i in range(1,attn_weights.size(0)):
                cls_attn_cum = torch.matmul(attn_weights[i],cls_attn_cum)
            attn_weights = cls_attn_cum

        # attn for query [cls]
        attn_weights = attn_weights[:,:,0,1:] # L,B,H,T-1
        (B,heads_num,tokens_length) = attn_weights.size() # not include [cls]
        rand_size = int(tokens_length**0.5)

        cls_attn_topk_value,cls_attn_topk_idx = torch.topk(attn_weights,self.topk)
        cls_attn_topk_idx = cls_attn_topk_idx.flatten(-2,-1)

        # unmask
        cls_attn_topk_mask = torch.zeros(B,tokens_length,device=attn_weights.device)
        cls_attn_topk_mask = cls_attn_topk_mask.scatter_(1,cls_attn_topk_idx,1).view((-1,rand_size,rand_size))

        return cls_attn_topk_mask

    def forward(self,x,keep_mask=False,unmask=None,require_unmask=False,no_mask=False):
        x = self.patch_embed(x)
        z,mask,attn_weights = self.forward_features(x,keep_mask,unmask,no_mask=no_mask,require_unmask=require_unmask)
        if require_unmask:
            unmask = self._get_unmask(attn_weights)
        else:
            unmask = None

        if self.head_dist is not None:
            x, x_dist = self.head(z[:,0]), self.head_dist(z[:,1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                x = (x, x_dist)
            else:
                x =  (x + x_dist) / 2
        else:
            x = self.head(z[:,0])
        
        return z[:,0],z[:,self.patch_index:],x,mask,unmask

    # simmim 
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))

    model = build_model_with_cfg(
        ViTforMiM, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



@register_model
def vit_mim_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model