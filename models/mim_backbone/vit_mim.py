from functools import partial
import math
import logging
from matplotlib import use

from timm.models.vision_transformer import VisionTransformer,_init_vit_weights,checkpoint_filter_fn,default_cfgs
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.helpers import named_apply,build_model_with_cfg

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import MaskGenerator

_logger = logging.getLogger(__name__)

class ViTforMiM(VisionTransformer):
    def __init__(self, use_mae=False,mask_ratio=0.6,mask_patch_size=16,**kwargs):
        super().__init__(**kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

        # for mim
        self.use_mae = use_mae
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if not use_mae else None
        self.mask_patch_size = mask_patch_size
        self.img_size = kwargs['img_size']
        self.patch_size = kwargs['patch_size']
        self.in_chans = kwargs['in_chans']
        self.random_masking = MaskGenerator(self.img_size,self.mask_patch_size,self.patch_size,self.mask_ratio,self.use_mae)
        if self.dist_token is not None:
            self.patch_index = 2
        else:
            self.patch_index = 1
        if not use_mae:
            self._trunc_normal_(self.mask_token, std=.02)
        else:
            torch.nn.init.normal_(self.mask_token, std=.02)

    # simmim
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward_features(self, x,keep_mask=False):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # the mae first add the pos_embed and then masked the patches, simmim is the opposite
        if self.use_mae:
            x = self.pos_drop(x + self.pos_embed)
            if self.training:
                x[:,self.patch_index:], mask = self.random_masking(x[:,self.patch_index:])
            else:
                mask = None
        else:
            if self.training or keep_mask:
                x[:,self.patch_index:], mask = self.random_masking(x[:,self.patch_index:],self.mask_token)
            else:
                mask = None
            x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x),mask
        else:
            return x,mask

    def forward(self,x,keep_mask=False):
        x = self.patch_embed(x)
        # 分类
        z,mask = self.forward_features(x,keep_mask)
        if self.head_dist is not None:
            x, x_dist = self.head(z[:,0]), self.head_dist(z[:,1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                x = (x, x_dist)
            else:
                x =  (x + x_dist) / 2
        else:
            x = self.head(z[:,0])
        
        return z[:,self.patch_index:],x,mask


    # simmim 
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        ViTforMiM, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
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