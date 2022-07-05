from copy import deepcopy
import logging

from timm.models.swin_transformer import SwinTransformer,checkpoint_filter_fn,default_cfgs
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg,overlay_external_default_cfg

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

class SwinforPict(SwinTransformer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        z = self.avgpool(x.transpose(1, 2))  # B C 1
        z = torch.flatten(z, 1)
        return z,x

    def forward(self, x):
        z,x = self.forward_features(x)
        x = self.head(x)
        return x

def create_swin_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        SwinforPict, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model