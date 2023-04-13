from timm.models.swin_transformer import SwinTransformer
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg
from timm.models.vision_transformer import checkpoint_filter_fn,resolve_pretrained_cfg

import torch.nn as nn
import torch

from ..utils import MaskGenerator

class SwinforMiM(SwinTransformer):
    def __init__(self, mim_enable=False,use_mae=False,mask_ratio=0.6,mask_patch_size=16,**kwargs):
        super().__init__(**kwargs)

        self.patch_head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.mim_enable = mim_enable

        weight_init = kwargs['weight_init'] if 'weight_init' in kwargs.keys() else ''

        if self.mim_enable:
            # for mim
            self.use_mae = use_mae
            self.mask_ratio = mask_ratio
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if not use_mae else None
            self.img_size = kwargs['img_size']
            self.patch_size = kwargs['patch_size']
            self.in_chans = kwargs['in_chans']
            self.mask_patch_size = mask_patch_size
            self.random_masking = MaskGenerator(self.img_size,self.mask_patch_size,self.patch_size,self.mask_ratio,self.use_mae)
        
        self.init_weights(weight_init)

    def forward_features(self, x,keep_mask=False,get_multi_feat=False):
        x = self.patch_embed(x)

        features = []
        mask = None

        if self.mim_enable:
        # the mae first add the pos_embed and then masked the patches, simmim is the opposite
            if self.use_mae:
                if self.absolute_pos_embed is not None:
                    x = x + self.absolute_pos_embed
                x = self.pos_drop(x)
                if self.training:
                    x, mask = self.random_masking(x)
                else:
                    mask = None
            else:
                if self.training or keep_mask:
                    x, mask = self.random_masking(x,self.mask_token)
                else:
                    mask = None
                if self.absolute_pos_embed is not None:
                    x = x + self.absolute_pos_embed
                x = self.pos_drop(x)
        else:
            if self.absolute_pos_embed is not None:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

        for i,layer in enumerate(self.layers.children()):
            x = layer(x)
            # swin_b 224:
            # (b,784,256) (b,196,512) (b,49,1024) (b,49,1024)
            # swin_b 384:
            # (b,2304,256) (b,576,512) (b,144,1024) (b,144,1024)
            if get_multi_feat:
                features.append(x)

        x = self.norm(x)  # B L C
        return x,mask,features

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self,x):
        z,mask,mul_feat = self.forward_features(x)

        

        x = self.forward_head(z)
        return z,x,mask

def _create_swin_transformer(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        SwinforMiM, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model


@register_model
def swin_mim_base_patch4_window12_384_in22k(pretrained=False, **kwargs):
    """ Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer('swin_base_patch4_window12_384_in22k', pretrained=pretrained, **model_kwargs)

@register_model
def swin_mim_base_patch4_window7_224_in22k(pretrained=False, **kwargs):
    """ Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer('swin_base_patch4_window7_224_in22k', pretrained=pretrained, **model_kwargs)