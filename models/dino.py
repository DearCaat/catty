import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from .utils import MultiCropWrapper

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils import ModelEmaV3,cosine_scheduler

# copyright dino@facebook, ref: https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

def build_dino(**kwargs):
    config = kwargs['config']
    assert config.MODEL.BACKBONE != ''

    if not config.TEST.LINEAR_PROB.ENABLE:
        # 使用自定义的模型构建函数是因为可以处理不同模型API对于参数的不同需求
        student = kwargs['student']
        teacher = kwargs['teacher']

        embed_dim = student.num_features
        student = MultiCropWrapper(student,DINOHead(
            embed_dim,
            config.DINO.OUT_DIM,
            config.DINO.USE_BN_IN_HEAD,
            config.DINO.NORM_LAST_LAYER,
        ))
        teacher = MultiCropWrapper(teacher,DINOHead(
            embed_dim,
            config.DINO.OUT_DIM,
            config.DINO.USE_BN_IN_HEAD,
            config.DINO.NORM_LAST_LAYER,
        ))
        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False
        
        return {'main':student,'teacher':teacher}
    else:
        student = kwargs['student']

        if config.DINO.RESUME is not None:
            new_cpt = {}
            try:
                cpt = torch.load(config.DINO.RESUME, map_location = 'cpu')
            except:
                cpt = torch.load(os.path.join(config.OUTPUT,'model',config.DINO.RESUME), map_location = 'cpu')
            for i,_key in enumerate(cpt['state_dict'].keys()):
                _keys= _key.split('.',1)
                if _keys[0] == 'backbone':
                    new_cpt[_keys[1]] = cpt['state_dict'][_key]
            res = student.load_state_dict(new_cpt,strict=False)
            print(cpt['epoch'])
        return {'main':student}

@register_model
def dino(**kwargs):
    return build_dino(**kwargs)