import torch
import torch.nn as nn
from timm.models import create_model
from .pvt import *
from .pvt_v2 import *
from .cswin import *
from .mim import *
from .simmim import *
from .wsplin import *
from .stn import *
from .dino import *
from .simsiam import *
from .ioplin import *
from .utils import LinearProbWrapper
from .pict import *

# from ._vit import *

ONE_BACKBONE_GROUP = ('wsplin','stn','ioplin','simsiam','pict','clmim')
TWO_BACKBONE_SAME_ARCH_GROUP = ('dino')

def build_model(config,is_backbone=False,num_classes=None,logger=None):
    if is_backbone:
        model_name = config.MODEL.BACKBONE
        features_only = False
    else:
        model_name = config.MODEL.NAME.lower()
        features_only = False
    num_classes = num_classes or config.MODEL.NUM_CLASSES

    # Use the official impl to use the gradient cheackpoint
    if model_name.startswith('vgg'):
        model = create_model(
            model_name,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=num_classes,
            features_only=features_only
        )
        models = {'main':model}
    elif model_name.startswith(('res','incep')):
        model = create_model(
            model_name,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=num_classes,
            drop_rate=None if int(config.MODEL.DROP_RATE) == -1 else config.MODEL.DROP_RATE,
            drop_path_rate=None if int(config.MODEL.DROP_PATH_RATE) == -1 else config.MODEL.DROP_PATH_RATE,
            features_only=features_only
        )
        models = {'main':model}
    elif model_name.startswith('tf_effi'):
        model = create_model(
            model_name,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=num_classes,
            drop_rate=None if int(config.MODEL.DROP_RATE) == -1 else config.MODEL.DROP_RATE,
            drop_path_rate=None if int(config.MODEL.DROP_PATH_RATE) == -1 else config.MODEL.DROP_PATH_RATE,
            features_only=features_only
        )
        models = {'main':model}
    elif model_name.startswith('mim') or model_name.startswith('simmim'):
        model = create_model(
            model_name,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=num_classes,
            drop_rate=None if int(config.MODEL.DROP_RATE) == -1 else config.MODEL.DROP_RATE,
            drop_path_rate=None if int(config.MODEL.DROP_PATH_RATE) == -1 else config.MODEL.DROP_PATH_RATE,
            img_size = config.DATA.IMG_SIZE[0],
            use_mae = config.MIM.USE_MAE,
            norm_pix_loss = config.MIM.NORM_PIX_LOSS,
            config = config
        )
        models = {'main':model}
    # The base framework which includes two backbones, student and teacher. And the arch of S and T are same.
    elif model_name.startswith(TWO_BACKBONE_SAME_ARCH_GROUP):
        student = build_model(config,is_backbone=True)['main']
        teacher = build_model(config,is_backbone=True)['main']

        models = create_model(
            model_name,
            student = student,
            teacher = teacher,
            config = config,
            logger=logger
        )
    # The base framework which includes the one backbone
    elif model_name.startswith(ONE_BACKBONE_GROUP):
        backbone = build_model(config,is_backbone=True)['main']
        models = create_model(
            model_name,
            backbone = backbone,
            config = config,
            logger=logger
        )
    else:
        # drop_rate=config.MODEL.DROP_RATE,
        # drop_path_rate=config.MODEL.DROP_PATH_RATE
        model = create_model(
            model_name,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=num_classes,
            drop_rate=None if int(config.MODEL.DROP_RATE) == -1 else config.MODEL.DROP_RATE,
            drop_path_rate=None if int(config.MODEL.DROP_PATH_RATE) == -1 else config.MODEL.DROP_PATH_RATE,
            img_size = config.DATA.IMG_SIZE[0],
            features_only=features_only
        )
        models = {'main':model}

    # 处理线性测试的情况
    if config.TEST.LINEAR_PROB.ENABLE and not is_backbone:
        models['main'] = LinearProbWrapper(models['main'],config)

    return models