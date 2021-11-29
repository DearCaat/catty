import torch
import torch.nn as nn
from timm.models import create_model
from .swin import *
from .deit import *
from .rdd_trans import *

def build_model(config):
    model_name=config.MODEL.NAME
    # Use the official impl to use the gradient cheackpoint
    if model_name.startswith('cluster'):
        model = create_model(
            config.MODEL.NAME,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            cluster_name = config.RDD_TRANS.CLUSTER.NAME,
            num_cluster = config.RDD_TRANS.CLUSTER.NUM_CLUSTER,
            ips_active_connection = config.RDD_TRANS.CLUSTER.IPS_ACTIVE_CONNECTION,
            ips_k_at_hop = config.RDD_TRANS.CLUSTER.IPS_K_AT_HOP,
            cluster_distance = config.RDD_TRANS.CLUSTER.CLUSTER_DISTANCE,
            cluster_thr = config.RDD_TRANS.CLUSTER.THR 
        )
    else:
        model = create_model(
            config.MODEL.NAME,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE
        )
    return model