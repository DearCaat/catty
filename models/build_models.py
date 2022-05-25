import torch
import torch.nn as nn
from timm.models import create_model
from .pvt import *
from .pvt_v2 import *
from ._vit import *

def build_model(config):
    model_name=config.MODEL.NAME
    # Use the official impl to use the gradient cheackpoint
    if model_name.startswith('rdd_trans'):
        model = create_model(
            config.MODEL.NAME,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=config.MODEL.NUM_CLASSES,
            ins_num_classes=config.RDD_TRANS.INST_NUM_CLASS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            cluster_name = config.RDD_TRANS.CLUSTER.NAME,
            num_cluster = config.RDD_TRANS.CLUSTER.NUM_CLUSTER,
            ips_active_connection = config.RDD_TRANS.CLUSTER.IPS_ACTIVE_CONNECTION,
            ips_k_at_hop = config.RDD_TRANS.CLUSTER.IPS_K_AT_HOP,
            cluster_distance = config.RDD_TRANS.CLUSTER.CLUSTER_DISTANCE.lower(),
            cluster_thr = config.RDD_TRANS.CLUSTER.THR,
            select_cluster_thr = config.RDD_TRANS.CLUSTER.SELECT_THR,
            nor_index = config.RDD_TRANS.CLUSTER.NOR_INDEX if hasattr(config.RDD_TRANS.CLUSTER, 'NOR_INDEX') else config.DATA.NOR_CLS_INDEX,
            cluster_rbf_distance = config.RDD_TRANS.CLUSTER.RBF_DISTANCE,
            cluster_rbf_gamma = config.RDD_TRANS.CLUSTER.RBF_GAMMA,
            cluster_n_compoents = config.RDD_TRANS.CLUSTER.N_COMPOENTS,
            persistent_center = config.RDD_TRANS.CLUSTER.PERSISTENT_CENTER,
            cluster_flip_sel = config.RDD_TRANS.TEST_CLU_FLIP_SEL
        )
        models = {'main':model}
    elif model_name.startswith('vgg'):
        model = create_model(
            config.MODEL.NAME,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=config.MODEL.NUM_CLASSES,
        )
        models = {'main':model}
    else:
        # drop_rate=config.MODEL.DROP_RATE,
        # drop_path_rate=config.MODEL.DROP_PATH_RATE
        model = create_model(
            config.MODEL.NAME,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=None if int(config.MODEL.DROP_RATE) == -1 else config.MODEL.DROP_RATE,
            drop_path_rate=None if int(config.MODEL.DROP_PATH_RATE) == -1 else config.MODEL.DROP_PATH_RATE
        )
        models = {'main':model}
    return models