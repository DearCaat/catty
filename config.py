# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
_C.DATA.VAL_BATCH_SIZE = 96
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Path to pretrained model
_C.DATA.PRETRAINED_DIR = ''
# Dataset name     tfds/cqu_bpdd  cfd crack500 cracktre200
_C.DATA.DATASET = 'cqu_bpdd'
# 数据集中的正常图片所在的类别索引 cqu_bpdd ：6
_C.DATA.NOR_CLS_INDEX = 6
_C.DATA.GRAY = True
# Input image size (h,w)  cqu_bpdd (900,1200) cfd(300,450)
_C.DATA.IMG_SIZE = (224,224)
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.TFRECORD_MODE = False
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# NVIDIA DALI data loader
_C.DATA.DALI = False
# dataset train split (default: train)
_C.DATA.TRAIN_SPLIT = 'train'
# dataset validation split (default: validation)
_C.DATA.VAL_SPLIT = 'val'
# dataset test split (default: test)
_C.DATA.TEST_SPLIT = 'test'
# epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).
_C.DATA.EPOCH_REPEATS = 0
# Default timm thumb image loader
_C.DATA.TIMM = True

_C.DATA.PATCH_SIZE=300
# for cfd 150 cracktree200 150 cqu_bpdd 300
_C.DATA.STRIDE=300
_C.DATA.CROP_SIZE=300

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type        #swin_small_patch4_window7_224  efficientnetv2_rw_s  deit_base_patch16_224
_C.MODEL.NAME = 'cluster_swin_small_patch4_window7_224'
# Model name
_C.MODEL.BACKBONE = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 8
# Dropout rate     effi-b3 0.3`
_C.MODEL.DROP_RATE = 0
# Drop path rate   effi-b3 0.2  swin_s 0.3
_C.MODEL.DROP_PATH_RATE = 0.3
# Label Smoothing
#_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.LABEL_SMOOTHING = 0
# Start with pretrained version of specified network (if avail)
_C.MODEL.PRETRAINED = True

_C.MODEL.NUM_PATCHES=17

# -----------------------------------------------------------------------------
# RDD_TRANS settings
# -----------------------------------------------------------------------------
_C.RDD_TRANS = CN()
_C.RDD_TRANS.EMA_DECAY = 0.9997
_C.RDD_TRANS.EMA_DECAY_SCHEDULER = None #warmup    warmup_flat
_C.RDD_TRANS.EMA_DECAY_SCHEDULER_FLAT_RATIO = 0.01
_C.RDD_TRANS.INIT_STAGE_EPOCH = 0
_C.RDD_TRANS.EMA_FORCE_CPU = False
_C.RDD_TRANS.NOR_THR = 0.05
_C.RDD_TRANS.TEST_THR = 0.995
_C.RDD_TRANS.INST_NUM_CLASS = 2
_C.RDD_TRANS.NOT_INST_TEST = True

_C.RDD_TRANS.CLUSTER = CN()  # Kmeans因为要指定簇数量，因此不适用于该方法，该方法不同类别图片的簇数量理应不相等，而且不同种类病害的簇中心也不相同
_C.RDD_TRANS.CLUSTER.NAME='gcn'
_C.RDD_TRANS.CLUSTER.CLUSTER_DISTANCE = 'euclidean'  # kmeans default euclidean, gcn cosine
# kmeans paras 
_C.RDD_TRANS.CLUSTER.NUM_CLUSTER = None
# _C.RDD_TRANS.CLUSTER.NUM_INIT = 10    # default
# _C.RDD_TRANS.CLUSTER.INIT = 'k-means++' # default
# gcn paras
_C.RDD_TRANS.CLUSTER.IPS_ACTIVE_CONNECTION = 5
_C.RDD_TRANS.CLUSTER.IPS_K_AT_HOP = (5,0)  # 先不考虑第二跳，因为效率问题
_C.RDD_TRANS.CLUSTER.THR = 0.75


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 20
_C.TRAIN.WARMUP_EPOCHS = 2
_C.TRAIN.WEIGHT_DECAY = 0
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-7
# Clip gradient norm                                     
#_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.CLIP_GRAD = 0
# Gradient clipping mode. One of ("norm", "value", "agc")                                                                                   
_C.TRAIN.CLIP_MODE = 'norm'
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# STEP interval to decay LR, used in flat_cosine
_C.TRAIN.LR_SCHEDULER.DECAY_STEPS_RATIO=0.5
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 2
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
 #'Lookahead_adamw'
_C.TRAIN.OPTIMIZER.NAME =  'Lookahead_adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

#Loss
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.LAMBDA_L1 = 1e-3

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Norm mean and std, default is [IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD], but old wsplin model use [(0.455,0.455,0.455),(0.225,0.225,0.225)]  effi-b3 use [(0.5,0.5,0.5),(0.5,0.5,0.5)]
_C.AUG.NORM = [IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD]
# Disable all training augmentation, override other train aug args
_C.AUG.NO_AUG = True
# Random resize scale (default: 0.08 1.0)
_C.AUG.SCALE = [0.08, 1.0]
# Random resize aspect ratio (default: 0.75 1.33)
_C.AUG.RATIO = [3./4., 4./3.]
# Horizontal flip training aug probability
_C.AUG.HFLIP = 0.5
# Vertical flip training aug probability
_C.AUG.VFLIP = 0.
# Color jitter factor   0.4
_C.AUG.COLOR_JITTER = 0  
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-n2-mstd0.5'
# Random erase prob
_C.AUG.REPROB = 0   #0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
'''# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0

# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'''
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 0.
# Number of augmentation splits (default: 0, valid: 0 or >=2)
_C.AUG.SPLITS=0

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# use NVIDIA Apex AMP or Native AMP for mixed precision training
# overwritten by command line argument
_C.AMP = True
# Use NVIDIA Apex AMP mixed precision default O1
_C.APEX_AMP = False
# Use Native Torch AMP mixed precision
_C.NATIVE_AMP = True
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# name of experiment, overwritten by command line argument
_C.EXP_NAME = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 50
# Fixed random seed
_C.SEED = 42
# use the thumb data to train
_C.THUMB_MODE = False
# binary train and binary test
_C.BINARYTRAIN_MODE = False
# the dir of tested data
_C.LOAD_TEST_DIR = ''
# log training and validation metrics to wandb
_C.LOG_WANDB = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
_C.DISTRIBUTED = False
_C.WORLD_SIZE = 0

_C.TRAIN_MODE = 't_e'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if args.cfg:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.log_wandb:
        config.LOG_WANDB = args.log_wandb
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.tfrecord:
        config.DATA.TFRECORD_MODE = True
    if args.title:
        config.EXP_NAME = args.title
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.no_amp:
        config.AMP = False
    if args.output:
        config.OUTPUT = args.output
    if args.model_name:
        config.MODEL.NAME = args.model_name
    if args.thumb:
        config.THUMB_MODE = True
    if args.binary_train:
        config.BINARYTRAIN_MODE = True
        config.MODEL.NUM_CLASSES = 2
    if args.load_test_dir:
        config.LOAD_TEST_DIR = args.load_test_dir
    if args.epochs:
        config.TRAIN.EPOCHS = args.epochs
    if args.local_rank:
        config.LOCAL_RANK = args.local_rank
    if args.pretrained_backbone:
        config.DATA.PRETRAINED_DIR = args.pretrained_backbone
    if args.train_mode:
        config.TRAIN_MODE = args.train_mode

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.DISTRIBUTED = int(os.environ['WORLD_SIZE']) > 1
        config.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.EXP_NAME)

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    if not args=='':
        update_config(config, args)

    return config