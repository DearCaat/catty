from .transform import build_transform
from timm.data import create_dataset
from .utils import _search_split
import os 
from .datasets import *
def _build_dataset(config,_type='train'):

    name = config.DATA.DATALOADER_NAME.lower().split('_')[1]

    if _type == 'train':
        split = config.DATA.TRAIN_SPLIT
        is_train = True
    elif _type == 'val':
        split = config.DATA.VAL_SPLIT
        is_train = False
    elif _type == 'test':
        split = config.DATA.TEST_SPLIT
        is_train = False

    transform,target_transform = build_transform(config,is_train)

    if name == 'img':
        return ImageDataset(root=_search_split(config.DATA.DATA_PATH, config.DATA.TRAIN_SPLIT),transform=transform,target_transform=target_transform)
    elif name == 'timm':
        if config.DATA.DATALOADER_NAME.lower().split('_')[0] == 'timm':
            transform = None

        return create_dataset(
        config.DATA.DATASET,
        root=config.DATA.DATA_PATH, split=split, is_training=is_train,
        batch_size=config.DATA.BATCH_SIZE,repeats=config.DATA.EPOCH_REPEATS,transform=transform)

    elif name == 'tfds':
        return IterableImageDataset(parser=config.DATA.DATASET.lower(),root=config.DATA.DATA_PATH,split=split,gray=config.DATA.GRAY,shuffle=True,transform=transform,patch_size=config.DATA.PATCH_SIZE,stride=config.DATA.STRIDE,thumb=config.THUMB_MODE)
    elif name == 'multiview':
        return MulitiViewImageDataset(root=_search_split(config.DATA.DATA_PATH, config.DATA.TRAIN_SPLIT),transform=transform,is_multi_view=config.AUG.MULTI_VIEW,size=config.DATA.IMG_SIZE,timm_trans=config.AUG.TIMM_TRANS,target_transform=target_transform)
    elif name == 'album':
        if os.path.isdir(config.DATA.DATA_PATH):
            # look for split specific sub-folder in root
            root = _search_split(config.DATA.DATA_PATH, split)
        return ALImageDataset(root, parser=config.DATA.DATASET.lower(), class_map=None, transform=transform,target_transform=target_transform)
    elif name == 'patch':
        return PatchImageDataset(root=_search_split(config.DATA.DATA_PATH, config.DATA.TRAIN_SPLIT),transform=transform,target_transform=target_transform,last_transform=config.WSPLIN.LAST_TRANSFORM,is_ip=config.WSPLIN.IS_IP,patch_size=config.WSPLIN.PATCH_SIZE,stride=config.WSPLIN.STRIDE)
    else:
        raise NotImplementedError

def build_dataset(config,_type='train_val'):
    
    _type = _type.lower()
    _type = _type.split('_')

    assert all(_t in ('train','val','test') for _t in _type)

    datasets = ()
    for _t in _type:
        datasets += (_build_dataset(config,_t),)
    if len(datasets) == 1:
        datasets = datasets[0]
    return datasets
