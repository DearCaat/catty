from timm.data import Mixup

from dataloader.ioplin import ioplin_dataloader
from .iNet_torch import *
from .pim import *
def build_loader(config,is_train):
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    
    name = config.DATA.DATALOADER_NAME.lower().split('_')[0]

    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    if name == 'dali':
        import nvidia.dali as dali
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator,LastBatchPolicy
        from nvidia.dali.pipeline import Pipeline
        import nvidia.dali.types as types
        import nvidia.dali.ops as ops
        import nvidia.dali.fn as fn
        import nvidia.dali.tfrecord as tfrec
        from nvidia.dali.fn import readers
        print('dali mode')
    elif name == 'timm':
        dataloader = timm_dataloader(config,is_train)
    elif name == 'ioplin':
        dataloader = ioplin_dataloader(config,is_train)
    elif name == 'torch':
        dataloader = pytorch_dataloader(config=config,is_train=is_train)
    elif name == 'pim':
        dataloader = build_pim_loader(config)
    else:
        raise NotImplementedError
    # elif name == 'simmim':
    #     dataloader = 
    dataloader += (mixup_fn,) if is_train else ()
    return dataloader