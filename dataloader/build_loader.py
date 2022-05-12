from timm.data import Mixup


def build_loader(is_train,config):
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    if config.DATA.DALI:
        import nvidia.dali as dali
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator,LastBatchPolicy
        from nvidia.dali.pipeline import Pipeline
        import nvidia.dali.types as types
        import nvidia.dali.ops as ops
        import nvidia.dali.fn as fn
        import nvidia.dali.tfrecord as tfrec
        from nvidia.dali.fn import readers
        print('dali mode')
    elif config.DATA.TIMM:
        if is_train:
            dataset_train, dataset_val, loader_train, loader_val = timm_dataloader(config=config,is_train=True)
            return dataset_train, dataset_val, loader_train, loader_val,mixup_fn
        else:
            dataset_test, loader_test = timm_dataloader(config=config,is_train=False)
            return dataset_test, loader_test
    else:
        if is_train:
            dataset_train, dataset_val, loader_train, loader_val =pytorch_dataloader(is_train=True,config=config)
            return dataset_train, dataset_val, loader_train, loader_val,mixup_fn
        else:
            dataset_test,loader_test = pytorch_dataloader(is_train=False,config=config)
            return dataset_test,loader_test