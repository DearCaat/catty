from timm.data import create_dataset,create_loader
import torch

def timm_dataloader(config,is_train):
    if is_train:
        dataset_train = create_dataset(
        config.DATA.DATASET,
        root=config.DATA.DATA_PATH, split=config.DATA.TRAIN_SPLIT, is_training=True,
        batch_size=config.DATA.BATCH_SIZE,repeats=config.DATA.EPOCH_REPEATS)
        dataset_val = create_dataset(
            config.DATA.DATASET, root=config.DATA.DATA_PATH, split=config.DATA.VAL_SPLIT, is_training=False, batch_size=config.DATA.BATCH_SIZE)
        loader_train = create_loader(
            dataset_train,
            input_size=config.DATA.IMG_SIZE,
            batch_size=config.DATA.BATCH_SIZE,
            is_training=True,
            use_prefetcher=config.DATA.TIMM_PREFETCHER,
            no_aug=config.AUG.NO_AUG,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            scale=config.AUG.SCALE,
            ratio=config.AUG.RATIO,
            hflip=config.AUG.HFLIP,
            vflip=config.AUG.VFLIP,
            color_jitter=config.AUG.COLOR_JITTER,
            auto_augment=config.AUG.AUTO_AUGMENT,
            num_aug_splits=config.AUG.SPLITS,
            interpolation=config.DATA.INTERPOLATION,
            mean=config.AUG.NORM[0],
            std=config.AUG.NORM[1],
            num_workers=config.DATA.NUM_WORKERS,
            distributed=config.DISTRIBUTED,
            pin_memory=config.DATA.PIN_MEMORY,
        )

        loader_val = create_loader(
            dataset_val,
            input_size=config.DATA.IMG_SIZE,
            batch_size=config.DATA.VAL_BATCH_SIZE,
            is_training=False,
            use_prefetcher=config.DATA.TIMM_PREFETCHER,
            interpolation=config.DATA.INTERPOLATION,
            mean=config.AUG.NORM[0],
            std=config.AUG.NORM[1],
            num_workers=config.DATA.NUM_WORKERS,
            distributed=config.DISTRIBUTED,
            crop_pct=config.TEST.CROP,
            pin_memory=config.DATA.PIN_MEMORY,
        )
        return dataset_train, dataset_val, loader_train, loader_val
    else:
        dataset_test = create_dataset(
        config.DATA.DATASET, root=config.DATA.DATA_PATH, split=config.DATA.TEST_SPLIT, is_training=False, batch_size=config.DATA.BATCH_SIZE)

        loader_test = create_loader(
            dataset_test,
            input_size=config.DATA.IMG_SIZE,
            batch_size=config.DATA.VAL_BATCH_SIZE,
            is_training=False,
            use_prefetcher=config.DATA.TIMM_PREFETCHER,
            interpolation=config.DATA.INTERPOLATION,
            mean=config.AUG.NORM[0],
            std=config.AUG.NORM[1],
            num_workers=config.DATA.NUM_WORKERS,
            distributed=config.DISTRIBUTED,
            crop_pct=config.TEST.CROP,
            pin_memory=config.DATA.PIN_MEMORY,
        )
        return dataset_test,loader_test

def pytorch_dataloader(is_train,config):
    if is_train:
        if config.DATA.TFRECORD_MODE:
            dataset_train, dataset_val = tfds_dataset(True,config)
        else:
            transform_train = build_transform(is_train=True,config=config)
            transform_val = build_transform(is_train=False,config=config)
            dataset_train = MulitiViewImageDataset(root=_search_split(config.DATA.DATA_PATH, config.DATA.TRAIN_SPLIT),transform=transform_train,is_multi_view=config.AUG.MULTI_VIEW,size=config.DATA.IMG_SIZE,timm_trans=config.AUG.TIMM_TRANS)
            dataset_val = MulitiViewImageDataset(root=_search_split(config.DATA.DATA_PATH, config.DATA.VAL_SPLIT),transform=transform_val,size=config.DATA.IMG_SIZE,timm_trans=config.AUG.TIMM_TRANS)
        if config.DISTRIBUTED:
            sampler=torch.utils.data.distributed.DistributedSampler(dataset_train)
            loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.DATA.BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,drop_last=config.DATA.DROP_LAST,persistent_workers=True,sampler=sampler)
        else:
            loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.DATA.BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,drop_last=config.DATA.DROP_LAST,persistent_workers=True,shuffle=True)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config.DATA.VAL_BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,persistent_workers=True)
        
        
        return dataset_train, dataset_val, loader_train, loader_val
        
    else:
        if config.DATA.TFRECORD_MODE:
            #dataset_test = tfrecord_dataset(is_train=False,config=config)
            dataset_test = tfds_dataset(is_train=False,config=config)
        else:
            #占位，考虑文件夹数据集
            transform_test = build_transform(is_train=False,config=config)
            # 之前做WSPLIN其它数据库测试用
            #dataset_test = PatchImageDataset(parser=config.DATA.DATASET.lower(),root=config.DATA.DATA_PATH,transform=transform_test,patch_size=config.DATA.PATCH_SIZE,stride=config.DATA.STRIDE,eval=True,class_to_idx={'diseased':1,'normal':0},thumb=config.THUMB_MODE)
            dataset_test = MulitiViewImageDataset(root=_search_split(config.DATA.DATA_PATH, config.DATA.TEST_SPLIT),transform=transform_test,size=config.DATA.IMG_SIZE,timm_trans=config.AUG.TIMM_TRANS)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.DATA.VAL_BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,persistent_workers=True)
        return dataset_test,loader_test
