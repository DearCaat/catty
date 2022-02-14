import os
from numpy.lib.function_base import append
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.utils.data as data
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,DEFAULT_CROP_PCT
from timm.data import Mixup
from timm.data import create_transform,create_parser,create_dataset,create_loader
import numpy as np
from PIL import Image
import logging
#from utils import get_shape
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


#Cacha 功能没有实现，现在只能在队列中随机shuffle

_logger = logging.getLogger(__name__)

_ERROR_RETRY = 50
class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            class_map='',
            load_bytes=False,
            repeats=0,
            transform=None,
            patch_size=0,
            stride=0,
            eval = False,
            thumb = False,
            **kwargs
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training, batch_size=batch_size, repeats=repeats,**kwargs)
        else:
            self.parser = parser
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self._consecutive_errors = 0
        self.eval = eval
        self.thumb = thumb

    def __iter__(self):
        for img, target in self.parser:
            #first transform
            #if self.eval:
            if self.transform is not None:
                for i in range(len(self.transform)-1) :
                    img = Image.fromarray(self.transform[i](image=np.asarray(img))['image'])
                if not self.thumb:
                    extractor = PatchExtractor(img=img, patch_size=self.patch_size, stride=self.stride)
                    patches = extractor.extract_patches()
                    imgs = torch.zeros((len(patches)), 3, self.patch_size, self.patch_size)
                    for i in range(len(patches)):
                        imgs[i] = self.transform[-1](patches[i])
                else:
                    imgs = self.transform[-1](img)
            ''' else:
                #last transform
                if not self.thumb:
                    extractor = PatchExtractor(img=img, patch_size=self.patch_size, stride=self.stride)
                    patches = extractor.extract_patches()
                    imgs = torch.zeros((len(patches)), 3, self.patch_size, self.patch_size)
                    if self.transform is not None:
                        for i in range(len(patches)):
                            imgs[i] = self.transform(patches[i])
                # for thumb data
                else:
                    imgs = self.transform(img)'''
            if target is None:
                target = torch.tensor(-1, dtype=torch.long)
            yield imgs, torch.tensor(target,dtype=torch.long)

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)

class PatchImageDataset(data.Dataset):
    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
            patch_size=0,
            stride=0,
            eval = False,
            thumb = False,
            **kwargs
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map,**kwargs)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self._consecutive_errors = 0
        self.eval = eval
        self.patch_size = patch_size
        self.stride = stride
        self.thumb = thumb

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('L')
            img = img.convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        # for test
        if self.eval:
            if self.transform is not None:
                for i in range(len(self.transform)-1) :
                    img = self.transform[i](img)
                if not self.thumb:
                    extractor = PatchExtractor(img=img, patch_size=self.patch_size, stride=self.stride)
                    patches = extractor.extract_patches()
                    imgs = torch.zeros((len(patches)), 3, self.patch_size, self.patch_size)
                    for i in range(len(patches)):
                        imgs[i] = self.transform[-1](patches[i])
                else:
                    imgs = self.transform[-1](img)
        # for train
        else:
            #last transform
            if not self.thumb:
                extractor = PatchExtractor(img=img, patch_size=self.patch_size, stride=self.stride)
                patches = extractor.extract_patches()
                imgs = torch.zeros((len(patches)), 3, self.patch_size, self.patch_size)
                if self.transform is not None:
                    for i in range(len(patches)):
                        imgs[i] = self.transform(patches[i])
            # for thumb data
            else:
                imgs = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return imgs, torch.tensor(target,dtype=torch.long)

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)

def build_transform(is_train,config):
    if is_train:
        #first transform
        if config.AUG.MULTI_VIEW is not None:
            transform_strong = create_transform(
                                input_size=config.DATA.IMG_SIZE,
                                is_training=True,
                                no_aug = config.AUG.NO_AUG,
                                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                                re_prob=config.AUG.REPROB,
                                re_mode=config.AUG.REMODE,
                                re_count=config.AUG.RECOUNT,
                                interpolation=config.DATA.INTERPOLATION)
            transform_weak = A.Compose([
                                A.Resize(height=config.DATA.IMG_SIZE[0],width=config.DATA.IMG_SIZE[1],interpolation=cv2.INTER_CUBIC),
                                A.RandomBrightnessContrast(p=0.5),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                A.ShiftScaleRotate(rotate_limit=15.0, p=0.7),
                                A.Normalize(config.AUG.NORM[0], config.AUG.NORM[1]),
                                ToTensorV2()
                                ])
            transform_no_aug = create_transform(input_size=config.DATA.IMG_SIZE,
                                is_training=True,
                                no_aug = True)
            if config.AUG.MULTI_VIEW == 'strong_weak':
                return [transform_strong,transform_weak]
            elif config.AUG.MULTI_VIEW == 'strong_none':
                return [transform_weak,transform_no_aug]
            elif config.AUG.MULTI_VIEW == 'weak_none':
                return [transform_weak,transform_no_aug]
            else:
                raise NotImplementedError
        
        transform = create_transform(
                                input_size=config.DATA.IMG_SIZE,
                                is_training=True,
                                no_aug = config.AUG.NO_AUG,
                                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                                re_prob=config.AUG.REPROB,
                                re_mode=config.AUG.REMODE,
                                re_count=config.AUG.RECOUNT,
                                interpolation=config.DATA.INTERPOLATION)
        return transform

    else:
        t1 = []
        t2 = []
        t1 = A.Compose([
            A.Resize(height=config.DATA.IMG_SIZE[0],width=config.DATA.IMG_SIZE[1],interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=15.0, p=0.7)
        ])
        t2.append(A.Normalize(config.AUG.NORM[0], config.AUG.NORM[1]))
        t2.append(ToTensorV2())
        #return [transform_1,transform_2]
        return [t1,transforms.Compose(t2)]

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
            dataset_val,loader_test = pytorch_dataloader(is_train=False,config=config)
            return dataset_val,loader_test
#采用timm库，作了小小改动，支持多gpu，支持多线程，支持shuffle，不支持cache
def tfds_dataset(is_train,config):
    if is_train:
        transform_train = build_transform(True,config)
        train_dataset = IterableImageDataset(parser=config.DATA.DATASET.lower(),root=config.DATA.DATA_PATH,split='train',gray=config.DATA.GRAY,shuffle=True,transform=transform_train,patch_size=config.DATA.PATCH_SIZE,stride=config.DATA.STRIDE,thumb=config.THUMB_MODE)

        transform_test = build_transform(False,config)
        val_dataset = IterableImageDataset(parser=config.DATA.DATASET.lower(),root=config.DATA.DATA_PATH,split='test',gray=config.DATA.GRAY,transform=transform_test,patch_size=config.DATA.PATCH_SIZE,stride=config.DATA.STRIDE,eval=eval,thumb=config.THUMB_MODE)
        return train_dataset,val_dataset
    else:
        transform_test = build_transform(False,config)
        test_dataset = IterableImageDataset(parser=config.DATA.DATASET.lower(),root=config.DATA.DATA_PATH,split='test',gray=config.DATA.GRAY,transform=transform_test,patch_size=config.DATA.PATCH_SIZE,stride=config.DATA.STRIDE,eval=eval,thumb=config.THUMB_MODE)
        return test_dataset

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
            #占位，考虑文件夹数据集
            #暂时不考虑训练
            #dataset_val = create_dataset(name=config.DATA.DATASET,root=config.DATA.DATA_PATH,thumb=config.THUMB_MODE)
            transform_train = build_transform(is_train=True,config=config)
            transform_val = build_transform(is_train=False,config=config)
            dataset_train = MulitiViewImageDataset(root=_search_split(config.DATA.DATA_PATH, config.DATA.TEST_SPLIT),transform=transform_train,is_multi_view=config.AUG.MULTI_VIEW)
            dataset_val = MulitiViewImageDataset(root=_search_split(config.DATA.DATA_PATH, config.DATA.VAL_SPLIT),transform=transform_val,is_multi_view=config.AUG.MULTI_VIEW)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.DATA.BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,drop_last=config.DATA.DROP_LAST,persistent_workers=True)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config.DATA.BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,persistent_workers=True)
        
        
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
            dataset_test = MulitiViewImageDataset(root=_search_split(config.DATA.DATA_PATH, config.DATA.TEST_SPLIT),transform=transform_test,is_multi_view=config.AUG.MULTI_VIEW)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.DATA.BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,persistent_workers=True)
        return dataset_test,loader_test

# 适用于常见的多数据增强的方法，暂时考虑两个视角
class MulitiViewImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
            is_multi_view=None
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform 
        self._consecutive_errors = 0
        self.is_multi_view = is_multi_view

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            if self.is_multi_view is not None:
                for idx,transform in enumerate(self.transform):
                    try:
                        # pytorch transforms
                        if idx == 0:
                            imgs = transform(img=img).unsqueeze(0)
                        imgs = torch.cat((imgs,transform(img=img).unsqueeze(0)))
                    except:
                        # albumentations
                        if idx == 0:
                            imgs = transform(image=np.asarray(img))['image'].unsqueeze(0)
                        imgs = torch.cat((imgs,transform(image=np.asarray(img))['image'].unsqueeze(0)))
            else:
                imgs = self.transform(img=img)    
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return imgs, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)

def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root
    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root

# DALI data loader，以后考虑DALI的实现
