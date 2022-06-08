from timm.data import create_transform
from timm.data.transforms import str_to_interp_mode
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import torch
import numpy as np
from .tranfg_autoda import AutoAugImageNetPolicy

def _build_transform(config,is_train,type=None):
    _name = config.DATA.DATALOADER_NAME.lower().split('_')[2]
    if type is None:
        if _name == 'timm':
            return create_transform(
                            input_size=config.DATA.IMG_SIZE,
                            is_training=is_train,
                            use_prefetcher=config.DATA.TIMM_PREFETCHER,
                            no_aug = config.AUG.NO_AUG,
                            scale=config.AUG.SCALE,
                            ratio=config.AUG.RATIO,
                            hflip=config.AUG.HFLIP,
                            vflip=config.AUG.VFLIP,
                            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                            interpolation=config.DATA.INTERPOLATION,
                            mean=config.AUG.NORM[0],
                            std=config.AUG.NORM[1],
                            re_prob=config.AUG.REPROB,
                            re_mode=config.AUG.REMODE,
                            re_count=config.AUG.RECOUNT,
                            crop_pct=config.TEST.CROP,
                            )
        elif _name == 'custom':
            tf_timm = create_transform(
                            input_size=config.DATA.IMG_SIZE,
                            is_training=is_train,
                            use_prefetcher=config.DATA.TIMM_PREFETCHER,
                            no_aug = config.AUG.NO_AUG,
                            scale=config.AUG.SCALE,
                            ratio=config.AUG.RATIO,
                            hflip=config.AUG.HFLIP,
                            vflip=config.AUG.VFLIP,
                            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                            interpolation=config.DATA.INTERPOLATION,
                            mean=config.AUG.NORM[0],
                            std=config.AUG.NORM[1],
                            re_prob=config.AUG.REPROB,
                            re_mode=config.AUG.REMODE,
                            re_count=config.AUG.RECOUNT,
                            crop_pct=config.TEST.CROP,
                            separate=is_train,
                            )
            if is_train:
                [_tf1,tf2,tf3] = tf_timm
                if config.TEST.CROP != 1.0 and config.TEST.CROP != 0.:
                    tf1 = [transforms.Resize(list((np.array(config.DATA.IMG_SIZE) / config.TEST.CROP).astype(int)), str_to_interp_mode(config.DATA.INTERPOLATION)),
                        transforms.RandomCrop(config.DATA.IMG_SIZE)]
                else:
                    tf1 = [transforms.Resize(config.DATA.IMG_SIZE)]
                if config.AUG.HFLIP > 0:
                    tf1 += [transforms.RandomHorizontalFlip(config.AUG.HFLIP)]
                if config.AUG.VFLIP:
                    tf1 += [transforms.RandomVerticalFlip(config.AUG.VFLIP)]
                if config.AUG.TRANSFG_AA:
                    tf1 += [AutoAugImageNetPolicy()]
                tf1 = tf1 + tf2.transforms + tf3.transforms
                return transforms.Compose(tf1)
            else:
                return tf_timm
            
        elif _name == 'pim':
            # 448:600
            # 384:510
            # 768:
            if is_train:
                # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
                # RandAugment(n=2, m=3, img_size=sub_data_size)
                return transforms.Compose([
                            transforms.Resize(list((np.array(config.DATA.IMG_SIZE) / config.TEST.CROP).astype(int)), str_to_interp_mode(config.DATA.INTERPOLATION)),
                            transforms.RandomCrop(config.DATA.IMG_SIZE),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=torch.tensor(config.AUG.NORM[0]),
                                std=torch.tensor(config.AUG.NORM[1])
                            ),
                    ])
            else:
                return transforms.Compose([
                            transforms.Resize(list((np.array(config.DATA.IMG_SIZE) / config.TEST.CROP).astype(int)), str_to_interp_mode(config.DATA.INTERPOLATION)),
                            transforms.CenterCrop(config.DATA.IMG_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=torch.tensor(config.AUG.NORM[0]),
                                std=torch.tensor(config.AUG.NORM[1])
                            ),
                    ])
        
        elif _name == 'pict':
            t1 = [A.Resize(height=config.DATA.IMG_SIZE[0],width=config.DATA.IMG_SIZE[1])]
            if is_train and not config.AUG.NO_AUG:
                t2 = [
                        A.RandomBrightnessContrast(p=0.7),
                        A.HorizontalFlip(p=0.7),
                        A.VerticalFlip(p=0.7),
                        A.ShiftScaleRotate(rotate_limit=15.0, p=0.7),
                        A.OneOf([
                            A.Emboss(p=1),
                            A.Sharpen(p=1),
                            A.Blur(p=1)
                                ], p=0.7),
                    ]
            else:
                t2=[]
            t3 = [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()]
            return A.Compose(t1+t2+t3)
                        
def build_transform(config,is_train):
    if config.AUG.MULTI_VIEW is not None:
        _type = _type.lower().split('_')

        assert all(_t in ('strong','weak','none') for _t in _type)
        _transforms = ()
        for _t in _type:
            _transforms += (_build_transform(config,is_train,_t),)
    else:
        _transforms = _build_transform(config,is_train)
    
    return _transforms
