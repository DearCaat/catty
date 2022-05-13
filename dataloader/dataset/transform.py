from timm.data import create_transform
import albumentations as A
from torchvision import transforms


def build_transform(is_train,config):
    if is_train:
        #first transform
        if config.AUG.MULTI_VIEW is not None:
            if config.AUG.TIMM_TRANS:

                transform_strong = create_transform(
                                    input_size=config.DATA.IMG_SIZE,
                                    is_training=True,
                                    no_aug = config.AUG.NO_AUG,
                                    color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                                    auto_augment='rand-m7-n4-mstd0.5',
                                    re_prob=config.AUG.REPROB,
                                    re_mode=config.AUG.REMODE,
                                    re_count=config.AUG.RECOUNT,
                                    interpolation=config.DATA.INTERPOLATION)
                transform_weak = create_transform(
                                    input_size=config.DATA.IMG_SIZE,
                                    is_training=True,
                                    no_aug = config.AUG.NO_AUG,
                                    color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                                    auto_augment='rand-m3-n2-mstd0.5',
                                    re_prob=config.AUG.REPROB,
                                    re_mode=config.AUG.REMODE,
                                    re_count=config.AUG.RECOUNT,
                                    interpolation=config.DATA.INTERPOLATION)
                transform_no_aug = create_transform(
                                    input_size=config.DATA.IMG_SIZE,
                                    is_training=True,
                                    no_aug = True,
                                    interpolation=config.DATA.INTERPOLATION)
            else:
                transform_strong = A.Compose([
                                    #A.Resize(height=config.DATA.IMG_SIZE[0],width=config.DATA.IMG_SIZE[1]),
                                    A.RandomBrightnessContrast(p=0.7),
                                    A.HorizontalFlip(p=0.7),
                                    A.VerticalFlip(p=0.7),
                                    A.ShiftScaleRotate(rotate_limit=15.0, p=0.7),
                                    A.OneOf([
                                        A.Emboss(p=1),
                                        A.Sharpen(p=1),
                                        A.Blur(p=1)
                                            ], p=0.7)
                                    ])
                transform_weak = A.Compose([
                                    #A.Resize(height=config.DATA.IMG_SIZE[0],width=config.DATA.IMG_SIZE[1]),
                                    A.RandomBrightnessContrast(p=0.5),
                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5),
                                    A.ShiftScaleRotate(rotate_limit=15.0, p=0.7),
                                    #ToTensorV2()
                                    ])
                # A.Resize(height=config.DATA.IMG_SIZE[0],width=config.DATA.IMG_SIZE[1],interpolation=cv2.INTER_CUBIC),
                # A.Normalize(config.AUG.NORM[0], config.AUG.NORM[1]),
                transform_no_aug = A.Compose([
                                    #A.Resize(height=config.DATA.IMG_SIZE[0],width=config.DATA.IMG_SIZE[1]),
                                    #ToTensorV2()
                                    ])
            if config.AUG.MULTI_VIEW == 'strong_weak':
                return [transform_strong,transform_weak]
            elif config.AUG.MULTI_VIEW == 'strong_none':
                return [transform_weak,transform_no_aug]
            elif config.AUG.MULTI_VIEW == 'weak_none':
                return [transform_weak,transform_no_aug]
            else:
                raise NotImplementedError
        if config.AUG.TIMM_TRANS:
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
        else:
            if not config.AUG.NO_AUG:
                transform = A.Compose([
                                #A.Resize(height=config.DATA.IMG_SIZE[0],width=config.DATA.IMG_SIZE[1]),
                                A.RandomBrightnessContrast(p=0.7),
                                A.HorizontalFlip(p=0.7),
                                A.VerticalFlip(p=0.7),
                                A.ShiftScaleRotate(rotate_limit=15.0, p=0.7),
                                A.OneOf([
                                    A.Emboss(p=1),
                                    A.Sharpen(p=1),
                                    A.Blur(p=1)
                                        ], p=0.7)
                                ])
            else:
                transform = A.Compose([])
        # transform = transforms.Compose([
        #                 transforms.Resize((510, 510), Image.BILINEAR),
        #                 transforms.RandomCrop(config.DATA.IMG_SIZE),
        #                 transforms.RandomHorizontalFlip(),
        #                 transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
        #                 transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
        #         ])
        #transform = A.Compose([])
        return transform

    else:
        if config.AUG.TIMM_TRANS:
            t2 = create_transform(
                            input_size=config.DATA.IMG_SIZE,
                            is_training=False,
                            no_aug = config.AUG.NO_AUG,
                            interpolation=config.DATA.INTERPOLATION,
                            crop_pct=config.TEST.CROP)
        else:
            if config.TEST.CROP is not None and config.TEST.CROP != 1 and config.TEST.CROP != 0:
                t2 = transforms.Compose([
                                transforms.Resize((510, 510), Image.BILINEAR),
                                transforms.CenterCrop(config.DATA.IMG_SIZE),
                        ])
            else:
                t2 = A.Compose([
                #A.Resize(height=config.DATA.IMG_SIZE[0],width=config.DATA.IMG_SIZE[1]),
            ])
        return t2