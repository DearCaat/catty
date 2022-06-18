import torch.utils.data as data
from .transform import build_transform
from timm.data import create_dataset,create_parser
from .utils import _search_split
import logging
from timm.data.transforms import str_to_interp_mode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import torch
from PIL import Image
import os 

_logger = logging.getLogger(__name__)
_ERROR_RETRY = 50

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

    transform = build_transform(config,is_train)

    if name == 'timm':
        if config.DATA.DATALOADER_NAME.lower().split('_')[0] == 'timm':
            transform = None

        return create_dataset(
        config.DATA.DATASET,
        root=config.DATA.DATA_PATH, split=split, is_training=is_train,
        batch_size=config.DATA.BATCH_SIZE,repeats=config.DATA.EPOCH_REPEATS,transform=transform)
    elif name == 'tfds':
        return IterableImageDataset(parser=config.DATA.DATASET.lower(),root=config.DATA.DATA_PATH,split=split,gray=config.DATA.GRAY,shuffle=True,transform=transform,patch_size=config.DATA.PATCH_SIZE,stride=config.DATA.STRIDE,thumb=config.THUMB_MODE)
    elif name == 'multiview':
        return MulitiViewImageDataset(root=_search_split(config.DATA.DATA_PATH, config.DATA.TRAIN_SPLIT),transform=transform,is_multi_view=config.AUG.MULTI_VIEW,size=config.DATA.IMG_SIZE,timm_trans=config.AUG.TIMM_TRANS)
    elif name == 'album':
        if os.path.isdir(config.DATA.DATA_PATH):
            # look for split specific sub-folder in root
            root = _search_split(config.DATA.DATA_PATH, split)
        return ALImageDataset(root, parser=config.DATA.DATASET.lower(), class_map=None, transform=transform)

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
            is_multi_view=None,
            size=None,
            timm_trans=False,
            binary_mode=False,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform 
        self._consecutive_errors = 0
        self.is_multi_view = is_multi_view
        self.size = size
        self.timm_trans = timm_trans

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
        if not self.timm_trans:
            img = transforms.Resize(size=self.size,interpolation=str_to_interp_mode('bicubic'))(img=img)
        
        if self.transform is not None:
            if self.is_multi_view is not None:
                for idx,transform in enumerate(self.transform):
                    if self.timm_trans:
                        if idx == 0:
                            imgs = transform(img=img).unsqueeze(0)
                        else:
                            imgs = torch.cat((imgs,transform(img=img).unsqueeze(0)))
                    # try:
                    #     # pytorch transforms
                    #     if idx == 0:
                    #         imgs = transform(img=img).unsqueeze(0)
                    #     else:
                    #         imgs = torch.cat((imgs,transform(img=img).unsqueeze(0)))
                    # except:
                    else:
                    # albumentations
                        if idx == 0:
                            img_tmp = transform(image=np.asarray(img))['image']
                            img_tmp = transforms.ToTensor()(img_tmp)
                            img_tmp = transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD))(img_tmp)
                            imgs = img_tmp.unsqueeze(0)
                        else:
                            img_tmp = transform(image=np.asarray(img))['image']
                            img_tmp = transforms.ToTensor()(img_tmp)
                            img_tmp = transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD))(img_tmp)
                            imgs = torch.cat((imgs,img_tmp.unsqueeze(0)))
            else:
                # default albumentations
                try:
                    imgs = self.transform(image=np.asarray(img))['image']
                except:
                    imgs = self.transform(img=img)
                if not self.timm_trans:
                    imgs = transforms.ToTensor()(imgs)
                    imgs = transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD))(imgs)
        
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

#采用timm库，作了小小改动，支持多gpu，支持多线程，支持shuffle，不支持cache
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

class ALImageDataset(data.Dataset):
    
    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

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
            # for albumentations 

            img = self.transform(image=np.asarray(img))['image']
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


