import argparse
import os
import shutil
import time
import math

from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import AdamW
from PIL import Image
from torchvision.transforms import transforms
from networks import *
from optimizer import RangerLars
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score,precision_recall_curve

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from datasets import *
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import numpy as np
import random

try:
    import nvidia.dali as dali
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator,LastBatchPolicy
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.types as types
    import nvidia.dali.ops as ops
    import nvidia.dali.fn as fn
    import nvidia.dali.tfrecord as tfrec
    from nvidia.dali.fn import readers
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('base', metavar='DIR',default='/raid/data/huangsheng/RoadDisease/CQU-BPDD',
                        help='base dir path')
    parser.add_argument('--data_dir', metavar='DIR',default='/raid/data/huangsheng/RoadDisease/CQU-BPDD',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='effi-b3',
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: effi-b3)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--classes', default=8, type=int, metavar='N',
                        help='num of class')
    parser.add_argument('-b', '--batch-size', default=6, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--gpu',  default=0, type=int,
                        metavar='GPU', help='gpu id')
    parser.add_argument('--lr', '--learning-rate', default=8e-4, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--sparse_ratio', type=float, default=.0)
    parser.add_argument('--is_pretrain', type=int, default=1)
    parser.add_argument('--l1_loss', type=float, default=8e-4)
    parser.add_argument('--keep_batchnorm_fp32', type=int, default=0)
    parser.add_argument('--title', type=str, default='wsplin_noNormal')
    parser.add_argument('--apex_distributed', type=int, default=0)
    parser.add_argument('--sync_bn', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='/data/zhangxiaoxian/')
    parser.add_argument('--is_thumb', dest='is_thumb', action='store_true',
                        help='train and evaluate model on thumb data')
    parser.add_argument('--binary', dest='binary', action='store_true',
                        help='evaluate model on binary data')
    parser.add_argument('--binary_test', dest='binary_test', action='store_true',
                        help='evaluate model on binary data')
    parser.add_argument('--load_model', dest='load_model', action='store_true',
                        help='evaluate model on binary data')
    parser.add_argument('--load_test', type=str,default='',
                        help='evaluate model on binary data')
    parser.add_argument('--miniset', dest='miniset', action='store_true',
                        help='use the miniset to train')
    parser.add_argument('--attention', dest='attention', action='store_true',
                        help='use the miniset to train')
    parser.add_argument('--randsm', dest='randsm', action='store_true',
                        help='use the miniset to train')
    parser.add_argument('--noDA', dest='noDA', action='store_true',
                        help='use the miniset to train')
    parser.add_argument('--noIP', dest='noIP', action='store_true',
                        help='use the miniset to train')
                        

    args = parser.parse_args()
    return args

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

    
class TfRecordPipe(Pipeline):
    #def __init__(self, batch_size, num_threads, device_id, path, index_path,crop,
                # shard_id=0, num_shards=1, dali_cpu=False,stride=300,shape=[900,1200],is_training=True):
    def __init__(self,batch_size, num_threads, device_id, path, index_path,crop,data_dir='',
                 shard_id=0, num_shards=1, dali_cpu=True,stride=[300,300],shape=[900,1200],is_training=True,is_thumb=True):
        super(TfRecordPipe, self).__init__(batch_size,
                                num_threads,
                                device_id,)
        #self.input = ops.FileReader(file_root=data_dir,
                                #    random_shuffle=True,
                                    #pad_last_batch=True)
        self.stride=[300,300]      #[300,300]   cracktree200[150,200]
        self.shape=[900,1200]     # shape[900,1200]   cracktree200[600,900]
        self.is_thumb = is_thumb
        self.is_training = is_training
        '''self.input = readers.tfrecord(path = path,
                            index_path = index_path,
                            features = {"image" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                        "label": tfrec.FixedLenFeature([1], tfrec.float32,-1.1)},
                            shard_id=shard_id,
                            num_shards=num_shards,
                            random_shuffle=is_training,
                            pad_last_batch=True,
                            name="Reader",
                            initial_fill=102400)'''
        self.f_input = ops.FileReader(file_root=data_dir,
                            shard_id=shard_id,
                            num_shards=num_shards,
                            random_shuffle=is_training,
                            pad_last_batch=True,
                            name="Reader")
        #eii = DataIterator(path,batch_size)
        #self.iterator = iter(eii)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.GRAY)
        self.wp,self.hp = get_shape(self.shape,self.stride)
        self.res_300 = ops.Resize(device=dali_device,
                              resize_x=crop,
                              resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)
        self.res_600 = ops.Resize(device=dali_device,
                              resize_x=600,
                              resize_y=600,
                              interp_type=types.INTERP_TRIANGULAR)
        self.res = ops.Resize(device=dali_device,
                              resize_x=1200,
                              resize_y=900,
                              interp_type=types.INTERP_TRIANGULAR)
        nor = [0.455*255,0.225*255] if is_thumb else [0.455*255,0.225*255]
        self.nor = ops.CropMirrorNormalize(device=dali_device,
                                #mean=[0.485,0.456,0.406],
                                #std=[0.229,0.224,0.225])
                                mean=nor[0],
                                std=nor[1])

        self.reshape = ops.Reshape(device=dali_device,shape=((900,1200,1)))
        self.reshape_300 = ops.Reshape(device=dali_device,shape=((300,300,1)))
        self.vflip = ops.Flip(device=dali_device,vertical=1,horizontal=0)
        self.hflip = ops.Flip(device=dali_device,vertical=0,horizontal=1)
        self.rotate = ops.Rotate(device = dali_device, interp_type = types.INTERP_LINEAR,fill_value=0,keep_size=True)
        self.crop = ops.Crop(device=dali_device,crop=[300,300])
        self.slice = ops.Slice(device=dali_device,shape=[300,300])
        self.cat = ops.Cat(device=dali_device)
        self.stack = ops.Stack(device=dali_device)
        if self.is_training:
            self.ran = ops.random.CoinFlip(probability=0.)
        else:
            self.ran = ops.random.CoinFlip(probability=0.)
        self.noised = ops.random.CoinFlip(probability=.1,shape=(900,1200,1))
        self.ran_range = ops.random.Uniform(range=(-30,30),dtype=types.DALIDataType.FLOAT)
        if shard_id == 0:
            print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):   
        #images = self.input["image"]
        #labels = self.input["label"].gpu()
        images,labels = self.f_input()
        labels = labels.gpu()
        images = self.decode(images)
        #images = images.gpu()
        
        #if self.is_training:
        
        
        if self.is_thumb:
            #if self.is_training:
            #images = self.rotate(images,angle=self.ran_range())
            images = self.res_300(images)
            images = self.reshape_300(images)
            image = self.nor(images,mirror=self.ran())
            image = self.cat(image,image,image,axis = 0)
        else:
            images = self.res(images)
            images = self.reshape(images)
            #if self.is_training:
            #images = self.rotate(images,angle=self.ran_range())
            #else:
                #images = self.rotate(images,angle=15)
            images_600 = self.res_600(images)
            images16 = self.nor(self.res_300(images))
            
            imgs = []
            for i in range(self.hp):
                for m in range(self.wp):
                    imgs.append(self.nor(self.slice(images, start=(types.Constant([m*self.stride[1],i*self.stride[0]])))))
            for i in range(2):
                for m in range(2):
                    imgs.append(self.nor(self.slice(images_600, start=(types.Constant([m*300,i*300])))))
            imgs.append(images16)
            image = self.stack(imgs[0],imgs[1],imgs[2],imgs[3],imgs[4],imgs[5],imgs[6],imgs[7],imgs[8],imgs[9],imgs[10],imgs[11],imgs[12],imgs[13],imgs[14],imgs[15],imgs[16])
            #image = self.stack(imgs[0],imgs[1],imgs[2],imgs[3],imgs[4],imgs[5],imgs[6],imgs[7],imgs[8],imgs[9],imgs[10],imgs[11])
         
            image = self.cat(image,image,image,axis = 1)
        return image.gpu(),labels

def get_shape(shape,stride):
        wp = int((shape[1] - 300) / stride[1] + 1)
        hp = int((shape[0] - 300) / stride[0] + 1)
        return wp, hp        
def main():
    global LAMBDA_L1,LAMBDA_BIN
    LAMBDA_L1 = 1e-3
    LAMBDA_BIN = 1e-3
    
    global best_prec1, args,best_auc
    best_prec1 = 0
    best_auc = 0
    best_f1 = 0
    args = parse()

    if not len(args.data_dir):
        raise Exception("error: No data set provided")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    # make apex optional
    if args.opt_level is not None or args.apex_distributed or args.sync_bn:
        try:
            global DDP, amp, optimizers, parallel
            from apex.parallel import DistributedDataParallel as DDP
            from apex import amp, optimizers, parallel
            args.apex_distributed = 0
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                 init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        args.world_size = 1
        device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)

    args.total_batch_size = args.world_size * args.batch_size
    #args.lr = args.lr*float(args.batch_size*args.world_size*17)/256.
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    #model = WSPLIN_IP(args.local_rank,args.is_pretrain,args.arch)
    if args.binary:
        num_class = 2
        args.classes = 2
    else:
        num_class = args.classes
    if args.is_thumb:
        if args.arch=='vit':
            model = ViT(image_size = 300,
                   patch_size = 300,
                   num_classes = num_class,
                   num_patch=17,
                   dim = 1024,
                   depth = 6,
                   heads = 16,
                   mlp_dim = 2048,
                   dropout = 0.1,
                   emb_dropout = 0.1)
        elif args.arch=='effi-b3':
            MODEL_NAME = 'efficientnet-b3'
            if not args.evaluate:
                model = EfficientNet.from_pretrained(MODEL_NAME,num_classes=num_class)
            else:
                model = EfficientNet.from_name(MODEL_NAME,num_classes=num_class)
            
    else:
        model = WSPLIN_IP(shard_id=args.gpu,is_pretrain=args.is_pretrain,arch=args.arch,num_classes=num_class,binary=args.binary,mini=args.miniset,attention=args.attention,patches=math.ceil(17*args.sparse_ratio))
    
    
    model.cuda()

    # Scale learning rate based on global batch size
    
    if args.is_thumb:
        #optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        optimizer = RangerLars(model.parameters(),lr=args.lr, weight_decay=0)
        
    else:
        #optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        optimizer = RangerLars(model.parameters(),lr=args.lr, weight_decay=1e-5)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.opt_level is not None:
        model, optimizer = amp.initialize(model, optimizer,
                                opt_level='O1',
                                          )
        #scaler = GradScaler()
        scaler = ''

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        if args.apex_distributed:
            model = DDP(model, delay_allreduce=True)
        else:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
            
        # not args.is_thumb:
        model = model.module
        # Optionally resume from a checkpoint
    if args.resume:
        #for multi gpu test
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                #model.load_state_dict(torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu)))
                #model.load_state_dict(torch.load(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                if args.load_model:
                    model.load_state_dict(checkpoint) #/data/tangwenhao/model/best_model_parameters_04.pt
                else:
                    args.start_epoch = checkpoint['epoch']
                    global best_prec1
                    best_prec1 = checkpoint['best_prec1']
                    if args.binary:
                        global best_auc
                        best_auc = checkpoint['best_auc']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()         
     # Data loading code
    if args.miniset:
        train_tfr = os.path.join(args.data_dir, 'cqu_bpdd_train_mini.tfrecord')
        train_idx = os.path.join(args.data_dir, 'cqu_bpdd_train_mini.idx')
    else:
        train_tfr = os.path.join(args.data_dir, 'cqu_bpdd_train.tfrecord')
        train_idx = os.path.join(args.data_dir, 'cqu_bpdd_train.idx')
    
    val_tfr = os.path.join(args.data_dir, 'cqu_bpdd_val.tfrecord')
    val_idx = os.path.join(args.data_dir, 'cqu_bpdd_val.idx')
   
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    count_class = np.array([1000.,425.,1000.,478.,1200.,518.,5000.,519.],dtype='float32')
    belt = 1.5
    weight  = 1/((count_class**(1/belt)) / sum(count_class**(1/belt)) / (1/8))
    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weight)).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    crop_size = 300
    if args.evaluate:
        test_tfr = os.path.join(args.data_dir, 'cqu_bpdd_test.tfrecord')
        test_idx = os.path.join(args.data_dir, 'cqu_bpdd_test.idx')
        test_dir = os.path.join(args.data_dir, 'test')
        # pipe = TfRecordPipe(batch_size=args.batch_size,
        #             num_threads=args.workers,
        #             device_id=args.gpu,
        #             path = test_tfr,
        #             index_path = test_idx,
        #             crop=crop_size,
        #             shard_id=args.local_rank,
        #             num_shards=args.world_size,
        #             is_training=False,
        #             is_thumb = args.is_thumb,
        #             data_dir = test_dir)
        # pipe.build()
        # test_loader = DALIClassificationIterator(pipe,reader_name='Reader',last_batch_policy=LastBatchPolicy.PARTIAL)
        test_transform = A.Compose([
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.ShiftScaleRotate(rotate_limit=15.0, p=0.7)
            ])
        test_transform = None
        test_dataset = ImageWiseDataset(test_dir, test_transform,stride=STRIDE,noIP=args.noIP)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,pin_memory=True)

        if args.load_test=='':
            #Random SM
            if args.randsm:
                for i in range(5):
                    _index = [random.sample(range(12),math.ceil(12 * args.sparse_ratio)),random.sample(range(12,16),math.ceil(4 * args.sparse_ratio)),[-1]]
                    index = [ind for st in _index for ind in st]
                    top_1,top_5,losses,auc,pred,label = validate(test_loader, model, criterion,save_pre=True,index=index)
                    _save_path = os.path.join(args.base,'result/')+args.title+str(i)+'.npz'
                    np.savez(_save_path,pred=np.array(pred),label=np.array(label))
            else:
                index = None
                top_1,top_5,losses,auc,pred,label = validate(test_loader, model, criterion,save_pre=True,index=index)
                _save_path = os.path.join(args.base,'result/')+args.title+'.npz'
                np.savez(_save_path,pred=np.array(pred),label=np.array(label))
        else:
            dic=np.load(args.load_test)
            label=np.array(dic['label'])
            pred=np.array(dic['pred'])
            
        pred=pred.reshape((-1,args.classes))
        if pred.shape[1]>2:
            pred = 1-pred[:,6]
            for m in range(len(label)):
                if int(label[m])==6:
                    label[m] = 0
                else:
                    label[m] = 1
        else:
            pred = pred[:,1]
        precision,recall,thr=precision_recall_curve(label, pred)
        stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]  
        patr=getDataByStick([precision,recall],stick)
        f = open("log.txt","a")
        print(patr,file=f)
        f.close()
        if args.local_rank==0:
            f = open("log.txt","a")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),file = f)
            print(args.title+" validate",file=f)
            print(' * Prec@1 {top1:.3f} Prec@5 {top5:.3f} AUC {auc:.3f}'
            .format(top1=top_1, top5=top_5,auc=auc),file=f)
            f.close()
        return
    '''A.OneOf([
            A.Emboss(p=1),
            A.Sharpen(p=1),
            A.Blur(p=1)
        ], p=0.5)'''

    train_transform = A.Compose([]) if args.noDA else A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # chagne rotate limit from 25 to 15 on 20/7/2020
        A.ShiftScaleRotate(rotate_limit=15.0, p=0.7),
    ])
    val_transform = A.Compose([
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.ShiftScaleRotate(rotate_limit=15.0, p=0.7)
    ])
    val_transform = None
    # pipe = TfRecordPipe(batch_size=args.batch_size,
    #                 num_threads=args.workers,
    #                 device_id=args.gpu,
    #                 path = train_tfr,
    #                 index_path = train_idx,
    #                 crop=crop_size,
    #                 shard_id=args.local_rank,
    #                 num_shards=args.world_size,
    #                 is_training=True,
    #                 is_thumb = args.is_thumb,
    #                 data_dir=train_dir)
    # pipe.build()
    # train_loader = DALIClassificationIterator(pipe,reader_name='Reader',last_batch_policy=LastBatchPolicy.DROP)

    # pipe = TfRecordPipe(batch_size=args.batch_size,
    #                 num_threads=args.workers,
    #                 device_id=args.gpu,
    #                 path = val_tfr,
    #                 index_path = val_idx,
    #                 crop=crop_size,
    #                 shard_id=args.local_rank,
    #                 num_shards=args.world_size,
    #                 is_training=False,
    #                 is_thumb = args.is_thumb,
    #                 data_dir=val_dir)
    # pipe.build()
    # val_loader = DALIClassificationIterator(pipe,reader_name='Reader',last_batch_policy=LastBatchPolicy.PARTIAL)

    train_dataset = ImageWiseDataset(train_dir, train_transform,stride=STRIDE,noIP=args.noIP)
    val_dataset = ImageWiseDataset(val_dir, val_transform,stride=STRIDE,noIP=args.noIP)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,pin_memory=True)
    train_size = len(train_loader)
    num_train_steps = int(math.ceil((train_size)*args.batch_size / args.total_batch_size)) * args.epochs
    #flat_steps = int(math.ceil((train_loader._size)*args.batch_size / args.total_batch_size)) * 0.75
    flat_steps = float(num_train_steps*0.75)-1
    scheduler = flat_cosine_schedule(
        optimizer,
        num_flat_steps = flat_steps,
        num_training_steps = num_train_steps
    )  
    total_time = AverageMeter()
    if args.local_rank==0:
        f = open("log.txt","a")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),file = f)
        print(args.title+" train",file=f)
        f.close()
    
    for epoch in range(args.start_epoch, args.epochs):
        #Random SM
        if args.randsm:
            _index = [random.sample(range(12),math.ceil(12 * args.sparse_ratio)),random.sample(range(12,16),math.ceil(4 * args.sparse_ratio)),[-1]]
            index = [ind for st in _index for ind in st]
        else:
            index = None
    
        # train for one epoch
        avg_train_time,avg_train_top1,avg_train_loss = train(train_loader, model, criterion, optimizer, epoch,scheduler,args.epochs,scaler,index)
        total_time.update(avg_train_time)
 

        # evaluate on validation set
        
        [prec1, prec5,val_loss,auc,f1] = validate(val_loader, model, criterion,index=index)

        # remember best prec@1 and save checkpoint
        if args.local_rank==0:
            if args.binary:
                is_best = auc > best_auc and epoch >0
                best_auc = max(auc, best_auc)
            else:
                #is_best = prec1 > best_prec1 and epoch > 0
                is_best = f1 > best_f1 and epoch > 0
            best_prec1 = max(prec1, best_prec1)
            best_f1 = max(f1,best_f1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_auc': best_auc,
                'optimizer' : optimizer.state_dict(),
            }, is_best,dir=args.save_dir,title=args.title,model=model)
            #save_checkpoint(model.state_dict(), is_best,dir=args.save_dir,title=args.title,model=model)
            #if epoch == args.epochs - 1:
            f = open("log.txt","a")
            if epoch>-1 and args.local_rank==0:
                print('##Epoch {0}\n'
                      '##Train Top-1 {1}\n'
                      '##Loss {2}\n'
                      '##Test Top1 {3}\n'
                      '##Top-5 {4}  \n'
                      '##Loss  {5}\n'
                      '##AUC  {6}\n'.format(
                      epoch,
                      avg_train_top1,
                      avg_train_loss,
                      prec1,
                      prec5,
                      val_loss,
                      auc),
                      file=f)
            f.close()
        #train_loader.reset()
        #val_loader.reset()
    if args.local_rank==0:
        f = open("log.txt","a")
        print("##Best Test Top1 {0}",best_prec1,file=f)
        f.close()
    os.remove(args.save_dir+args.title+'_checkpoint.pth.tar')
    best_model_dir = args.save_dir+args.title+'_model_best.pth.tar'
    checkpoint = torch.load(best_model_dir, map_location = 'cpu')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = RangerLars(model.parameters(),lr=args.lr, weight_decay=0)
    if args.opt_level is not None:
        model, optimizer = amp.initialize(model, optimizer,
                                opt_level='O1',
                                          )
    test_tfr = os.path.join(args.data_dir, 'cqu_bpdd_test.tfrecord')
    test_idx = os.path.join(args.data_dir, 'cqu_bpdd_test.idx')
    test_dir = os.path.join(args.data_dir, 'test')
    del train_loader,val_loader
    # pipe = TfRecordPipe(batch_size=args.batch_size,
    #             num_threads=args.workers,
    #             device_id=args.gpu,
    #             path = test_tfr,
    #             index_path = test_idx,
    #             crop=crop_size,
    #             shard_id=args.local_rank,
    #             num_shards=args.world_size,
    #             is_training=False,
    #             is_thumb = args.is_thumb,
    #             data_dir = test_dir)
    # pipe.build()
    # test_loader = DALIClassificationIterator(pipe,reader_name='Reader',last_batch_policy=LastBatchPolicy.PARTIAL)
    test_transform = A.Compose([
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.ShiftScaleRotate(rotate_limit=15.0, p=0.7)
        ])
    test_transform = None
    test_dataset = ImageWiseDataset(test_dir, test_transform, stride=STRIDE,noIP=args.noIP)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,pin_memory=True)

    #Random SM
    if args.randsm:
        for i in range(5):
            _index = [random.sample(range(12),math.ceil(12 * args.sparse_ratio)),random.sample(range(12,16),math.ceil(4 * args.sparse_ratio)),[-1]]
            index = [ind for st in _index for ind in st]
            top_1,top_5,losses,auc,pred,label = validate(test_loader, model, criterion,save_pre=True,index=index)
            _save_path = os.path.join(args.base,'result/')+args.title+str(i)+'.npz'
            np.savez(_save_path,pred=np.array(pred),label=np.array(label))
    else:
        index = None
        top_1,top_5,losses,auc,pred,label = validate(test_loader, model, criterion,save_pre=True,index=index)
        _save_path = os.path.join(args.base,'result/')+args.title+'.npz'
        np.savez(_save_path,pred=np.array(pred),label=np.array(label))
    pred=pred.reshape((-1,args.classes))
    if pred.shape[1]>2:
        pred = 1-pred[:,6]
        for m in range(len(label)):
            if int(label[m])==6:
                label[m] = 0
            else:
                label[m] = 1
    else:
        pred = pred[:,1]
    precision,recall,thr=precision_recall_curve(label, pred)
    stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]  
    patr=getDataByStick([precision,recall],stick)
    f = open("log.txt","a")
    print(patr,file=f)
    f.close()
    if args.local_rank==0:
        f = open("log.txt","a")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),file = f)
        print(args.title+" validate",file=f)
        print(' * Prec@1 {top1:.3f} Prec@5 {top5:.3f} AUC {auc:.3f}'
        .format(top1=top_1, top5=top_5,auc=auc),file=f)
        f.close()

def train(train_loader, model, criterion, optimizer, epoch,scheduler,max_epochs,scaler=None,index=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    a = 0
    sparse_index=args.sparse_ratio
    alph = epoch/(max_epochs-1)
    for i, data in enumerate(train_loader):
        a = a +1
        #input = data[0]["data"]
        #target = data[0]["label"].squeeze(-1).long()
        input,target = data[0].cuda(non_blocking=True),data[1].cuda(non_blocking=True)
        
        if input.size(0) <= 0:
            continue
        
        if index:
            for j in range(len(input)):
                input_tmp = input[j,index].clone()
                input_tmp = torch.unsqueeze(input_tmp,dim=0)
                if j == 0:
                    input_ = input_tmp.clone()
                else:
                    input_ = torch.cat((input_,input_tmp))
            del input,input_tmp
            input = input_.clone().cuda()
            del input_
        elif sparse_index > 0 :
            for j in range(len(input)):
                '''input_tmp = input[j,[int(np.median(i)) for i in np.split(np.array(list(range(12))),math.ceil(len(input[j,:12]) * sparse_index))]].clone()                
                input_tmp = torch.cat((input_tmp,input[j,[int(np.median(i)) for i in np.split(np.array(list(range(12,16))),math.ceil(len(input[j,12:16]) * sparse_index))]],torch.unsqueeze(input[j,-1],dim=0)))'''
                '''input_tmp = input[j,random.sample(range(12),math.ceil(len(input[j,:12]) * sparse_index))].clone()
                input_tmp = torch.cat((input_tmp,input[j,random.sample(range(12,16),math.ceil(len(input[j,12:16]) * sparse_index))],torch.unsqueeze(input[j,-1],dim=0)))'''
                # #input_tmp = input[j,[1,3,4,5,6,7,8,9,10]].clone()     
                # input_tmp = input[j,[1,3,5,7,9,10]].clone() 
                # #input_tmp = input[j,[1,5,10]].clone()             
                # input_tmp = torch.cat((input_tmp,input[j,[14]],torch.unsqueeze(input[j,-1],dim=0)))
                if sparse_index == 0.5:
                    input_tmp = input[j,[1,3,5,7,9,10]].clone()           
                    input_tmp = torch.cat((input_tmp,input[j,[13,14]],torch.unsqueeze(input[j,-1],dim=0)))
                elif sparse_index == 0.25:
                    input_tmp = input[j,[1,5,10]].clone()             
                    input_tmp = torch.cat((input_tmp,input[j,[14]],torch.unsqueeze(input[j,-1],dim=0)))
                elif sparse_index == 0.75:
                    input_tmp = input[j,[1,3,4,5,6,7,8,9,10]].clone()        
                    input_tmp = torch.cat((input_tmp,input[j,[13,14,15]],torch.unsqueeze(input[j,-1],dim=0)))
                input_tmp = torch.unsqueeze(input_tmp,dim=0)
                if j == 0:
                    input_ = input_tmp.clone()
                else:
                    input_ = torch.cat((input_,input_tmp))
            del input,input_tmp
            input=input_.clone().cuda()
            del input_
        
        
        #if args.binary:
        m = 0
        target_bin = target.clone()
        for m in range(len(target_bin)):
            if int(target_bin[m])==6:
                target_bin[m] = 0
            else:
                target_bin[m] = 1
        
        train_loader_len = int(len(train_loader) / args.batch_size)
        
        # compute output
        
        #feature_maps = model.feature_extractor(input)
        #with autocast():
        if args.is_thumb:
            predictions = model(input)
            bs = input.size(0)
            del input
            if args.binary:
                classify_loss = criterion(predictions, target_bin)
            else:
                classify_loss = criterion(predictions, target)
            loss = classify_loss
        else:
            feature_maps = model.feature_extractor(input)
            bs = input.size(0)
            del input
            predictions = model.classifier(feature_maps,bs)

            #pre_bin = torch.stack((predictions[:,6],(torch.sum(predictions,dim=1)-predictions[:,6])),dim=1)
            #classify_loss_bin = criterion(pre_bin , target_bin)
            if args.binary:
                classify_loss = criterion(predictions, target_bin)

            else:
                classify_loss = criterion(predictions, target)
            #l1_loss = LAMBDA_L1 * l1_regularizer(feature_maps)
            #score_ = torch.nn.functional.softmax(feature_maps.view(feature_maps.size(0),-1,args.classes),2)
            #l1_loss = 1e-2 * l1_regularizer(score_[:,:,1].view(feature_maps.size(0),-1))

            #dis_feat = feature_maps[list(np.where(target.cpu().numpy()!=6))[0]]
            #print(dis_feat.size())
            dis_feat = feature_maps
            if dis_feat.size(0)>0:
                #l1_loss = 1e-2 * l1_regularizer(dis_feat.view(dis_feat.size(0),-1,args.classes)[:,:,1])
                l1_loss = args.l1_loss * l1_regularizer(dis_feat)
            else:
                l1_loss = 0
            #print(l1_loss)
            #l1_loss = LAMBDA_L1 * l1_regularizer(dis_feat)
            #thu_loss = criterion(feature_maps.view(feature_maps.size(0),-1,args.classes)[:,-1],target_bin)
            '''nor_index=feature_maps[list(np.where(target.cpu().numpy()==6))[0]]
            if len(nor_index)>0:
                nor_loss=criterion(nor_index.view(-1,args.classes),torch.ones((nor_index.size(0)*17)).long().cuda()*0)
            else:
                nor_loss=0'''
            #print(l1_loss)
            loss = classify_loss+l1_loss
            #loss = classify_loss+nor_loss
            del feature_maps,dis_feat
        
        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            #scaler.scale(loss).backward()
        else:
             loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        #scaler.step(optimizer)
        #scaler.update()
        optimizer.step()
        scheduler.step()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            if args.binary:
                topk = (1,1)
            else:
                topk = (1,5)
                
            prec1, prec5 = accuracy(predictions.data, target, topk=topk)


            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), bs)
            top1.update(to_python_float(prec1), bs)
            top5.update(to_python_float(prec5), bs)

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, train_loader_len,
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
        
        # Pop range "Body of iteration {}".format(i)
    prec1, prec5 = accuracy(predictions, target, topk=topk)
    if args.distributed:
        reduced_loss = reduce_tensor(loss.data)
        prec1 = reduce_tensor(prec1)
        prec5 = reduce_tensor(prec5)
    else:
        reduced_loss = loss.data
    top1.update(to_python_float(prec1), bs)
    losses.update(to_python_float(reduced_loss), bs)
    return batch_time.avg, top1.avg,losses.avg
    
def getDataByStick(data,stick):
    '''
    data    [data_y,data_x]   y坐标数据，x坐标数据
    stick                     坐标刻度
    '''
    j = -1
    diff = 9999
    _stick = stick.copy()
    for a in range(0,len(data[0])):
        if len(_stick):
            diff_tem = abs(data[1][a]-_stick[j])
            if diff_tem < diff:
                diff = diff_tem
            if diff_tem > diff:
                print(data[0][a-1],data[1][a-1])
                if len(_stick)>1:
                    diff = abs(data[1][a]-_stick[j-1])
                else:
                    diff=9999
                del _stick[j]
                
def validate(val_loader, model, criterion,save_pre=False,index=None):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    phase_pred = np.array([])
    phase_label = np.array([])
    save_pred = np.array([])
    save_label = np.array([])
    
    # switch to evaluate mode
    model.eval()
    sparse_index = args.sparse_ratio
    end = time.time()
    val_loader_len = int(len(val_loader) / args.batch_size)
    for i, data in enumerate(val_loader):
        #input = data[0]["data"]
        #target = data[0]["label"].squeeze(-1).long()
        input,target_ori = data[0].cuda(non_blocking=True),data[1].cuda(non_blocking=True)
        if input.size(0) <= 0:
            continue
        if index:
            for j in range(len(input)):
                input_tmp = input[j,index].clone()
                input_tmp = torch.unsqueeze(input_tmp,dim=0)
                if j == 0:
                    input_ = input_tmp.clone()
                else:
                    input_ = torch.cat((input_,input_tmp))
            del input,input_tmp
            input = input_.clone().cuda()
            del input_
        elif sparse_index > 0 :
            for j in range(len(input)):
                '''input_tmp = input[j,[int(np.median(i)) for i in np.split(np.array(list(range(12))),math.ceil(len(input[j,:12]) * sparse_index))]].clone()                 
                input_tmp = torch.cat((input_tmp,input[j,[int(np.median(i)) for i in np.split(np.array(list(range(12,16))),math.ceil(len(input[j,12:16]) * sparse_index))]],torch.unsqueeze(input[j,-1],dim=0)))'''
                #input_tmp = input[j,[1,3,4,5,6,7,8,9,10]].clone()
                # input_tmp = input[j,[1,3,5,7,9,10]].clone()
                # #input_tmp = input[j,[1,5,10]].clone()                  
                # input_tmp = torch.cat((input_tmp,input[j,[14,15]],torch.unsqueeze(input[j,-1],dim=0)))
                if sparse_index == 0.5:
                    input_tmp = input[j,[1,3,5,7,9,10]].clone()           
                    input_tmp = torch.cat((input_tmp,input[j,[13,14]],torch.unsqueeze(input[j,-1],dim=0)))
                elif sparse_index == 0.25:
                    input_tmp = input[j,[1,5,10]].clone()             
                    input_tmp = torch.cat((input_tmp,input[j,[14]],torch.unsqueeze(input[j,-1],dim=0)))
                elif sparse_index == 0.75:
                    input_tmp = input[j,[0,1,4,5,6,7,8,9,10]].clone()        
                    input_tmp = torch.cat((input_tmp,input[j,[13,14,15]],torch.unsqueeze(input[j,-1],dim=0)))
                input_tmp = torch.unsqueeze(input_tmp,dim=0)
                if j == 0:
                    input_ = input_tmp.clone()
                else:
                    input_ = torch.cat((input_,input_tmp))
            del input,input_tmp
            input=input_.clone().cuda()
            del input_
            
        if args.binary or args.binary_test:
            if args.binary_test:
                target_bin = target_ori.clone()
                target_tmp = target_bin
            else:
                target_tmp = target_ori
            if args.binary:
                topk = (1,1)
            else:
                topk = (1,5)
            m = 0
            #print(target)
            for m in range(len(target_tmp)):
                if int(target_tmp[m])==6:
                    target_tmp[m] = 0
                else:
                    target_tmp[m] = 1
        else:
            topk = (1,5)
        # compute output
        with torch.no_grad():
           # with autocast():
            output = model(input,bs=input.size(0))
            if args.binary:
                loss = criterion(output, target_tmp)
            else:
                loss = criterion(output, target_ori)
        
        #if save_pre:
        output_soft = nn.functional.softmax(output,dim=1)
        save_pred = np.append(save_pred,output_soft.cpu().numpy())
        save_label = np.append(save_label,target_ori.data.cpu().numpy())
        
        if args.binary or args.binary_test:           
            if args.binary:
                prec1, prec5 = accuracy(output, target_tmp, topk=topk)
                output = nn.functional.softmax(output,dim=1)
                preds = output[:,1]
            else:
                output_soft = nn.functional.softmax(output,dim=1)
                preds = 1-output_soft[:,6]
                prec1, prec5 = accuracy(output, target_ori, topk=topk)
                #prec1, prec5 = accuracy(torch.cat((output[:,6].view(-1,1),(torch.sum(output,dim=1)-output[:,6]).view(-1,1)),1), target, topk=topk)
            phase_label = np.append(phase_label, target_tmp.data.cpu().numpy())
            phase_pred = np.append(phase_pred, preds.cpu().numpy())
        else:
            prec1, prec5 = accuracy(output, target_ori, topk=topk)
            
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data
        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                   i, val_loader_len,
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    auc = 0
    ma_f1 = f1_score(save_label,np.argmax(save_pred.reshape(-1,8),axis=1),average='macro')
    if args.binary or args.binary_test:
        auc = roc_auc_score(phase_label, phase_pred)
    
    if args.local_rank == 0:
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} AUC {auc:.3f} F1{f1:.3f} '
            .format(top1=top1, top5=top5,auc=auc,f1=ma_f1))
    auc=0
    if save_pre:
        return [top1.avg, top5.avg,losses.avg,auc,save_pred,save_label]
    else:
        return [top1.avg, top5.avg,losses.avg,auc,ma_f1]


def save_checkpoint(state, is_best, dir,title,model):
    c_filename = dir+title+'_checkpoint.pth.tar'
    b_filename = dir+title+'_model_best.pth.tar'
    
    torch.save(state, c_filename)

    if is_best:
        shutil.copyfile(c_filename, b_filename)
        #torch.save(state, b_filename)

def l1_regularizer(input_features):
    l1_loss = torch.norm(input_features, p=1)
    input_features = input_features.view(input_features.size(0),-1,args.classes)
    if input_features.size(0)>0:
        #return l1_loss / (input_features.size(0) * input_features.size(1))
        return l1_loss / (input_features.size(0))
    else:
        return l1_loss

def l2_regularizer(input_features):
    l2_loss = torch.norm(input_features, p=2)
    if input_features.size(0)>0:
        return l2_loss / input_features.size(0)
    else:
        return l2_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def flat_cosine_schedule(optimizer: Optimizer, num_flat_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    def lr_lambda(current_step):
        if current_step < num_flat_steps:
            return 1.0
        progress = float(current_step - num_flat_steps) / float(max(1, num_training_steps - num_flat_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
