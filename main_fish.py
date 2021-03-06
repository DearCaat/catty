import os
from networkx.algorithms import cluster

from torch.nn.modules import module
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse
import datetime
import numpy as np
from copy import deepcopy
import math

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import *
from timm.loss import *
from timm.models import  model_parameters

from config import get_config
from collections import OrderedDict
from models import build_model
from data import build_loader
from utils import ModelEmaV3, get_sigmod_num
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import *

from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score
from contextlib import suppress
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

def parse_option():
    parser = argparse.ArgumentParser('WSPLIN training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', nargs='+')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--trainer', type=str, help='name of trainer')
    parser.add_argument('--tfrecord', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--epochs', type=int, help="train epochs")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--no-amp',  action='store_true',
                        help='disable mixed precision, default use native amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--model-name',  type=str, help='model name')
    parser.add_argument('--project',  type=str, help='experiment project')
    parser.add_argument('--title',  type=str, help='experiment title')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--thumb', action='store_true', help='Use thumb data')
    parser.add_argument('--ema', action='store_true', help='Use thumb data')
    parser.add_argument('--pin-memory', action='store_true', help='Use thumb data')
    parser.add_argument('--binary-train', action='store_true', help='train the model with binary setting')
    parser.add_argument('--load-test-dir', type=str, metavar='PATH',help='the file of tested data')
    parser.add_argument('--pretrained-backbone', type=str, metavar='PATH',help='the file of pretrained model')
    parser.add_argument('--train-mode', type=str, default='t_e', choices=['train', 'eval', 't-e','predict'],
                        help='train: only train, '
                             'eval: only test , '
                             't_e: first train the model, and use it to eval'
                             'predict: only output predict score')

    # distributed training
    parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)
    return args, config


def main(config):
    if config.TRAIN_MODE=='train' or config.TRAIN_MODE=='t_e':
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(is_train=True,config=config)
    else:
        dataset_test,data_loader_test = build_loader(is_train=False,config=config)
        
    logger.info(f"Creating model:{config.MODEL.NAME}/{config.MODEL.BACKBONE}")
    model = build_model(config)
    if config.RDD_TRANS.TEACHER_INIT is not None and (config.RDD_TRANS.PERSUDO_LABEL or config.RDD_TRANS.CLUSTER.NAME=='gcn') and not config.THUMB_MODE:
        model_teacher = build_model(config)
        model_teacher.cuda()
        cpt = torch.load(config.RDD_TRANS.TEACHER_INIT, map_location='cpu')
        std = cpt['state_dict']
        if config.RDD_TRANS.INST_NUM_CLASS == config.MODEL.NUM_CLASSES:
            std_ins = dict([('weight',std['head.weight']),('bias',std['head.bias'])])
            model_teacher.head_instance.load_state_dict(std_ins, strict=True)
            model_teacher.head[0].load_state_dict(std_ins, strict=True)
        model_teacher.instance_feature_extractor.load_state_dict(std, strict=False)
        logger.info(f"Teacher model inited")
    elif (config.RDD_TRANS.PERSUDO_LABEL or config.RDD_TRANS.CLUSTER.NAME=='gcn') and not config.RDD_TRANS.TEACHER_INIT and not config.THUMB_MODE:
        model_teacher = model

    else:
         model_teacher = None
        
    #if not config.THUMB_MODE:
        # model_teacher = build_model(config)
        # model_teacher.cuda()
        # cpt = torch.load('/home/tangwenhao/rdd/output/swin_test/model/swin_small_patch4_window7_224_best_model.pth', map_location='cpu')
        # std = cpt['state_dict']
        # std['head_instance.weight'] = std['head.weight']
        # std['head_instance.bias'] = std['head.bias']
        # model_teacher.load_state_dict(std, strict=True)
    # setup augmentation batch splits for contrastive loss or split bn
    '''num_aug_splits = 0
    if config.AUG.SPLITS > 0:
        assert config.AUG.SPLITS > 1, 'A split of 1 makes no sense'
        num_aug_splits = config.AUG.SPLITS

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))'''

    model.cuda()
    logger.info(f"model:{config.MODEL.NAME}/{config.MODEL.BACKBONE}")

    optimizer = build_optimizer(config, model)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if config.LOCAL_RANK == 0:
            logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        # ?????????timm??????????????????????????????????????????????????????gradient accumulation
        loss_scaler = NativeScaler_V2()
        if config.LOCAL_RANK == 0:
            logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if config.LOCAL_RANK == 0:
            logger.info('AMP not enabled. Training in float32.')

    # setup learning rate schedule and starting epoch
    if config.TRAIN_MODE=='eval' or config.TRAIN_MODE=='predict':
        lr_scheduler = ''
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    max_accuracy = 0.0
    best_auc = 0.0
    max_f1 = 0.0
    max_patr90 = .0
    max_f1_ema = .0
    max_accuracy_ema = .0
    best_auc_ema = .0
    max_patr90_ema = .0

    if model_teacher is not None:
        teacher_ema = ModelEmaV3(model_teacher, decay=config.RDD_TRANS.EMA_DECAY, device='cpu' if config.RDD_TRANS.EMA_FORCE_CPU else None, diff_layers=[] if config.RDD_TRANS.EMA_DIFF is None else config.RDD_TRANS.EMA_DIFF,decay_diff=config.RDD_TRANS.EMA_DECAY_DIFF)
        # if config.MODEL.RESUME:
         #    load_checkpoint(config, teacher_ema.module, optimizer, lr_scheduler, logger)
    else:
        teacher_ema = None
    model_ema = None
    if config.MODEL_EMA:    
        model_ema = ModelEmaV3(model, decay=config.EMA_DECAY, device='cpu' if config.EMA_FORCE_CPU else None)
        #if config.MODEL.RESUME:
            #load_checkpoint(config, model_ema.module, optimizer, lr_scheduler, logger)
    # setup exponential moving average of model weights, SWA could be used here too
    '''model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)'''

    # setup distributed training
    if config.DISTRIBUTED:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if config.LOCAL_RANK == 0:
                logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if config.LOCAL_RANK  == 0:
                logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[config.LOCAL_RANK ])  # can use device str in Torch >= 1.1
        model_without_ddp = model.module
        # NOTE: EMA model does not need to be wrapped by DDP
    else:
        model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    # setup loss function
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()

    if config.MODEL.RESUME:
        if config.MODEL.RESUME.lower() == 'auto':
            load_best_model(config, model_without_ddp, logger)
            acc1, acc5, loss, auc,pred,label,eval_metrics = validate(config, data_loader_test, model_without_ddp,save_pre=True,amp_autocast=amp_autocast,criterion=criterion)
            logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {acc1:.2f}% {auc:.2f}%")
            _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_'+config.DATA.DATASET.split('/')[-1])+'.npz'
            np.savez(_save_path,pred=pred,label=label)
            if 'cqu_bpdd' in config.DATA.DATASET:
                for m in range(len(label)):
                    if int(label[m])==6:
                        label[m] = 0
                    else:
                        label[m] = 1
            pred=pred.reshape((-1,config.MODEL.NUM_CLASSES))
            if config.MODEL.NUM_CLASSES > 2:
                pred = 1-pred[:,6]
            else:
                pred = pred[:,1]
            
            precision,recall,thr=precision_recall_curve(label, pred)
            stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]  
            patr=getDataByStick([precision,recall],stick)
            logger.info(patr)

            acc1_ema,auc_ema,eval_metrics_ema,patr_ema = 0,0,None,None
            #ema
            if model_ema is not None or teacher_ema is not None:
                load_best_model(config, model_ema.module if model_ema is not None else teacher_ema.module, logger,is_ema=True)
                
                acc1_ema, acc5_ema,loss_ema,auc_ema,pred,label,eval_metrics_ema = validate(config, data_loader_test, model_ema.module if model_ema is not None else teacher_ema.module,amp_autocast=amp_autocast,criterion=criterion,save_pre=True)
                
                logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {acc1_ema:.2f}% {auc_ema:.2f}%")
                _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_ema_'+config.DATA.DATASET.split('/')[-1])+'.npz'
                np.savez(_save_path,pred=pred,label=label)
                if 'cqu_bpdd' in config.DATA.DATASET:
                    for m in range(len(label)):
                        if int(label[m])==6:
                            label[m] = 0
                        else:
                            label[m] = 1
                pred=pred.reshape((-1,config.MODEL.NUM_CLASSES))
                if config.MODEL.NUM_CLASSES > 2:
                    pred = 1-pred[:,6]
                else:
                    pred = pred[:,1]
                precision,recall,thr=precision_recall_curve(label, pred)
                stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]  
                patr_ema=getDataByStick([precision,recall],stick)
                logger.info(patr_ema)
            
            if config.LOG_WANDB and has_wandb:
                _summary = OrderedDict([('test_top1',acc1),
                                        ('test_f1',eval_metrics['macro_f1']),
                                        ('test_auc',auc),
                                        ('test_patr90',patr[1][0] if patr else 0),
                                        ('test_ema_top1',acc1_ema),
                                        ('test_ema_f1',eval_metrics_ema['macro_f1'] if eval_metrics else 0),
                                        ('test_ema_auc',auc_ema),
                                        ('test_ema_patr90',patr_ema[1][0] if patr_ema else 0),])
                wandb.log(_summary)
            return
        else:
            max_accuracy,best_auc, best_f1 = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
            if config.TRAIN_MODE=='eval':
                if config.LOAD_TEST_DIR:
                    dic=np.load(config.LOAD_TEST_DIR)
                    label=dic['label']
                    pred=dic['pred']
                    #if 'cqu_bpdd' in config.DATA.DATASET:
                    for m in range(len(label)):
                        if int(label[m])==config.DATA.DATA_NOR_INDEX:
                            label[m] = 0
                        else:
                            label[m] = 1
                    pred=pred.reshape((-1,config.MODEL.NUM_CLASSES))
                    if config.MODEL.NUM_CLASSES > 2:
                        pred = 1-pred[:,6]
                    else:
                        pred = pred[:,1]
                    precision,recall,thr=precision_recall_curve(label, pred)
                    stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]  
                    patr=getDataByStick([precision,recall],stick)
                    logger.info(patr)
                else:
                    acc1, acc5, loss, auc,pred,label,eval_metrics = validate(config, data_loader_test, model,save_pre=True,amp_autocast=amp_autocast,criterion=criterion)
                    logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {acc1:.2f}% {auc:.2f}%")
                    _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_'+config.DATA.DATASET.split('/')[-1])+'.npz'
                    np.savez(_save_path,pred=pred,label=label)
                    #if 'cqu_bpdd' in config.DATA.DATASET:
                    for m in range(len(label)):
                        if int(label[m])==config.DATA.DATA_NOR_INDEX:
                            label[m] = 0
                        else:
                            label[m] = 1
                    pred=pred.reshape((-1,config.MODEL.NUM_CLASSES))
                    if config.MODEL.NUM_CLASSES > 2:
                        pred = 1-pred[:,6]
                    else:
                        pred = pred[:,1]
                    precision,recall,thr=precision_recall_curve(label, pred)
                    stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]  
                    patr=getDataByStick([precision,recall],stick)
                    logger.info(patr)
                return
            elif config.TRAIN_MODE=='predict':
                pred,label= predict(config, data_loader_test, model,amp_autocast=amp_autocast)
                _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_predict_'+config.DATA.DATASET.split('/')[-1])+'.npz'
                np.savez(_save_path,pred=pred,label=label)

                return

    logger.info("Start training")
    start_time = time.time()
    loss_rec = np.array([])

    thr_list = np.array([config.RDD_TRANS.NOR_THR for i in range(config.RDD_TRANS.INST_NUM_CLASS)])

    if config.LOG_WANDB and has_wandb:
        wandb.watch(model, log_freq=100)
    # try:
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if not config.DATA.TFRECORD_MODE and config.DISTRIBUTED:
            data_loader_train.sampler.set_epoch(epoch)

        loss_r,train_metrics = train_one_epoch(config, model_without_ddp, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,amp_autocast=amp_autocast, loss_scaler=loss_scaler,teacher_ema=teacher_ema,model_ema=model_ema,thr_list=thr_list)
        #np.append(loss_rec,loss_r)
        acc1, acc5,loss,auc,eval_metrics = validate(config, data_loader_val, model_without_ddp,amp_autocast=amp_autocast,criterion=criterion)
        if teacher_ema is not None:
            acc1_ema, acc5_ema,loss_ema,auc_ema,eval_metrics_ema = validate(config, data_loader_val, teacher_ema.module,amp_autocast=amp_autocast,criterion=criterion)
        elif model_ema is not None:
            acc1_ema, acc5_ema,loss_ema,auc_ema,eval_metrics_ema = validate(config, data_loader_val, model_ema.module.cuda(),amp_autocast=amp_autocast,criterion=criterion)
            if config.EMA_FORCE_CPU:
                model_ema.module.to('cpu')
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        for i in range(len(thr_list)):
            logger.info(f"Thr:  {thr_list[i]:.2f}")
        
        # save the ema checkpoint
        if model_ema is not None or teacher_ema is not None:
            f1_ema = eval_metrics_ema['macro_f1']
            patr90_ema = eval_metrics_ema['p@r90']
            if config.TEST.BEST_METRIC.lower() == 'auc':
                is_best_ema = (auc_ema > best_auc_ema) and epoch>0
            elif config.TEST.BEST_METRIC.lower() == 'top1':
                is_best_ema = (acc1_ema > max_accuracy_ema) and epoch>0
            elif config.TEST.BEST_METRIC.lower() == 'f1':
                is_best_ema = (f1_ema > max_f1_ema) and epoch>0
            elif config.TEST.BEST_METRIC.lower() == 'p@r90':
                is_best_ema = (patr90_ema > max_patr90_ema) and epoch>0
            else:
                raise AttributeError

            max_accuracy_ema = max(max_accuracy_ema, acc1_ema) if epoch > 0 else 0
            best_auc_ema = max(best_auc_ema,auc_ema) if epoch > 0 else 0
            max_f1_ema = max(max_f1_ema,f1_ema)
            max_patr90_ema = max(max_patr90_ema,patr90_ema)

            if config.LOCAL_RANK == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_ema.module if model_ema is not None else teacher_ema.module, max_accuracy_ema, optimizer, lr_scheduler, logger,is_best_ema,best_auc_ema,max_f1_ema,None,is_ema=True,best_patr90=max_patr90_ema)
        else:
            eval_metrics_ema = None

        f1 = eval_metrics['macro_f1']
        patr90 = eval_metrics['p@r90']

        if config.TEST.BEST_METRIC.lower() == 'auc':
            is_best = (auc > best_auc) and epoch>0
        elif config.TEST.BEST_METRIC.lower() == 'top1':
            is_best = (acc1 > max_accuracy) and epoch>0
        elif config.TEST.BEST_METRIC.lower() == 'f1':
            is_best = (f1 > max_f1) and epoch>0
        elif config.TEST.BEST_METRIC.lower() == 'p@r90':
            is_best = (patr90 > max_patr90) and epoch>0
        else:
            raise AttributeError

        max_accuracy = max(max_accuracy, acc1) if epoch > 0 else 0
        best_auc = max(best_auc,auc) if epoch > 0 else 0
        max_f1 = max(max_f1,f1)
        max_patr90 = max(max_patr90,patr90)

        if eval_metrics_ema is not None:
            eval_metrics.update(eval_metrics_ema)
        update_summary(
                    epoch, train_metrics, eval_metrics,os.path.join(config.OUTPUT, 'summary.csv'),
                    write_header=False, log_wandb=config.LOG_WANDB and has_wandb)

        if config.LOCAL_RANK == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger,is_best,best_auc,max_f1,model_ema,best_patr90=max_patr90)

    logger.info(f'Max accuracy: {max_accuracy:.2f}%\t'
                f'Max f1: {max_f1:.2f}%\t'
                f'Max ema_f1: {max_f1_ema:.2f}%\t'
                f'Best AUC: {best_auc:.2f}%')
    if config.LOG_WANDB and has_wandb:
        _summary = OrderedDict([('best_top1',max_accuracy),
                                ('best_f1',max_f1),
                                ('best_auc',best_auc),
                                ('p@r90',max_patr90),
                                ('best_ema_top1',max_accuracy_ema),
                                ('best_ema_f1',max_f1_ema),
                                ('best_ema_auc',best_auc_ema)])
        wandb.log(_summary)
    # except:
    #     print(torch.cuda.memory_stats())
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    if config.TRAIN_MODE=='t_e':
        dataset_test,data_loader_test = build_loader(is_train=False,config=config)

        load_best_model(config, model_without_ddp, logger)
        acc1, acc5, loss, auc,pred,label,eval_metrics = validate(config, data_loader_test, model_without_ddp,save_pre=True,amp_autocast=amp_autocast,criterion=criterion)
        logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {acc1:.2f}% {auc:.2f}%")
        _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_'+config.DATA.DATASET.split('/')[-1])+'.npz'
        np.savez(_save_path,pred=pred,label=label)
        if 'cqu_bpdd' in config.DATA.DATASET:
            for m in range(len(label)):
                if int(label[m])==6:
                    label[m] = 0
                else:
                    label[m] = 1
        pred=pred.reshape((-1,config.MODEL.NUM_CLASSES))
        if config.MODEL.NUM_CLASSES > 2:
            pred = 1-pred[:,6]
        else:
            pred = pred[:,1]
        
        precision,recall,thr=precision_recall_curve(label, pred)
        stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]  
        patr=getDataByStick([precision,recall],stick)
        logger.info(patr)

        acc1_ema,auc_ema,eval_metrics_ema,patr_ema = 0,0,None,None
        #ema
        if model_ema is not None or teacher_ema is not None:
            load_best_model(config, model_ema.module if model_ema is not None else teacher_ema.module, logger,is_ema=True)
            
            acc1_ema, acc5_ema,loss_ema,auc_ema,pred,label,eval_metrics_ema = validate(config, data_loader_test, model_ema.module if model_ema is not None else teacher_ema.module,amp_autocast=amp_autocast,criterion=criterion,save_pre=True)
            
            logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {acc1_ema:.2f}% {auc_ema:.2f}%")
            _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_ema_'+config.DATA.DATASET.split('/')[-1])+'.npz'
            np.savez(_save_path,pred=pred,label=label)
            if 'cqu_bpdd' in config.DATA.DATASET:
                for m in range(len(label)):
                    if int(label[m])==6:
                        label[m] = 0
                    else:
                        label[m] = 1
            pred=pred.reshape((-1,config.MODEL.NUM_CLASSES))
            if config.MODEL.NUM_CLASSES > 2:
                pred = 1-pred[:,6]
            else:
                pred = pred[:,1]
            precision,recall,thr=precision_recall_curve(label, pred)
            stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]  
            patr_ema=getDataByStick([precision,recall],stick)
            logger.info(patr_ema)
        
        if config.LOG_WANDB and has_wandb:
            _summary = OrderedDict([('test_top1',acc1),
                                    ('test_f1',eval_metrics['macro_f1']),
                                    ('test_auc',auc),
                                    ('test_patr90',patr[1][0] if patr else 0),
                                    ('test_ema_top1',acc1_ema),
                                    ('test_ema_f1',eval_metrics_ema['macro_f1'] if eval_metrics else 0),
                                    ('test_ema_auc',auc_ema),
                                    ('test_ema_patr90',patr_ema[1][0] if patr_ema else 0),])
            wandb.log(_summary)

def train_one_epoch(config,model, criterion, data_loader, optimizer, epoch, mixup_fn=None, lr_scheduler=None,amp_autocast=suppress,loss_scaler=None,model_ema=None,teacher_ema=None, thr_list=[]):
    model.train()
    torch.cuda.empty_cache()
    loss_teacher = None

    if not config.THUMB_MODE and teacher_ema is not None:
        loss_teacher = SoftTargetCrossEntropy_v2()
        # loss_teacher = torch.nn.CrossEntropyLoss()
        #if not config.RDD_TRANS.EMA_FORCE_CPU:
        loss_teacher.cuda()

    optimizer.zero_grad()
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_teacher_meter = AverageMeter()
    dis_ins_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    cluster_num_meter = AverageMeter()
    cluster_ema_num_meter = AverageMeter()
    #acc1_meter = AverageMeter()
    loss_rec = np.array([])
    dis_ratio_list = [[] for i in range(len(thr_list))]
    dis_rec = np.array([AverageMeter() for i in range(len(thr_list))])
    selec_rec = np.array([AverageMeter() for i in range(len(thr_list))])

    start = time.time()
    end = time.time()
    last_idx = len(data_loader) - 1
    persudo_inst = False
    gcn = False
    center_inst = torch.zeros(size=(config.RDD_TRANS.INST_NUM_CLASS,config.RDD_TRANS.INST_NUM_CLASS))

    for idx, (samples, targets) in enumerate(data_loader):
        last_batch = idx == last_idx
        
        # timm dataloader prefetcher will do this
        if not config.DATA.TIMM or not config.DATA.TIMM_PREFETCHER:
            samples = samples.cuda(non_blocking=config.DATA.PIN_MEMORY)
            targets = targets.cuda(non_blocking=config.DATA.PIN_MEMORY)

        # ??????????????????????????????????????????????????????????????????????????????loss??????
        if config.BINARYTRAIN_MODE or (not config.THUMB_MODE and config.RDD_TRANS.INST_NUM_CLASS):
            targets_bin = targets.clone()
            targets_bin[targets==config.DATA.DATA_NOR_INDEX] = 0
            targets_bin[targets!=config.DATA.DATA_NOR_INDEX] = 1
        
        # timm dataloader prefetcher will do this
        if mixup_fn is not None and ((not config.DATA.TIMM_PREFETCHER and config.DATA.TIMM) or not config.DATA.TIMM):
            samples, targets = mixup_fn(samples, targets)

        p = 1
        b = targets.size(0)
        cluster_num_ema=[0]
        with amp_autocast():
            if config.THUMB_MODE:
                predictions = model(samples)
                #print(predictions)
                del samples
                if config.BINARYTRAIN_MODE:
                    classify_loss = criterion(predictions, targets_bin)
                else:
                    classify_loss = criterion(predictions, targets)
                loss = classify_loss
            #custom model
            else:
                if config.AUG.MULTI_VIEW is not None:
                    [samples_student,samples_teacher] = samples.transpose(0, 1)
                else:
                    samples_student,samples_teacher = samples.clone(),samples.clone()
                if config.RDD_TRANS.EMA_FORCE_CPU:
                    samples_teacher = samples_teacher.cpu()
                del samples
                torch.cuda.empty_cache()
                if gcn:
                    output,o_inst,stu_edge,h1_mask_stu,cluster_num = model(samples_student)
                else:
                    output,o_inst,cluster_num = model(samples_student)
                
                if config.RDD_TRANS.SHARPEN_STUDENT:
                    o_inst /= config.RDD_TRANS.SHARPEN_STUDENT

                torch.cuda.empty_cache()    
                if gcn:
                    with torch.no_grad():
                        tea_edge,h1_mask_tea = teacher_ema.module(samples_teacher,is_teacher=True)
                        if config.RDD_TRANS.EMA_FORCE_CPU:
                            tea_edge.cuda(non_blocking=True)
                    # ?????????????????????????????????????????????????????????????????????????????????????????????loss?????????????????????????????????????????????clone??????????????????????????????
                    tea_logist_edge = torch.zeros(size=(tea_edge.size(0),o_inst.size(1),o_inst.size(1),tea_edge.size(-1))).cuda(non_blocking=True).half()
                    stu_logist_edge = torch.zeros(size=(tea_edge.size(0),o_inst.size(1),o_inst.size(1),tea_edge.size(-1))).cuda(non_blocking=True).half()

                    tea_logist_edge[h1_mask_tea] = tea_edge.flatten(end_dim=-2)
                    stu_logist_edge[h1_mask_stu] = stu_edge.flatten(end_dim=-2)

                    tps_stu = 1 if config.RDD_TRANS.SHARPEN_STUDENT is None else config.RDD_TRANS.SHARPEN_STUDENT
                    tps_tea = 1 if config.RDD_TRANS.SHARPEN_TEACHER is None else config.RDD_TRANS.SHARPEN_TEACHER

                    tea_logist_edge /= tps_tea
                    stu_logist_edge /= tps_stu
                    #print(tea_logist_edge)
                   # print(F.softmax(tea_logist_edge,dim=-1))
                    #print(F.softmax(stu_logist_edge,dim=-1))
                    loss_pl = loss_teacher(stu_logist_edge,tea_logist_edge)
                else:
                    loss_gcn = 0
                    #loss_pl = 0
                # fish
                with torch.no_grad():
                    tea_output,tea_o_inst,tea_cluster_num = teacher_ema.module(samples_teacher,is_teacher=True)
                    loss_pl = loss_teacher(o_inst,tea_o_inst)

                # ??????
                loss_gcn = 0
                loss_pl = 0
                
                cluster_num_meter.update(sum(cluster_num) / b,b)

                if config.BINARYTRAIN_MODE:
                    classify_loss = criterion(output, targets_bin)
                else:
                    classify_loss = criterion(output, targets)
                loss =  config.RDD_TRANS.CLASSIFY_LOSS * classify_loss + loss_pl

        
        if not config.DISTRIBUTED:
            loss_meter.update(loss.item(), targets.size(0))
            
            if loss_teacher is not None:
                    loss_teacher_meter.update(loss_pl.item(), targets.size(0))
            else:
                loss_teacher_meter.update(0, targets.size(0))

            if persudo_inst:
                dis_ins_meter.update(dis_ins / (targets.size(0) * p),targets.size(0))
            else:
                dis_ins_meter.update(0 / (targets.size(0) * p),targets.size(0))

       # with torch.autograd.detect_anomaly():
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            #loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if loss_scaler is not None:
                loss_scaler(
                    loss,create_graph=second_order,acc_gradient=True)
            else:
                loss.backward(create_graph=second_order)
                if config.TRAIN.CLIP_GRAD > 0:
                    dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in config.TRAIN.CLIP_MODE>0),
                        value=config.TRAIN.CLIP_GRAD, mode=config.TRAIN.CLIP_MODE)
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                if loss_scaler is not None:
                    loss_scaler.opt_step(optimizer,clip_grad=None if config.TRAIN.CLIP_GRAD == 0 else config.TRAIN.CLIP_GRAD, clip_mode=config.TRAIN.CLIP_MODE,
                    parameters=model_parameters(model, exclude_head='agc' in config.TRAIN.CLIP_MODE>0))
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    if not config.THUMB_MODE:
                        cluster_ema_num_meter.update(sum(cluster_num_ema),b)
                    if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup' or config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                        #teacher_ema.decay_diff = (epoch * num_steps + idx) / (config.TRAIN.EPOCHS * num_steps)
                        model_ema.decay = 0
                    if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                        model_ema.decay = config.RDD_TRANS.EMA_DECAY if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH else model_ema.decay
                    model_ema.update(model)
                if config.TRAIN.LR_SCHEDULER.NAME=='flat_cosine':
                    lr_scheduler.step(epoch * num_steps + idx)
                else:
                    lr_scheduler.step_update(epoch * num_steps + idx)
        else: 
            #loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss, optimizer,
                    clip_grad=None if config.TRAIN.CLIP_GRAD == 0 else config.TRAIN.CLIP_GRAD, clip_mode=config.TRAIN.CLIP_MODE,
                    parameters=model_parameters(model, exclude_head='agc' in config.TRAIN.CLIP_MODE>0),
                    create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                if config.TRAIN.CLIP_GRAD > 0:
                    dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in config.TRAIN.CLIP_MODE),
                        value=config.TRAIN.CLIP_GRAD, mode=config.TRAIN.CLIP_MODE)
                optimizer.step()

            if model_ema is not None:
                if not config.THUMB_MODE:
                    cluster_ema_num_meter.update(sum(cluster_num_ema),b)
                if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup' or config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                    #teacher_ema.decay_diff = (epoch * num_steps + idx) / (config.TRAIN.EPOCHS * num_steps)
                    model_ema.decay = 0
                if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                    model_ema.decay = config.RDD_TRANS.EMA_DECAY if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH else model_ema.decay
                model_ema.update(model)
            if config.TRAIN.LR_SCHEDULER.NAME is not None:
                if config.TRAIN.LR_SCHEDULER.NAME=='flat_cosine':
                    lr_scheduler.step(epoch * num_steps + idx)
                else:
                    lr_scheduler.step_update(epoch * num_steps + idx)
        if teacher_ema is not None:
            if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup' or config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                #teacher_ema.decay_diff = (epoch * num_steps + idx) / (config.TRAIN.EPOCHS * num_steps)
                teacher_ema.decay_diff = 0
            if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                teacher_ema.decay_diff = config.RDD_TRANS.EMA_DECAY if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH else teacher_ema.decay_diff
            #if epoch > 0:
                # ?????????????????????????????????????????????????????????????????????????????????
                #teacher_ema.decay_diff = 0.9997
                # ??????10???epoch??????????????????
                #teacher_ema.decay = 1 if epoch < 10 else config.RDD_TRANS.EMA_DECAY

            teacher_ema.update(model)
        #acc1, _ = accuracy(predictions, targets, topk=(1,1))
        
        torch.cuda.synchronize()
        #acc1_meter.update(acc1.item(), targets.size(0))
        batch_time.update(time.time() - end)
        np.append(loss_rec,loss_meter.avg)
        end = time.time()

        if last_batch or idx % config.PRINT_FREQ == 0:
            if config.DISTRIBUTED:
                reduced_loss = reduce_tensor(loss.data)
                loss_meter.update(reduced_loss.item(), targets.size(0))

                if loss_teacher is not None:
                    loss_teacher_meter.update(reduce_tensor(loss_pl.data), targets.size(0))
                else:
                    loss_teacher_meter.update(0, targets.size(0))

                if persudo_inst:
                    dis_ins_meter.update(dis_ins / (targets.size(0) * p),targets.size(0))
                else:
                    dis_ins_meter.update(0 / (targets.size(0) * p),targets.size(0))

            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss_tea {loss_teacher_meter.val:.4f} ({loss_teacher_meter.avg:.4f})\t'
                f'dis_ins {dis_ins_meter.val:.4f} ({dis_ins_meter.avg:.4f}) \t'
                f'patch_num {patch_num_meter.val:.4f} ({patch_num_meter.avg:.4f}) \t'
                f'cluster_num {cluster_num_meter.val:.4f} ({cluster_num_meter.avg:.4f}) \t'
                f'cluster_ema_num {cluster_ema_num_meter.val:.4f} ({cluster_ema_num_meter.avg:.4f}) \t'
                #f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'mem {memory_used:.0f}MB')
    if persudo_inst and epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
        for i in range(len(thr_list)):
            dis_ratio_tmp= np.sort(np.array(dis_ratio_list[i]))
            if len(dis_ratio_tmp) == 0:
                dis_ratio_tmp = [0]
            thr_list[i] = config.RDD_TRANS.THR_REL_EMA_DECAY * thr_list[i] + (1-config.RDD_TRANS.THR_REL_EMA_DECAY) * dis_ratio_tmp[math.floor(len(dis_ratio_tmp) * config.RDD_TRANS.THR_REL_UPDATE_RATIO)]

    epoch_time = time.time() - start
    for i in range(len(dis_rec)):
        logger.info(f"Dis:  {dis_rec[i].avg:.2f}")
    for i in range(len(selec_rec)):

        logger.info(f"Selected:  {selec_rec[i].avg:.2f}")
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    torch.cuda.empty_cache()
    return loss,OrderedDict([('loss', loss_meter.avg),('tea_loss',loss_teacher_meter.avg)])

# ??????
def predict(config, data_loader, model,amp_autocast=suppress, log_suffix=''):
    model.eval()
    torch.cuda.empty_cache()

    batch_time = AverageMeter()
    
    save_pred = np.array([])
    save_label = np.array([])

    end = time.time()
    last_idx = len(data_loader) - 1
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            last_batch = idx == last_idx
            # timm dataloader prefetcher will do this
            if not config.DATA.TIMM or not config.DATA.TIMM_PREFETCHER:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            #if config.EVAL_MODE:
            targets_bin = targets.clone()

            #if 'cqu_bpdd' in config.DATA.DATASET:
            targets_bin[targets==config.DATA.DATA_NOR_INDEX] = 0
            targets_bin[targets!=config.DATA.DATA_NOR_INDEX] = 1
            # compute output
            with amp_autocast():
                output = model(images)
            if isinstance(output, (tuple, list)):
                if config.RDD_TRANS.PERSUDO_LEARNING:
                    output_ins = output[1]
                    cluster_num = output[-1]
                    output_soft_ins = torch.nn.functional.softmax(output_ins,dim=-1)
                    output = output[0]
                else:
                    index = 0 
                    cluster_num = output[-1]
                    output = output[index]
            output_soft = torch.nn.functional.softmax(output,dim=-1)
            
            if not config.THUMB_MODE and config.RDD_TRANS.INST_TEST and config.RDD_TRANS.PERSUDO_LEARNING:
                #??????max-pool?????????
                print('das')
                b,p,cls = output_ins.shape
                max_score,max_index_cls = torch.max(output_soft_ins,dim=-1)  # B P
                # ??????????????????????????????????????????????????? 
                mask = (max_score - config.RDD_TRANS.TEST_THR > 0) & (max_index_cls - config.DATA.CLS_NOR_INDEX != 0)
                mask_max_nor = max_index_cls == config.DATA.CLS_NOR_INDEX
                max_index = []
                for i in range(b):
                    # ??????????????????????????????????????????,???????????????????????????????????????????????????????????????
                    if mask[i].any()==True:
                        score_tmp = max_score[i,mask[i]]  # B
                        _,max_inx_tmp = torch.max(score_tmp,dim=-1)
                        max_index.append(torch.nonzero(mask[i])[max_inx_tmp,0].tolist())
                    
                    else:
                        # ??????????????????????????????????????????????????????????????????
                        # ???????????????????????????????????????????????????
                        if config.RDD_TRANS.TEST_MAX_POOL:
                            _,index = torch.max(max_score[i],dim=-1)
                        else:
                        # ????????????????????????????????????????????????????????????
                            _,index = torch.max(output_soft_ins[i,:,config.DATA.CLS_NOR_INDEX],dim=-1)
                        max_index.append(index)

                output_ins = output_ins[[i for i in range(b)],max_index,:]
                output_soft_ins = output_soft_ins[[i for i in range(b)],max_index,:]

                del max_score,mask

                if config.RDD_TRANS.BAG_TEST and not config.RDD_TRANS.INST_TEST:
                    output = output
                elif config.RDD_TRANS.INST_TEST and not config.RDD_TRANS.BAG_TEST:
                    output = output_ins
                    output_soft = output_soft_ins
                    del output_ins,output_soft_ins
                # ?????????????????????????????????????????????
                elif config.RDD_TRANS.INST_TEST and config.RDD_TRANS.BAG_TEST:
                    if config.RDD_TRANS.INST_NUM_CLASS != config.MODEL.NUM_CLASSES:
                        output = (output_ins[:,:7] + output) / 2
                        output_soft = (output_soft + output_soft_ins[:,:7]) / 2
                    else:
                        output = (output_ins + output) / 2
                        output_soft = (output_soft + output_soft_ins) / 2
                    del output_ins,output_soft_ins

            #output_soft = torch.nn.functional.softmax(output,dim=-1)
    
            save_pred = np.append(save_pred,output_soft.cpu().numpy())
            save_label = np.append(save_label,targets.cpu().numpy())
            
            torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if last_batch or idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
            
        save_pred = save_pred.reshape(-1,config.MODEL.NUM_CLASSES)

    torch.cuda.empty_cache()

    return save_pred,save_label


def validate(config, data_loader, model,save_pre=False,amp_autocast=suppress, log_suffix='',criterion=None):
    model.eval()
    torch.cuda.empty_cache()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    cluster_num_meter = AverageMeter()
    
    save_pred = np.array([])
    save_label = np.array([])

    end = time.time()
    last_idx = len(data_loader) - 1
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            last_batch = idx == last_idx
            # timm dataloader prefetcher will do this
            if not config.DATA.TIMM or not config.DATA.TIMM_PREFETCHER and next(model.parameters()).is_cuda:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            topk = (1,5)
            #if config.EVAL_MODE:
            targets_bin = targets.clone()
            if config.BINARYTRAIN_MODE:
                topk = (1,1)
            m = 0
            # if 'cqu_bpdd' in config.DATA.DATASET:
            targets_bin[targets==config.DATA.DATA_NOR_INDEX] = 0
            targets_bin[targets!=config.DATA.DATA_NOR_INDEX] = 1

            # compute output
            with amp_autocast():
                output = model(images)
            if isinstance(output, (tuple, list)):
                if config.RDD_TRANS.PERSUDO_LEARNING:
                    output_ins = output[1]
                    cluster_num = output[-1]
                    output_soft_ins = torch.nn.functional.softmax(output_ins,dim=-1)
                    output = output[0]
                else:
                    index = 0 
                    cluster_num = output[-1]
                    output = output[index]
            output_soft = torch.nn.functional.softmax(output,dim=-1)

            if not config.THUMB_MODE and config.RDD_TRANS.INST_TEST and config.RDD_TRANS.PERSUDO_LEARNING:
                #??????max-pool?????????
                b,p,cls = output_ins.shape
                max_score,max_index_cls = torch.max(output_soft_ins,dim=-1)  # B P
                # ??????????????????????????????????????????????????? 
                mask = (max_score - config.RDD_TRANS.TEST_THR > 0) & (max_index_cls - config.DATA.CLS_NOR_INDEX != 0)
                mask_max_nor = max_index_cls == config.DATA.CLS_NOR_INDEX
                max_index = []
                for i in range(b):
                    # ??????????????????????????????????????????,???????????????????????????????????????????????????????????????
                    if mask[i].any()==True:
                        score_tmp = max_score[i,mask[i]]  # B
                        _,max_inx_tmp = torch.max(score_tmp,dim=-1)
                        max_index.append(torch.nonzero(mask[i])[max_inx_tmp,0].tolist())
                    
                    else:
                        # ??????????????????????????????????????????????????????????????????
                        # ???????????????????????????????????????????????????
                        if config.RDD_TRANS.TEST_MAX_POOL:
                            _,index = torch.max(max_score[i],dim=-1)
                        else:
                        # ????????????????????????????????????????????????????????????
                            _,index = torch.max(output_soft_ins[i,:,config.DATA.CLS_NOR_INDEX],dim=-1)
                        max_index.append(index)

                output_ins = output_ins[[i for i in range(b)],max_index,:]
                output_soft_ins = output_soft_ins[[i for i in range(b)],max_index,:]

                del max_score,mask

                if config.RDD_TRANS.BAG_TEST and not config.RDD_TRANS.INST_TEST:
                    output = output
                elif config.RDD_TRANS.INST_TEST and not config.RDD_TRANS.BAG_TEST:
                    output = output_ins
                    output_soft = output_soft_ins
                    del output_ins,output_soft_ins
                # ?????????????????????????????????????????????
                elif config.RDD_TRANS.INST_TEST and config.RDD_TRANS.BAG_TEST:
                    if config.RDD_TRANS.INST_NUM_CLASS != config.MODEL.NUM_CLASSES:
                        output = (output_ins[:,:7] + output) / 2
                        output_soft = (output_soft + output_soft_ins[:,:7]) / 2
                    else:
                        output = (output_ins + output) / 2
                        output_soft = (output_soft + output_soft_ins) / 2
                    del output_ins,output_soft_ins

            if config.BINARYTRAIN_MODE: 
                loss = criterion(output, targets_bin)
            else:
                loss = criterion(output, targets)
                
            # mean-->softmax  or  softmax-->mean  ?
            if config.RDD_TRANS.TEST_FIRST_MEAN:
                del output_soft
                output_soft = torch.nn.functional.softmax(output,dim=-1)

            save_pred = np.append(save_pred,output_soft.cpu().numpy())
            save_label = np.append(save_label,targets.cpu().numpy())
            
            
            # if config.BINARYTRAIN_MODE:
            #     acc1, acc5 = accuracy(output, target_bin, topk=topk)
            #     output = torch.nn.functional.softmax(output,dim=1)
            #     preds = output[:,1]
            # else:
            #     acc1, acc5 = accuracy(output, target, topk=topk)
            #     output = torch.nn.functional.softmax(output,dim=1)
            #     preds = 1-output[:,6]
            
            # phase_label = np.append(phase_label, target_bin.data.cpu().numpy())
            # phase_pred = np.append(phase_pred, preds.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, targets, topk=topk)
            if config.DISTRIBUTED:
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
                loss = reduce_tensor(loss)

            torch.cuda.synchronize()

            loss_meter.update(loss.item(), targets.size(0))
            acc1_meter.update(acc1.item(), targets.size(0))
            acc5_meter.update(acc5.item(), targets.size(0))
            if not config.THUMB_MODE:
                cluster_num_meter.update(sum(cluster_num) / targets.size(0),targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if last_batch or idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Cluster_num {cluster_num_meter.val:.3f} ({cluster_num_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
            
        save_pred = save_pred.reshape(-1,config.MODEL.NUM_CLASSES)
        auc = 0
        if config.BINARYTRAIN_MODE:
            ma_f1 = f1_score(np.array(save_label!=config.DATA.DATA_NOR_INDEX,dtype=int),np.argmax(save_pred,axis=1),average='binary')
            mi_f1 = ma_f1
            try:
                auc = roc_auc_score(np.array(save_label!=config.DATA.DATA_NOR_INDEX,dtype=int), save_pred[:,1])
                precision,recall,thr=precision_recall_curve(np.array(save_label!=config.DATA.DATA_NOR_INDEX,dtype=int), save_pred[:,1])
                stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]  
                patr=getDataByStick([precision,recall],stick)
            except:
                print(save_pred)
        else:
            ma_f1 = f1_score(save_label,np.argmax(save_pred,axis=1),average='macro')
            mi_f1 = f1_score(save_label,np.argmax(save_pred,axis=1),average='micro')
            try:
                auc = roc_auc_score(np.array(save_label!=config.DATA.DATA_NOR_INDEX,dtype=int), 1-save_pred[:,6])
                precision,recall,thr=precision_recall_curve(np.array(save_label!=config.DATA.DATA_NOR_INDEX,dtype=int), save_pred[:,1])
                stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]  
                patr=getDataByStick([precision,recall],stick)
            except:
                print(save_pred)
        
        #if config.BINARYTRAIN_MODE:
        
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} AUC {auc*100:.3f} F1@Macro {ma_f1*100:.3f} F1@Micro {mi_f1*100:.3f} P@R90 {patr[1][0]:.3f}')
    metrics = OrderedDict([('loss', loss_meter.avg), ('top1', acc1_meter.avg), ('top5', acc5_meter.avg),('auc',auc),('p@r90',patr[1][0]),('macro_f1',ma_f1),('micro_f1',mi_f1)])
    torch.cuda.empty_cache()
    if save_pre:
        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg, auc,save_pred,save_label,metrics
    else:    
        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg, auc,metrics

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')

    _, config = parse_option()
    os.makedirs(config.OUTPUT, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT,'result'), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT,'log'), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT,'model'), exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=config.LOCAL_RANK, name=f"{config.EXP_NAME}")
    
    if config.LOG_WANDB:
        if has_wandb:
            wandb.init(project=config.PROJECT_NAME, config=config,entity="dearcat",name=config.EXP_NAME)
        else: 
            logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")
    # resolve AMP arguments based on PyTorch / Apex availability
    config.defrost()
    use_amp = None
    if config.AMP:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if  has_native_amp and not config.APEX_AMP:
            config.NATIVE_AMP = True
        elif  has_apex:
            config.APEX_AMP = True
    else:
        config.NATIVE_AMP = False
        config.APEX_AMP = False
    if config.NATIVE_AMP and has_native_amp:
        use_amp = 'native'
    elif config.APEX_AMP and has_apex:
        use_amp = 'apex'
    elif config.APEX_AMP or config.NATIVE_AMP:
        logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6") 
    # if 'WORLD_SIZE' in os.environ:
    #     config.DISTRIBUTED = int(os.environ['WORLD_SIZE']) > 1
    config.freeze()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    world_size = 1
    rank = 0  # global rank
    if config.DISTRIBUTED:
        device = 'cuda:%d' % config.LOCAL_RANK
        torch.cuda.set_device(config.LOCAL_RANK)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (rank, world_size))
    else:
        logger.info('Training with a single process on 1 GPUs.')
    assert rank >= 0
    
        
    random_seed(config.SEED, rank)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    '''linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()'''

    if config.LOCAL_RANK == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)