import os

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
from utils import ModelEmaV3
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_best_model, load_checkpoint, save_checkpoint, get_grad_norm,  reduce_tensor,l1_regularizer,getDataByStick
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
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
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
    parser.add_argument('--title',  type=str, help='experiment title')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--thumb', action='store_true', help='Use thumb data')
    parser.add_argument('--binary-train', action='store_true', help='train the model with binary setting')
    parser.add_argument('--load-test-dir', type=str, metavar='PATH',help='the file of tested data')
    parser.add_argument('--pretrained-backbone', type=str, metavar='PATH',help='the file of pretrained model')
    parser.add_argument('--train-mode', type=str, default='t_e', choices=['train', 'eval', 't-e'],
                        help='train: only train, '
                             'eval: only test , '
                             't_e: first train the model, and use it to eval')

    # distributed training
    parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)
    return args, config


def main(config):
    if not config.TRAIN_MODE=='eval':
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(is_train=True,config=config)
    else:
        dataset_test,data_loader_test = build_loader(is_train=False,config=config)
        
    logger.info(f"Creating model:{config.MODEL.NAME}/{config.MODEL.BACKBONE}")
    model = build_model(config)
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
        loss_scaler = NativeScaler()
        if config.LOCAL_RANK == 0:
            logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if config.LOCAL_RANK == 0:
            logger.info('AMP not enabled. Training in float32.')

    # setup learning rate schedule and starting epoch
    if config.TRAIN_MODE=='eval':
        lr_scheduler = ''
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    max_accuracy = 0.0
    best_auc = 0.0
    max_f1 = 0.0
    max_f1_ema = .0
    max_accuracy_ema = .0
    best_auc_ema = .0

    if config.MODEL.RESUME:
        criterion = torch.nn.CrossEntropyLoss()
        criterion.cuda()
        max_accuracy,best_auc = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        if config.TRAIN_MODE=='eval':
            if config.LOAD_TEST_DIR:
                dic=np.load(config.LOAD_TEST_DIR)
                label=dic['label']
                pred=dic['pred']
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
            else:
                acc1, acc5, loss, auc,pred,label,eval_metrics = validate(config, data_loader_test, model,save_pre=True,amp_autocast=amp_autocast,criterion=criterion)
                logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {acc1:.2f}% {auc:.2f}%")
                _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_'+config.DATA.DATASET.split('/')[1])+'.npz'
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
            return
    teacher_ema = None
    model_ema = None
    if not config.THUMB_MODE:
        cpt = torch.load('/home/tangwenhao/rdd/output/swin_test/model/swin_small_patch4_window7_224_best_model.pth', map_location='cpu')
        std = cpt['state_dict']
        std['head_instance.weight'] = std['head.weight']
        std['head_instance.bias'] = std['head.bias']
        model.load_state_dict(std)
        #teacher_ema = ModelEmaV3(model_teacher, decay=config.RDD_TRANS.EMA_DECAY, device='cpu' if config.RDD_TRANS.EMA_FORCE_CPU else None, diff_layers=[])
        model_ema = ModelEmaV3(model, decay=config.RDD_TRANS.EMA_DECAY, device='cpu' if config.RDD_TRANS.EMA_FORCE_CPU else None)

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

    logger.info("Start training")
    start_time = time.time()
    loss_rec = np.array([])

    thr_list = np.array([config.RDD_TRANS.NOR_THR for i in range(config.MODEL.NUM_CLASSES)])

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if not config.DATA.TFRECORD_MODE and config.DISTRIBUTED:
            data_loader_train.sampler.set_epoch(epoch)
        loss_r,train_metrics = train_one_epoch(config, model_without_ddp, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,amp_autocast=amp_autocast, loss_scaler=loss_scaler,teacher_ema=teacher_ema,model_ema=model_ema,thr_list=thr_list)
        #np.append(loss_rec,loss_r)

        acc1, acc5,loss,auc,eval_metrics = validate(config, data_loader_val, model_without_ddp,amp_autocast=amp_autocast,criterion=criterion)
        if teacher_ema is not None:
            acc1_ema, acc5_ema,loss_ema,auc_ema,eval_metrics_ema = validate(config, data_loader_val, teacher_ema.module,amp_autocast=amp_autocast,criterion=criterion)
        if model_ema is not None:
            acc1_ema, acc5_ema,loss_ema,auc_ema,eval_metrics_ema = validate(config, data_loader_val, model_ema.module,amp_autocast=amp_autocast,criterion=criterion)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        for i in range(len(thr_list)):
            logger.info(f"Thr:  {thr_list[i]:.2f}")
        
        # save the ema checkpoint
        if model_ema is not None or teacher_ema is not None:
            f1_ema = eval_metrics_ema['macro_f1']
            is_best_ema = f1_ema > max_f1_ema
            max_accuracy_ema = max(max_accuracy_ema, acc1_ema) if epoch > 0 else 0
            best_auc_ema = max(best_auc_ema,auc_ema) if epoch > 0 else 0
            max_f1_ema = max(max_f1_ema,f1_ema)
            if config.LOCAL_RANK == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_ema.module if model_ema is not None else teacher_ema.module, max_accuracy_ema, optimizer, lr_scheduler, logger,is_best_ema,best_auc_ema,None,is_ema=True)

        f1 = eval_metrics['macro_f1']
        is_best = (auc > best_auc if config.BINARYTRAIN_MODE else f1 > max_f1) and epoch>0
        max_accuracy = max(max_accuracy, acc1) if epoch > 0 else 0
        best_auc = max(best_auc,auc) if epoch > 0 else 0
        max_f1 = max(max_f1,f1)

        update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(config.OUTPUT, 'summary.csv'),
                    write_header=False, log_wandb=config.LOG_WANDB and has_wandb)

        if config.LOCAL_RANK == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger,is_best,best_auc,model_ema)

        logger.info(f'Max accuracy: {max_accuracy:.2f}%\t'
                    f'Max f1: {max_f1:.2f}%\t'
                    f'Best AUC: {best_auc:.2f}%')

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
            patr=getDataByStick([precision,recall],stick)
            logger.info(patr)


def train_one_epoch(config,model, criterion, data_loader, optimizer, epoch, mixup_fn=None, lr_scheduler=None,amp_autocast=suppress,loss_scaler=None,model_ema=None,teacher_ema=None, thr_list=[]):
    model.train()
    loss_teacher = None
    if teacher_ema is not None:
        teacher_ema.module.train()
    if not config.THUMB_MODE:
        loss_teacher = torch.nn.CrossEntropyLoss()
        loss_teacher.cuda()

    optimizer.zero_grad()
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_teacher_meter = AverageMeter()
    dis_ins_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    #acc1_meter = AverageMeter()
    loss_rec = np.array([])
    dis_ratio_list = [[] for i in range(len(thr_list))]
    dis_rec = np.array([AverageMeter() for i in range(len(thr_list))])
    selec_rec = np.array([AverageMeter() for i in range(len(thr_list))])

    start = time.time()
    end = time.time()
    last_idx = len(data_loader) - 1
    
    for idx, (samples, targets) in enumerate(data_loader):
        last_batch = idx == last_idx
        
        # timm dataloader prefetcher will do this
        if not config.DATA.TIMM:
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        
        if config.BINARYTRAIN_MODE:
            m = 0
            target_bin = targets.clone()
            for m in range(len(target_bin)):
                if int(target_bin[m])==6:
                    target_bin[m] = 0
                else:
                    target_bin[m] = 1
        
        # timm dataloader prefetcher will do this
        if mixup_fn is not None and not config.DATA.TIMM:
            samples, targets = mixup_fn(samples, targets)
        
        with amp_autocast():
            if config.THUMB_MODE:
                p = 1
                predictions,_ = model(samples)
                #print(predictions)
                del samples
                if config.BINARYTRAIN_MODE:
                    classify_loss = criterion(predictions, target_bin)
                else:
                    classify_loss = criterion(predictions, targets)
                loss = classify_loss
            #custom model
            else:
                output,o_inst = model(samples)
                with torch.no_grad():
                    _,pl_inst = model_ema.module(samples)
                   #_,pl_inst = teacher_ema.module(samples)
                b,p,cls = pl_inst.shape
                output_pl = torch.nn.functional.softmax(pl_inst,dim=2)
                t_cpu = targets.cpu()
                ins_t = targets.unsqueeze(-1).repeat((1,p))
                dis_ins = 0
                output_bag_label = output_pl[torch.functional.F.one_hot(ins_t,num_classes=config.MODEL.NUM_CLASSES) == 1].view(b,p)        #包所属标签下的置信度
                if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                    out_tmp_sort,_ = torch.sort(output_bag_label,dim=-1,descending=True)         # [b p]
                    min_nor_thr = out_tmp_sort[[i for i in range(b)],np.floor(p*thr_list[t_cpu])] # [b 1]
                    min_nor_thr = min_nor_thr.unsqueeze(-1).repeat((1,p))        # [b p]
                else:
                    min_nor_thr = torch.ones(size=(b,p))
                min_nor_thr = min_nor_thr.cuda()

                max_score,label_pl = torch.max(output_pl,dim=2)
                #label_pl = torch.argmax(output_pl,dim=2)
                label_tmp = label_pl.clone()
                #获得正常包索引和病害包索引
                bs_index_nor = targets==6
                bs_index_dis = bs_index_nor==False
                ps_mask_nor = (ins_t - 6 == 0)
                ps_mask_dis = ps_mask_nor==False

                #正常包里面的所有实例设为正常
                #label_pl[bs_index_nor,:] = 6
                #病害包里面的所有实例先置为包病害标签
                #label_pl[bs_index_dis] = ins_t[bs_index_dis]
                #将部分实例设为正常
                #mask_ins = ((output_pl[:,:,6] - .99 > 0) | (output_pl[:,:,6] - min_nor_thr > 0)) & (label_tmp - 6  == 0) & (targets.unsqueeze(-1).repeat((1,p)) -6 != 0)

                #将所有实例设为正常
                label_pl[:,:] = 6
                thr_min_conf = 0.9 + ((epoch * num_steps + idx) / (config.TRAIN.EPOCHS * num_steps)) * 0.1
                thr_min_conf = 0.99 if thr_min_conf > 0.99 else thr_min_conf
                #把网络判断为不是正常的部分实例置为包病害标签
                if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                    #判断为相应病害的实例置为包标签
                    #mask_ins =  (label_tmp - ins_t  == 0)

                    #将病害包里判断为不是正常的实例都置为包标签 
                    mask_ins =   ps_mask_dis & (((output_pl[:,:,6] - 0.001 < 0) & (label_tmp - 6  != 0))  | (output_bag_label - min_nor_thr > 0))
                    label_pl[mask_ins] = ins_t[mask_ins]
                    #选取部分置信度比较高的实例参与loss计算   
                    mask_ins = (mask_ins | (ps_mask_dis & (label_pl==6) & (output_pl[:,:,6] - 0.95 > 0))) | ((output_pl[:,:,6] - 0.99 > 0) & ps_mask_nor)
                else:
                    #全部都用
                    #mask_ins = (label_tmp - label_tmp == 0)
                    #只要正常图片
                    mask_ins = (ins_t - 6 == 0) & (output_bag_label - 0.9 > 0)

                    #label_pl[bs_index_dis] = ins_t[bs_index_dis]

                #统计病害实例
                dis_count = torch.count_nonzero(label_pl - 6,dim=1) 
                dis_ins += torch.sum(dis_count)

                for i in range(len(thr_list)):
                    i_index = targets==i
                    dis_tmp = dis_count[i_index]
                    selec_tmp = mask_ins[i_index]

                    if len(selec_tmp)>0:
                        selec_rec[i].update( len(selec_tmp[selec_tmp==True]) / (len(selec_tmp)*p),b)

                    if len(dis_tmp) > 0 and epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                        thr_tmp = torch.sum(dis_tmp) / (p * len(dis_tmp))
                        dis_ratio_list[i].append(thr_tmp.cpu())

                        #thr_list[i] = 0.9999 * thr_list[i] + (1-0.9999)* thr_tmp
                        #thr_list[i] = 0.1 if thr_list[i] < 0.1 else thr_list[i]
                        #thr_list[i] = thr_tmp
                        dis_rec[i].update(thr_tmp,b)

                #选择部分patch来计算loss
                #if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                label_pl = label_pl[mask_ins]
                o_inst = o_inst[mask_ins]
                #     label_pl = label_pl[bs_index_nor]
                #     label_pl = label_pl.view(-1)
                #     o_inst = o_inst[bs_index_nor]
                #     o_inst = o_inst.view(-1,cls)
               #else:

               #计算loss
                
                label_pl = label_pl.view(-1)
                o_inst = o_inst.view(-1,cls)
                patch_num_meter.update(len(label_pl) / (p*b),b)
                if o_inst.size(0)>0:
                    loss_pl = loss_teacher(o_inst,label_pl)
                else:
                    loss_pl = 0
                classify_loss = criterion(output, targets)
                loss = loss_pl

                del samples
        
        if not config.DISTRIBUTED:
            loss_meter.update(loss.item(), targets.size(0))
            if config.THUMB_MODE:
                dis_ins_meter.update(0 / (targets.size(0) * p),targets.size(0))
            else:
                dis_ins_meter.update(dis_ins / (targets.size(0) * p),targets.size(0))
            if loss_teacher is not None and o_inst.size(0)>0:
                loss_teacher_meter.update(loss_pl.item(), targets.size(0))
            else:
                loss_teacher_meter.update(0, targets.size(0))
       # with torch.autograd.detect_anomaly():
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            #loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
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
                        model_parameters(model, exclude_head='agc' in config.TRAIN.CLIP_MODE>0),
                        value=config.TRAIN.CLIP_GRAD, mode=config.TRAIN.CLIP_MODE)
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
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
        if teacher_ema is not None:
            if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup' or config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                #teacher_ema.decay_diff = (epoch * num_steps + idx) / (config.TRAIN.EPOCHS * num_steps)
                teacher_ema.decay_diff = 0
            if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                teacher_ema.decay_diff = config.RDD_TRANS.EMA_DECAY if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH else teacher_ema.decay_diff
            if epoch > 1:
                # 将前面特征层与后面实例分类层分开更新，分类层更新率更高
                #teacher_ema.decay_diff = 0.9997
                # 前面10个epoch不更新特征层
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
                loss_meter.update(reduced_loss.item(), input.size(0))
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
                
                #f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'mem {memory_used:.0f}MB')
    if not config.THUMB_MODE:
        for i in range(len(thr_list)):
            dis_ratio_tmp= np.sort(np.array(dis_ratio_list[i]))
            thr_list[i] = 0.75 * thr_list[i] + 0.25 * dis_ratio_tmp[math.floor(len(dis_ratio_tmp) * 0.01)]

    epoch_time = time.time() - start
    for i in range(len(dis_rec)):
        logger.info(f"Dis:  {dis_rec[i].avg:.2f}")
    for i in range(len(selec_rec)):

        logger.info(f"Selected:  {selec_rec[i].avg:.2f}")
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return loss,OrderedDict([('loss', loss_meter.avg)])

def validate(config, data_loader, model,save_pre=False,amp_autocast=suppress, log_suffix='',criterion=None):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    save_pred = np.array([])
    save_label = np.array([])

    end = time.time()
    last_idx = len(data_loader) - 1
    with torch.no_grad():
        for idx, (images, target) in enumerate(data_loader):
            last_batch = idx == last_idx
            # timm dataloader prefetcher will do this
            if not config.DATA.TIMM:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            topk = (1,5)
            #if config.EVAL_MODE:
            target_bin = target.clone()
            if config.BINARYTRAIN_MODE:
                topk = (1,1)
            m = 0
            if 'cqu_bpdd' in config.DATA.DATASET:
                for m in range(len(target_bin)):
                    if int(target_bin[m])==6:
                        target_bin[m] = 0
                    else:
                        target_bin[m] = 1

            # compute output
            with amp_autocast():
                output = model(images)
            if isinstance(output, (tuple, list)):
                index = 0 if config.THUMB_MODE else 1
                output = output[index]

            output_soft = torch.nn.functional.softmax(output,dim=-1)
            if not config.THUMB_MODE:
                #使用max-pool来测试，取所有图块中得分最大的图块的置信度
                b,p,cls = output.shape
                max_score,max_index_cls = torch.max(output_soft,dim=-1)

                mask = (max_score - config.RDD_TRANS.TEST_THR > 0) & (max_index_cls - 6 != 0)
                if mask.any() == True:
                    max_index = []
                    for i in range(b):
                        if mask[i].any()==True:
                            score_tmp = max_score[i,mask[i]]
                            _,max_inx_tmp = torch.max(score_tmp,dim=-1)
                            max_index.append(torch.nonzero(mask[i])[max_inx_tmp,0].tolist())
                        else:
                            _,index = torch.max(max_score[i],dim=-1)
                            max_index.append(index)
                else:
                    max_score,max_index = torch.max(max_score,dim=-1)

                output_soft = output_soft[[i for i in range(b)],max_index,:]
                output = output[[i for i in range(b)],max_index,:]

            if config.BINARYTRAIN_MODE:
                loss = criterion(output, target_bin)
            else:
                loss = criterion(output, target)
                
            save_pred = np.append(save_pred,output_soft.cpu().numpy())
            save_label = np.append(save_label,target.cpu().numpy())
            
            
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
            acc1, acc5 = accuracy(output, target, topk=topk)
            if config.DISTRIBUTED:
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
                loss = reduce_tensor(loss)


            torch.cuda.synchronize()

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

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
                    f'Mem {memory_used:.0f}MB')

        save_pred = save_pred.reshape(-1,config.MODEL.NUM_CLASSES)
        ma_f1 = f1_score(save_label,np.argmax(save_pred,axis=1),average='macro')
        mi_f1 = f1_score(save_label,np.argmax(save_pred,axis=1),average='micro')
        auc = 0
        #if config.BINARYTRAIN_MODE:
        auc = roc_auc_score(np.array(save_label!=6,dtype=int), 1-save_pred[:,6])
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} AUC {auc*100:.3f} F1@Macro {ma_f1*100:.3f} F1@Micro {mi_f1*100:.3f}')
    metrics = OrderedDict([('loss', loss_meter.avg), ('top1', acc1_meter.avg), ('top5', acc5_meter.avg),('auc',auc),('macro_f1',ma_f1),('micro_f1',mi_f1)])
    if save_pre:
        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg, auc,save_pred,save_label,metrics
    else:    
        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg, auc,metrics

if __name__ == '__main__':
    _, config = parse_option()
    os.makedirs(config.OUTPUT, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT,'result'), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT,'log'), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT,'model'), exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=config.LOCAL_RANK, name=f"{config.EXP_NAME}")
    
    if config.LOG_WANDB:
        if has_wandb:
            wandb.init(project=config.EXP_NAME, config=config)
            print('a')
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
    if config.NATIVE_AMP and has_native_amp:
        use_amp = 'native'
    elif config.APEX_AMP and has_apex:
        use_amp = 'apex'
    elif config.APEX_AMP or config.NATIVE_AMP:
        logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6") 
    config.freeze()

    if 'WORLD_SIZE' in os.environ:
        config.DISTRIBUTED = int(os.environ['WORLD_SIZE']) > 1
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
    
        
    #random_seed(config.SEED, rank)
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
    #logger.info(config.dump())

    main(config)