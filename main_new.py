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
from trainer import build_trainer
from data import build_loader
from utils import ModelEmaV3, get_sigmod_num
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
    parser.add_argument('-b','--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--trainer', type=str, help='trainer name')
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
    parser.add_argument('--ema', action='store_true', help='Use thumb data')
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

    train_one_epoch,predict,validate = build_trainer(config)

    if config.TRAIN_MODE=='train' or config.TRAIN_MODE=='t_e':
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
    if config.TRAIN_MODE=='eval' or config.TRAIN_MODE=='predict':
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
            return
        elif config.TRAIN_MODE=='predict':
            pred,label= predict(config, data_loader_test, model,amp_autocast=amp_autocast)
            _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_predict_'+config.DATA.DATASET.split('/')[-1])+'.npz'
            np.savez(_save_path,pred=pred,label=label)

            return
    teacher_ema = None
    model_ema = None
    if not config.THUMB_MODE or config.MODEL_EMA:
        # cpt = torch.load('/home/tangwenhao/rdd/model/swin_small_patch4_window7_224_best_model.pth', map_location='cpu')
        # std = cpt['state_dict']
        # std['head_instance.weight'] = std['head.weight']
        # std['head_instance.bias'] = std['head.bias']
        # model.load_state_dict(std)
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
                f'Max ema_f1: {max_f1_ema:.2f}%\t'
                f'Best AUC: {best_auc:.2f}%')
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
            wandb.init(project=config.EXP_NAME, config=config)
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
    #logger.info(config.dump())

    main(config)