import os
from collections import OrderedDict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import argparse
import datetime
import numpy as np
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.utils import *
from timm.loss import *

from config import get_config
from models import build_model
from engine import build_trainer
from dataloader import build_loader
from utils import ModelEmaV3, _save_checkpoint_V2, load_best_model_V2, load_checkpoint_V2
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from criterion import build_criterion
from logger import create_logger
from utils import load_best_model, load_checkpoint, save_checkpoint, get_grad_norm,  reduce_tensor,l1_regularizer,getDataByStick,list2dict

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
    parser.add_argument('--project',  type=str, help='experiment project')
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

    train_one_epoch,predict,validate,best_metrics = build_trainer(config)

    if config.TRAIN_MODE=='train' or config.TRAIN_MODE=='t_e':
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config=config,is_train=True)
    else:
        dataset_test,data_loader_test = build_loader(config=config,is_train=False)
        
    logger.info(f"Creating model:{config.MODEL.NAME}/{config.MODEL.BACKBONE}")
    models = build_model(config)

    # setup augmentation batch splits for contrastive loss or split bn
    '''num_aug_splits = 0
    if config.AUG.SPLITS > 0:
        assert config.AUG.SPLITS > 1, 'A split of 1 makes no sense'
        num_aug_splits = config.AUG.SPLITS

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))'''

    # 考虑多模型情况下的，GPU存储
    for model_name in config.MODEL.TOGPU_MODEL_NAME:
        models[model_name].cuda()

    logger.info(f"model:{config.MODEL.NAME}/{config.MODEL.BACKBONE}")

    optimizer = build_optimizer(config, models['main'])

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    # 半精度暂时只考虑训练
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        models['main'], optimizer = amp.initialize(models['main'], optimizer, opt_level='O1')
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

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if config.MODEL_EMA:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV3(models['main'], decay=config.EMA_DECAY, device='cpu' if config.EMA_FORCE_CPU else None)
        best_metrics_ema = deepcopy(best_metrics) 

    # setup loss function
    criterions = build_criterion(config)
    for cri_ids in range(len(criterions)):
        criterions[cri_ids].cuda()

    if config.MODEL.RESUME:

        best_metrics,best_metrics_ema = load_checkpoint_V2(config, models, optimizer, lr_scheduler, logger,model_ema)

        if config.TRAIN_MODE=='eval':
            if '_ema_' in config.MODEL.RESUME:
                ema_prefix = 'ema'
                models_without_ddp['main'] = model_ema.module
            else:
                ema_prefix = ''
            loss_eval, eval_metrics,pred,label = validate(config, data_loader_test, models_without_ddp,save_pre=True,amp_autocast=amp_autocast,criterion=criterions,logger=logger)

            _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_'+ema_prefix+'_'+config.DATA.DATASET.split('/')[-1])+'.npz'
            np.savez(_save_path,pred=pred,label=label)
            return
        elif config.TRAIN_MODE=='predict':
            pred,label= predict(config, data_loader_test, model,amp_autocast=amp_autocast)
            _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_predict_'+config.DATA.DATASET.split('/')[-1])+'.npz'
            np.savez(_save_path,pred=pred,label=label)
            return

    # setup distributed training
    if config.DISTRIBUTED:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if config.LOCAL_RANK == 0:
                logger.info("Using NVIDIA APEX DistributedDataParallel.")
            models['main'] = ApexDDP(models['main'], delay_allreduce=True)
        else:
            if config.LOCAL_RANK  == 0:
                logger.info("Using native Torch DistributedDataParallel.")
            models['main'] = NativeDDP(models['main'], device_ids=[config.LOCAL_RANK ])  # can use device str in Torch >= 1.1
        models_without_ddp = models
        models_without_ddp['main'] = models['main'].module
        # NOTE: EMA model does not need to be wrapped by DDP
    else:
        models_without_ddp = models
    if model_ema is not None:
        models_without_ddp_ema = deepcopy(models_without_ddp)
        models_without_ddp_ema['main'] =  model_ema.module
        

    n_parameters = sum(p.numel() for p in models['main'].parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(models_without_ddp['main'], 'flops'):
        flops = models_without_ddp['main'].flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    logger.info("Start training")
    start_time = time.time()
    loss_rec = np.array([])

    # wandb log. watch, record gradient of the model training
    if config.LOG_WANDB and has_wandb:
        wandb.watch(models_without_ddp['main'], log_freq=100)

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if not config.DATA.TFRECORD_MODE and config.DISTRIBUTED:
            data_loader_train.sampler.set_epoch(epoch)

        loss_r,train_metrics = train_one_epoch(config, models_without_ddp, criterions, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,amp_autocast=amp_autocast, loss_scaler=loss_scaler,model_ema=model_ema,logger=logger)
        #np.append(loss_rec,loss_r)
        loss,eval_metrics = validate(config, data_loader_val, models_without_ddp,amp_autocast=amp_autocast,criterion=criterions,logger=logger)

        eval_metrics_ema = {}
        if model_ema is not None:
            loss_ema,eval_metrics_ema = validate(config, data_loader_val, models_without_ddp_ema,amp_autocast=amp_autocast,criterion=criterions,logger=logger)

        best_model_metirc = list2dict(config.TEST.BEST_MODEL_METRIC) 
        eval_best_metric_ema = eval_metrics_ema[best_model_metirc['main']] if model_ema is not None else 0.
        logger.info(f"The {best_model_metirc['main']} of the network on the {len(dataset_val)} test images: {eval_metrics[best_model_metirc['main']]:.2f}% {eval_best_metric_ema:.2f}%")

        # save the checkpoint
        save_checkpoint(config,epoch,models_without_ddp,best_metrics,optimizer,lr_scheduler,logger,model_ema,eval_metrics,is_ema=False,best_metrics_ema=best_metrics_ema)
        
        # save the ema checkpoint
        if model_ema is not None:
            # ema模型暂时只考虑main模型的最佳模型存储
            save_checkpoint(config,epoch,models_without_ddp,best_metrics,optimizer,lr_scheduler,logger,model_ema,eval_metrics_ema,is_ema=True,best_metrics_ema=best_metrics_ema)
        
        update_summary(
            epoch, train_metrics, eval_metrics, os.path.join(config.OUTPUT, 'summary.csv'),
            write_header=False, log_wandb=config.LOG_WANDB and has_wandb,eval_metrics_ema=eval_metrics_ema)

    for bt_metric in list(best_metrics.keys()):
        logger.info(f'Best {bt_metric}: {best_metrics[bt_metric]:.2f}%\t')
    if config.LOG_WANDB and has_wandb:
        best_metrics = OrderedDict([('eval_best_'+k,v) for k,v in best_metrics.items()])
        wandb.log(best_metrics)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    if config.TRAIN_MODE=='t_e':
        dataset_test,data_loader_test = build_loader(config=config,is_train=False)

        load_best_model_V2(config, models_without_ddp, logger)
        loss_eval, eval_metrics,pred,label = validate(config, data_loader_test, models_without_ddp,save_pre=True,amp_autocast=amp_autocast,criterion=criterions,logger=logger)

        _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_'+config.DATA.DATASET.split('/')[-1])+'.npz'
        np.savez(_save_path,pred=pred,label=label)
        #ema
        eval_metrics_ema = {}
        if model_ema is not None:
            load_best_model_V2(config, models_without_ddp_ema, logger,is_ema=True)
            loss_eval_ema, eval_metrics_ema,pred_ema,label_ema = validate(config, data_loader_test, models_without_ddp_ema,amp_autocast=amp_autocast,criterion=criterions,save_pre=True,logger=logger)

            _save_path = os.path.join(config.OUTPUT,'result',config.EXP_NAME+'_ema_'+config.DATA.DATASET.split('/')[-1])+'.npz'
            np.savez(_save_path,pred=pred_ema,label=label_ema)

        eval_best_metric_ema = eval_metrics_ema[best_model_metirc['main']] if model_ema is not None else 0.
        logger.info(f"The {best_model_metirc['main']} of the network on the {len(dataset_val)} test images: {eval_metrics[best_model_metirc['main']]:.1f}% {eval_best_metric_ema:.1f}%")

        if config.LOG_WANDB and has_wandb:
            eval_metrics = OrderedDict([('test_'+k,v) for k,v in eval_metrics.items()])
            eval_metrics_ema = OrderedDict([('test_ema_'+k,v) for k,v in eval_metrics_ema.items()])
            eval_metrics.update(eval_metrics_ema)
            wandb.log(eval_metrics)

# 此函数是为了处理ema和多模型保存而创建，ema模型的save_ckpt和正常模型流程基本一致，故将其抽象出来
def save_checkpoint(config,epoch,models_without_ddp,best_metrics,optimizer,lr_scheduler,logger,ema_model,eval_metrics,is_ema,best_metrics_ema):
    _best_metrics = best_metrics_ema if is_ema else best_metrics
    best_model_metirc = list2dict(config.TEST.BEST_MODEL_METRIC) 
    # main始终在最前面，最佳评价指标字典必须和最佳模型名称列表一一对应
    assert config.MODEL.SAVE_BEST_MODEL_NAME[0] == 'main' and list(best_model_metirc.keys()) == config.MODEL.SAVE_BEST_MODEL_NAME

    for best_model_name in config.MODEL.SAVE_BEST_MODEL_NAME:
        prefix = best_model_name

        best_metric_name = best_model_metirc[best_model_name].lower()

        is_best = eval_metrics[best_metric_name] > _best_metrics[best_metric_name]
        _best_metrics[best_metric_name] = max(_best_metrics[best_metric_name],eval_metrics[best_metric_name])
      
        if config.LOCAL_RANK == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            _save_checkpoint_V2(config,epoch,models_without_ddp,best_metrics,optimizer,lr_scheduler,logger,is_best,ema_model,prefix,best_model_name,is_ema,best_metrics_ema)
        # ema 暂且只考虑一个模型的存储
        if is_ema:
            break
    # 更新best_metrics里面所有的指标
    for key, value in _best_metrics.items():
        _best_metrics[key] = max(_best_metrics[key],eval_metrics[key])

    return 0

if __name__ == '__main__':

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
                        "Install NVIDIA apex or upgrade to PyTorch 1.6") 
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