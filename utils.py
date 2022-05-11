import os
from sklearn import metrics
import torch
import torch.distributed as dist
import shutil
from copy import deepcopy
import math
import torch.nn.functional as F
import torch.nn as nn
from timm.utils.clip_grad import dispatch_clip_grad

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
def load_best_model(config,model,logger,is_ema=False):
    if is_ema:
        ckpt_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ema_ckpt.pth')
    else:
        ckpt_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ckpt.pth')
    # if os.path.exists(ckpt_path):
    #     os.remove(ckpt_path)
    if is_ema:
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ema_best_model.pth')
    else:
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_best_model.pth')
    logger.info(f"==============> Loading the best model....................")
    checkpoint = torch.load(best_path, map_location='cpu')
    if 'epoch' in checkpoint:
        logger.info(f"==============> Epoch {checkpoint['epoch']}....................")
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(msg)
    if config.APEX_AMP and checkpoint['config'].APEX_AMP:
        amp.initialize(model, opt_level='O1')

def load_checkpoint(config, model,optimizer=None, lr_scheduler=None,logger=None,is_ema=False):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if is_ema:
        if 'ema' in checkpoint:
            msg = model.load_state_dict(checkpoint['ema'], strict=False)
            logger.info(msg)
            return 0
    # 是否只读取模型
    if 'state_dict' in checkpoint:
        msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(msg)
        max_accuracy = 0.0
        best_auc = 0.0
        best_f1 = 0.0
        if config.TRAIN_MODE=='train' or config.TRAIN_MODE=='t_e' :
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                config.defrost()
                config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
                config.freeze()
                if 'amp' in checkpoint and config.APEX_AMP and checkpoint['config'].APEX_AMP:
                    amp.initialize(model, opt_level='O1')
                    amp.load_state_dict(checkpoint['amp'])
                logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
                if 'max_accuracy' in checkpoint and 'best_auc' in checkpoint:
                    max_accuracy = checkpoint['max_accuracy']
                    best_auc = checkpoint['best_auc']
                if 'best_f1' in checkpoint:
                    best_f1 = checkpoint['best_f1']
    else:
        msg = model.load_state_dict(checkpoint, strict=False)
        logger.info(msg)
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy,best_auc,best_f1

def load_checkpoint_V2(config, model,optimizer=None, lr_scheduler=None,logger=None,is_ema=False):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if is_ema:
        if 'ema' in checkpoint:
            msg = model.load_state_dict(checkpoint['ema'], strict=False)
            logger.info(msg)
            return 0
    # 是否只读取模型
    if 'state_dict' in checkpoint:
        msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(msg)
        max_accuracy = 0.0
        best_auc = 0.0
        best_f1 = 0.0
        if config.TRAIN_MODE=='train' or config.TRAIN_MODE=='t_e' :
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                config.defrost()
                config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
                config.freeze()
                if 'amp' in checkpoint and config.APEX_AMP and checkpoint['config'].APEX_AMP:
                    amp.initialize(model, opt_level='O1')
                    amp.load_state_dict(checkpoint['amp'])
                logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
                if 'max_accuracy' in checkpoint and 'best_auc' in checkpoint:
                    max_accuracy = checkpoint['max_accuracy']
                    best_auc = checkpoint['best_auc']
                if 'best_f1' in checkpoint:
                    best_f1 = checkpoint['best_f1']
    else:
        msg = model.load_state_dict(checkpoint, strict=False)
        logger.info(msg)
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy,best_auc,best_f1

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger,is_best,best_auc,best_f1,ema,is_ema=False,best_patr90=0.0):
    save_state = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'max_accuracy': max_accuracy,
                  'best_auc': best_auc,
                  'best_f1': best_f1,
                  'p@r90':best_patr90,
                  'epoch': epoch,
                  'config': config,
                  'ema':ema.module.state_dict() if ema is not None else None}
    save_state_best = {'state_dict': model.state_dict(),
                        'max_accuracy': max_accuracy,
                        'best_auc': best_auc,
                        'best_f1': best_f1,
                        'p@r90':best_patr90,
                        'epoch': epoch,
                        'config': config}
    if config.TRAIN.LR_SCHEDULER.NAME is not None:
        save_state['lr_scheduler'] = lr_scheduler.state_dict()
    if config.APEX_AMP:
        amp.initialize(model, opt_level='O1')
        save_state['amp'] = amp.state_dict()

    if is_ema:
        save_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ema_ckpt.pth')
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ema_best_model.pth')
        history_best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_his_ema_best_model.pth')
    else:
        save_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ckpt.pth')
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_best_model.pth')
        history_best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_his_best_model.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    
    if is_best:
        torch.save(save_state_best, best_path)
        # shutil.copyfile(save_path, best_path)
    if epoch == config.TRAIN.EPOCHS - 1:
        if os.path.exists(history_best_path):
            checkpoint = torch.load(best_path, map_location='cpu')
            checkpoint_his = torch.load(history_best_path, map_location='cpu')
            if config.TEST.BEST_METRIC.lower() == 'f1':
                if 'best_f1' in checkpoint_his:
                    if checkpoint['best_f1'] > checkpoint_his['best_f1']:
                        shutil.copyfile(best_path, history_best_path)
                else:
                    shutil.copyfile(best_path, history_best_path)
            elif config.TEST.BEST_METRIC.lower() == 'p@r90':
                if 'p@r90' in checkpoint_his:
                    if checkpoint['p@r90'] > checkpoint_his['p@r90']:
                        shutil.copyfile(best_path, history_best_path)
                else:
                    shutil.copyfile(best_path, history_best_path)
            elif config.TEST.BEST_METRIC.lower() == 'top1':
                if checkpoint['max_accuracy'] > checkpoint_his['max_accuracy']:
                    shutil.copyfile(best_path, history_best_path)
            elif config.TEST.BEST_METRIC.lower() == 'auc':
                if checkpoint['best_auc'] > checkpoint_his['best_auc']:
                    shutil.copyfile(best_path, history_best_path)
        else:
            shutil.copyfile(best_path, history_best_path)

    logger.info(f"{save_path} saved !!!")

def save_checkpoint_V2(config, epoch, model, best_metrics,optimizer, lr_scheduler, logger,is_best,ema,is_ema=False):
    save_state = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'best_metrics':best_metrics,
                  'epoch': epoch,
                  'config': config,
                  'ema':ema.module.state_dict() if ema is not None else None}
    save_state_best = {'state_dict': model.state_dict(),
                        'best_metrics':best_metrics,
                        'epoch': epoch,
                        'config': config}
    if config.TRAIN.LR_SCHEDULER.NAME is not None:
        save_state['lr_scheduler'] = lr_scheduler.state_dict()
    if config.APEX_AMP:
        amp.initialize(model, opt_level='O1')
        save_state['amp'] = amp.state_dict()

    if is_ema:
        save_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ema_ckpt.pth')
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ema_best_model.pth')
        history_best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_his_ema_best_model.pth')
    else:
        save_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ckpt.pth')
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_best_model.pth')
        history_best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_his_best_model.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    
    if is_best:
        torch.save(save_state_best, best_path)
        # shutil.copyfile(save_path, best_path)
    if epoch == config.TRAIN.EPOCHS - 1:
        if os.path.exists(history_best_path):
            checkpoint = torch.load(best_path, map_location='cpu')
            checkpoint_his = torch.load(history_best_path, map_location='cpu')
            metrics = checkpoint['best_metrics']
            metrics_his = checkpoint_his['best_metrics']
            best_metric_name = config.TEST.BEST_METRIC.lower()

            if best_metric_name in metrics_his:
                if metrics[best_metric_name] > metrics_his[best_metric_name]:
                    shutil.copyfile(best_path, history_best_path)
                else:
                    shutil.copyfile(best_path, history_best_path)
        else:
            shutil.copyfile(best_path, history_best_path)

    logger.info(f"{save_path} saved !!!")

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
    
def l1_regularizer(input_features):
    l1_loss = torch.norm(input_features, p=1)
    return l1_loss

def l2_regularizer(input_features):
    l2_loss = torch.norm(input_features, p=2)
    return l2_loss

def decimal_num(number):
    if type(number) == int:
        return 0
    else:
        num = 1
        while number * 10 ** num != int(number * 10 ** num ):
            num += 1
        return num
   
def getDataByStick(data,stick):
    '''
    data    [data_y,data_x]   y坐标数据，x坐标数据
    stick                     坐标刻度
    '''
    j = -1
    diff = 9999
    _stick = stick.copy()
    _list = []
    for a in range(0,len(data[0])):
        if len(_stick):
            diff_tem = abs(data[1][a]-_stick[j])
            if diff_tem < diff:
                diff = diff_tem
            if diff_tem > diff:
                #print(data[0][a-1],data[1][a-1])
                _list.append((data[0][a-1],data[1][a-1]))
                if len(_stick)>1:
                    diff = abs(data[1][a]-_stick[j-1])
                else:
                    diff=9999
                del _stick[j]
    return _list

def get_sigmod_num(start=0,curr_step=0,all_step=0,end=0.999,alph=10):
    '''
    alph  平缓系数,数字越小则曲线越平缓
    '''
    thr_min_conf = start + (round((1 / (1+math.exp(-alph*(curr_step / all_step)))-0.5 )*2,3)) * (end-start)
    return thr_min_conf
class ModelEmaV3(torch.nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, decay_diff=0.9999,device=None,diff_layers = [],ban_para = [],init_para=[]):
        super(ModelEmaV3, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.decay_diff = decay_diff
        self.diff_layers = diff_layers
        self.ban_para = ban_para
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for k,ema_v, model_v in zip(self.module.state_dict().keys(),self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                if k.split('.')[0] not in self.ban_para:
                    if k.split('.')[0] in self.diff_layers:
                        ema_v.copy_(update_fn(ema_v, model_v,self.decay_diff))
                    else:
                        ema_v.copy_(update_fn(ema_v, model_v,self.decay))

    def update(self, model):
        self._update(model, update_fn=lambda e, m, decay: decay * e + (1. - decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def log_loss(tea,stu,config):
    tps_stu = 1 if config.RDD_TRANS.SHARPEN_STUDENT is None else config.RDD_TRANS.SHARPEN_STUDENT
    tps_tea = 1 if config.RDD_TRANS.SHARPEN_TEACHER is None else config.RDD_TRANS.SHARPEN_TEACHER
    tea = tea.detach()
    stu =  stu / tps_stu
    tea = torch.nn.functional.softmax(tea / tps_tea,dim=-1)
    return -(tea*F.log_softmax(stu,dim=-1).sum(dim=-1).mean())

# 相较于timm的版本，我在这里对target也做softmax
class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy_v2, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target,dim=-1) * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

# 相较于timm的版本，我想要实现gradient accumulation，需要把梯度计算和更新参数步骤分开
class NativeScaler_V2:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer=None, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False,acc_gradient=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if not acc_gradient:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def opt_step(self,optimizer,clip_grad=None, clip_mode='norm', parameters=None):
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)