import os
import torch
import torch.distributed as dist
import shutil
from copy import deepcopy

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
def load_best_model(config,model,logger,is_ema=False):
    if is_ema:
        ckpt_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_ema_ckpt.pth')
    else:
        ckpt_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_ckpt.pth')
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if is_ema:
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_ema_best_model.pth')
    else:
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_best_model.pth')
    logger.info(f"==============> Loading the best model....................")
    checkpoint = torch.load(best_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(msg)
    if config.APEX_AMP and checkpoint['config'].APEX_AMP:
        amp.initialize(model, opt_level='O1')

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    best_auc = 0.0
    if not config.TRAIN_MODE=='eval' and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
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

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy,best_auc

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger,is_best,best_auc,ema,is_ema=False):
    save_state = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'best_auc': best_auc,
                  'epoch': epoch,
                  'config': config,
                  'ema':ema.module.state_dict() if ema is not None else None}
    if config.APEX_AMP:
        amp.initialize(model, opt_level='O1')
        save_state['amp'] = amp.state_dict()

    if is_ema:
        save_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_ema_ckpt.pth')
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_ema_best_model.pth')
    else:
        save_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_ckpt.pth')
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_best_model.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)
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
    for a in range(0,len(data[0])):
        if len(_stick):
            if round(data[1][a],decimal_num(_stick[j])) == _stick[j]:
                diff_tem = abs(data[1][a]-_stick[j])
                if diff_tem < diff:
                    diff = diff_tem
                if diff_tem > diff:
                    print(data[0][a-1],data[1][a-1])
                    diff = 9999
                    del _stick[j]    

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
    def __init__(self, model, decay=0.9999, device=None,diff_layers = []):
        super(ModelEmaV3, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.decay_diff = decay
        self.diff_layers = diff_layers
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for k,ema_v, model_v in zip(self.module.state_dict().keys(),self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                if k.split('.')[0] in self.diff_layers:
                    ema_v.copy_(update_fn(ema_v, model_v,self.decay_diff))
                else:
                    ema_v.copy_(update_fn(ema_v, model_v,self.decay))

    def update(self, model):
        self._update(model, update_fn=lambda e, m, decay: decay * e + (1. - decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)