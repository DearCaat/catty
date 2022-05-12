import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

def _build_criterion(name,config):
    if name == 'crossentropy':
        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    elif name == 'kl':
        criterion = SoftTargetCrossEntropy_v2()
    return criterion

def build_criterion(config):
    '''
    loss_name : XX_XX_XX
    '''
    criterions = []
    losses_name = config.TRAIN.LOSS.NAME.lower().split('_')

    for loss_name in losses_name:
        criterions.append(_build_criterion(loss_name,config))

    return criterions

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