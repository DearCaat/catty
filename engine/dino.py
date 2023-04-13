import numpy as np
import sys,os
from collections import OrderedDict

from timm.utils import *
import torch
import torch.nn as nn

from .iNet_cls import INetClsEngine

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils import cosine_scheduler

class DINOEngine(INetClsEngine):
    def __init__(self,config,**kwargs):
        super().__init__(config)
        # 除了主损失以外的metric，主损失会每个iter进行log
        # 每个iter更新的指标需要初始化成AverageMeter
        self.train_metrics = OrderedDict([
        ('loss_cl',AverageMeter()),
        ('loss_cls',AverageMeter()),
        ])
        
        if not config.TEST.LINEAR_PROB.ENABLE:
            self.test_metrics = OrderedDict([
                ('loss',AverageMeter()),
            ])
            self.momentum_schedule=cosine_scheduler(config.DINO.MOMENTUM_TEACHER,1,config.TRAIN.EPOCHS,config.DATA.LEN_DATALOADER_TRAIN)

            self.train_metrics_epoch_log =['loss_cl','loss_cls']
            self.train_metrics_iter_log =['loss_cl','loss_cls']

    def cancel_gradients_last_layer(self,epoch, _model, freeze_last_layer):
        if epoch >= freeze_last_layer:
            return
        for n, p in _model.named_parameters():
            if "last_layer" in n:
                p.grad = None

    def update_per_iter(self,config,epoch,idx,models,**kwargs):
        if not config.TEST.LINEAR_PROB.ENABLE:
            # update teacher model
            with torch.no_grad():
                m = self.momentum_schedule[idx]  # momentum parameter
                for param_q, param_k in zip(models['main'].parameters(), models['teacher'].parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        return

    # do something to gradients, this function is call after cal the gradients(include the clip gradients), and before the optimize
    def change_grad_func(self,config,epoch,models,idx,**kwargs):
        if not config.TEST.LINEAR_PROB.ENABLE:
            self.cancel_gradients_last_layer(epoch,models['main'],config.DINO.FREEZE_LAST_LAYER)
        

    def cal_loss_func(self,config,models,idx,samples,targets,epoch,num_steps,criterions,**kwargs):
        if config.TEST.LINEAR_PROB.ENABLE:
            predictions = models['main'](samples)
            loss = criterions[0](predictions,targets)
            metrics_values = OrderedDict([
            ])
        else:
            student_output = models['main'](samples)
            teacher_output = models['teacher'](samples[:2])

            if not config.CL.ONLY_CL:
                loss_cls = criterions[0](student_output,targets)
                loss_cl = criterions[1](student_output, teacher_output, epoch)
                loss = loss_cls + config.CL.LOSS_ALPHA*loss_cl
            else:
                loss_cls = 0.
                loss_cl = criterions[0](student_output, teacher_output,epoch)
                loss = loss_cl
            metrics_values = OrderedDict([
                ('loss_cl',[loss_cl,targets.size(0)]),
                ('loss_cls',[loss_cls,targets.size(0)]),
            ])

        return loss,metrics_values,OrderedDict([])
