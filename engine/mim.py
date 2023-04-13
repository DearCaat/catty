import numpy as np
from collections import OrderedDict

from timm.utils import *
import torch

from utils import cosine_scheduler
from .iNet_cls import INetClsEngine
class MIMEngine(INetClsEngine):
    def __init__(self,config,**kwargs):
        super().__init__(config)
        # 除了主损失以外的metric，主损失会每个iter进行log
        # 每个iter更新的指标需要初始化成AverageMeter
        self.train_metrics = OrderedDict([
        ('loss_cl',AverageMeter()),
        ('loss_mim',AverageMeter()),
        ('loss_cls',AverageMeter()),
        ('loss_mim_pix',AverageMeter()),
        ])

        self.test_metrics = OrderedDict([
        ('acc1',AverageMeter()),
        ('acc1_tea',AverageMeter()),
        ('acc5',AverageMeter()),
        ('macro_f1',.0),
        ('micro_f1',.0),])

        self.test_metrics_iter_log =['acc1','acc5','acc1_tea']
        self.train_metrics_epoch_log =['loss_mim','loss_cls','loss_cl','loss_mim_pix']
        self.train_metrics_iter_log =['loss_mim','loss_cls','loss_cl','loss_mim_pix']

        self.momentum_schedule=cosine_scheduler(config.MIM.MOMENTUM_TEACHER,1,config.TRAIN.EPOCHS,config.DATA.LEN_DATALOADER_TRAIN) if config.MIM.SCHEDULER_TEACHER is not None else config.MIM.MOMENTUM_TEACHER

    def update_per_iter(self,config,epoch,idx,models,**kwargs):
        # update teacher model
        with torch.no_grad():
            if isinstance(self.momentum_schedule,(list,tuple,np.ndarray)):
                m = self.momentum_schedule[epoch*config.DATA.LEN_DATALOADER_TRAIN+idx]  # momentum parameter
            else:
                m = self.momentum_schedule

            for param_q, param_k in zip(models['main'].parameters(), models['teacher'].parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        return

    def cal_loss_func(self,config,models,idx,samples,targets,epoch,num_steps,criterions,**kwargs):
        unmask = None

        if config.AUG.MULTI_VIEW is not None or config.DATA.DATALOADER_NAME.split('_')[2] in ('dinofgia','multiviewfgia'):
            if config.MIM.MULTI_INPUT:
                samples_t,samples_s = samples,samples
            else:
                samples_t = samples[1]
                samples_s = samples[0]
        else:
            samples_t,samples_s = samples,samples

        # mask student
        with torch.no_grad():
            if config.MIM.SELECT_ENABLE:
                if config.MIM.MULTI_INPUT:
                    logits_t,rec_t,unmask = []
                    for _sample in samples_t:
                        _,_l_t,_r_t,_u,_,_ = models['teacher'](_sample)
                        logits_t.append(_l_t)
                        rec_t.append(_r_t)
                        unmask.append(_u)
                else:
                    _,logits_t,rec_t,unmask,_,_ = models['teacher'](samples_t,require_unmask=config.MIM.SELECT_ENABLE,need_cl=config.MIM.CL_ENABLE,need_mask=config.MIM.MASK_TEACHER)

        if config.MIM.MULTI_INPUT:
            pass
        else:
            logits_cls,logits_s,rec_s,unmask,mask,loss_mim_pix = models['main'](samples_s,need_mask=config.MIM.MASK_STUDENT,require_unmask=(not config.MIM.MASK_STUDENT) and config.MIM.SELECT_ENABLE,need_cl=config.MIM.CL_ENABLE,need_rec=config.MIM.MIM_PIX_ENABLE)
            
        # mask teacher
        with torch.no_grad():
            if not config.MIM.MASK_STUDENT:
                if config.MIM.MULTI_INPUT:
                    logits_t,rec_t,unmask = []
                    for _sample in samples_t:
                        _,_l_t,_r_t,_u = models['teacher'](_sample)
                        logits_t.append(_l_t)
                        rec_t.append(_r_t)
                        unmask.append(_u)
                else:
                    _,logits_t,rec_t,unmask,mask,_ = models['teacher'](samples_t,unmask=unmask,need_mask=config.MIM.MASK_TEACHER,need_cl=config.MIM.CL_ENABLE,need_rec=config.MIM.MIM_PIX_ENABLE)

        loss_cls = criterions[0](logits_cls,targets)

        if config.MIM.CL_ENABLE:
            loss_cl = criterions[1](logits_s,logits_t)
        else:
            loss_cl = 0.

        if config.MIM.MIM_ENABLE:
            loss_mim = criterions[1](rec_s,rec_t,mask)
        else:
            loss_mim = 0.

        loss =  config.MIM.CLS_LOSS_ALPHA*loss_cls + config.MIM.CL_LOSS_ALPHA * loss_cl +config.MIM.MIM_LOSS_ALPHA * loss_mim +config.MIM.MIM_PIX_LOSS_ALPHA*loss_mim_pix

        metrics_values = OrderedDict([
            ('loss_mim',[loss_mim,targets.size(0)]),
            ('loss_cl',[loss_cl,targets.size(0)]),
            ('loss_mim_pix',[loss_mim_pix,targets.size(0)]),
            ('loss_cls',[loss_cls,targets.size(0)]),
        ])

        return loss,metrics_values,OrderedDict([])

    def measure_per_iter(self,config,models,samples,targets,criterions,**kwargs):

            # compute output
            output = models['main'](samples)
            output_tea = models['teacher'](samples)

            if isinstance(output, (tuple, list)):
                prediction = output[0]
            else:
                prediction = output

            if isinstance(output_tea, (tuple, list)):
                prediction_tea = output_tea[0]
            else:
                prediction_tea = output_tea

            output_soft = torch.nn.functional.softmax(prediction,dim=-1)

            loss = criterions[0](prediction, targets)
                
            pred = output_soft.cpu().numpy()
            label = targets.cpu().numpy()

            # topk acc cls
            acc1,acc5 = accuracy(prediction, targets, topk=self.topk)
            acc1_tea,acc5_tea = accuracy(prediction_tea, targets, topk=self.topk)

            metrics_values = OrderedDict([
            ('acc1',[acc1,targets.size(0)]),
            ('acc5',[acc5,targets.size(0)]),
            ('acc1_tea',[acc1_tea,targets.size(0)]),
            ])

            others = OrderedDict([])

            return loss,pred,label,metrics_values,others