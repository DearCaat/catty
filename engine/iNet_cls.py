import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score
import torch
from timm.utils import *

from engine.rdd_trans import predict

class INetClsEngine:
    def __init__(self,config,**kwargs):
        # 除了主损失以外的metric，主损失会每个iter进行log
        # 每个iter更新的指标需要初始化成AverageMeter
        # 默认情况下，训练函数内不进行任何评价指标计算
        self.train_metrics = OrderedDict([])
        self.train_metrics_epoch_log =[]
        self.train_metrics_iter_log =[]

        self.topk = (1,1) if config.MODEL.NUM_CLASSES < 5 else (1,5)

        self.test_metrics = OrderedDict([
        ('acc1',AverageMeter()),
        ('acc5',AverageMeter()),
        ('macro_f1',.0),
        ('micro_f1',.0),])

        self.test_metrics_epoch_log =['macro_f1','micro_f1']
        self.test_metrics_iter_log =['acc1','acc5']

        if config.TEST.BINARY_MODE or config.MODEL.NUM_CLASSES == 2:
            self.test_metrics.update(OrderedDict([
                ('auc',.0),
            ]))
            self.test_metrics_epoch_log += ['auc']
        
    def cal_loss_func(self,config,models,idx,samples,targets,epoch,num_steps,criterions,**kwargs):
        # torch.autograd.set_detect_anomaly(True)
        predictions = models['main'](samples)
        loss = criterions[0](predictions,targets)
        metrics_values = OrderedDict([
        ])

        return loss,metrics_values,OrderedDict([])

    def update_per_iter(self,config,epoch,idx,**kwargs):
        return

    def update_per_epoch(self,config,epoch,**kwargs):
        return

    def measure_per_iter(self,config,models,samples,targets,criterions,**kwargs):
        # compute output
        output = models['main'](samples)

        if isinstance(output, (tuple, list)):
            predition = output[0]
        else:
            predition = output

        output_soft = torch.nn.functional.softmax(predition,dim=-1)

        loss = criterions[0](predition, targets)
            
        pred = output_soft.cpu().numpy()
        label = targets.cpu().numpy()

        # topk acc cls
        acc1,acc5 = accuracy(output, targets, topk=self.topk)

        metrics_values = OrderedDict([
        ('acc1',[acc1,targets.size(0)]),
        ('acc5',[acc5,targets.size(0)]),
        ])

        others = OrderedDict([])

        return loss,pred,label,metrics_values,others
    def measure_per_epoch(self,config,**kwargs):
        assert config.MODEL.NUM_CLASSES == 2 and config.TEST.BINARY_MODE

        metrics_values = OrderedDict([])
        
        _binary_test = config.MODEL.NUM_CLASSES == 2 or config.TEST.BINARY_MODE
        label = kwargs['label']
        pred = kwargs['pred']

        ma_f1 = f1_score(label,np.argmax(pred,axis=1),average='macro')
        mi_f1 = f1_score(label,np.argmax(pred,axis=1),average='micro')

        if _binary_test:
            if config.TEST.BINARY_MODE:
                label = label!=config.DATA.DATA_NOR_INDEX
                pred = 1-pred[:,config.DATA.DATA_NOR_INDEX]
            elif config.MODEL.NUM_CLASSES == 2:
                ma_f1 = f1_score(label,np.argmax(pred,axis=1),average='binary')
                mi_f1 = ma_f1
                pred = 1-pred[:,config.DATA.CLS_NOR_INDEX]
            
            auc = roc_auc_score(label,pred)
            metrics_values.update(OrderedDict([('auc',auc),]))

        metrics_values.update(OrderedDict([
        ('macro_f1',ma_f1),
        ('micro_f1',mi_f1)
        ]))
        others = OrderedDict([
        ])

        return metrics_values,others
