import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score

from timm.utils import *

class INetClsEngine:
    def __init__(self,config,**kwargs):
        # 除了主损失以外的metric，主损失会每个iter进行log
        # 每个iter更新的指标需要初始化成AverageMeter
        self.train_metrics = OrderedDict([
        ('acc1',AverageMeter()),
        ('acc5',AverageMeter()),
        ])
        self.train_metrics_epoch_log =[]
        self.train_metrics_iter_log =[]

        self.test_metrics = OrderedDict([
        ('acc1',AverageMeter()),
        ('acc5',AverageMeter()),])

        self.test_metrics_epoch_log =[]
        self.test_metrics_iter_log =['acc1','acc5']

        if config.TEST.BINARY_MODE:
            self.test_metrics.update(OrderedDict([
                ('auc',.0),
                ('macro_f1',.0),
                ('micro_f1',.0),
            ]))
            self.test_metrics_epoch_log += ['auc','macro_f1','micro_f1']
        

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

    def measure_per_iter(self,config,output,targets,**kwargs):
        topk = (1,1) if config.MODEL.NUM_CLASSES < 5 else (1,5)

        # topk acc cls
        acc1,acc5 = accuracy(output, targets, topk=topk)

        metrics_values = OrderedDict([
        ('acc1',[acc1,targets.size(0)]),
        ('acc5',[acc5,targets.size(0)]),
        ])

        others = OrderedDict([])

        return metrics_values,others
    def measure_per_epoch(self,config,**kwargs):
        metrics_values = OrderedDict([])
        if config.TEST.BINARY_MODE:
            auc = 0
            label = kwargs['label']
            pred = kwargs['pred']
            if config.BINARYTRAIN_MODE:
                ma_f1 = f1_score(np.array(label!=config.DATA.NOR_CLS_INDEX,dtype=int),np.argmax(pred,axis=1),average='binary')
                mi_f1 = ma_f1

                auc = roc_auc_score(np.array(label!=config.DATA.NOR_CLS_INDEX,dtype=int), pred[:,1])

            else:
                ma_f1 = f1_score(label,np.argmax(pred,axis=1),average='macro')
                mi_f1 = f1_score(label,np.argmax(pred,axis=1),average='micro')

                auc = roc_auc_score(np.array(label!=config.DATA.NOR_CLS_INDEX,dtype=int), 1-pred[:,config.DATA.NOR_CLS_INDEX])

            metrics_values.update(OrderedDict([
            ('auc',auc),
            ('macro_f1',ma_f1),
            ('micro_f1',mi_f1)
            ]))
        others = OrderedDict([
        ])

        return metrics_values,others
