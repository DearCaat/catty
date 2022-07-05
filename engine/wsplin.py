import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score
import math

from timm.utils import *

from.iNet_cls import INetClsEngine
class WSPLINEngine(INetClsEngine):
    def __init__(self,config,**kwargs):
        super().__init__(config)
        if config.WSPLIN.RANDSM:
            _index = [random.sample(range(12),math.ceil(12 * config.WSPLIN.SPARSE_RATIO)),random.sample(range(12,16),math.ceil(4 * config.WSPLIN.SPARSE_RATIO)),[-1]]
            self.index = [ind for st in _index for ind in st]
        else:
            self.index = None
        # 添加p@r评价指标
        self.test_metrics.update(OrderedDict([
                ('p@r90',.0),
                ('p@r95',.0),
            ]))

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
        label = kwargs['label']
        pred = kwargs['pred']

        ma_f1 = f1_score(label,np.argmax(pred,axis=1),average='macro')
        mi_f1 = f1_score(label,np.argmax(pred,axis=1),average='micro')
        if config.MODEL.NUM_CLASSES == 2:
            ma_f1 = f1_score(label,np.argmax(pred,axis=1),average='binary')
            
        if config.TEST.BINARY_MODE:
            auc = roc_auc_score(np.array(label!=config.DATA.DATA_NOR_INDEX,dtype=int), 1-pred[:,config.DATA.DATA_NOR_INDEX])

            metrics_values.update(OrderedDict([('auc',auc),]))
        
        metrics_values.update(OrderedDict([
        ('macro_f1',ma_f1),
        ('micro_f1',mi_f1)
        ]))
        others = OrderedDict([
        ])

        return metrics_values,others
