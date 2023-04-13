from collections import OrderedDict

from timm.utils import *

from .iNet_cls import INetClsEngine

class SSLEngine(INetClsEngine):
    def __init__(self,config,**kwargs):
        super().__init__(config)
        # SSL training
        if not config.TEST.LINEAR_PROB.ENABLE:
            self.test_metrics = OrderedDict([
                ('loss',AverageMeter()),
            ])
            self.train_metrics = OrderedDict([
                ('loss_ssl',AverageMeter()),
            ])
            self.train_metrics_epoch_log =['loss_ssl']
            self.train_metrics_iter_log =['loss_ssl']
        # Linear prob test
        else:
            self.train_metrics = OrderedDict([
                ('loss',AverageMeter()),
            ])
            self.train_metrics_epoch_log =['loss']
            self.train_metrics_iter_log =['loss']

    def update_per_iter(self,config,epoch,idx,models,**kwargs):
        pass
        return 

    # do something to gradients, this function is call after cal the gradients(include the clip gradients), and before the optimize
    def change_grad_func(self,config,epoch,models,idx,**kwargs):
        pass 
        return 
        
    def cal_loss_func(self,config,models,idx,samples,targets,epoch,num_steps,criterions,**kwargs):
        if config.TEST.LINEAR_PROB.ENABLE:
            predictions = models['main'](samples)
            loss = criterions[0](predictions,targets)
            metrics_values = OrderedDict([
            ])
        else:
            loss_ssl = models['main'](samples[0],samples[1])['loss']
            loss = loss_ssl
            metrics_values = OrderedDict([
                ('loss_ssl',[loss_ssl,targets.size(0)]),
            ])

        return loss,metrics_values,OrderedDict([])
