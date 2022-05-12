from .base import BaseTrainer
from .imagenet_cls import *
from .rdd_trans import *
from .ts_cam import *
from .wsplin import *
from .ioplin import *

def build_trainer(config):
    if config.TRAINER.NAME.lower() == 'rdd_trans':
        thr_list = np.array([config.RDD_TRANS.NOR_THR for i in range(config.RDD_TRANS.INST_NUM_CLASS)])
        trainer = RddTransTrainer(
            thr_list=thr_list,
            dis_ratio_list=[[] for i in range(len(thr_list))],
            criterion_teacher = config.RDD_TRANS.TEACHER_LOSS
        )
    base = BaseTrainer(trainer=trainer)
    return base.train_one_epoch,base.predict,base.validate,base.best_metrics