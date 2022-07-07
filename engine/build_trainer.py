from .base import BaseTrainer
from .iNet_cls import *
from .rdd_trans import *
from .ts_cam import *
from .wsplin import *
from .ioplin import *
from .mim import *

def build_trainer(config):
    if config.TRAINER.NAME.lower() == 'rdd_trans':
        thr_list = np.array([config.RDD_TRANS.NOR_THR for i in range(config.RDD_TRANS.INST_NUM_CLASS)])
        engine = RddTransTrainer(
            thr_list=thr_list,
            dis_ratio_list=[[] for i in range(len(thr_list))],
            criterion_teacher = config.RDD_TRANS.TEACHER_LOSS
        )
    elif config.TRAINER.NAME.lower() == 'inet_cls':
        engine = INetClsEngine(config)
    elif config.TRAINER.NAME.lower() == 'mim':
        engine = MIMEngine(config)
    else:
        raise NotImplementedError
    base = BaseTrainer(engine=engine)
    return base.train_one_epoch,base.predict,base.validate,base.best_metrics