from .base import BaseTrainer
from .imagenet_cls import *
from .rdd_trans import *
from .ts_cam import *
from .wsplin import *
from .ioplin import *

def build_trainer(config):
    if config.TRAINER.NAME.lower() == 'rdd_trans':
        trainer = RddTransTrainer()
    base = BaseTrainer(trainer=trainer)
    return base.train_one_epoch,base.predict,base.validate