from contextlib import suppress
from select import select
import time
from typing import Dict
import numpy as np
import math
import datetime
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score
from utils import get_sigmod_num

import torch
from torch import Tensor
from timm.utils import *
from timm.models import  model_parameters


class RddTransTrainer:
    def __init__(self,**kwargs):
        self.thr_list,self.dis_ratio_list,self.criterion_teacher = kwargs['thr_list'],kwargs['dis_ratio_list'],kwargs['criterion_teacher']

        self.train_metrics = OrderedDict([
        ('loss_teacher_meter',AverageMeter()),
        ('dis_ins_meter',AverageMeter()),
        ('patch_num_meter',AverageMeter()),
        ('cluster_num_meter',AverageMeter()),
        ('cluster_ema_num_meter',AverageMeter()),
        ('dis_rec_meters',np.array([AverageMeter() for i in range(len(self.thr_list))])),
        ('selec_rec_meters',np.array([AverageMeter() for i in range(len(self.thr_list))]))
        ])
        self.train_metrics_epoch_log =['dis_rec_meters','selec_rec_meters']
        self.train_metrics_iter_log =['loss_teacher_meter','dis_ins_meter','patch_num_meter','cluster_num_meter','cluster_ema_num_meter']

        self.test_metrics = OrderedDict([
        ('acc1_meter',AverageMeter()),
        ('acc5_meter',AverageMeter()),
        ('cluster_num_meter',AverageMeter()),
        ('auc',.0),
        ('macro_f1',.0),
        ('micro_f1',.0)
        ])
        self.test_metrics_epoch_log =['auc','macro_f1','micro_f1']
        self.test_metrics_iter_log =['acc1_meter','acc5_meter','patch_num_meter','cluster_num_meter']


    def cal_loss_func(self,config,model,idx,samples,targets,targets_bin,epoch,num_steps,criterion,**kwargs,):

        criterion_teacher = self.criterion_teacher if self.criterion_teacher is not None else criterion
        dis_ins = 0

        output,o_inst,_,cluster_num = model(samples)
        # 设定正常图片在类别中的索引
        pl_nor_cls_index = 0 if config.RDD_TRANS.INST_NUM_CLASS == 2 else config.DATA.NOR_CLS_INDEX
        
            #_,pl_inst = teacher_ema.module(samples)
        #output_pl = torch.nn.functional.softmax(pl_inst,dim=2)
        if config.RDD_TRANS.INST_NUM_CLASS != 2:
            targets_pl = targets
        else:
            if targets_bin is None:
                targets_bin = targets.clone()
                targets_bin[targets==config.DATA.NOR_CLS_INDEX] = 0
                targets_bin[targets!=config.DATA.NOR_CLS_INDEX] = 1
            targets_pl = targets_bin

        if config.RDD_TRANS.PERSUDO_LEARNING: 
            with torch.no_grad():
                _,pl_inst,output_pl,cluster_num_ema = model_ema.module(samples)
                torch.cuda.empty_cache()
            b,p,cls = pl_inst.shape
            t_cpu = targets_pl.cpu()
            ins_t = targets_pl.unsqueeze(-1).repeat((1,p))

            output_bag_label = output_pl[torch.functional.F.one_hot(ins_t,num_classes=config.RDD_TRANS.INST_NUM_CLASS) == 1].view(b,p)        #包所属标签下的置信度
            if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                out_tmp_sort,_ = torch.sort(output_bag_label,dim=-1,descending=True)         # [b p]
                min_nor_thr = out_tmp_sort[[i for i in range(b)],np.floor(p*self.thr_list[t_cpu])] # [b 1]
                min_nor_thr = min_nor_thr.unsqueeze(-1).repeat((1,p))        # [b p]
            else:
                min_nor_thr = torch.ones(size=(b,p))
            min_nor_thr = min_nor_thr.cuda(non_blocking=True)

            _,label_pl = torch.max(output_pl,dim=2)
            #label_pl = torch.argmax(output_pl,dim=2)
            #label_tmp = label_pl.clone()
            #获得正常包索引和病害包索引
            #bs_index_nor = targets_pl==pl_nor_cls_index
            #bs_index_dis = bs_index_nor==False
            ps_mask_nor = (ins_t - pl_nor_cls_index == 0)
            ps_mask_dis = ps_mask_nor==False

            #将所有实例设为正常
            label_pl[:,:] = pl_nor_cls_index 
            # sigmod函数来保证前期增加速率远远高于后期，优于线性增长
            thr_min_conf = get_sigmod_num(0.9,(epoch-config.RDD_TRANS.INIT_STAGE_EPOCH) * num_steps + idx,(config.TRAIN.EPOCHS * num_steps))
            thr_min_dis_conf = get_sigmod_num(0.5,(epoch-config.RDD_TRANS.INIT_STAGE_EPOCH) * num_steps + idx,(config.TRAIN.EPOCHS * num_steps))
            #把网络判断为不是正常的部分实例置为包病害标签
            if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                #判断为相应病害的实例置为包标签
                #mask_ins =  (label_tmp - ins_t  == 0)
                #将病害包里判断为不是正常的实例都置为包标签 & (output_bag_label > thr_min_conf)
                mask_ins =   ps_mask_dis & (((output_pl[:,:,pl_nor_cls_index] < (1-thr_min_conf)) & (output_bag_label > thr_min_dis_conf) )  | (output_bag_label - min_nor_thr >  0))
                label_pl[mask_ins] = ins_t[mask_ins]
                #选取部分置信度比较高的实例参与loss计算   label_tmp - config.DATA.NOR_CLS_INDEX  == 0
                mask_ins = (mask_ins | (ps_mask_dis & (label_pl==pl_nor_cls_index) & (output_pl[:,:,pl_nor_cls_index] - thr_min_dis_conf > 0))) | ((output_pl[:,:,pl_nor_cls_index] > thr_min_dis_conf) & ps_mask_nor)
            else:
                #全部都用
                #mask_ins = (label_tmp - label_tmp == 0)
                #只要正常图片 (output_bag_label - 0.9 > 0)
                mask_ins = ps_mask_nor 

                #label_pl[bs_index_dis] = ins_t[bs_index_dis]

            #统计病害实例
            dis_count = torch.count_nonzero(label_pl - pl_nor_cls_index,dim=1) 
            dis_ins += torch.sum(dis_count)

            selec_rec = [ () for i in range(len(self.thr_list)) ]
            dis_rec = [ () for i in range(len(self.thr_list)) ]
            for i in range(len(self.thr_list)):
                i_index = targets_pl==i
                dis_tmp = dis_count[i_index]
                selec_tmp = mask_ins[i_index]

                if len(selec_tmp)>0:
                    selec_rec[i] = ( len(selec_tmp[selec_tmp==True]) / (len(selec_tmp)*p),b)

                if len(dis_tmp) > 0 and epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                    thr_tmp = torch.sum(dis_tmp) / (p * len(dis_tmp))
                    self.dis_ratio_list[i].append(thr_tmp.cpu())

                    #thr_list[i] = 0.9999 * thr_list[i] + (1-0.9999)* thr_tmp
                    #thr_list[i] = 0.1 if thr_list[i] < 0.1 else thr_list[i]
                    #thr_list[i] = thr_tmp
                    dis_rec[i] = (thr_tmp,b)

            #选择部分patch来计算loss
            #if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
            label_pl = label_pl[mask_ins]
            o_inst = o_inst[mask_ins]
            #     label_pl = label_pl[bs_index_nor]
            #     label_pl = label_pl.view(-1)
            #     o_inst = o_inst[bs_index_nor]
            #     o_inst = o_inst.view(-1,cls)
        #else:

        #计算loss
            label_pl = label_pl.view(-1)
            o_inst = o_inst.view(-1,cls)
            if o_inst.size(0)>0:
                loss_pl = criterion_teacher(o_inst,label_pl)
            else:
                loss_pl = 0
        else:
            loss_pl = 0

        classify_loss = criterion(output, targets)
        loss = classify_loss + loss_pl

        metrics_values = OrderedDict([
        ('classify_loss_meter',(classify_loss,targets.size(0))),
        
        ('loss_teacher_meter', (torch.tensor(loss_pl),targets.size(0))),
        ('dis_ins_meter',(torch.tensor(dis_ins / (targets.size(0) * p)),targets.size(0))),
        ('patch_num_meter',(torch.tensor(len(label_pl) / (p*b)),b)),
        ('cluster_num_meter', (torch.tensor(sum(cluster_num) / b),b)),
        ('cluster_ema_num_meter',(torch.tensor(sum(cluster_num_ema)),b)),
        ('dis_rec_meters',torch.tensor(dis_rec)),
        ('selec_rec_meters',torch.tensor(selec_rec)),
        ])

        return loss,metrics_values,OrderedDict([('dis_ratio_list',self.dis_ratio_list)])

    def update_per_iter(self,config,epoch,idx,**kwargs):
        teacher_ema = kwargs['teacher_ema']
        model = kwargs['model']

        if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup' or config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                #teacher_ema.decay_diff = (epoch * num_steps + idx) / (config.TRAIN.EPOCHS * num_steps)
                teacher_ema.decay_diff = 0
        if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
            teacher_ema.decay_diff = config.RDD_TRANS.EMA_DECAY if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH else teacher_ema.decay_diff
        if epoch > 1:
            # 将前面特征层与后面实例分类层分开更新，分类层更新率更高
            #teacher_ema.decay_diff = 0.9997
            # 前面10个epoch不更新特征层
            #teacher_ema.decay = 1 if epoch < 10 else config.RDD_TRANS.EMA_DECAY
            teacher_ema.update(model)

    def update_per_epoch(self,config,epoch,**kwargs):
        if config.RDD_TRANS.PERSUDO_LEARNING and epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
            for i in range(len(self.thr_list)):
                dis_ratio_tmp= np.sort(np.array(self.dis_ratio_list[i]))
                self.thr_list[i] = 0.75 * self.thr_list[i] + 0.25 * dis_ratio_tmp[math.floor(len(dis_ratio_tmp) * 0.01)]
        
        # 每一轮更新内部参数
        self.dis_ratio_list=[[] for i in range(len(self.thr_list))]
    def measure_per_iter(self,output,targets,**kwargs):
        topk = (1,5)
        cluster_num = kwargs['cluster_num']

        # topk acc cls
        acc1,acc5 = accuracy(output, targets, topk=topk)

        metrics_values = OrderedDict([
        ('acc1_meter',(acc1,targets.size(0))),
        ('acc5_meter',(acc5,targets.size(0))),
        ('cluster_num_meter',(torch.tensor(sum(cluster_num) / targets.size(0)),targets.size(0))),
        ])
        others = OrderedDict([
        ('Null',None)
        ])
        return metrics_values,others
    def measure_per_epoch(self,config,**kwargs):
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

        metrics_values = OrderedDict([
        ('auc',auc),
        ('macro_f1',ma_f1),
        ('micro_f1',mi_f1)
        ])
        others = OrderedDict([
        ('Null',None)
        ])

        return metrics_values,others

def train_one_epoch(config,model, criterion, data_loader, optimizer, epoch, mixup_fn=None, lr_scheduler=None,amp_autocast=suppress,loss_scaler=None,model_ema=None,teacher_ema=None, thr_list=[],logger=None):
    model.train()
    torch.cuda.empty_cache()
    loss_teacher = None

    loss_teacher = torch.nn.CrossEntropyLoss()
    loss_teacher.cuda()
   
    optimizer.zero_grad()
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_teacher_meter = AverageMeter()
    dis_ins_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    cluster_num_meter = AverageMeter()
    cluster_ema_num_meter = AverageMeter()
    #acc1_meter = AverageMeter()
    loss_rec = np.array([])
    dis_ratio_list = [[] for i in range(len(thr_list))]
    dis_rec = np.array([AverageMeter() for i in range(len(thr_list))])
    selec_rec = np.array([AverageMeter() for i in range(len(thr_list))])

    start = time.time()
    end = time.time()
    last_idx = len(data_loader) - 1
    persudo_inst = config.RDD_TRANS.PERSUDO_LEARNING

    for idx, (samples, targets) in enumerate(data_loader):
        last_batch = idx == last_idx
        
        # timm dataloader prefetcher will do this
        if not config.DATA.TIMM or not config.DATA.TIMM_PREFETCHER:
            samples = samples.cuda(non_blocking=config.DATA.PIN_MEMORY)
            targets = targets.cuda(non_blocking=config.DATA.PIN_MEMORY)

        # 当伪标签为二分类时或二分类训练时，需要二分类标签进行loss计算
        if config.BINARYTRAIN_MODE or config.RDD_TRANS.INST_NUM_CLASS:
            targets_bin = targets.clone()
            targets_bin[targets==config.DATA.NOR_CLS_INDEX] = 0
            targets_bin[targets!=config.DATA.NOR_CLS_INDEX] = 1
        
        # timm dataloader prefetcher will do this
        if mixup_fn is not None and not config.DATA.TIMM:
            samples, targets = mixup_fn(samples, targets)

        p = 1
        b = targets.size(0)
        cluster_num_ema=[0]
        with amp_autocast():
            output,o_inst,_,cluster_num = model(samples)
            torch.cuda.empty_cache()
            # 设定正常图片在类别中的索引
            pl_nor_cls_index = 0 if config.RDD_TRANS.INST_NUM_CLASS == 2 else config.DATA.NOR_CLS_INDEX
            
                #_,pl_inst = teacher_ema.module(samples)
            #output_pl = torch.nn.functional.softmax(pl_inst,dim=2)
            if config.RDD_TRANS.INST_NUM_CLASS != 2:
                targets_pl = targets
            else:
                targets_pl = targets_bin
            if persudo_inst: 
                with torch.no_grad():
                    _,pl_inst,output_pl,cluster_num_ema = model_ema.module(samples)
                    torch.cuda.empty_cache()
                b,p,cls = pl_inst.shape
                t_cpu = targets_pl.cpu()
                ins_t = targets_pl.unsqueeze(-1).repeat((1,p))

                dis_ins = 0
                output_bag_label = output_pl[torch.functional.F.one_hot(ins_t,num_classes=config.RDD_TRANS.INST_NUM_CLASS) == 1].view(b,p)        #包所属标签下的置信度
                if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                    out_tmp_sort,_ = torch.sort(output_bag_label,dim=-1,descending=True)         # [b p]
                    min_nor_thr = out_tmp_sort[[i for i in range(b)],np.floor(p*thr_list[t_cpu])] # [b 1]
                    min_nor_thr = min_nor_thr.unsqueeze(-1).repeat((1,p))        # [b p]
                else:
                    min_nor_thr = torch.ones(size=(b,p))
                min_nor_thr = min_nor_thr.cuda(non_blocking=True)

                _,label_pl = torch.max(output_pl,dim=2)
                #label_pl = torch.argmax(output_pl,dim=2)
                #label_tmp = label_pl.clone()
                #获得正常包索引和病害包索引
                #bs_index_nor = targets_pl==pl_nor_cls_index
                #bs_index_dis = bs_index_nor==False
                ps_mask_nor = (ins_t - pl_nor_cls_index == 0)
                ps_mask_dis = ps_mask_nor==False

                #将所有实例设为正常
                label_pl[:,:] = pl_nor_cls_index 
                # sigmod函数来保证前期增加速率远远高于后期，优于线性增长
                thr_min_conf = get_sigmod_num(0.9,(epoch-config.RDD_TRANS.INIT_STAGE_EPOCH) * num_steps + idx,(config.TRAIN.EPOCHS * num_steps))
                thr_min_dis_conf = get_sigmod_num(0.5,(epoch-config.RDD_TRANS.INIT_STAGE_EPOCH) * num_steps + idx,(config.TRAIN.EPOCHS * num_steps))
                #把网络判断为不是正常的部分实例置为包病害标签
                if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                    #判断为相应病害的实例置为包标签
                    #mask_ins =  (label_tmp - ins_t  == 0)
                    #将病害包里判断为不是正常的实例都置为包标签 & (output_bag_label > thr_min_conf)
                    mask_ins =   ps_mask_dis & (((output_pl[:,:,pl_nor_cls_index] < (1-thr_min_conf)) & (output_bag_label > thr_min_dis_conf) )  | (output_bag_label - min_nor_thr >  0))
                    label_pl[mask_ins] = ins_t[mask_ins]
                    #选取部分置信度比较高的实例参与loss计算   label_tmp - config.DATA.NOR_CLS_INDEX  == 0
                    mask_ins = (mask_ins | (ps_mask_dis & (label_pl==pl_nor_cls_index) & (output_pl[:,:,pl_nor_cls_index] - thr_min_dis_conf > 0))) | ((output_pl[:,:,pl_nor_cls_index] > thr_min_dis_conf) & ps_mask_nor)
                else:
                    #全部都用
                    #mask_ins = (label_tmp - label_tmp == 0)
                    #只要正常图片 (output_bag_label - 0.9 > 0)
                    mask_ins = ps_mask_nor 

                    #label_pl[bs_index_dis] = ins_t[bs_index_dis]

                #统计病害实例
                dis_count = torch.count_nonzero(label_pl - pl_nor_cls_index,dim=1) 
                dis_ins += torch.sum(dis_count)

                for i in range(len(thr_list)):
                    i_index = targets_pl==i
                    dis_tmp = dis_count[i_index]
                    selec_tmp = mask_ins[i_index]

                    if len(selec_tmp)>0:
                        selec_rec[i].update( len(selec_tmp[selec_tmp==True]) / (len(selec_tmp)*p),b)

                    if len(dis_tmp) > 0 and epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                        thr_tmp = torch.sum(dis_tmp) / (p * len(dis_tmp))
                        dis_ratio_list[i].append(thr_tmp.cpu())

                        #thr_list[i] = 0.9999 * thr_list[i] + (1-0.9999)* thr_tmp
                        #thr_list[i] = 0.1 if thr_list[i] < 0.1 else thr_list[i]
                        #thr_list[i] = thr_tmp
                        dis_rec[i].update(thr_tmp,b)

                #选择部分patch来计算loss
                #if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
                label_pl = label_pl[mask_ins]
                o_inst = o_inst[mask_ins]
                #     label_pl = label_pl[bs_index_nor]
                #     label_pl = label_pl.view(-1)
                #     o_inst = o_inst[bs_index_nor]
                #     o_inst = o_inst.view(-1,cls)
            #else:

            #计算loss
                
                label_pl = label_pl.view(-1)
                o_inst = o_inst.view(-1,cls)
                patch_num_meter.update(len(label_pl) / (p*b),b)
                if o_inst.size(0)>0:
                    loss_pl = loss_teacher(o_inst,label_pl)
                else:
                    loss_pl = 0

            cluster_num_meter.update(sum(cluster_num) / b,b)
            classify_loss = criterion(output, targets)
            loss = classify_loss

            del samples
        
        if not config.DISTRIBUTED:
            loss_meter.update(loss.item(), targets.size(0))

            if persudo_inst:
                dis_ins_meter.update(dis_ins / (targets.size(0) * p),targets.size(0))
                if loss_teacher is not None and o_inst.size(0)>0:
                    loss_teacher_meter.update(loss_pl.item(), targets.size(0))
                else:
                    loss_teacher_meter.update(0, targets.size(0))
            else:
                dis_ins_meter.update(0 / (targets.size(0) * p),targets.size(0))
                loss_teacher_meter.update(0, targets.size(0))
       # with torch.autograd.detect_anomaly():
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            #loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if loss_scaler is not None:
                loss_scaler(
                    loss, optimizer,
                    clip_grad=None if config.TRAIN.CLIP_GRAD == 0 else config.TRAIN.CLIP_GRAD, clip_mode=config.TRAIN.CLIP_MODE,
                    parameters=model_parameters(model, exclude_head='agc' in config.TRAIN.CLIP_MODE>0),
                    create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                if config.TRAIN.CLIP_GRAD > 0:
                    dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in config.TRAIN.CLIP_MODE>0),
                        value=config.TRAIN.CLIP_GRAD, mode=config.TRAIN.CLIP_MODE)
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                if config.TRAIN.LR_SCHEDULER.NAME=='flat_cosine':
                    lr_scheduler.step(epoch * num_steps + idx)
                else:
                    lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            #loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss, optimizer,
                    clip_grad=None if config.TRAIN.CLIP_GRAD == 0 else config.TRAIN.CLIP_GRAD, clip_mode=config.TRAIN.CLIP_MODE,
                    parameters=model_parameters(model, exclude_head='agc' in config.TRAIN.CLIP_MODE>0),
                    create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                if config.TRAIN.CLIP_GRAD > 0:
                    dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in config.TRAIN.CLIP_MODE),
                        value=config.TRAIN.CLIP_GRAD, mode=config.TRAIN.CLIP_MODE)
                optimizer.step()

            if model_ema is not None:

                cluster_ema_num_meter.update(sum(cluster_num_ema),b)

                if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup' or config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                    #teacher_ema.decay_diff = (epoch * num_steps + idx) / (config.TRAIN.EPOCHS * num_steps)
                    model_ema.decay = 0
                if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                    model_ema.decay = config.RDD_TRANS.EMA_DECAY if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH else model_ema.decay
                model_ema.update(model)
            if config.TRAIN.LR_SCHEDULER.NAME is not None:
                if config.TRAIN.LR_SCHEDULER.NAME=='flat_cosine':
                    lr_scheduler.step(epoch * num_steps + idx)
                else:
                    lr_scheduler.step_update(epoch * num_steps + idx)
        if teacher_ema is not None:
            if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup' or config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                #teacher_ema.decay_diff = (epoch * num_steps + idx) / (config.TRAIN.EPOCHS * num_steps)
                teacher_ema.decay_diff = 0
            if config.RDD_TRANS.EMA_DECAY_SCHEDULER == 'warmup_flat':
                teacher_ema.decay_diff = config.RDD_TRANS.EMA_DECAY if epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH else teacher_ema.decay_diff
            if epoch > 1:
                # 将前面特征层与后面实例分类层分开更新，分类层更新率更高
                #teacher_ema.decay_diff = 0.9997
                # 前面10个epoch不更新特征层
                #teacher_ema.decay = 1 if epoch < 10 else config.RDD_TRANS.EMA_DECAY

                teacher_ema.update(model)
        #acc1, _ = accuracy(predictions, targets, topk=(1,1))
        
        torch.cuda.synchronize()
        #acc1_meter.update(acc1.item(), targets.size(0))
        batch_time.update(time.time() - end)
        np.append(loss_rec,loss_meter.avg)
        end = time.time()

        if last_batch or idx % config.PRINT_FREQ == 0:
            if config.DISTRIBUTED:
                reduced_loss = reduce_tensor(loss.data)
                loss_meter.update(reduced_loss.item(), input.size(0))
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss_tea {loss_teacher_meter.val:.4f} ({loss_teacher_meter.avg:.4f})\t'
                f'dis_ins {dis_ins_meter.val:.4f} ({dis_ins_meter.avg:.4f}) \t'
                f'patch_num {patch_num_meter.val:.4f} ({patch_num_meter.avg:.4f}) \t'
                f'cluster_num {cluster_num_meter.val:.4f} ({cluster_num_meter.avg:.4f}) \t'
                f'cluster_ema_num {cluster_ema_num_meter.val:.4f} ({cluster_ema_num_meter.avg:.4f}) \t'
                #f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'mem {memory_used:.0f}MB')
    if persudo_inst and epoch >= config.RDD_TRANS.INIT_STAGE_EPOCH:
        for i in range(len(thr_list)):
            dis_ratio_tmp= np.sort(np.array(dis_ratio_list[i]))
            thr_list[i] = 0.75 * thr_list[i] + 0.25 * dis_ratio_tmp[math.floor(len(dis_ratio_tmp) * 0.01)]

    epoch_time = time.time() - start
    for i in range(len(dis_rec)):
        logger.info(f"Dis:  {dis_rec[i].avg:.2f}")
    for i in range(len(selec_rec)):

        logger.info(f"Selected:  {selec_rec[i].avg:.2f}")
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    torch.cuda.empty_cache()
    return loss,OrderedDict([('loss', loss_meter.avg)])

def predict(config, data_loader, model,amp_autocast=suppress,logger=None):
    model.eval()
    torch.cuda.empty_cache()

    batch_time = AverageMeter()
    
    save_pred = np.array([])
    save_label = np.array([])

    end = time.time()
    last_idx = len(data_loader) - 1
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            last_batch = idx == last_idx
            # timm dataloader prefetcher will do this
            if not config.DATA.TIMM or not config.DATA.TIMM_PREFETCHER:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            #if config.EVAL_MODE:
            targets_bin = targets.clone()

            if 'cqu_bpdd' in config.DATA.DATASET:
                targets_bin[targets==config.DATA.NOR_CLS_INDEX] = 0
                targets_bin[targets!=config.DATA.NOR_CLS_INDEX] = 1
            # compute output
            with amp_autocast():
                output = model(images)
            if isinstance(output, (tuple, list)):
                index = 0 if config.RDD_TRANS.NOT_INST_TEST else 1

                output = output[index]

            output_soft = torch.nn.functional.softmax(output,dim=-1)
    
            save_pred = np.append(save_pred,output_soft.cpu().numpy())
            save_label = np.append(save_label,targets.cpu().numpy())
            
            torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if last_batch or idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
            
        save_pred = save_pred.reshape(-1,config.MODEL.NUM_CLASSES)

    torch.cuda.empty_cache()

    return save_pred,save_label

def validate(config, data_loader, model,save_pre=False,amp_autocast=suppress, logger=None,criterion=None):
    model.eval()
    torch.cuda.empty_cache()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    cluster_num_meter = AverageMeter()
    
    save_pred = np.array([])
    save_label = np.array([])

    end = time.time()
    last_idx = len(data_loader) - 1
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            last_batch = idx == last_idx
            # timm dataloader prefetcher will do this
            if not config.DATA.TIMM or not config.DATA.TIMM_PREFETCHER:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            topk = (1,5)
            #if config.EVAL_MODE:
            targets_bin = targets.clone()
            if config.BINARYTRAIN_MODE:
                topk = (1,1)
            m = 0
            if 'cqu_bpdd' in config.DATA.DATASET:
                targets_bin[targets==config.DATA.NOR_CLS_INDEX] = 0
                targets_bin[targets!=config.DATA.NOR_CLS_INDEX] = 1

            # compute output
            with amp_autocast():
                output = model(images)
            if isinstance(output, (tuple, list)):
                index = 0 if config.RDD_TRANS.NOT_INST_TEST else 1
                cluster_num = output[-1]
                output = output[index]

            output_soft = torch.nn.functional.softmax(output,dim=-1)

            # if not config.THUMB_MODE:
            #     #使用max-pool来测试，取所有图块中得分最大的图块的置信度
            #     b,p,cls = output.shape
            #     max_score,max_index_cls = torch.max(output_soft,dim=-1)

            #     mask = (max_score - config.RDD_TRANS.TEST_THR > 0) & (max_index_cls - config.DATA.NOR_CLS_INDEX != 0)
            #     if mask.any() == True:
            #         max_index = []
            #         for i in range(b):
            #             if mask[i].any()==True:
            #                 score_tmp = max_score[i,mask[i]]
            #                 _,max_inx_tmp = torch.max(score_tmp,dim=-1)
            #                 max_index.append(torch.nonzero(mask[i])[max_inx_tmp,0].tolist())
            #             else:
            #                 _,index = torch.max(max_score[i],dim=-1)
            #                 max_index.append(index)
            #     else:
            #         max_score,max_index = torch.max(max_score,dim=-1)

            #     output_soft = output_soft[[i for i in range(b)],max_index,:]
            #     output = output[[i for i in range(b)],max_index,:]
            #     del max_score,mask

            if config.BINARYTRAIN_MODE:
                loss = criterion(output, targets_bin)
            else:
                loss = criterion(output, targets)
                
            save_pred = np.append(save_pred,output_soft.cpu().numpy())
            save_label = np.append(save_label,targets.cpu().numpy())
            
            
            # if config.BINARYTRAIN_MODE:
            #     acc1, acc5 = accuracy(output, target_bin, topk=topk)
            #     output = torch.nn.functional.softmax(output,dim=1)
            #     preds = output[:,1]
            # else:
            #     acc1, acc5 = accuracy(output, target, topk=topk)
            #     output = torch.nn.functional.softmax(output,dim=1)
            #     preds = 1-output[:,config.DATA.NOR_CLS_INDEX]
            
            # phase_label = np.append(phase_label, target_bin.data.cpu().numpy())
            # phase_pred = np.append(phase_pred, preds.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, targets, topk=topk)
            if config.DISTRIBUTED:
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
                loss = reduce_tensor(loss)

            torch.cuda.synchronize()

            loss_meter.update(loss.item(), targets.size(0))
            acc1_meter.update(acc1.item(), targets.size(0))
            acc5_meter.update(acc5.item(), targets.size(0))
            if not config.THUMB_MODE:
                cluster_num_meter.update(sum(cluster_num) / targets.size(0),targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if last_batch or idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Cluster_num {cluster_num_meter.val:.3f} ({cluster_num_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
            
        save_pred = save_pred.reshape(-1,config.MODEL.NUM_CLASSES)
        auc = 0
        if config.BINARYTRAIN_MODE:
            ma_f1 = f1_score(np.array(save_label!=config.DATA.NOR_CLS_INDEX,dtype=int),np.argmax(save_pred,axis=1),average='binary')
            mi_f1 = ma_f1
            try:
                auc = roc_auc_score(np.array(save_label!=config.DATA.NOR_CLS_INDEX,dtype=int), save_pred[:,1])
            except:
                print(save_pred)
        else:
            ma_f1 = f1_score(save_label,np.argmax(save_pred,axis=1),average='macro')
            mi_f1 = f1_score(save_label,np.argmax(save_pred,axis=1),average='micro')
            try:
                auc = roc_auc_score(np.array(save_label!=config.DATA.NOR_CLS_INDEX,dtype=int), 1-save_pred[:,config.DATA.NOR_CLS_INDEX])
            except:
                print(save_pred)
        
        #if config.BINARYTRAIN_MODE:
        
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} AUC {auc*100:.3f} F1@Macro {ma_f1*100:.3f} F1@Micro {mi_f1*100:.3f}')
    metrics = OrderedDict([('loss', loss_meter.avg), ('top1', acc1_meter.avg), ('top5', acc5_meter.avg),('auc',auc),('macro_f1',ma_f1),('micro_f1',mi_f1)])
    torch.cuda.empty_cache()
    if save_pre:
        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg, auc,save_pred,save_label,metrics
    else:    
        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg, auc,metrics