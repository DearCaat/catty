from ast import Or
from contextlib import suppress
from distutils.command.config import config
from math import isnan
import time
import numpy as np
import datetime
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score

import torch
from timm.utils import *
from timm.models import  model_parameters

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

        predictions = models['main'](samples)
        loss = criterions[0](predictions,targets)
        if isnan(loss.item()):
            print(samples)
            print(predictions)
            print(targets)
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
        ('acc1',(acc1,targets.size(0))),
        ('acc5',(acc5,targets.size(0))),
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


def train_one_epoch(config,model, criterion, data_loader, optimizer, epoch, mixup_fn=None, lr_scheduler=None,amp_autocast=suppress,loss_scaler=None,model_ema=None,logger=None):
    model.train()
    torch.cuda.empty_cache()

    optimizer.zero_grad()
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    #acc1_meter = AverageMeter()
    loss_rec = np.array([])

    start = time.time()
    end = time.time()
    last_idx = len(data_loader) - 1

    for idx, (samples, targets) in enumerate(data_loader):
        last_batch = idx == last_idx
        
        # timm dataloader prefetcher will do this
        if not config.DATA.TIMM or not config.DATA.TIMM_PREFETCHER:
            samples = samples.cuda(non_blocking=config.DATA.PIN_MEMORY)
            targets = targets.cuda(non_blocking=config.DATA.PIN_MEMORY)

        # 当伪标签为二分类时或二分类训练时，需要二分类标签进行loss计算
        if config.BINARYTRAIN_MODE:
            targets_bin = targets.clone()
            targets_bin[targets==config.DATA.NOR_CLS_INDEX] = 0
            targets_bin[targets!=config.DATA.NOR_CLS_INDEX] = 1
        
        # timm dataloader prefetcher will do this
        if mixup_fn is not None and not config.DATA.TIMM:
            samples, targets = mixup_fn(samples, targets)

        with amp_autocast():
            predictions = model(samples)
            #print(predictions)
            del samples
            if config.BINARYTRAIN_MODE:
                classify_loss = criterion(predictions, targets_bin)
            else:
                classify_loss = criterion(predictions, targets)
            loss = classify_loss
            #custom model

        if not config.DISTRIBUTED:
            loss_meter.update(loss.item(), targets.size(0))

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
                model_ema.update(model)

            if config.TRAIN.LR_SCHEDULER.NAME is not None:
                if config.TRAIN.LR_SCHEDULER.NAME=='flat_cosine':
                    lr_scheduler.step(epoch * num_steps + idx)
                else:
                    lr_scheduler.step_update(epoch * num_steps + idx)

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
                #f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
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
                index = 0 if config.THUMB_MODE or config.RDD_TRANS.NOT_INST_TEST else 1

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

def validate(config, data_loader, model,save_pre=False,amp_autocast=suppress, log_suffix='',criterion=None,logger=None):
    model.eval()
    torch.cuda.empty_cache()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
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
                output = output[0]

            output_soft = torch.nn.functional.softmax(output,dim=-1)

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