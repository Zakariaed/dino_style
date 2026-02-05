import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import time
import datetime
import math

from training.losses import DINOLoss, iBOTLoss, KoLeoLoss, SPDMatrixLoss, CLSTokenLoss
from utils.logger import MetricLogger, SmoothedValue
from utils.distributed import is_main_process

class Trainer:
    def __init__(self, model, optimizer, lr_schedule, wd_schedule, args):
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule
        self.args = args
        
        # Initialize losses
        self.dino_loss = DINOLoss(
            out_dim=args.out_dim,
            ncrops=args.local_crops_number + 2,
            warmup_teacher_temp=args.warmup_teacher_temp,
            teacher_temp=args.teacher_temp,
            warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
            nepochs=args.epochs,
            student_temp=args.student_temp,
            center_momentum=args.center_momentum,
        ).cuda()
        
        self.ibot_loss = iBOTLoss(
            out_dim=args.out_dim,
            patch_out_dim=args.patch_out_dim,
            ncrops=args.local_crops_number + 2,
            warmup_teacher_temp=args.warmup_teacher_temp,
            teacher_temp=args.teacher_temp,
            warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
            nepochs=args.epochs,
            student_temp=args.student_temp,
            center_momentum=args.center_momentum,
        ).cuda()
        
        self.koleo_loss = KoLeoLoss().cuda()
        self.spd_loss = SPDMatrixLoss().cuda()
        self.cls_loss = CLSTokenLoss(temperature=args.student_temp).cuda()
        
        # Loss weights
        self.loss_weights = {
            'dino': args.dino_loss_weight,
            'ibot': args.ibot_loss_weight,
            'koleo': args.koleo_loss_weight,
            'spd': args.spd_loss_weight,
            'cls': args.cls_loss_weight,
        }
        
        # Mixed precision
        self.fp16_scaler = GradScaler() if args.use_fp16 else None
        
        # Logging
        self.metric_logger = MetricLogger(delimiter="  ")
        
    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        self.dino_loss.train()
        self.ibot_loss.train()
        
        header = f'Epoch: [{epoch}/{self.args.epochs}]'
        
        for it, (images, masks, _) in enumerate(self.metric_logger.log_every(data_loader, 10, header)):
            # Update learning rate and weight decay
            it_global = len(data_loader) * epoch + it
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.lr_schedule[it_global]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_schedule[it_global]
            
            # Move to GPU
            images = [im.cuda(non_blocking=True) for im in images]
            if masks is not None:
                masks = masks.cuda(non_blocking=True)
            
            # Forward pass
            with autocast(enabled=(self.fp16_scaler is not None)):
                student_output, teacher_output, student_ibot_output, teacher_ibot_output, student_feat, teacher_feat = \
                    self.model(images, masks, return_features=True)
                
                # Compute losses
                loss_dict = {}
                
                # 1. DINO Loss
                loss_dict['dino'] = self.dino_loss(student_output, teacher_output, epoch)
                
                # 2. iBOT Loss
                if masks is not None:
                    # Flatten masks for all global crops
                    masks_flat = masks.reshape(-1, masks.shape[-2] * masks.shape[-1])
                    loss_dict['ibot'] = self.ibot_loss(student_ibot_output, teacher_ibot_output, masks_flat, epoch)
                else:
                    loss_dict['ibot'] = torch.tensor(0.).cuda()
                
                # 3. KoLeo Loss (on student features)
                loss_dict['koleo'] = self.koleo_loss(student_feat)
                
                # 4. CLS Token Loss
                student_cls = student_feat[:len(teacher_feat)]  # Match teacher batch size
                loss_dict['cls'] = self.cls_loss(student_cls, teacher_feat)
                
                # 5. SPD Matrix Loss
                loss_dict['spd'] = self.spd_loss(student_feat[:len(teacher_feat)], teacher_feat)
                
                # Total weighted loss
                total_loss = sum(self.loss_weights[k] * v for k, v in loss_dict.items())
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.fp16_scaler is not None:
                self.fp16_scaler.scale(total_loss).backward()
                if self.args.clip_grad:
                    self.fp16_scaler.unscale_(self.optimizer)
                    param_norms = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()
            else:
                total_loss.backward()
                if self.args.clip_grad:
                    param_norms = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.optimizer.step()
            
            # Logging
            torch.cuda.synchronize()
            self.metric_logger.update(loss=total_loss.item())
            for k, v in loss_dict.items():
                self.metric_logger.update(**{f'loss_{k}': v.item()})
            self.metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            self.metric_logger.update(wd=self.optimizer.param_groups[0]["weight_decay"])
        
        # Gather stats from all processes
        self.metric_logger.synchronize_between_processes()
        print(f"Averaged stats: {self.metric_logger}")
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}

def get_params_groups(model):
    """Separate parameters for weight decay"""
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay on bias and LayerNorm
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """Cosine learning rate schedule with warmup"""
    warmup_schedule = torch.linspace(start_warmup_value, base_value, warmup_epochs * niter_per_ep)
    iters = torch.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + torch.cos(math.pi * iters / len(iters)))
    schedule = torch.cat((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
