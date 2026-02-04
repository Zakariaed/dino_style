import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.train_config import get_args_parser
from models.student_teacher import StudentTeacherModel
from data.dataset import build_dataset, build_dataloader
from training.train import Trainer, get_params_groups, cosine_scheduler
from utils.distributed import init_distributed_mode, is_main_process
from utils.checkpoint import save_checkpoint, load_checkpoint

def main(args):
    init_distributed_mode(args)
    
    # Set random seed
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    
    # Build dataset and dataloader
    print(f"Building dataset from {args.data_path}")
    dataset = build_dataset(args, is_train=True)
    data_loader, sampler = build_dataloader(dataset, args, is_train=True)
    print(f"Dataset contains {len(dataset):,} images")
    
    # Build model
    print("Building student-teacher model...")
    model = StudentTeacherModel(args)
    model = model.cuda()
    
    # Distributed training
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)
    
    # Get parameter groups (separate for weight decay)
    params_groups = get_params_groups(model)
    
    # Build optimizer
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate schedule
    lr_schedule = cosine_scheduler(
        base_value=args.lr * (args.batch_size_per_gpu * args.world_size) / 256.,
        final_value=args.min_lr,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    
    # Weight decay schedule
    wd_schedule = cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from epoch {start_epoch}")
    
    # Create trainer
    trainer = Trainer(model, optimizer, lr_schedule, wd_schedule, args)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        
        # Train one epoch
        train_stats = trainer.train_one_epoch(data_loader, epoch)
        
        # Save checkpoint
        if is_main_process():
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
            }
            
            # Save regular checkpoint
            if (epoch + 1) % args.saveckp_freq == 0 or epoch == args.epochs - 1:
                save_checkpoint(save_dict, os.path.join(args.output_dir, f'checkpoint_{epoch:04d}.pth'))
            
            # Always save last checkpoint
            save_checkpoint(save_dict, os.path.join(args.output_dir, 'checkpoint_last.pth'))
            
            # Log stats
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Add seed
    args.seed = 0
    
    main(args)