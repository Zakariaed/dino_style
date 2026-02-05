import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('DINO-style Student-Teacher Training', add_help=False)
    
    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str, help='Student architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch size for ViT')
    parser.add_argument('--out_dim', default=65536, type=int, help='Output dimension of DINO head')
    parser.add_argument('--patch_out_dim', default=8192, type=int, help='Output dimension for patch tokens')
    parser.add_argument('--norm_last_layer', default=True, type=bool, help='Normalize last layer')
    parser.add_argument('--use_bn_in_head', default=False, type=bool, help='Use BN in head')
    parser.add_argument('--shared_head_teacher', default=True, type=bool, help='Share head for iBOT')
    
    # Teacher parameters
    parser.add_argument('--teacher_pretrained_weights', default='', type=str, help='Path to pretrained teacher weights')
    
    # Temperature parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help='Initial teacher temperature')
    parser.add_argument('--teacher_temp', default=0.07, type=float, help='Final teacher temperature')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Warmup epochs for teacher temp')
    parser.add_argument('--student_temp', default=0.1, type=float, help='Student temperature')
    
    # Loss weights
    parser.add_argument('--dino_loss_weight', default=1.0, type=float, help='Weight for DINO loss')
    parser.add_argument('--ibot_loss_weight', default=1.0, type=float, help='Weight for iBOT loss')
    parser.add_argument('--koleo_loss_weight', default=0.1, type=float, help='Weight for KoLeo loss')
    parser.add_argument('--spd_loss_weight', default=1.0, type=float, help='Weight for SPD matrix loss')
    parser.add_argument('--cls_loss_weight', default=1.0, type=float, help='Weight for CLS token loss')
    
    # Center momentum
    parser.add_argument('--center_momentum', default=0.9, type=float, help='Center momentum for DINO loss')
    
    # Augmentation parameters
    parser.add_argument('--global_crops_scale', default=(0.4, 1.0), type=tuple, help='Scale for global crops')
    parser.add_argument('--local_crops_scale', default=(0.05, 0.4), type=tuple, help='Scale for local crops')
    parser.add_argument('--local_crops_number', default=8, type=int, help='Number of local crops')
    parser.add_argument('--global_crops_size', default=224, type=int, help='Size of global crops')
    parser.add_argument('--local_crops_size', default=96, type=int, help='Size of local crops')
    
    # Masking parameters (iBOT)
    parser.add_argument('--mask_patch_size', default=32, type=int, help='Mask patch size for iBOT')
    parser.add_argument('--mask_ratio', default=0.4, type=float, help='Mask ratio for iBOT')
    
    # Training parameters
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Warmup epochs')
    parser.add_argument('--weight_decay', default=0.04, type=float, help='Initial weight decay')
    parser.add_argument('--weight_decay_end', default=0.4, type=float, help='Final weight decay')
    parser.add_argument('--clip_grad', default=3.0, type=float, help='Gradient clipping')
    parser.add_argument('--use_fp16', default=True, type=bool, help='Use mixed precision training')
    
    # Optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, help='Optimizer')
    
    # Data
    parser.add_argument('--data_path', default='/path/to/imagenet/train', type=str, help='Path to dataset')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers')
    
    # Distributed
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank for distributed training')
    parser.add_argument('--dist_url', default='env://', type=str, help='URL for distributed training')
    
    # Checkpointing
    parser.add_argument('--output_dir', default='./outputs', type=str, help='Output directory')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint frequency')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--print_freq', default=10, type=int, help='Print frequency')
    
    return parser
