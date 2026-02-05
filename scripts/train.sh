#!/bin/bash

# Training script for DINO-style Student-Teacher framework

# Configuration
NUM_GPUS=8
BATCH_SIZE_PER_GPU=64
EPOCHS=100
DATA_PATH="/path/to/imagenet/train"
TEACHER_WEIGHTS="/path/to/dinov3_large.pth"
OUTPUT_DIR="./outputs/experiment_001"

# Create output directory
mkdir -p $OUTPUT_DIR

# Launch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    main.py \
    --data_path $DATA_PATH \
    --teacher_pretrained_weights $TEACHER_WEIGHTS \
    --output_dir $OUTPUT_DIR \
    --arch vit_small \
    --patch_size 16 \
    --out_dim 65536 \
    --patch_out_dim 8192 \
    --epochs $EPOCHS \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --lr 0.0005 \
    --min_lr 1e-6 \
    --warmup_epochs 10 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --clip_grad 3.0 \
    --use_fp16 True \
    --optimizer adamw \
    --global_crops_scale 0.4 1.0 \
    --local_crops_scale 0.05 0.4 \
    --local_crops_number 8 \
    --global_crops_size 224 \
    --local_crops_size 96 \
    --mask_patch_size 32 \
    --mask_ratio 0.4 \
    --dino_loss_weight 1.0 \
    --ibot_loss_weight 1.0 \
    --koleo_loss_weight 0.1 \
    --spd_loss_weight 1.0 \
    --cls_loss_weight 1.0 \
    --warmup_teacher_temp 0.04 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 30 \
    --student_temp 0.1 \
    --center_momentum 0.9 \
    --num_workers 10 \
    --saveckp_freq 20 \
    --print_freq 10

echo "Training completed!"
