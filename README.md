# DINO-Style Student-Teacher Framework

A PyTorch implementation of a DINO-style knowledge distillation framework with a frozen teacher and trainable student.

## Key Features

- **Frozen Teacher**: ViT-Large/16 initialized with DINOv3 pretrained weights (no EMA updates)
- **Trainable Student**: ViT-Small/16 learning from frozen teacher
- **5 Loss Functions**:
  1. DINO Loss - Self-distillation loss
  2. iBOT Loss - Masked image modeling
  3. KoLeo Loss - Feature uniformity regularization
  4. CLS Token Loss - Specific CLS token distillation
  5. SPD Matrix Loss - Log-Euclidean distance between SPD matrices (replaces Gram matrix)

## Project Structure

```
.
├── models/
│   ├── vision_transformer.py    # ViT architectures (Small, Base, Large)
│   ├── dino_head.py             # DINO and iBOT projection heads
│   └── student_teacher.py       # Student-Teacher wrapper model
├── training/
│   ├── losses.py                # All loss implementations
│   └── train.py                 # Training loop and scheduler
├── data/
│   ├── augmentation.py          # Multi-crop augmentation and masking
│   └── dataset.py               # Dataset and dataloader
├── evaluation/
│   └── linear_probe.py          # Linear evaluation protocol
├── utils/
│   ├── distributed.py           # Distributed training utilities
│   ├── logger.py                # Logging utilities
│   └── checkpoint.py            # Checkpoint save/load
├── configs/
│   └── train_config.py          # Training configuration
└── main.py                      # Main training script
```

## Installation

```bash
pip install torch torchvision
```

## Usage

### Training

```bash
# Single GPU
python main.py \
  --data_path /path/to/imagenet/train \
  --teacher_pretrained_weights /path/to/dinov3_large.pth \
  --output_dir ./outputs \
  --epochs 100 \
  --batch_size_per_gpu 64

# Multi-GPU (using torch.distributed.launch)
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  main.py \
  --data_path /path/to/imagenet/train \
  --teacher_pretrained_weights /path/to/dinov3_large.pth \
  --output_dir ./outputs \
  --epochs 100 \
  --batch_size_per_gpu 64
```

### Key Arguments

**Model:**
- `--arch`: Student architecture (default: `vit_small`)
- `--patch_size`: Patch size (default: `16`)
- `--teacher_pretrained_weights`: Path to DINOv3 Large pretrained weights

**Loss Weights:**
- `--dino_loss_weight`: DINO loss weight (default: `1.0`)
- `--ibot_loss_weight`: iBOT loss weight (default: `1.0`)
- `--koleo_loss_weight`: KoLeo loss weight (default: `0.1`)
- `--spd_loss_weight`: SPD matrix loss weight (default: `1.0`)
- `--cls_loss_weight`: CLS token loss weight (default: `1.0`)

**Data Augmentation:**
- `--global_crops_scale`: Scale range for global crops (default: `(0.4, 1.0)`)
- `--local_crops_number`: Number of local crops (default: `8`)
- `--global_crops_size`: Size of global crops (default: `224`)
- `--local_crops_size`: Size of local crops (default: `96`)

**iBOT Masking:**
- `--mask_patch_size`: Mask patch size (default: `32`)
- `--mask_ratio`: Masking ratio (default: `0.4`)

**Training:**
- `--epochs`: Number of epochs (default: `100`)
- `--lr`: Learning rate (default: `0.0005`)
- `--batch_size_per_gpu`: Batch size per GPU (default: `64`)

### Evaluation

Linear probing evaluation:

```bash
python evaluation/linear_probe.py \
  --checkpoint_path ./outputs/checkpoint_0099.pth \
  --train_data_path /path/to/imagenet/train \
  --val_data_path /path/to/imagenet/val \
  --batch_size 256 \
  --epochs 100
```

## Architecture Details

### Student Network
- **Backbone**: ViT-Small/16 (384 dim, 12 layers, 6 heads)
- **Trainable**: Yes
- **Learns from**: Frozen teacher via multiple loss objectives

### Teacher Network
- **Backbone**: ViT-Large/16 (1024 dim, 24 layers, 16 heads)
- **Trainable**: No (completely frozen)
- **Initialization**: DINOv3 pretrained weights
- **No EMA**: Teacher weights remain static throughout training

## Loss Functions

### 1. DINO Loss
Standard DINO self-distillation with centering and sharpening:
- Student predictions on all crops (global + local)
- Teacher predictions on global crops only
- Cross-entropy with temperature scheduling

### 2. iBOT Loss
Masked image modeling with patch-level distillation:
- ~40% of patches masked with block-wise strategy
- Student predicts masked patches
- Teacher provides targets from unmasked image

### 3. KoLeo Loss
Kozachenko-Leonenko entropy regularization:
- Encourages uniform distribution of features
- Prevents feature collapse
- Applied to student features

### 4. CLS Token Loss
Direct distillation on [CLS] token embeddings:
- Cosine similarity between student and teacher CLS tokens
- Focuses on global image representation

### 5. SPD Matrix Loss
Log-Euclidean distance between SPD matrices:
- Computes covariance-like SPD matrices from features
- Uses Log-Euclidean metric on SPD manifold
- Replaces Gram matrix anchoring from DINOv3

## Multi-Crop Strategy

Following DINOv3:
- **2 Global crops**: 224×224, scale (0.4, 1.0)
- **8 Local crops**: 96×96, scale (0.05, 0.4)
- Different augmentations per crop type
- Teacher only sees global crops
- Student sees all crops

## Training Tips

1. **Batch Size**: Total batch size = `batch_size_per_gpu × num_gpus`
   - Recommended: 512-1024 for best results
   
2. **Learning Rate**: Scaled with batch size
   - Base LR: 0.0005 for batch size 256
   - Actual LR = base_lr × (total_batch_size / 256)

3. **Temperature Schedule**: 
   - Warmup from 0.04 to 0.07 over 30 epochs
   - Student temp fixed at 0.1

4. **Memory**: 
   - ViT-S student: ~6GB per GPU (batch=64)
   - ViT-L teacher (frozen): ~10GB

5. **Convergence**: 
   - Expect improvements up to 100-300 epochs
   - Save checkpoints regularly for evaluation

## Citation

If you use this framework, please cite the relevant papers:

```bibtex
@article{caron2021dino,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J'egou, Herv'e and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  journal={ICCV},
  year={2021}
}

@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth'ee and Moutakanni, Th'eo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## License

This project is released under the MIT License.