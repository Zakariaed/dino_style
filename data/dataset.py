import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np

class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None, mask_generator=None):
        """
        Args:
            root: Path to ImageNet dataset
            transform: DataAugmentationDINO instance
            mask_generator: MaskGenerator instance for iBOT
        """
        self.root = root
        self.transform = transform
        self.mask_generator = mask_generator
        
        # Load ImageNet
        self.dataset = datasets.ImageFolder(root)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        
        # Apply multi-crop augmentation
        if self.transform is not None:
            crops = self.transform(image)
        else:
            crops = [image]
        
        # Generate mask for iBOT (only for global crops)
        masks = []
        if self.mask_generator is not None:
            for _ in range(2):  # 2 global crops
                masks.append(self.mask_generator())
        
        return crops, masks, target

def collate_fn(batch):
    """Custom collate function for multi-crop data"""
    crops_batch = [[] for _ in range(len(batch[0][0]))]
    masks_batch = []
    targets_batch = []
    
    for crops, masks, target in batch:
        for i, crop in enumerate(crops):
            crops_batch[i].append(crop)
        if len(masks) > 0:
            masks_batch.append(masks)
        targets_batch.append(target)
    
    # Stack crops
    crops_batch = [torch.stack(crops) for crops in crops_batch]
    
    # Stack masks
    if len(masks_batch) > 0:
        # Convert to tensor (batch_size, n_global_crops, H, W)
        masks_batch = torch.from_numpy(np.stack([np.stack(m) for m in masks_batch]))
    else:
        masks_batch = None
    
    targets_batch = torch.tensor(targets_batch)
    
    return crops_batch, masks_batch, targets_batch

def build_dataset(args, is_train=True):
    """Build dataset with augmentations"""
    from data.augmentation import DataAugmentationDINO, MaskGenerator
    
    if is_train:
        transform = DataAugmentationDINO(
            global_crops_scale=args.global_crops_scale,
            local_crops_scale=args.local_crops_scale,
            local_crops_number=args.local_crops_number,
            global_crops_size=args.global_crops_size,
            local_crops_size=args.local_crops_size,
        )
        mask_generator = MaskGenerator(
            input_size=args.global_crops_size,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.patch_size,
            mask_ratio=args.mask_ratio,
        )
    else:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        mask_generator = None
    
    dataset = ImageNetDataset(
        root=args.data_path,
        transform=transform,
        mask_generator=mask_generator,
    )
    
    return dataset

def build_dataloader(dataset, args, is_train=True):
    """Build dataloader with distributed sampler"""
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=is_train)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    return loader, sampler
