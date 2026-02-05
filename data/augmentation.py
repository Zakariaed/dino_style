import random
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps
import numpy as np

class GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.prob:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

class Solarization:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img

class DataAugmentationDINO:
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4),
                 local_crops_number=8, global_crops_size=224, local_crops_size=96):
        
        # Color jitter for global crops
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        # Normalization
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Global crop 1 (with more augmentation)
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=1.0),
            normalize,
        ])
        
        # Global crop 2 (with solarization)
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            normalize,
        ])
        
        # Local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class MaskGenerator:
    """Block-wise masking generator for iBOT"""
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=16, 
                 mask_ratio=0.4):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask

