import torch
import os

def save_checkpoint(state, filename):
    """Save checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint and return epoch"""
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Get epoch
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch