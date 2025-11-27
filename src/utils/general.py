"""
General utilities
"""

import torch
import numpy as np
import random
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device
    
    Args:
        device: Device string ('cuda', 'cpu', or None for auto)
        
    Returns:
        device: PyTorch device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model(model: torch.nn.Module):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: torch.nn.Module):
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs
):
    """
    Save model checkpoint
    
    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optional optimizer to save state
        **kwargs: Additional items to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        **kwargs
    }
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)


# TODO: Add gradient flow visualization
# TODO: Add model pruning utilities
# TODO: Add ONNX export utilities
