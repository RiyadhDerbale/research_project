"""
Dataset classes for classification tasks
"""

import os
from typing import Optional, Callable, Tuple, List
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image


class ImageFolderClassification(Dataset):
    """
    Standard image folder dataset for classification
    
    Expected structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
    
    Args:
        root: Root directory path
        split: Dataset split ('train', 'val', or 'test')
        transform: Image transformations
        target_transform: Label transformations
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.root = Path(root) / split
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        if not self.root.exists():
            raise ValueError(f"Dataset path does not exist: {self.root}")
        
        # Use torchvision's ImageFolder internally
        self.dataset = datasets.ImageFolder(
            str(self.root),
            transform=transform,
            target_transform=target_transform
        )
        
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_class_weights(self):
        """Compute class weights for imbalanced datasets"""
        targets = [label for _, label in self.dataset.samples]
        class_counts = torch.bincount(torch.tensor(targets))
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum()  # Normalize
        return weights


class DummyClassificationDataset(Dataset):
    """
    Dummy dataset for testing (generates random data)
    
    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        image_size: Image size (H, W)
        num_channels: Number of channels (default: 3)
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 10,
        image_size: Tuple[int, int] = (32, 32),
        num_channels: int = 3,
        transform: Optional[Callable] = None
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_channels = num_channels
        self.transform = transform
        
        self.classes = [f"class_{i}" for i in range(num_classes)]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image
        image = torch.randn(self.num_channels, *self.image_size)
        
        # Generate random label
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        if self.transform:
            # Convert to PIL for transforms
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        
        return image, label


def get_classification_transforms(
    split: str,
    image_size: int = 224,
    normalize: bool = True
) -> transforms.Compose:
    """
    Get standard transforms for classification
    
    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        transform: Composed transforms
    """
    if split == "train":
        transform_list = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]
    else:
        transform_list = [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return transforms.Compose(transform_list)


# TODO: Add support for CSV-based datasets with image paths
# TODO: Add support for multi-label classification
# TODO: Add few-shot learning dataset wrapper
# TODO: Add data augmentation policies (AutoAugment, RandAugment)
