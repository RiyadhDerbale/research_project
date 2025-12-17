"""
Dataset classes for segmentation tasks
"""
import os
from typing import Optional, Callable, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    """
    Image segmentation dataset
    
    Expected structure:
        root/
            images/
                train/
                    img1.jpg
                    img2.jpg
                val/
                    img3.jpg
            masks/
                train/
                    img1.png
                    img2.png
                val/
                    img3.png
    
    Args:
        root: Root directory path
        split: Dataset split ('train', 'val', or 'test')
        image_transform: Transformations for images
        mask_transform: Transformations for masks
        joint_transform: Transformations applied to both image and mask
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None
    ):
        self.root = Path(root)
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform
        
        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split
        
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory does not exist: {self.mask_dir}")
        
        # Get image list
        self.image_files = sorted(list(self.image_dir.glob("*")))
        self.mask_files = sorted(list(self.mask_dir.glob("*")))
        
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(
                f"Number of images ({len(self.image_files)}) != "
                f"number of masks ({len(self.mask_files)})"
            )
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx])
        
        # Apply joint transforms (e.g., random crop, flip)
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        
        # Apply individual transforms
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default: convert to tensor and squeeze
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask
    
    def get_class_weights(self, num_classes: int):
        """Compute class weights based on pixel frequencies"""
        class_counts = torch.zeros(num_classes)
        
        for mask_file in self.mask_files:
            mask = np.array(Image.open(mask_file))
            for c in range(num_classes):
                class_counts[c] += (mask == c).sum()
        
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum()
        return weights


class DummySegmentationDataset(Dataset):
    """
    Dummy segmentation dataset for testing
    
    Args:
        num_samples: Number of samples
        num_classes: Number of segmentation classes
        image_size: Image size (H, W)
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        num_classes: int = 2,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image
        image = torch.randn(3, *self.image_size)
        
        # Generate random mask
        mask = torch.randint(0, self.num_classes, self.image_size)
        
        if self.transform:
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        
        return image, mask


class JointTransform:
    """Apply same random transform to both image and mask"""
    
    def __init__(self, transforms_list):
        self.transforms = transforms_list
    
    def __call__(self, image, mask):
        seed = np.random.randint(2147483647)
        
        for t in self.transforms:
            torch.manual_seed(seed)
            np.random.seed(seed)
            image = t(image)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            mask = t(mask)
        
        return image, mask


def get_segmentation_transforms(
    split: str,
    image_size: int = 256,
    normalize: bool = True
) -> Tuple[transforms.Compose, Optional[transforms.Compose]]:
    """
    Get transforms for segmentation
    
    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size
        normalize: Whether to normalize images
        
    Returns:
        image_transform: Transforms for images
        mask_transform: Transforms for masks (None uses default)
    """
    if split == "train":
        image_transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
    else:
        image_transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    
    if normalize:
        image_transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    image_transform = transforms.Compose(image_transform_list)
    
    return image_transform, None


# TODO: Add support for instance segmentation
# TODO: Add support for panoptic segmentation
# TODO: Add multi-scale training support
# TODO: Add online augmentation policies
