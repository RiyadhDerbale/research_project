"""
Segmentation trainer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm

from ..utils.logging import get_logger
from ..evaluation.metrics import compute_segmentation_metrics

logger = get_logger(__name__)


class SegmentationTrainer:
    """
    Trainer for segmentation tasks
    
    Args:
        model: Segmentation model
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        num_classes: Number of segmentation classes
        scheduler: Optional learning rate scheduler
        use_wandb: Whether to log to Weights & Biases
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int,
        scheduler: Optional[Any] = None,
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.use_wandb = use_wandb
        
        self.current_epoch = 0
        self.best_val_iou = 0.0
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, logging disabled")
                self.use_wandb = False
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1)
            })
        
        metrics = {
            'train_loss': total_loss / len(train_loader)
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_masks = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Collect predictions
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())
        
        # Concatenate all predictions and masks
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # Compute metrics
        metrics = compute_segmentation_metrics(
            all_masks.numpy(),
            all_preds.numpy(),
            num_classes=self.num_classes
        )
        metrics['val_loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str
    ) -> Dict[str, Any]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            
        Returns:
            history: Training history
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['train_loss']:.4f}, "
                f"Val Loss={val_metrics['val_loss']:.4f}, "
                f"Val IoU={val_metrics['mean_iou']:.4f}, "
                f"Val Dice={val_metrics['mean_dice']:.4f}"
            )
            
            # Save history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_iou'].append(val_metrics['mean_iou'])
            history['val_dice'].append(val_metrics['mean_dice'])
            
            # WandB logging
            if self.use_wandb:
                self.wandb.log({
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch
                })
            
            # Save best model
            if val_metrics['mean_iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['mean_iou']
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_iou': self.best_val_iou,
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
        
        logger.info(f"Training completed. Best Val IoU: {self.best_val_iou:.4f}")
        
        return history


# TODO: Add class-weighted loss for imbalanced segmentation
# TODO: Add boundary-aware loss functions
# TODO: Add multi-scale training
# TODO: Add test-time augmentation
