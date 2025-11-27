"""
Training script for classification models
Usage: python train_classification.py [options]
"""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import wandb

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import build_model
from src.datasets import DummyClassificationDataset, get_classification_transforms
from src.training import ClassificationTrainer
from src.utils import set_seed, get_device, setup_logger, get_experiment_dir


@hydra.main(version_base=None, config_path="../configs", config_name="classification")
def main(cfg: DictConfig):
    """Main training function"""
    
    # Setup
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    
    # Create experiment directory
    exp_dir = get_experiment_dir(cfg, base_dir=cfg.experiment.base_dir)
    logger = setup_logger(log_file=str(exp_dir / "train.log"))
    
    logger.info(f"Starting classification training")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Device: {device}")
    
    # Initialize wandb
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.experiment.name,
            config=dict(cfg)
        )
    
    # Build model
    logger.info(f"Building model: {cfg.model.name}")
    model = build_model(
        task="classification",
        model_name=cfg.model.name,
        num_classes=cfg.data.num_classes,
        **cfg.model.params
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    logger.info("Creating datasets")
    
    # TODO: Replace with real dataset loading
    train_transform = get_classification_transforms('train', cfg.data.image_size)
    val_transform = get_classification_transforms('val', cfg.data.image_size)
    
    train_dataset = DummyClassificationDataset(
        num_samples=cfg.data.train_samples,
        num_classes=cfg.data.num_classes,
        image_size=(cfg.data.image_size, cfg.data.image_size),
        transform=train_transform
    )
    
    val_dataset = DummyClassificationDataset(
        num_samples=cfg.data.val_samples,
        num_classes=cfg.data.num_classes,
        image_size=(cfg.data.image_size, cfg.data.image_size),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay
    )
    
    # Create loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create scheduler
    scheduler = None
    if cfg.train.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs
        )
    
    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_wandb=cfg.wandb.enabled
    )
    
    # Train
    logger.info("Starting training loop")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.train.epochs,
        save_dir=str(exp_dir / "checkpoints")
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    
    # Finish wandb
    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
