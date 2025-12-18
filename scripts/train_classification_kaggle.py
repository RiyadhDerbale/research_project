"""
Training script for classification models - Kaggle Notebook Version
Usage: Run cells in Kaggle notebook
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='IPython')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Check numpy compatibility early
try:
    import numpy as np
    print(f"✅ numpy version: {np.__version__}")
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        print("❌ numpy compatibility issue detected!")
        print("   Run: !pip uninstall -y numpy && pip install numpy==1.24.3")
        print("   Then restart the kernel.")
        raise

import os
import sys
from pathlib import Path

# Setup paths for Kaggle
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    sys.path.append('/kaggle/working')
    sys.path.append('/kaggle/input')
else:
    # Local environment
    PROJECT_ROOT = Path.cwd()
    if (PROJECT_ROOT / 'src').exists():
        sys.path.append(str(PROJECT_ROOT))
    elif (PROJECT_ROOT.parent / 'src').exists():
        sys.path.append(str(PROJECT_ROOT.parent))

import torch
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf

from src.models import build_model
from src.datasets import ImageFolderClassification, get_classification_transforms
from src.training import ClassificationTrainer
from src.utils import set_seed, get_device, setup_logger


def is_kaggle():
    """Check if running on Kaggle"""
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def find_kaggle_dataset():
    """Auto-detect Dogs vs Cats dataset path on Kaggle"""
    if not is_kaggle():
        return './data/classification'
    
    # Try common dataset paths
    dataset_paths = [
        '/kaggle/input/dogs-vs-cats-redux-kernels-edition',
        '/kaggle/input/dogs-vs-cats',
        '/kaggle/input/dog-vs-cat',
        '/kaggle/input/dogs-cats-images',
        '/kaggle/working/data/classification'
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"✅ Found dataset at: {path}")
            if os.path.exists(path):
                print(f"   Contents: {os.listdir(path)[:5]}...")  # Show first 5 items
            return path
    
    # Fallback: show available datasets
    print("❌ Dataset not found! Available datasets:")
    for item in os.listdir('/kaggle/input/'):
        print(f"   - /kaggle/input/{item}")
    
    raise ValueError("Dogs vs Cats dataset not found. Please add it to your Kaggle notebook.")


def train_classification():
    """Main training function without Hydra (for notebooks)"""
    
    # Auto-detect dataset path
    data_root = find_kaggle_dataset()
    
    # Load config manually (no Hydra needed)
    config_dict = {
        'model': {
            'name': 'resnet18',
            'params': {}
        },
        'data': {
            'root': data_root,
            'num_classes': 2,
            'image_size': 224
        },
        'train': {
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'num_workers': 2 if is_kaggle() else 4,
            'seed': 42,
            'device': 'cuda',
            'use_scheduler': True
        },
        'experiment': {
            'name': 'classification_training',
            'base_dir': '/kaggle/working' if is_kaggle() else './experiments'
        },
        'wandb': {
            'enabled': True,
            'project': 'classification'
        }
    }
    
    cfg = OmegaConf.create(config_dict)
    
    # Setup
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    
    # Create experiment directory
    exp_dir = Path(cfg.experiment.base_dir) / "experiment"
    exp_dir.mkdir(exist_ok=True, parents=True)
    
    logger = setup_logger(log_file=str(exp_dir / "train.log"))
    
    logger.info(f"Starting classification training")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Running on Kaggle: {is_kaggle()}")
    
    # Initialize wandb in offline mode (no API key required)
    if cfg.wandb.enabled:
        os.environ["WANDB_MODE"] = "offline"
        
        try:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.experiment.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=str(exp_dir)
            )
            logger.info("Wandb initialized in OFFLINE mode (no API key needed)")
            logger.info(f"Wandb logs will be saved to: {exp_dir / 'wandb'}")
        except Exception as e:
            logger.warning(f"Wandb initialization failed: {e}. Continuing without wandb.")
            cfg.wandb.enabled = False
    
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
    
    # Get transforms
    train_transform = get_classification_transforms('train', cfg.data.image_size)
    test_transform = get_classification_transforms('val', cfg.data.image_size)
    
    # Load datasets
    train_dataset = ImageFolderClassification(
        root=cfg.data.root,
        split='train',
        transform=train_transform
    )
    
    test_dataset = ImageFolderClassification(
        root=cfg.data.root,
        split='test',
        transform=test_transform
    )
    
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    logger.info(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=not is_kaggle()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=not is_kaggle()
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay
    )
    
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
        val_loader=test_loader,
        num_epochs=cfg.train.epochs,
        save_dir=str(exp_dir / "checkpoints")
    )
    
    logger.info("Training completed!")
    logger.info(f"Best test accuracy: {trainer.best_val_acc:.2f}%")
    
    # Finish wandb
    if cfg.wandb.enabled:
        wandb.finish()
    
    return history, trainer


if __name__ == "__main__":
    # This version works directly without Hydra
    history, trainer = train_classification()
    print(f"\nTraining completed! Best accuracy: {trainer.best_val_acc:.2f}%")
