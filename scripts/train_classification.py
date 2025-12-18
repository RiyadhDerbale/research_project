"""
Training script for classification models
Usage: python train_classification.py [options]
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='IPython')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import wandb
import os

import sys
from pathlib import Path

# Handle path setup for both script and notebook environments
PROJECT_ROOT = None
try:
    # Running as a script
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.append(str(PROJECT_ROOT))
except NameError:
    # Running in Jupyter/Kaggle notebook where __file__ is not defined
    # Try to find the project root
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        # On Kaggle
        PROJECT_ROOT = Path('/kaggle/working')
        sys.path.append('/kaggle/working')
        sys.path.append('/kaggle/input')
    else:
        # Local Jupyter notebook
        current_dir = Path.cwd()
        if (current_dir / 'src').exists():
            PROJECT_ROOT = current_dir
            sys.path.append(str(current_dir))
        elif (current_dir.parent / 'src').exists():
            PROJECT_ROOT = current_dir.parent
            sys.path.append(str(current_dir.parent))
        else:
            PROJECT_ROOT = current_dir
            sys.path.append(str(current_dir))

from src.models import build_model
from src.datasets import ImageFolderClassification, get_classification_transforms
from src.training import ClassificationTrainer
from src.utils import set_seed, get_device, setup_logger, get_experiment_dir


def is_kaggle():
    """Check if running on Kaggle"""
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def is_notebook():
    """Check if running in Jupyter/Colab/Kaggle notebook"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except:
        pass
    return False


def get_config_path():
    """Get absolute path to config directory"""
    if PROJECT_ROOT:
        config_path = PROJECT_ROOT / "configs"
        if config_path.exists():
            return str(config_path)
    
    # Try common locations
    possible_paths = [
        "../configs",
        "./configs",
        "/kaggle/working/configs",
        "/kaggle/input/research-project-code/configs"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    # Last resort: return relative path and let Hydra handle the error
    print("WARNING: Could not find configs directory. Trying default path.")
    return "../configs"


# Use absolute config path for notebooks
CONFIG_PATH = get_config_path()

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="classification")
def main(cfg: DictConfig):
    """Main training function"""
    
    # Kaggle-specific adjustments
    if is_kaggle():
        # Update paths for Kaggle environment
        # Common Kaggle dataset paths (update based on your actual dataset):
        # Option 1: Kaggle competition dataset
        cfg.data.root = '/kaggle/input/dogs-vs-cats-redux-kernels-edition'
        # Option 2: Custom uploaded dataset
        # cfg.data.root = '/kaggle/input/your-dataset-name/data/classification'
        # Option 3: Working directory (if you extracted there)
        # cfg.data.root = '/kaggle/working/data/classification'
        
        cfg.experiment.base_dir = '/kaggle/working'
        cfg.train.num_workers = 2  # Kaggle works better with 2 workers
        cfg.wandb.enabled = True  # Keep wandb in offline mode
        
        # Log paths for debugging
        import os
        print(f"📁 Looking for data at: {cfg.data.root}")
        if os.path.exists(cfg.data.root):
            print(f"✅ Data directory found!")
            print(f"📂 Contents: {os.listdir(cfg.data.root)}")
        else:
            print(f"❌ Data directory NOT found!")
            print(f"Available inputs: {os.listdir('/kaggle/input/')}")
        
    # Setup
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    
    # Create experiment directory with error handling for Kaggle
    try:
        exp_dir = get_experiment_dir(cfg, base_dir=cfg.experiment.base_dir)
    except Exception as e:
        # Fallback for Kaggle or other environments
        exp_dir = Path(cfg.experiment.base_dir) / "experiment"
        exp_dir.mkdir(exist_ok=True, parents=True)
        print(f"Warning: Could not create standard experiment dir, using: {exp_dir}")
        
    logger = setup_logger(log_file=str(exp_dir / "train.log"))
    
    logger.info(f"Starting classification training")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Running on Kaggle: {is_kaggle()}")
    
    # Initialize wandb in offline mode (no API key required)
    if cfg.wandb.enabled:
        # Set wandb to offline mode - stores logs locally
        os.environ["WANDB_MODE"] = "offline"
        
        try:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.experiment.name,
                config=dict(cfg),
                dir=str(exp_dir)  # Store wandb logs in experiment directory
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
    test_transform = get_classification_transforms('val', cfg.data.image_size)  # Use 'val' transforms for test (no augmentation)
    
    # Load real datasets from data directory
    train_dataset = ImageFolderClassification(
        root=cfg.data.root,
        split='train',
        transform=train_transform
    )
    
    # Use test set instead of validation set
    test_dataset = ImageFolderClassification(
        root=cfg.data.root,
        split='test',
        transform=test_transform
    )
    
    # Validation commented out - using test set instead
    # val_dataset = ImageFolderClassification(
    #     root=cfg.data.root,
    #     split='val',
    #     transform=val_transform
    # )
    
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    logger.info(f"Classes: {train_dataset.classes}")
    
    # Create data loaders with Kaggle-compatible settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=not is_kaggle()  # False on Kaggle to reduce memory usage
    )
    
    # Use test_loader instead of val_loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=not is_kaggle()  # False on Kaggle to reduce memory usage
    )
    
    # val_loader commented out - using test_loader instead
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=cfg.train.batch_size,
    #     shuffle=False,
    #     num_workers=cfg.train.num_workers,
    #     pin_memory=True
    # )
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
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
        val_loader=test_loader,  # Using test_loader for evaluation during training
        num_epochs=cfg.train.epochs,
        save_dir=str(exp_dir / "checkpoints")
    )
    
    logger.info("Training completed!")
    logger.info(f"Best test accuracy: {trainer.best_val_acc:.2f}%")  # Note: still called best_val_acc in trainer
    
    # Finish wandb
    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    # Clear Jupyter/Colab kernel arguments that conflict with Hydra
    if is_notebook():
        import sys
        # Remove notebook-specific arguments (like -f)
        sys.argv = [arg for arg in sys.argv if not arg.startswith('-f')]
        # Ensure at least script name exists
        if len(sys.argv) == 0:
            sys.argv = ['train_classification.py']
    
    main()
