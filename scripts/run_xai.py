"""
Generate XAI attribution maps for classification or segmentation
Usage: python run_xai.py task=classification model_path=path/to/model.pth
"""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import build_model
from src.datasets import DummyClassificationDataset, DummySegmentationDataset, get_classification_transforms
from src.xai import AttributionEngine, visualize_multiple_attributions
from src.utils import set_seed, get_device, setup_logger


@hydra.main(version_base=None, config_path="../configs", config_name="xai")
def main(cfg: DictConfig):
    """Main XAI function"""
    
    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger = setup_logger()
    
    logger.info(f"Running XAI for task: {cfg.task}")
    logger.info(f"Device: {device}")
    
    # Build model
    logger.info(f"Building model: {cfg.model.name}")
    model = build_model(
        task=cfg.task,
        model_name=cfg.model.name,
        num_classes=cfg.data.num_classes,
        **cfg.model.params
    )
    
    # Load checkpoint
    logger.info(f"Loading model from: {cfg.model_path}")
    checkpoint = torch.load(cfg.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset
    logger.info("Creating dataset")
    if cfg.task == "classification":
        transform = get_classification_transforms('test', cfg.data.image_size)
        dataset = DummyClassificationDataset(
            num_samples=cfg.num_samples,
            num_classes=cfg.data.num_classes,
            image_size=(cfg.data.image_size, cfg.data.image_size),
            transform=transform
        )
    else:
        from src.datasets import get_segmentation_transforms
        transform, _ = get_segmentation_transforms('test', cfg.data.image_size)
        dataset = DummySegmentationDataset(
            num_samples=cfg.num_samples,
            num_classes=cfg.data.num_classes,
            image_size=(cfg.data.image_size, cfg.data.image_size),
            transform=transform
        )
    
    # Create attribution engine
    logger.info("Initializing attribution engine")
    attribution_engine = AttributionEngine(model, device, task=cfg.task)
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process samples
    logger.info(f"Generating attributions for {cfg.num_samples} samples")
    
    for idx in range(min(cfg.num_samples, len(dataset))):
        if cfg.task == "classification":
            image, label = dataset[idx]
        else:
            image, mask = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            if cfg.task == "classification":
                pred_class = output.argmax(dim=1).item()
            else:
                pred_class = output.argmax(dim=1).flatten().mode()[0].item()
        
        logger.info(f"Sample {idx}: Predicted class = {pred_class}")
        
        # Compute attributions
        attributions = attribution_engine.get_multiple_attributions(
            image,
            methods=cfg.xai.methods,
            target=pred_class
        )
        
        # Visualize
        save_path = output_dir / f"sample_{idx}_attributions.png"
        visualize_multiple_attributions(
            image,
            attributions,
            save_path=str(save_path)
        )
        
        logger.info(f"Saved attribution visualization to {save_path}")
    
    logger.info("XAI generation completed!")


if __name__ == "__main__":
    main()
