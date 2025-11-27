"""
Run TCAV concept analysis
Usage: python run_concepts.py [options]
"""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import build_model
from src.concepts import TCAV
from src.utils import set_seed, get_device, setup_logger


@hydra.main(version_base=None, config_path="../configs", config_name="concepts")
def main(cfg: DictConfig):
    """Main concept analysis function"""
    
    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger = setup_logger()
    
    logger.info("Running TCAV concept analysis")
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
    model = model.to(device)
    
    # TODO: Load concept images and random images
    # TODO: Extract target layer from model
    # TODO: Create TCAV instance
    # TODO: Train CAVs for each concept
    # TODO: Compute TCAV scores
    # TODO: Save results
    
    logger.info("Concept analysis completed!")
    logger.info("TODO: Implement full TCAV pipeline with your concept datasets")


if __name__ == "__main__":
    main()
