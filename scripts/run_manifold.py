"""
Run manifold analysis (UMAP/PCA + FAISS indexing)
Usage: python run_manifold.py [options]
"""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import build_model
from src.datasets import DummyClassificationDataset, get_classification_transforms
from src.manifold import ManifoldAnalyzer, extract_features_from_model
from src.utils import set_seed, get_device, setup_logger


@hydra.main(version_base=None, config_path="../configs", config_name="manifold")
def main(cfg: DictConfig):
    """Main manifold analysis function"""
    
    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger = setup_logger()
    
    logger.info("Running manifold analysis")
    logger.info(f"Device: {device}")
    logger.info(f"Method: {cfg.manifold.method}")
    
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
    
    # Create dataset
    logger.info("Creating dataset")
    transform = get_classification_transforms('test', cfg.data.image_size)
    dataset = DummyClassificationDataset(
        num_samples=cfg.num_samples,
        num_classes=cfg.data.num_classes,
        image_size=(cfg.data.image_size, cfg.data.image_size),
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # Extract features
    logger.info("Extracting features from model")
    features, labels = extract_features_from_model(
        model, dataloader, device, use_embeddings=True
    )
    
    # Create manifold analyzer
    logger.info(f"Fitting {cfg.manifold.method.upper()}")
    analyzer = ManifoldAnalyzer(
        n_components=cfg.manifold.n_components,
        method=cfg.manifold.method
    )
    
    # Fit and transform
    embeddings = analyzer.fit_transform(features)
    
    # Visualize if 2D
    if cfg.manifold.n_components == 2:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6
        )
        plt.colorbar(scatter, label='Class')
        plt.title(f'{cfg.manifold.method.upper()} Visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        save_path = output_dir / f'{cfg.manifold.method}_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
        plt.close()
    
    # Build FAISS index if requested
    if cfg.manifold.build_index:
        logger.info("Building FAISS index")
        analyzer.build_faiss_index(features, use_gpu=False)
        logger.info(f"FAISS index built with {len(features)} vectors")
    
    logger.info("Manifold analysis completed!")


if __name__ == "__main__":
    main()
