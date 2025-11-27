"""
Generate LLM-based explanations for model predictions
Usage: python run_llm_explanations.py [options]
"""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import build_model
from src.datasets import DummyClassificationDataset, get_classification_transforms
from src.xai import AttributionEngine
from src.llm import LLMExplainer
from src.utils import set_seed, get_device, setup_logger


@hydra.main(version_base=None, config_path="../configs", config_name="llm")
def main(cfg: DictConfig):
    """Main LLM explanation function"""
    
    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger = setup_logger()
    
    logger.info("Generating LLM-based explanations")
    logger.info(f"Device: {device}")
    logger.info(f"LLM Provider: {cfg.llm.provider}")
    
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
    model.eval()
    
    # Create LLM explainer
    logger.info("Initializing LLM explainer")
    llm_explainer = LLMExplainer(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        api_key=cfg.llm.get('api_key'),
        temperature=cfg.llm.temperature
    )
    
    # Create attribution engine for XAI context
    attribution_engine = AttributionEngine(model, device, task=cfg.task)
    
    # Create dataset
    logger.info("Creating dataset")
    transform = get_classification_transforms('test', cfg.data.image_size)
    dataset = DummyClassificationDataset(
        num_samples=cfg.num_samples,
        num_classes=cfg.data.num_classes,
        image_size=(cfg.data.image_size, cfg.data.image_size),
        transform=transform
    )
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process samples
    logger.info(f"Generating explanations for {cfg.num_samples} samples")
    
    for idx in range(min(cfg.num_samples, len(dataset))):
        image, label = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)[0]
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()
            
            # Top-k predictions
            top_k = torch.topk(probs, k=min(3, len(probs)))
            top_k_classes = [(f"class_{i}", p.item()) for i, p in zip(top_k.indices, top_k.values)]
        
        logger.info(f"Sample {idx}: Predicted class = {pred_class}, Confidence = {confidence:.3f}")
        
        # Get attribution summary (simplified)
        attribution = attribution_engine.get_attribution(
            image,
            method="integrated_gradients",
            target=pred_class
        )
        
        attribution_summary = f"Top contributing regions identified with peak attribution score of {attribution.max():.3f}"
        
        # Generate LLM explanation
        if cfg.task == "classification":
            explanation = llm_explainer.explain_classification(
                predicted_class=f"class_{pred_class}",
                confidence=confidence,
                top_k_classes=top_k_classes,
                attribution_summary=attribution_summary,
                concepts=None,  # TODO: Add concept detection
                uncertainty=None  # TODO: Add uncertainty estimation
            )
        else:
            # For segmentation
            class_dist = {f"class_{i}": 0.1 for i in range(cfg.data.num_classes)}
            explanation = llm_explainer.explain_segmentation(
                class_distribution=class_dist,
                attribution_summary=attribution_summary
            )
        
        # Save explanation
        explanation_file = output_dir / f"sample_{idx}_explanation.txt"
        with open(explanation_file, 'w') as f:
            f.write(f"Sample {idx}\n")
            f.write(f"Predicted: class_{pred_class} (confidence: {confidence:.3f})\n")
            f.write(f"Ground truth: class_{label}\n\n")
            f.write("="*50 + "\n")
            f.write("LLM Explanation:\n")
            f.write("="*50 + "\n\n")
            f.write(explanation)
        
        logger.info(f"Saved explanation to {explanation_file}")
    
    logger.info("LLM explanation generation completed!")


if __name__ == "__main__":
    main()
