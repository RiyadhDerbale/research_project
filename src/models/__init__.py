"""
Model factory and model registry
"""

from typing import Dict, Any
import torch.nn as nn

from .classification import SimpleCNN, ResNetClassifier
from .segmentation import UNetMini, SimpleCNNSegmenter


# Model registry
CLASSIFICATION_MODELS = {
    "simple_cnn": SimpleCNN,
    "resnet18": lambda **kwargs: ResNetClassifier(backbone="resnet18", **kwargs),
    "resnet34": lambda **kwargs: ResNetClassifier(backbone="resnet34", **kwargs),
    "resnet50": lambda **kwargs: ResNetClassifier(backbone="resnet50", **kwargs),
}

SEGMENTATION_MODELS = {
    "unet_mini": UNetMini,
    "simple_segmenter": SimpleCNNSegmenter,
}


def build_model(task: str, model_name: str, **kwargs) -> nn.Module:
    """
    Build model from registry
    
    Args:
        task: Task type ('classification' or 'segmentation')
        model_name: Name of the model
        **kwargs: Model-specific arguments
        
    Returns:
        model: Instantiated model
        
    Example:
        >>> model = build_model('classification', 'simple_cnn', num_classes=10)
        >>> model = build_model('segmentation', 'unet_mini', num_classes=2)
    """
    if task == "classification":
        if model_name not in CLASSIFICATION_MODELS:
            raise ValueError(
                f"Unknown classification model: {model_name}. "
                f"Available: {list(CLASSIFICATION_MODELS.keys())}"
            )
        model_cls = CLASSIFICATION_MODELS[model_name]
    elif task == "segmentation":
        if model_name not in SEGMENTATION_MODELS:
            raise ValueError(
                f"Unknown segmentation model: {model_name}. "
                f"Available: {list(SEGMENTATION_MODELS.keys())}"
            )
        model_cls = SEGMENTATION_MODELS[model_name]
    else:
        raise ValueError(f"Unknown task: {task}. Use 'classification' or 'segmentation'")
    
    return model_cls(**kwargs)


def get_model_info(task: str, model_name: str) -> Dict[str, Any]:
    """
    Get model information
    
    Args:
        task: Task type
        model_name: Model name
        
    Returns:
        info: Dict with model metadata
    """
    if task == "classification":
        registry = CLASSIFICATION_MODELS
    elif task == "segmentation":
        registry = SEGMENTATION_MODELS
    else:
        raise ValueError(f"Unknown task: {task}")
    
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_cls = registry[model_name]
    
    return {
        "name": model_name,
        "task": task,
        "class": model_cls,
    }


# TODO: Add model registration decorator for custom models
# TODO: Add automatic model discovery from directory
# TODO: Add model capability querying (supports_uncertainty, supports_features, etc.)
