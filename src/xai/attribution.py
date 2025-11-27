"""
Attribution methods for explainable AI (XAI)
Supports both classification and segmentation models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    LayerGradCam,
    Saliency
)

from ..utils.logging import get_logger

logger = get_logger(__name__)


class AttributionEngine:
    """
    Unified attribution engine for classification and segmentation
    
    Args:
        model: PyTorch model
        device: Device to run on
        task: Task type ('classification' or 'segmentation')
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        task: str = "classification"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.task = task
        
        # Initialize attribution methods
        self.methods = {}
        self._init_methods()
    
    def _init_methods(self):
        """Initialize Captum attribution methods"""
        self.methods['integrated_gradients'] = IntegratedGradients(self.model)
        self.methods['gradient_shap'] = GradientShap(self.model)
        self.methods['deeplift'] = DeepLift(self.model)
        self.methods['saliency'] = Saliency(self.model)
        
        logger.info(f"Initialized attribution methods: {list(self.methods.keys())}")
    
    def get_attribution(
        self,
        image: torch.Tensor,
        method: str = "integrated_gradients",
        target: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute attribution map
        
        Args:
            image: Input image [1, C, H, W] or [C, H, W]
            method: Attribution method name
            target: Target class (for classification) or None (for segmentation)
            **kwargs: Method-specific arguments
            
        Returns:
            attribution: Attribution map as numpy array
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad = True
        
        if method not in self.methods:
            raise ValueError(
                f"Unknown method: {method}. Available: {list(self.methods.keys())}"
            )
        
        attr_method = self.methods[method]
        
        # For classification, use target class
        # For segmentation, aggregate over all pixels
        if self.task == "classification":
            if target is None:
                # Use predicted class
                with torch.no_grad():
                    output = self.model(image)
                    target = output.argmax(dim=1).item()
        
        # Compute attribution
        if method == "integrated_gradients":
            attribution = attr_method.attribute(
                image,
                target=target,
                n_steps=kwargs.get('n_steps', 50)
            )
        elif method == "gradient_shap":
            # Generate baseline (zero or random)
            baselines = torch.zeros_like(image)
            attribution = attr_method.attribute(
                image,
                baselines=baselines,
                target=target,
                n_samples=kwargs.get('n_samples', 5)
            )
        elif method == "deeplift":
            attribution = attr_method.attribute(
                image,
                target=target
            )
        elif method == "saliency":
            attribution = attr_method.attribute(
                image,
                target=target
            )
        else:
            raise NotImplementedError(f"Method {method} not implemented")
        
        # Convert to numpy
        attribution = attribution.squeeze().cpu().detach().numpy()
        
        # Aggregate channels if needed
        if attribution.ndim == 3:  # [C, H, W]
            attribution = np.abs(attribution).sum(axis=0)
        
        return attribution
    
    def get_gradcam(
        self,
        image: torch.Tensor,
        layer: nn.Module,
        target: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM
        
        Args:
            image: Input image
            layer: Target layer for Grad-CAM
            target: Target class
            
        Returns:
            gradcam: Grad-CAM heatmap
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        if target is None:
            with torch.no_grad():
                output = self.model(image)
                if self.task == "classification":
                    target = output.argmax(dim=1).item()
                else:
                    # For segmentation, use dominant class
                    target = output.argmax(dim=1).flatten().mode()[0].item()
        
        # Initialize Grad-CAM
        gradcam = LayerGradCam(self.model, layer)
        
        # Compute attribution
        attribution = gradcam.attribute(image, target=target)
        
        # Convert to numpy and squeeze
        heatmap = attribution.squeeze().cpu().detach().numpy()
        
        return heatmap
    
    def get_multiple_attributions(
        self,
        image: torch.Tensor,
        methods: List[str],
        target: Optional[int] = None
    ) -> dict:
        """
        Compute multiple attribution maps
        
        Args:
            image: Input image
            methods: List of method names
            target: Target class
            
        Returns:
            attributions: Dict of attribution maps
        """
        attributions = {}
        
        for method in methods:
            try:
                attr = self.get_attribution(image, method, target)
                attributions[method] = attr
                logger.info(f"Computed {method} attribution")
            except Exception as e:
                logger.error(f"Failed to compute {method}: {e}")
        
        return attributions


# TODO: Add LRP (Layer-wise Relevance Propagation)
# TODO: Add LIME support
# TODO: Add Shapley value computation
# TODO: Add attribution aggregation methods
# TODO: Add sensitivity analysis
