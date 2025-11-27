"""
Counterfactual generation methods
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable

from ..utils.logging import get_logger

logger = get_logger(__name__)


class InputPerturbationCF:
    """
    Simple counterfactual generation via input perturbations
    
    Args:
        model: PyTorch model
        device: Device
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
    
    def generate_targeted_cf(
        self,
        image: torch.Tensor,
        target_class: int,
        learning_rate: float = 0.01,
        max_iterations: int = 500,
        l2_weight: float = 0.1,
        early_stop_threshold: float = 0.95
    ) -> torch.Tensor:
        """
        Generate targeted counterfactual by optimizing input
        
        Args:
            image: Input image [1, C, H, W]
            target_class: Target class to reach
            learning_rate: Optimization learning rate
            max_iterations: Maximum optimization steps
            l2_weight: Weight for L2 distance regularization
            early_stop_threshold: Stop if target confidence reached
            
        Returns:
            counterfactual: Generated counterfactual image
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Initialize counterfactual as a copy
        cf = image.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([cf], lr=learning_rate)
        
        best_cf = cf.clone().detach()
        best_score = 0.0
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(cf)
            
            if self.task == "classification":
                # Classification loss
                target_score = torch.softmax(output, dim=1)[0, target_class]
                
                # Maximize target class probability, minimize distance
                loss = -target_score + l2_weight * torch.norm(cf - image, p=2)
            else:
                # Segmentation: maximize target class in segmentation map
                target_score = torch.softmax(output, dim=1)[:, target_class].mean()
                loss = -target_score + l2_weight * torch.norm(cf - image, p=2)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Clip to valid image range
            with torch.no_grad():
                cf.clamp_(0, 1)
            
            # Check early stopping
            current_score = target_score.item()
            if current_score > best_score:
                best_score = current_score
                best_cf = cf.clone().detach()
            
            if current_score >= early_stop_threshold:
                logger.info(
                    f"Early stop at iteration {iteration}, "
                    f"target confidence: {current_score:.3f}"
                )
                break
            
            if iteration % 100 == 0:
                logger.info(
                    f"Iter {iteration}: target_score={current_score:.3f}, "
                    f"loss={loss.item():.3f}"
                )
        
        logger.info(f"Best target confidence: {best_score:.3f}")
        
        return best_cf.detach()
    
    def generate_minimal_cf(
        self,
        image: torch.Tensor,
        original_class: int,
        learning_rate: float = 0.01,
        max_iterations: int = 500
    ) -> torch.Tensor:
        """
        Generate minimal counterfactual (flip prediction with minimal change)
        
        Args:
            image: Input image
            original_class: Original predicted class
            learning_rate: Optimization learning rate
            max_iterations: Maximum iterations
            
        Returns:
            counterfactual: Generated counterfactual
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Initialize
        cf = image.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([cf], lr=learning_rate)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            output = self.model(cf)
            
            if self.task == "classification":
                pred_class = output.argmax(dim=1).item()
                
                # If flipped, minimize distance
                if pred_class != original_class:
                    loss = torch.norm(cf - image, p=2)
                else:
                    # Minimize original class score + distance
                    loss = output[0, original_class] + 0.1 * torch.norm(cf - image, p=2)
            else:
                # For segmentation, flip dominant class
                pred_map = output.argmax(dim=1)
                original_match = (pred_map == original_class).float().mean()
                
                if original_match < 0.5:  # Flipped
                    loss = torch.norm(cf - image, p=2)
                else:
                    loss = original_match + 0.1 * torch.norm(cf - image, p=2)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                cf.clamp_(0, 1)
        
        return cf.detach()


class LatentCF:
    """
    Stub for latent-space counterfactual generation
    (requires VAE/GAN/Diffusion model)
    """
    
    def __init__(self, generative_model: Optional[nn.Module] = None):
        self.generative_model = generative_model
    
    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        """Generate image from latent code"""
        # TODO: Implement latent traversal
        # TODO: Implement latent interpolation
        # TODO: Implement semantic editing in latent space
        raise NotImplementedError("Latent CF requires a trained generative model")


# TODO: Add GAN-based counterfactuals
# TODO: Add diffusion-based counterfactuals
# TODO: Add causal counterfactuals
# TODO: Add diversity constraints
