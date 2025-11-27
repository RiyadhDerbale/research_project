"""
TCAV (Testing with Concept Activation Vectors) implementation
Concept-based explanations for neural networks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ConceptExtractor:
    """
    Extract concept activations from model layers
    
    Args:
        model: PyTorch model
        layer: Target layer to extract activations from
        device: Device to run on
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        device: torch.device
    ):
        self.model = model.to(device)
        self.model.eval()
        self.layer = layer
        self.device = device
        
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hook to capture activations"""
        def hook_fn(module, input, output):
            self.activations = output.detach()
        
        self.layer.register_forward_hook(hook_fn)
    
    def get_activations(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract activations for images
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            activations: Layer activations [B, D] or [B, C, H, W]
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            _ = self.model(images)
        
        # Flatten spatial dimensions if needed
        acts = self.activations
        if acts.dim() == 4:  # [B, C, H, W]
            acts = acts.mean(dim=[2, 3])  # Global average pooling
        
        return acts


class CAV:
    """
    Concept Activation Vector
    
    Trains a linear classifier to separate concept examples from random examples
    """
    
    def __init__(self, concept_name: str, layer_name: str):
        self.concept_name = concept_name
        self.layer_name = layer_name
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
    
    def train(
        self,
        concept_acts: np.ndarray,
        random_acts: np.ndarray
    ) -> float:
        """
        Train CAV classifier
        
        Args:
            concept_acts: Activations from concept examples [N, D]
            random_acts: Activations from random examples [M, D]
            
        Returns:
            accuracy: Classifier accuracy
        """
        # Prepare data
        X = np.vstack([concept_acts, random_acts])
        y = np.array([1] * len(concept_acts) + [0] * len(random_acts))
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        accuracy = self.classifier.score(X_test, y_test)
        
        self.is_trained = True
        
        logger.info(
            f"Trained CAV for '{self.concept_name}' at layer '{self.layer_name}' "
            f"with accuracy: {accuracy:.3f}"
        )
        
        return accuracy
    
    def get_direction(self) -> np.ndarray:
        """Get CAV direction (normal to decision boundary)"""
        if not self.is_trained:
            raise ValueError("CAV not trained yet")
        
        return self.classifier.coef_[0]
    
    def project(self, activations: np.ndarray) -> np.ndarray:
        """
        Project activations onto CAV direction
        
        Args:
            activations: Activations [N, D]
            
        Returns:
            projections: Projection scores [N]
        """
        direction = self.get_direction()
        return np.dot(activations, direction)


class TCAV:
    """
    Testing with Concept Activation Vectors
    
    Quantifies the influence of concepts on model predictions
    
    Args:
        model: PyTorch model
        layer: Target layer
        device: Device to run on
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        device: torch.device
    ):
        self.model = model
        self.layer = layer
        self.device = device
        self.extractor = ConceptExtractor(model, layer, device)
        
        self.cavs: Dict[str, CAV] = {}
    
    def create_cav(
        self,
        concept_name: str,
        concept_images: torch.Tensor,
        random_images: torch.Tensor,
        layer_name: str = "target_layer"
    ) -> CAV:
        """
        Create and train a CAV
        
        Args:
            concept_name: Name of the concept
            concept_images: Images containing the concept
            random_images: Random images (negative examples)
            layer_name: Name of the layer
            
        Returns:
            cav: Trained CAV
        """
        # Extract activations
        concept_acts = self.extractor.get_activations(concept_images).cpu().numpy()
        random_acts = self.extractor.get_activations(random_images).cpu().numpy()
        
        # Train CAV
        cav = CAV(concept_name, layer_name)
        cav.train(concept_acts, random_acts)
        
        # Store CAV
        self.cavs[concept_name] = cav
        
        return cav
    
    def compute_tcav_score(
        self,
        concept_name: str,
        test_images: torch.Tensor,
        target_class: int,
        n_samples: int = 50
    ) -> float:
        """
        Compute TCAV score: fraction of test images for which concept is positively influential
        
        Args:
            concept_name: Name of the concept
            test_images: Test images
            target_class: Target class to analyze
            n_samples: Number of samples to use
            
        Returns:
            tcav_score: TCAV score [0, 1]
        """
        if concept_name not in self.cavs:
            raise ValueError(f"CAV for '{concept_name}' not found")
        
        cav = self.cavs[concept_name]
        
        # Sample images
        if len(test_images) > n_samples:
            indices = torch.randperm(len(test_images))[:n_samples]
            test_images = test_images[indices]
        
        test_images = test_images.to(self.device)
        test_images.requires_grad = True
        
        # Compute gradients
        positive_count = 0
        
        for i in range(len(test_images)):
            img = test_images[i:i+1]
            
            # Forward pass
            _ = self.model(img)
            acts = self.extractor.activations
            
            # Compute gradient of target class w.r.t. activations
            output = self.model(img)
            target_score = output[0, target_class]
            
            # Backward
            self.model.zero_grad()
            target_score.backward(retain_graph=True)
            
            # Get gradients at target layer
            if acts.grad is not None:
                grad = acts.grad[0].flatten().cpu().numpy()
            else:
                continue
            
            # Project gradient onto CAV direction
            direction = cav.get_direction()
            
            # Resize direction if needed
            if len(direction) != len(grad):
                # Handle dimension mismatch
                continue
            
            # Directional derivative
            directional_deriv = np.dot(grad, direction)
            
            if directional_deriv > 0:
                positive_count += 1
        
        tcav_score = positive_count / len(test_images)
        
        logger.info(
            f"TCAV score for concept '{concept_name}' on class {target_class}: "
            f"{tcav_score:.3f}"
        )
        
        return tcav_score


# TODO: Add ACE (Automated Concept Extraction)
# TODO: Add concept completeness testing
# TODO: Add multi-layer TCAV analysis
# TODO: Add concept drift detection
