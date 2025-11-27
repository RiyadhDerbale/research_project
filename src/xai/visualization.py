"""
Visualization utilities for XAI
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple
import torch


def normalize_attribution(attribution: np.ndarray) -> np.ndarray:
    """Normalize attribution to [0, 1]"""
    attr_min = attribution.min()
    attr_max = attribution.max()
    
    if attr_max - attr_min < 1e-8:
        return np.zeros_like(attribution)
    
    return (attribution - attr_min) / (attr_max - attr_min)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image
    
    Args:
        image: Original image [H, W, C] (RGB, 0-255)
        heatmap: Attribution heatmap [H, W] (0-1)
        alpha: Overlay transparency
        colormap: OpenCV colormap
        
    Returns:
        overlay: Overlayed image
    """
    # Normalize heatmap
    heatmap = normalize_attribution(heatmap)
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Resize heatmap to match image
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def visualize_attribution(
    image: torch.Tensor,
    attribution: np.ndarray,
    title: str = "Attribution",
    save_path: Optional[str] = None
):
    """
    Visualize attribution alongside original image
    
    Args:
        image: Input image tensor [C, H, W]
        attribution: Attribution map [H, W]
        title: Plot title
        save_path: Path to save figure
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
        if image_np.ndim == 3:
            image_np = image_np.transpose(1, 2, 0)
        
        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = image
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Attribution heatmap
    im = axes[1].imshow(attribution, cmap='hot')
    axes[1].set_title(f"{title} Heatmap")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    image_uint8 = (image_np * 255).astype(np.uint8)
    overlay = overlay_heatmap(image_uint8, attribution)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_multiple_attributions(
    image: torch.Tensor,
    attributions: dict,
    save_path: Optional[str] = None
):
    """
    Visualize multiple attribution methods in a grid
    
    Args:
        image: Input image
        attributions: Dict of {method_name: attribution_map}
        save_path: Path to save figure
    """
    n_methods = len(attributions)
    
    # Convert image
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
        if image_np.ndim == 3:
            image_np = image_np.transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = image
    
    # Create grid
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
    
    # Original image (first column)
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Attribution methods
    for idx, (method_name, attr) in enumerate(attributions.items(), 1):
        # Heatmap
        im = axes[0, idx].imshow(attr, cmap='hot')
        axes[0, idx].set_title(f"{method_name}")
        axes[0, idx].axis('off')
        plt.colorbar(im, ax=axes[0, idx])
        
        # Overlay
        image_uint8 = (image_np * 255).astype(np.uint8)
        overlay = overlay_heatmap(image_uint8, attr)
        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f"{method_name} Overlay")
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# TODO: Add 3D visualization for volumetric data
# TODO: Add interactive visualization with Plotly
# TODO: Add video visualization for temporal data
# TODO: Add side-by-side comparison for multiple samples
