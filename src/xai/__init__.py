"""
XAI (Explainable AI) package
"""

from .attribution import AttributionEngine
from .visualization import (
    visualize_attribution,
    visualize_multiple_attributions,
    overlay_heatmap
)

__all__ = [
    "AttributionEngine",
    "visualize_attribution",
    "visualize_multiple_attributions",
    "overlay_heatmap",
]
