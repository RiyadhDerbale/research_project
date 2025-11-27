"""
Evaluation package
"""

from .metrics import (
    compute_classification_metrics,
    compute_segmentation_metrics,
    compute_confusion_matrix,
    compute_iou,
    compute_dice
)

__all__ = [
    "compute_classification_metrics",
    "compute_segmentation_metrics",
    "compute_confusion_matrix",
    "compute_iou",
    "compute_dice",
]
