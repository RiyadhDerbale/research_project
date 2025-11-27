"""
Evaluation metrics for classification and segmentation
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class ('weighted', 'macro', 'micro')
        
    Returns:
        metrics: Dictionary of metrics
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    metrics = {
        'val_acc': acc * 100,  # Convert to percentage
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> np.ndarray:
    """Compute confusion matrix"""
    return confusion_matrix(y_true, y_pred)


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = -1
) -> Dict[str, float]:
    """
    Compute Intersection over Union (IoU) for segmentation
    
    Args:
        pred: Predicted masks [H, W] or [B, H, W]
        target: Ground truth masks [H, W] or [B, H, W]
        num_classes: Number of classes
        ignore_index: Class index to ignore
        
    Returns:
        iou_dict: IoU per class and mean IoU
    """
    pred = pred.flatten()
    target = target.flatten()
    
    iou_per_class = []
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            iou = float('nan')  # No instances of this class
        else:
            iou = intersection / union
        
        iou_per_class.append(iou)
    
    # Compute mean IoU (ignoring NaN values)
    iou_per_class = np.array(iou_per_class)
    mean_iou = np.nanmean(iou_per_class)
    
    return {
        'iou_per_class': iou_per_class,
        'mean_iou': mean_iou
    }


def compute_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = -1
) -> Dict[str, float]:
    """
    Compute Dice coefficient for segmentation
    
    Args:
        pred: Predicted masks
        target: Ground truth masks
        num_classes: Number of classes
        ignore_index: Class index to ignore
        
    Returns:
        dice_dict: Dice per class and mean Dice
    """
    pred = pred.flatten()
    target = target.flatten()
    
    dice_per_class = []
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        pred_sum = pred_cls.sum()
        target_sum = target_cls.sum()
        
        if (pred_sum + target_sum) == 0:
            dice = float('nan')
        else:
            dice = (2 * intersection) / (pred_sum + target_sum)
        
        dice_per_class.append(dice)
    
    dice_per_class = np.array(dice_per_class)
    mean_dice = np.nanmean(dice_per_class)
    
    return {
        'dice_per_class': dice_per_class,
        'mean_dice': mean_dice
    }


def compute_segmentation_metrics(
    target: np.ndarray,
    pred: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    Compute all segmentation metrics
    
    Args:
        target: Ground truth masks
        pred: Predicted masks
        num_classes: Number of classes
        
    Returns:
        metrics: Dictionary of all metrics
    """
    iou_metrics = compute_iou(pred, target, num_classes)
    dice_metrics = compute_dice(pred, target, num_classes)
    
    metrics = {
        'mean_iou': iou_metrics['mean_iou'],
        'mean_dice': dice_metrics['mean_dice'],
        'iou_per_class': iou_metrics['iou_per_class'].tolist(),
        'dice_per_class': dice_metrics['dice_per_class'].tolist()
    }
    
    return metrics


# TODO: Add pixel accuracy metric
# TODO: Add boundary F1 score
# TODO: Add average precision for instance segmentation
# TODO: Add calibration metrics (ECE, MCE)
