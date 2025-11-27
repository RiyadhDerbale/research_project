"""
Research Project: Image Classification & Segmentation with XAI
Main package initialization
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from src.models.classification import SimpleCNN, ResNetClassifier
from src.models.segmentation import UNetMini, SimpleCNNSegmenter
from src.utils.config import setup_config
from src.utils.logging import setup_logger

__all__ = [
    "SimpleCNN",
    "ResNetClassifier",
    "UNetMini",
    "SimpleCNNSegmenter",
    "setup_config",
    "setup_logger",
]
