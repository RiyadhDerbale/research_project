"""
Classification models package
"""

from .simple_cnn import SimpleCNN
from .resnet import ResNetClassifier

__all__ = [
    "SimpleCNN",
    "ResNetClassifier",
]
