"""
Segmentation models package
"""

from .unet import UNetMini
from .simple_segmenter import SimpleCNNSegmenter

__all__ = [
    "UNetMini",
    "SimpleCNNSegmenter",
]
