"""
Simple CNN-based segmenter for lightweight segmentation tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNSegmenter(nn.Module):
    """
    Simple fully convolutional network for segmentation
    No skip connections - purely encoder-decoder architecture
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        num_classes: Number of segmentation classes
        base_channels: Base number of channels (default: 32)
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 32):
        super(SimpleCNNSegmenter, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        # Output
        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            return_features: If True, return intermediate features
            
        Returns:
            logits: Segmentation logits [B, num_classes, H, W]
            features: Dict of intermediate features (if return_features=True)
        """
        features = {}
        input_size = x.size()[2:]
        
        # Encoder
        x = self.enc1(x)
        if return_features:
            features['enc1'] = x
            
        x = self.enc2(x)
        if return_features:
            features['enc2'] = x
            
        x = self.enc3(x)
        if return_features:
            features['enc3'] = x
        
        # Decoder with upsampling
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.dec1(x)
        if return_features:
            features['dec1'] = x
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.dec2(x)
        if return_features:
            features['dec2'] = x
        
        # Final output
        logits = self.out(x)
        
        # Ensure output matches input size
        if logits.size()[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        if return_features:
            return logits, features
        return logits
    
    def get_embedding(self, x):
        """Extract embedding from encoder"""
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        
        # Global average pooling
        return x.mean(dim=[2, 3])


# TODO: Add dilated convolutions for larger receptive field
# TODO: Add spatial pyramid pooling
# TODO: Add class weighting in loss for imbalanced datasets
