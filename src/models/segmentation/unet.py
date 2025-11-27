"""
Lightweight U-Net for image segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool -> ConvBlock"""
    
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.pool_conv(x)


class UpBlock(nn.Module):
    """Upsampling block: ConvTranspose -> Concatenate -> ConvBlock"""
    
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetMini(nn.Module):
    """
    Lightweight U-Net for semantic segmentation
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        num_classes: Number of segmentation classes
        base_channels: Base number of channels (default: 32)
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 32):
        super(UNetMini, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder (downsampling)
        self.inc = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.down4 = DownBlock(base_channels * 8, base_channels * 16)
        
        # Decoder (upsampling)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels)
        
        # Output layer
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            return_features: If True, return intermediate features for XAI
            
        Returns:
            logits: Segmentation logits [B, num_classes, H, W]
            features: Dict of intermediate features (if return_features=True)
        """
        features = {}
        
        # Encoder
        x1 = self.inc(x)
        if return_features:
            features['enc1'] = x1
            
        x2 = self.down1(x1)
        if return_features:
            features['enc2'] = x2
            
        x3 = self.down2(x2)
        if return_features:
            features['enc3'] = x3
            
        x4 = self.down3(x3)
        if return_features:
            features['enc4'] = x4
            
        x5 = self.down4(x4)
        if return_features:
            features['bottleneck'] = x5
        
        # Decoder
        x = self.up1(x5, x4)
        if return_features:
            features['dec1'] = x
            
        x = self.up2(x, x3)
        if return_features:
            features['dec2'] = x
            
        x = self.up3(x, x2)
        if return_features:
            features['dec3'] = x
            
        x = self.up4(x, x1)
        if return_features:
            features['dec4'] = x
        
        logits = self.outc(x)
        
        if return_features:
            return logits, features
        return logits
    
    def get_embedding(self, x):
        """Extract bottleneck embedding"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Return flattened bottleneck
        return x5.mean(dim=[2, 3])  # Global average pooling


# TODO: Add attention gates to decoder
# TODO: Add deep supervision (auxiliary outputs at multiple scales)
# TODO: Add multi-scale input/output support
