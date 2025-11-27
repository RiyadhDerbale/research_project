"""
Simple CNN model for image classification
Suitable for small datasets and baseline experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple 3-layer CNN for image classification
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (default: 3 for RGB)
        dropout_rate: Dropout probability (default: 0.5)
    """
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout_rate: float = 0.5):
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Assuming input size 32x32 -> after 3 pools: 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            return_features: If True, return intermediate features for XAI
            
        Returns:
            logits: Class logits [B, num_classes]
            features: Dict of intermediate features (if return_features=True)
        """
        features = {}
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        if return_features:
            features['layer1'] = x
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        if return_features:
            features['layer2'] = x
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        if return_features:
            features['layer3'] = x
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        if return_features:
            features['fc1'] = x
        x = self.dropout(x)
        logits = self.fc2(x)
        
        if return_features:
            return logits, features
        return logits
    
    def get_embedding(self, x):
        """Extract embedding before final classification layer"""
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        return x


# TODO: Add residual connections for deeper variants
# TODO: Add attention mechanisms
# TODO: Add progressive layer freezing for transfer learning
