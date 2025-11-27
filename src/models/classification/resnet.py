"""
ResNet-based classifier using torchvision pretrained models
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict


class ResNetClassifier(nn.Module):
    """
    ResNet classifier with flexible backbone selection
    
    Args:
        num_classes: Number of output classes
        backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50')
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone weights
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(ResNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final FC layer
        self.backbone.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            return_features: If True, return intermediate features
            
        Returns:
            logits: Class logits [B, num_classes]
            features: Dict of intermediate features (if return_features=True)
        """
        if not return_features:
            return self.backbone(x)
        
        # Extract intermediate features
        features = {}
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        features['stem'] = x
        
        x = self.backbone.layer1(x)
        features['layer1'] = x
        
        x = self.backbone.layer2(x)
        features['layer2'] = x
        
        x = self.backbone.layer3(x)
        features['layer3'] = x
        
        x = self.backbone.layer4(x)
        features['layer4'] = x
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        features['avgpool'] = x
        
        logits = self.backbone.fc(x)
        
        return logits, features
    
    def get_embedding(self, x):
        """Extract embedding before final classification layer"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end (None = all)
        """
        layers = [self.backbone.layer1, self.backbone.layer2, 
                  self.backbone.layer3, self.backbone.layer4]
        
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last num_layers
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True


# TODO: Add EfficientNet backbone option
# TODO: Add Vision Transformer (ViT) backbone
# TODO: Add multi-scale feature extraction
