"""
Model Testing Script
Usage: python scripts/test_model.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import build_model
from src.datasets import ImageFolderClassification, get_classification_transforms
from torch.utils.data import DataLoader


class ModelTester:
    """Comprehensive model testing and evaluation"""
    
    def __init__(self, model_path, data_root, device='cuda'):
        """
        Initialize Model Tester
        
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            data_root: Path to dataset root directory
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_root = Path(data_root)
        
        print("="*70)
        print("🧪 Model Testing Suite")
        print("="*70)
        
        # Load model
        print(f"\n📥 Loading model from: {model_path}")
        self.model = build_model(
            task="classification",
            model_name="resnet18",
            num_classes=2
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Device: {self.device}")
        print(f"   Best training accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
        
        # Get transforms
        self.transform = get_classification_transforms('val', 224)
        
        # Class names
        self.class_names = ['cat', 'dog']
        
    def load_test_data(self, batch_size=32, num_workers=4):
        """Load test dataset"""
        print(f"\n📂 Loading test dataset from: {self.data_root}")
        
        # Create test dataset
        test_dataset = ImageFolderClassification(
            root=self.data_root,
            split='test',
            transform=self.transform
        )
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"✅ Test dataset loaded")
        print(f"   Total samples: {len(test_dataset)}")
        print(f"   Classes: {test_dataset.classes}")
        print(f"   Batch size: {batch_size}")
        
        return test_dataset, test_loader
    
    def evaluate_on_test_set(self, test_loader):
        """Comprehensive evaluation on test set"""
        print("\n🔍 Evaluating on test set...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Calculate accuracy
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = 100 * correct / total
        
        print(f"\n✅ Evaluation complete!")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Correct: {correct}/{total}")
        
        return {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy
        }
    
    def generate_classification_report(self, labels, predictions, save_path=None):
        """Generate detailed classification report"""
        print("\n📊 Classification Report:")
        print("="*70)
        
        report = classification_report(
            labels, 
            predictions, 
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Classification Report\n")
                f.write("="*70 + "\n")
                f.write(report)
            print(f"✅ Saved report to: {save_path}")
        
        return report
    
    def plot_confusion_matrix(self, labels, predictions, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy for each class
        accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(accuracy_per_class):
            plt.text(
                len(self.class_names), i, 
                f'{acc*100:.1f}%', 
                ha='left', va='center',
                fontsize=10, fontweight='bold'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved confusion matrix to: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, labels, probabilities, save_path=None):
        """Plot ROC curve"""
        # For binary classification
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved ROC curve to: {save_path}")
        
        plt.show()
        
        return roc_auc
    
    def analyze_predictions(self, test_dataset, predictions, labels, probabilities, 
                          num_samples=8, save_path=None):
        """Visualize predictions with confidence"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Get random samples
        indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), 
                                  replace=False)
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                sample_idx = indices[idx]
                img, label = test_dataset[sample_idx]
                pred = predictions[sample_idx]
                prob = probabilities[sample_idx]
                
                # Denormalize image
                img = img.permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # Plot
                ax.imshow(img)
                
                # Title with prediction info
                is_correct = pred == label
                color = 'green' if is_correct else 'red'
                confidence = prob[pred] * 100
                
                title = f"True: {self.class_names[label]}\n"
                title += f"Pred: {self.class_names[pred]} ({confidence:.1f}%)"
                title += f"\n{'✓ Correct' if is_correct else '✗ Wrong'}"
                
                ax.set_title(title, color=color, fontsize=10, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved prediction samples to: {save_path}")
        
        plt.show()


def main():
    """Main testing function"""
    
    # Paths
    model_path = PROJECT_ROOT / "models/classification/best_model.pth"
    data_root = PROJECT_ROOT / "data/classification"
    results_dir = PROJECT_ROOT / "results/testing"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("\nPlease:")
        print("1. Download best_model.pth from Kaggle")
        print(f"2. Place it in: {model_path.parent}")
        return
    
    # Initialize tester
    tester = ModelTester(
        model_path=model_path,
        data_root=data_root,
        device='cuda'
    )
    
    # Load test data
    test_dataset, test_loader = tester.load_test_data(batch_size=32, num_workers=4)
    
    # Evaluate on test set
    results = tester.evaluate_on_test_set(test_loader)
    
    # Generate classification report
    tester.generate_classification_report(
        results['labels'],
        results['predictions'],
        save_path=results_dir / "classification_report.txt"
    )
    
    # Plot confusion matrix
    tester.plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        save_path=results_dir / "confusion_matrix.png"
    )
    
    # Plot ROC curve
    roc_auc = tester.plot_roc_curve(
        results['labels'],
        results['probabilities'],
        save_path=results_dir / "roc_curve.png"
    )
    
    # Analyze predictions
    tester.analyze_predictions(
        test_dataset,
        results['predictions'],
        results['labels'],
        results['probabilities'],
        num_samples=8,
        save_path=results_dir / "prediction_samples.png"
    )
    
    # Summary
    print("\n" + "="*70)
    print("📊 TESTING SUMMARY")
    print("="*70)
    print(f"Overall Accuracy: {results['accuracy']:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"\n✅ All results saved to: {results_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
