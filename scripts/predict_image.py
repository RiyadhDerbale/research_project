"""
Interactive Image Prediction Script
Usage: python scripts/predict_image.py [image_path]
"""

import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import build_model
from src.datasets import get_classification_transforms
from src.xai import AttributionEngine, visualize_attribution, visualize_multiple_attributions


class ImagePredictor:
    """Simple image predictor for trained models"""
    
    def __init__(self, model_path, device='cuda', enable_xai=False):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            enable_xai: Whether to enable XAI attribution methods
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.enable_xai = enable_xai
        
        print("🤖 Loading model...")
        
        # Load model
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
        
        print(f"✅ Model loaded on {self.device}")
        
        # Get transforms
        self.transform = get_classification_transforms('val', 224)
        
        # Class names
        self.class_names = ['cat', 'dog']
        self.class_emojis = ['🐱', '🐶']
        
        # Initialize XAI engine if enabled
        self.attribution_engine = None
        if enable_xai:
            print("🔬 Initializing XAI attribution engine...")
            self.attribution_engine = AttributionEngine(
                model=self.model,
                device=self.device,
                task="classification"
            )
            print("✅ XAI engine ready")
    
    def predict(self, image_path):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with prediction results
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item() * 100
        
        result = {
            'predicted_class': self.class_names[pred_class],
            'predicted_emoji': self.class_emojis[pred_class],
            'confidence': confidence,
            'probabilities': {
                'cat': probs[0][0].item() * 100,
                'dog': probs[0][1].item() * 100
            },
            'image': img,
            'image_tensor': img_tensor.squeeze(0)
        }
        
        # Compute XAI attributions if enabled
        if self.enable_xai and self.attribution_engine:
            result['attributions'] = self.attribution_engine.get_multiple_attributions(
                img_tensor.squeeze(0),
                methods=['integrated_gradients', 'gradient_shap', 'saliency'],
                target=pred_class
            )
        
        return result
    
    def predict_and_display(self, image_path, save_path=None):
        """Predict and display result with visualization"""
        result = self.predict(image_path)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Display image
        ax1.imshow(result['image'])
        ax1.set_title(f"Input Image: {Path(image_path).name}", fontsize=12)
        ax1.axis('off')
        
        # Display prediction
        bars = ax2.barh(self.class_names, 
                       [result['probabilities']['cat'], result['probabilities']['dog']],
                       color=['#FF6B6B', '#4ECDC4'])
        ax2.set_xlabel('Confidence (%)', fontsize=12)
        ax2.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 2, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Add prediction text
        pred_text = f"{result['predicted_emoji']} Prediction: {result['predicted_class'].upper()}"
        pred_text += f"\nConfidence: {result['confidence']:.2f}%"
        fig.text(0.5, 0.95, pred_text, ha='center', fontsize=14, 
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved result to: {save_path}")
        
        plt.show()
        
        return result
    
    def predict_with_xai(self, image_path, save_path=None):
        """Predict and display result with XAI attribution maps"""
        if not self.enable_xai:
            print("⚠️ XAI is not enabled. Use --xai flag to enable it.")
            return self.predict_and_display(image_path, save_path)
        
        result = self.predict(image_path)
        
        # Check if attributions were computed
        if 'attributions' not in result:
            print("❌ No attributions found in result")
            return result
        
        attributions = result['attributions']
        n_methods = len(attributions)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(3, n_methods + 1, hspace=0.3, wspace=0.3)
        
        # Row 1: Original image and prediction bars
        ax_img = fig.add_subplot(gs[0, :2])
        ax_img.imshow(result['image'])
        ax_img.set_title(f"Input Image: {Path(image_path).name}", fontsize=14, fontweight='bold')
        ax_img.axis('off')
        
        ax_pred = fig.add_subplot(gs[0, 2:])
        bars = ax_pred.barh(self.class_names, 
                           [result['probabilities']['cat'], result['probabilities']['dog']],
                           color=['#FF6B6B', '#4ECDC4'])
        ax_pred.set_xlabel('Confidence (%)', fontsize=12)
        ax_pred.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
        ax_pred.set_xlim(0, 100)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax_pred.text(width + 2, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}%',
                        ha='left', va='center', fontsize=11, fontweight='bold')
        
        # Add prediction banner
        pred_text = f"{result['predicted_emoji']} Prediction: {result['predicted_class'].upper()} | Confidence: {result['confidence']:.2f}%"
        fig.text(0.5, 0.96, pred_text, ha='center', fontsize=16, 
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Row 2: Attribution heatmaps
        for idx, (method_name, attr_map) in enumerate(attributions.items()):
            ax = fig.add_subplot(gs[1, idx])
            im = ax.imshow(attr_map, cmap='hot')
            ax.set_title(f"{method_name.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 3: Overlay visualizations
        image_np = np.array(result['image'])
        for idx, (method_name, attr_map) in enumerate(attributions.items()):
            ax = fig.add_subplot(gs[2, idx])
            
            # Normalize attribution
            attr_norm = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
            attr_resized = np.array(Image.fromarray((attr_norm * 255).astype(np.uint8)).resize(
                (image_np.shape[1], image_np.shape[0])
            ))
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(attr_resized, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay
            overlay = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
            
            ax.imshow(overlay)
            ax.set_title(f"Overlay: {method_name.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle("🔬 XAI Explainability Analysis", fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved XAI result to: {save_path}")
        
        plt.show()
        
        return result
    
    def predict_batch(self, image_paths, save_path=None):
        """Predict on multiple images and display results"""
        num_images = len(image_paths)
        cols = 4
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        results = []
        
        for idx, img_path in enumerate(image_paths):
            try:
                # Make prediction
                result = self.predict(img_path)
                results.append({
                    'image_path': img_path,
                    **result
                })
                
                # Display
                if idx < len(axes):
                    axes[idx].imshow(result['image'])
                    
                    title = f"{result['predicted_emoji']} {result['predicted_class'].upper()}\n"
                    title += f"Confidence: {result['confidence']:.1f}%\n"
                    title += f"{Path(img_path).name}"
                    
                    axes[idx].set_title(title, fontsize=9, fontweight='bold')
                    axes[idx].axis('off')
            
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
        
        # Hide unused subplots
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved batch results to: {save_path}")
        
        plt.show()
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Predict dog or cat from images')
    parser.add_argument('images', nargs='*', help='Path(s) to image file(s)')
    parser.add_argument('--model', type=str, 
                       default='models/classification/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Path to save result image')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--xai', action='store_true',
                       help='Enable XAI attribution methods (Integrated Gradients, Gradient SHAP, Saliency)')
    parser.add_argument('--xai-methods', nargs='+', 
                       default=['integrated_gradients', 'gradient_shap', 'saliency'],
                       help='XAI methods to use')
    
    args = parser.parse_args()
    
    # Model path
    model_path = PROJECT_ROOT / args.model
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("\nPlease:")
        print("1. Download best_model.pth from Kaggle")
        print("2. Place it in: models/classification/")
        return
    
    # Initialize predictor
    predictor = ImagePredictor(model_path, device=args.device, enable_xai=args.xai)
    
    # If XAI is enabled, show info
    if args.xai:
        print("\n" + "="*70)
        print("🔬 XAI Mode Enabled")
        print("="*70)
        print(f"Methods: {', '.join(args.xai_methods)}")
        print("This will generate attribution maps showing which parts")
        print("of the image contributed most to the prediction.")
        print("="*70 + "\n")
    
    # If no images provided, use interactive mode
    if not args.images:
        print("\n" + "="*70)
        print("🎯 Interactive Prediction Mode")
        print("="*70)
        print("\nExamples:")
        print("  python scripts/predict_image.py path/to/image.jpg")
        print("  python scripts/predict_image.py image1.jpg image2.jpg image3.jpg")
        print("  python scripts/predict_image.py data/classification/test/cats/cat.1.jpg")
        print("\n🔬 XAI Examples (Explainable AI):")
        print("  python scripts/predict_image.py path/to/image.jpg --xai")
        print("  python scripts/predict_image.py image.jpg --xai --output results/xai_explanation.png")
        print("\n" + "="*70)
        
        # Try to find some sample images
        sample_dir = PROJECT_ROOT / "data/classification/test"
        if sample_dir.exists():
            # Get first 4 images from test set
            sample_images = []
            for subdir in sample_dir.iterdir():
                if subdir.is_dir():
                    images = list(subdir.glob("*.jpg"))[:2]
                    sample_images.extend(images)
            
            if sample_images:
                print(f"\n🖼️ Found {len(sample_images)} sample images in test set")
                print("   Running predictions on samples...")
                
                results = predictor.predict_batch(
                    sample_images[:8],
                    save_path=args.output if args.output else PROJECT_ROOT / "results/sample_predictions.png"
                )
                
                # Print results
                print("\n📊 Results:")
                for i, res in enumerate(results, 1):
                    print(f"   {i}. {Path(res['image_path']).name}")
                    print(f"      → {res['predicted_emoji']} {res['predicted_class']} ({res['confidence']:.1f}%)")
        else:
            print("\n⚠️ No test images found.")
            print(f"   Expected location: {sample_dir}")
        
        return
    
    # Process provided images
    print("\n" + "="*70)
    print("🔮 Making Predictions")
    print("="*70)
    
    if len(args.images) == 1:
        # Single image - detailed view
        image_path = Path(args.images[0])
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"\n📸 Processing: {image_path.name}")
        
        # Use XAI method if enabled
        if args.xai:
            result = predictor.predict_with_xai(
                str(image_path),
                save_path=args.output
            )
        else:
            result = predictor.predict_and_display(
                str(image_path),
                save_path=args.output
            )
        
        # Print result
        print("\n" + "="*70)
        print("🎯 PREDICTION RESULT")
        print("="*70)
        print(f"Image: {image_path.name}")
        print(f"Prediction: {result['predicted_emoji']} {result['predicted_class'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"\nProbabilities:")
        print(f"  🐱 Cat: {result['probabilities']['cat']:.2f}%")
        print(f"  🐶 Dog: {result['probabilities']['dog']:.2f}%")
        
        if args.xai and 'attributions' in result:
            print(f"\n🔬 XAI Attributions Generated:")
            for method in result['attributions'].keys():
                print(f"  ✓ {method.replace('_', ' ').title()}")
        
        print("="*70)
    
    else:
        # Multiple images - batch view
        valid_images = [img for img in args.images if Path(img).exists()]
        
        if not valid_images:
            print("❌ No valid images found")
            return
        
        print(f"\n📸 Processing {len(valid_images)} images...")
        results = predictor.predict_batch(
            valid_images,
            save_path=args.output
        )
        
        # Print results
        print("\n" + "="*70)
        print("🎯 PREDICTION RESULTS")
        print("="*70)
        for i, res in enumerate(results, 1):
            print(f"{i}. {Path(res['image_path']).name}")
            print(f"   → {res['predicted_emoji']} {res['predicted_class']} ({res['confidence']:.1f}%)")
        print("="*70)


if __name__ == "__main__":
    main()
