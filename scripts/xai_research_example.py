"""
Example: Using XAI for Model Analysis and Research

This script shows practical examples of using XAI to:
1. Analyze model predictions
2. Compare different models
3. Identify biases or issues
4. Generate figures for papers/presentations

Author: Research Project Team
"""

import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import build_model
from src.datasets import get_classification_transforms
from src.xai import AttributionEngine


def analyze_prediction(model_path, image_path, output_dir="results/xai_research"):
    """
    Comprehensive XAI analysis of a single prediction
    
    This shows:
    - What the model predicted
    - Which features it focused on
    - Consistency across XAI methods
    - Potential issues or biases
    """
    
    print("\n" + "="*70)
    print("📊 XAI Research Analysis")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(task="classification", model_name="resnet18", num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = get_classification_transforms('val', 224)
    img_tensor = transform(img).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item() * 100
    
    class_names = ['cat', 'dog']
    
    print(f"\n✅ Prediction: {class_names[pred_class].upper()} ({confidence:.2f}%)")
    print(f"   Cat probability: {probs[0][0].item()*100:.2f}%")
    print(f"   Dog probability: {probs[0][1].item()*100:.2f}%")
    
    # Initialize XAI
    xai = AttributionEngine(model, device, task="classification")
    
    # Compute attributions for BOTH classes (not just predicted)
    print("\n🔬 Computing attributions for both classes...")
    
    results = {}
    for class_idx, class_name in enumerate(class_names):
        print(f"\n   Analyzing attributions for: {class_name}")
        attrs = xai.get_multiple_attributions(
            img_tensor,
            methods=['integrated_gradients', 'gradient_shap', 'saliency'],
            target=class_idx
        )
        results[class_name] = attrs
    
    # Visualize comparison
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(3, 7, figsize=(24, 12))
    
    # Column 0: Original image (repeated 3 times)
    for i in range(3):
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
    
    # Rows = different XAI methods
    # Columns = [Original, Cat-IG, Cat-GS, Cat-Sal, Dog-IG, Dog-GS, Dog-Sal]
    methods = ['integrated_gradients', 'gradient_shap', 'saliency']
    method_names = ['Integrated Gradients', 'Gradient SHAP', 'Saliency']
    
    for row_idx, (method, method_name) in enumerate(zip(methods, method_names)):
        
        # Cat attributions
        for col_idx, class_name in enumerate(['cat', 'dog']):
            attr = results[class_name][method]
            
            # Normalize
            attr_norm = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
            
            # Plot heatmap
            ax_idx = 1 + col_idx * 3
            im = axes[row_idx, ax_idx].imshow(attr_norm, cmap='hot')
            axes[row_idx, ax_idx].set_title(f"{class_name.upper()}: {method_name}\nHeatmap", 
                                            fontsize=9, fontweight='bold')
            axes[row_idx, ax_idx].axis('off')
            plt.colorbar(im, ax=axes[row_idx, ax_idx], fraction=0.046, pad=0.04)
            
            # Plot overlay
            import cv2
            attr_resized = cv2.resize(attr_norm, (img.size[0], img.size[1]))
            attr_uint8 = (attr_resized * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(attr_uint8, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            img_np = np.array(img)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
            
            ax_idx = 2 + col_idx * 3
            axes[row_idx, ax_idx].imshow(overlay)
            axes[row_idx, ax_idx].set_title(f"{class_name.upper()}: {method_name}\nOverlay", 
                                           fontsize=9, fontweight='bold')
            axes[row_idx, ax_idx].axis('off')
    
    # Overall title
    pred_emoji = '🐱' if pred_class == 0 else '🐶'
    fig.suptitle(f"🔬 XAI Analysis: Predicted {pred_emoji} {class_names[pred_class].upper()} ({confidence:.1f}%)\n"
                 f"Comparing attributions for both classes across multiple XAI methods",
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    output_path = output_dir / f"xai_analysis_{Path(image_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comprehensive analysis to: {output_path}")
    
    plt.show()
    
    # Generate insights
    print("\n" + "="*70)
    print("🔍 INSIGHTS & INTERPRETATION")
    print("="*70)
    
    print("\n1. Prediction Quality:")
    if confidence > 90:
        print(f"   ✅ High confidence ({confidence:.1f}%) - Model is very certain")
    elif confidence > 70:
        print(f"   ⚠️ Medium confidence ({confidence:.1f}%) - Some uncertainty")
    else:
        print(f"   ❌ Low confidence ({confidence:.1f}%) - Model is uncertain")
    
    print("\n2. Attribution Consistency:")
    print("   Compare the heatmaps across different methods:")
    print("   - Similar patterns → Robust, consistent explanations")
    print("   - Different patterns → Model uncertainty or complexity")
    
    print("\n3. Class-specific Features:")
    print("   Compare Cat vs Dog attributions:")
    print("   - Cat attrs should highlight cat-specific features (whiskers, eyes)")
    print("   - Dog attrs should highlight dog-specific features (snout, ears)")
    
    print("\n4. Potential Issues to Check:")
    print("   ❌ Focus on background → Model may use spurious correlations")
    print("   ❌ Focus on watermarks → Dataset bias")
    print("   ❌ Uniform attribution → Model may be guessing")
    print("   ✅ Focus on animal features → Good generalization")
    
    print("\n" + "="*70)
    
    return results


def compare_models(model_paths, image_path, output_dir="results/xai_research"):
    """
    Compare how different model checkpoints explain their predictions
    
    Useful for:
    - Comparing different training epochs
    - Evaluating different architectures
    - Understanding model improvements
    """
    
    print("\n" + "="*70)
    print("🔄 Comparing Multiple Models")
    print("="*70)
    
    # TODO: Implement model comparison
    # Load multiple models, generate XAI for same image, compare
    print("\n💡 TIP: Implement this to compare model checkpoints")
    print("   Use case: Compare epoch 10 vs epoch 50 attention")
    

if __name__ == "__main__":
    """
    Example usage for research
    """
    
    # Configuration
    MODEL_PATH = PROJECT_ROOT / "models/classification/best_model.pth"
    TEST_IMAGE = PROJECT_ROOT / "data/classification/test/cats/cat.1.jpg"
    
    # Check if files exist
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("\n   Please train or download a model first:")
        print("   python scripts/train_classification.py")
        sys.exit(1)
    
    if not TEST_IMAGE.exists():
        print(f"❌ Test image not found: {TEST_IMAGE}")
        print("\n   Please add test images to: data/classification/test/")
        print("   Or specify a custom image path in the script")
        sys.exit(1)
    
    # Run analysis
    print("\n🚀 Starting XAI Research Analysis...")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Image: {TEST_IMAGE.name}")
    
    try:
        results = analyze_prediction(MODEL_PATH, TEST_IMAGE)
        
        print("\n" + "="*70)
        print("🎉 Analysis Complete!")
        print("="*70)
        print("\n📁 Check the results/ directory for generated figures")
        print("📊 Use these visualizations in your papers/presentations")
        print("🔬 Analyze the attribution patterns to understand your model")
        
        print("\n💡 Next Steps:")
        print("   1. Run analysis on misclassified examples")
        print("   2. Compare attributions across different model versions")
        print("   3. Identify systematic biases or issues")
        print("   4. Generate figures for your research paper")
        
        print("\n📚 Research Questions to Explore:")
        print("   • Does the model focus on relevant features?")
        print("   • Are attributions consistent across methods?")
        print("   • What causes misclassifications?")
        print("   • How do attributions change during training?")
        print("   • Are there dataset biases the model learned?")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Common issues:")
        print("   • Missing dependencies: pip install -r requirements.txt")
        print("   • Out of memory: Try --device cpu")
        print("   • Model mismatch: Check model architecture matches checkpoint")
