"""
Quick Demo: XAI Attribution Methods for Cat/Dog Classification

This script demonstrates how to use XAI methods to explain model predictions.
Run this to see attribution maps in action!
"""

import torch
from pathlib import Path
import sysq
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import build_model
from src.datasets import get_classification_transforms
from src.xai import AttributionEngine, visualize_multiple_attributions


def demo_xai():
    """Demo XAI attribution methods"""
    
    print("=" * 70)
    print("🔬 XAI Attribution Demo for Cat/Dog Classification")
    print("=" * 70)
    
    # 1. Load model
    print("\n1️⃣ Loading model...")
    model_path = PROJECT_ROOT / "models/classification/best_model.pth"
    
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("\nPlease download or train a model first:")
        print("  python scripts/train_classification.py")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = build_model(
        task="classification",
        model_name="resnet18",
        num_classes=2
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("   ✅ Model loaded")
    
    # 2. Initialize XAI engine
    print("\n2️⃣ Initializing XAI Attribution Engine...")
    
    xai_engine = AttributionEngine(
        model=model,
        device=device,
        task="classification"
    )
    print("   ✅ XAI engine ready")
    print(f"   Available methods: {list(xai_engine.methods.keys())}")
    
    # 3. Find a test image
    print("\n3️⃣ Finding test image...")
    test_dir = PROJECT_ROOT / "data/classification/test"
    
    test_image = None
    if test_dir.exists():
        # Try to find a cat image
        for subdir in test_dir.iterdir():
            if subdir.is_dir():
                images = list(subdir.glob("*.jpg"))
                if images:
                    test_image = images[0]
                    break
    
    if test_image is None:
        print("   ❌ No test images found")
        print(f"   Expected location: {test_dir}")
        print("\n   Please add some test images or specify a custom path:")
        print("   python scripts/predict_image.py YOUR_IMAGE.jpg --xai")
        return
    
    print(f"   📸 Using: {test_image.name}")
    
    # 4. Load and preprocess image
    print("\n4️⃣ Loading and preprocessing image...")
    from PIL import Image
    
    img = Image.open(test_image).convert('RGB')
    transform = get_classification_transforms('val', 224)
    img_tensor = transform(img).to(device)
    print("   ✅ Image preprocessed")
    
    # 5. Make prediction
    print("\n5️⃣ Making prediction...")
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item() * 100
    
    class_names = ['cat', 'dog']
    class_emojis = ['🐱', '🐶']
    
    print(f"   Prediction: {class_emojis[pred_class]} {class_names[pred_class].upper()}")
    print(f"   Confidence: {confidence:.2f}%")
    print(f"   Probabilities:")
    print(f"     🐱 Cat: {probs[0][0].item() * 100:.2f}%")
    print(f"     🐶 Dog: {probs[0][1].item() * 100:.2f}%")
    
    # 6. Compute attribution maps
    print("\n6️⃣ Computing XAI attribution maps...")
    print("   This may take a few moments...")
    
    methods = ['integrated_gradients', 'gradient_shap', 'saliency']
    attributions = {}
    
    for method in methods:
        print(f"   🔄 Computing {method}...")
        try:
            attr = xai_engine.get_attribution(
                img_tensor,
                method=method,
                target=pred_class
            )
            attributions[method] = attr
            print(f"   ✅ {method} complete")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    if not attributions:
        print("\n❌ No attributions were computed successfully")
        return
    
    # 7. Visualize results
    print("\n7️⃣ Generating visualization...")
    output_path = PROJECT_ROOT / "results/xai_demo.png"
    output_path.parent.mkdir(exist_ok=True)
    
    visualize_multiple_attributions(
        img_tensor,
        attributions,
        save_path=str(output_path)
    )
    
    print(f"   ✅ Saved visualization to: {output_path}")
    
    # 8. Summary
    print("\n" + "=" * 70)
    print("🎉 XAI Demo Complete!")
    print("=" * 70)
    print("\n📊 What we did:")
    print("  1. Loaded a trained Cat/Dog classifier")
    print("  2. Made a prediction on a test image")
    print("  3. Generated attribution maps using multiple XAI methods")
    print("  4. Visualized which image regions influenced the prediction")
    
    print("\n🔬 Understanding the Results:")
    print("  • Red/Hot regions = High importance (influenced prediction)")
    print("  • Blue/Cold regions = Low importance (little influence)")
    print("  • Different methods may highlight different features")
    
    print("\n💡 Next Steps:")
    print("  • Try on your own images:")
    print("    python scripts/predict_image.py YOUR_IMAGE.jpg --xai")
    print("\n  • Analyze misclassifications:")
    print("    python scripts/predict_image.py wrong_prediction.jpg --xai")
    print("\n  • Batch process test set:")
    print("    python scripts/run_xai.py task=classification")
    
    print("\n📖 Learn more: See QUICKSTART_XAI.md for detailed guide")
    print("=" * 70)


if __name__ == "__main__":
    try:
        demo_xai()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: Make sure you have:")
        print("  1. A trained model in models/classification/best_model.pth")
        print("  2. Test images in data/classification/test/")
        print("  3. Required packages: pip install -r requirements.txt")
