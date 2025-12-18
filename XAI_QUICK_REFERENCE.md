# 🔬 XAI Quick Reference Card

## One-Line Commands

### Basic Prediction with XAI

```bash
python scripts/predict_image.py IMAGE.jpg --xai
```

### Save XAI Results

```bash
python scripts/predict_image.py IMAGE.jpg --xai --output results/xai.png
```

### Run XAI Demo

```bash
python scripts/demo_xai.py
```

### Batch XAI Analysis

```bash
python scripts/run_xai.py task=classification model_path=models/classification/best_model.pth
```

## Available XAI Methods

| Method                 | Usage          | Speed  |
| ---------------------- | -------------- | ------ |
| `integrated_gradients` | Most reliable  | Medium |
| `gradient_shap`        | Game-theoretic | Medium |
| `saliency`             | Fastest        | Fast   |
| `deeplift`             | Layer-wise     | Fast   |

## Command-Line Options

```bash
--xai                    # Enable XAI mode
--xai-methods METHOD1 METHOD2  # Specify methods
--device cuda|cpu        # Choose device
--output PATH            # Save location
--model PATH             # Model checkpoint
```

## Quick Examples

```bash
# Single image with XAI
python scripts/predict_image.py cat.jpg --xai

# Custom methods
python scripts/predict_image.py cat.jpg --xai --xai-methods saliency integrated_gradients

# CPU mode (slower but works without GPU)
python scripts/predict_image.py cat.jpg --xai --device cpu

# Save to specific location
python scripts/predict_image.py cat.jpg --xai --output my_analysis.png
```

## Understanding the Output

🔴 **Red/Hot** = High importance → These pixels influenced prediction  
🔵 **Blue/Cold** = Low importance → These pixels had little effect

✅ **Good**: Model focuses on face, ears, body  
❌ **Bad**: Model focuses on background, watermarks

## Common Issues

| Problem          | Solution                          |
| ---------------- | --------------------------------- |
| Missing packages | `pip install -r requirements.txt` |
| Out of memory    | Add `--device cpu`                |
| Too slow         | Use `--xai-methods saliency`      |
| No visualization | Check `results/` folder           |

## Quick Integration

```python
from src.xai import AttributionEngine

# Setup
xai = AttributionEngine(model, device="cuda", task="classification")

# Get attributions
attrs = xai.get_multiple_attributions(
    image,
    methods=['integrated_gradients', 'saliency'],
    target=class_id
)

# Visualize
from src.xai import visualize_multiple_attributions
visualize_multiple_attributions(image, attrs, save_path="xai.png")
```

## Documentation

- 📖 Full Guide: `QUICKSTART_XAI.md`
- 📊 Summary: `XAI_INTEGRATION_SUMMARY.md`
- 🧪 Demo: `python scripts/demo_xai.py`
- 💻 Code: `src/xai/attribution.py`

---

**Get Started:** `python scripts/predict_image.py YOUR_IMAGE.jpg --xai` 🚀
