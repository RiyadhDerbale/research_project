# 🔬 XAI (Explainable AI) Quickstart Guide

This guide shows you how to use Explainable AI methods to understand your model's predictions.

## What is XAI?

Explainable AI (XAI) methods help you understand **why** your model made a specific prediction by showing which parts of the input image were most important for the decision.

## Available XAI Methods

Our framework includes several attribution methods from the Captum library:

1. **Integrated Gradients** - Accumulates gradients along a path from baseline to input
2. **Gradient SHAP** - Uses gradient information with Shapley values
3. **Saliency Maps** - Shows where the gradient is strongest
4. **DeepLift** - Compares activations to reference values
5. **Grad-CAM** - Class activation mapping for convolutional layers

## Quick Start

### 1. Basic Prediction (No XAI)

```bash
python scripts/predict_image.py data/classification/test/cats/cat.1.jpg
```

### 2. Prediction with XAI Attribution Maps

```bash
python scripts/predict_image.py data/classification/test/cats/cat.1.jpg --xai
```

This will:

- Make a prediction
- Generate attribution maps using multiple XAI methods
- Show heatmaps highlighting important image regions
- Display overlay visualizations

### 3. Save XAI Results

```bash
python scripts/predict_image.py path/to/image.jpg --xai --output results/xai_analysis.png
```

### 4. Specify Custom XAI Methods

```bash
python scripts/predict_image.py image.jpg --xai --xai-methods integrated_gradients saliency
```

## Using the Standalone XAI Script

For more advanced XAI analysis on your trained models:

```bash
# Classification XAI
python scripts/run_xai.py task=classification model_path=models/classification/best_model.pth

# Segmentation XAI
python scripts/run_xai.py task=segmentation model_path=models/segmentation/best_model.pth
```

## Understanding XAI Output

### Attribution Heatmaps

- **Red/Hot** colors = High importance (these pixels strongly influenced the prediction)
- **Blue/Cold** colors = Low importance (these pixels had little influence)

### Overlay Visualizations

- Shows the attribution map overlayed on the original image
- Makes it easy to see which parts of the cat/dog the model focused on

### What to Look For

✅ **Good Model Behavior:**

- Focuses on relevant features (face, ears, body)
- Consistent across different methods
- Makes intuitive sense

❌ **Potential Issues:**

- Focuses on background or irrelevant features
- Very different results across methods
- Highlights unexpected regions

## Example Use Cases

### 1. Debug Misclassifications

```bash
# If model predicts wrong, use XAI to see what it focused on
python scripts/predict_image.py misclassified_image.jpg --xai
```

### 2. Verify Model is Learning Correct Features

```bash
# Check that model looks at animal features, not background
python scripts/predict_image.py data/classification/test/cats/*.jpg --xai
```

### 3. Compare Different Model Predictions

```bash
# Use different model checkpoints and compare their attention
python scripts/predict_image.py image.jpg --model models/model_v1.pth --xai
python scripts/predict_image.py image.jpg --model models/model_v2.pth --xai
```

## Advanced Configuration

Edit `configs/xai.yaml` to customize:

- Number of integration steps
- Baseline selection
- Attribution aggregation methods
- Visualization settings

## Integration with Your Code

You can use the XAI engine in your own scripts:

```python
from src.xai import AttributionEngine, visualize_attribution
from src.models import build_model

# Load your model
model = build_model(task="classification", model_name="resnet18", num_classes=2)
model.load_state_dict(torch.load("path/to/model.pth"))

# Initialize XAI engine
xai_engine = AttributionEngine(model=model, device="cuda", task="classification")

# Get attributions
attributions = xai_engine.get_multiple_attributions(
    image_tensor,
    methods=['integrated_gradients', 'gradient_shap', 'saliency'],
    target=predicted_class
)

# Visualize
visualize_attribution(image_tensor, attributions['integrated_gradients'])
```

## Tips for Better XAI Analysis

1. **Use multiple methods** - Different methods can reveal different aspects
2. **Check consistency** - If all methods agree, the model is confident about important features
3. **Analyze failures** - XAI is most valuable when predictions are wrong
4. **Compare classes** - Generate attributions for both cat and dog predictions
5. **Document findings** - Save XAI outputs for presentations and papers

## Requirements

Make sure you have the required packages:

```bash
pip install captum opencv-python
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Next Steps

- 📊 Run batch XAI analysis on test set
- 🎓 Use XAI insights to improve your model
- 📝 Include XAI visualizations in your research paper
- 🔍 Explore other XAI methods (LIME, SHAP values)
- 🧪 Try XAI on segmentation models

## Common Issues

### Issue: "Captum not installed"

```bash
pip install captum
```

### Issue: "OpenCV not installed"

```bash
pip install opencv-python
```

### Issue: Out of memory with XAI

- Use CPU instead: `--device cpu`
- Reduce image size in config
- Process images one at a time

### Issue: XAI takes too long

- Reduce `n_steps` parameter for Integrated Gradients
- Use faster methods like Saliency
- Use GPU if available

## Learn More

- [Captum Documentation](https://captum.ai/)
- [XAI Methods Comparison](https://captum.ai/tutorials)
- [Interpretability in Deep Learning](https://christophm.github.io/interpretable-ml-book/)

---

Happy Explaining! 🔬✨
