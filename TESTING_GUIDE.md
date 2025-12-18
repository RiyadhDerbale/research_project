# Model Testing Quick Start Guide

## 📋 Prerequisites

1. **Download trained model from Kaggle**

   - Download `best_model.pth` from your Kaggle notebook
   - Place it in: `models/classification/best_model.pth`

2. **Install dependencies** (if not already installed)

   ```powershell
   pip install scikit-learn seaborn tqdm
   ```

3. **Prepare test images**
   - Test images should be in: `data/classification/test/`
   - Or provide custom image paths

---

## 🚀 Usage Options

### Option 1: Full Test Suite (Comprehensive Analysis)

Run complete evaluation on your test dataset:

```powershell
python scripts/test_model.py
```

**What it does:**

- ✅ Evaluates model on entire test set
- ✅ Generates classification report (precision, recall, F1)
- ✅ Creates confusion matrix visualization
- ✅ Plots ROC curve and calculates AUC
- ✅ Shows prediction samples with confidence
- ✅ Saves all results to `results/testing/`

**Output files:**

- `results/testing/classification_report.txt`
- `results/testing/confusion_matrix.png`
- `results/testing/roc_curve.png`
- `results/testing/prediction_samples.png`

---

### Option 2: Predict Single Image

Test on one image with detailed visualization:

```powershell
python scripts/predict_image.py path/to/your/image.jpg
```

**Example:**

```powershell
python scripts/predict_image.py data/classification/test/cats/cat.1.jpg
```

**Output:**

- Shows image with prediction
- Displays confidence bar chart
- Prints detailed probabilities

---

### Option 3: Predict Multiple Images

Test on multiple images at once:

```powershell
python scripts/predict_image.py image1.jpg image2.jpg image3.jpg
```

**Example:**

```powershell
python scripts/predict_image.py data/classification/test/cats/*.jpg
```

**Output:**

- Grid view of all predictions
- Confidence for each image
- Save with `--output results/my_predictions.png`

---

### Option 4: Interactive Mode (Auto-finds samples)

Run without arguments to test on sample images:

```powershell
python scripts/predict_image.py
```

**What it does:**

- Automatically finds images in test set
- Predicts on first 8 images
- Shows results in grid view

---

## 📊 Example Workflows

### 1. Quick Test (Single Image)

```powershell
# Test on a cat image
python scripts/predict_image.py data/classification/test/cats/cat.1.jpg

# Test on a dog image
python scripts/predict_image.py data/classification/test/dogs/dog.1.jpg
```

### 2. Batch Testing (Multiple Images)

```powershell
# Test on specific images
python scripts/predict_image.py ^
    data/classification/test/cats/cat.1.jpg ^
    data/classification/test/dogs/dog.1.jpg ^
    data/classification/test/cats/cat.2.jpg

# Save results
python scripts/predict_image.py image1.jpg image2.jpg --output results/my_test.png
```

### 3. Full Evaluation (Research Analysis)

```powershell
# Run comprehensive test suite
python scripts/test_model.py

# Check results
cd results/testing
dir
```

### 4. Test with Custom Images

```powershell
# Download any cat/dog image from internet
# Save to: custom_images/

# Test it
python scripts/predict_image.py custom_images/my_cat.jpg
```

---

## 🎯 Expected Output Examples

### Single Image Prediction:

```
🤖 Loading model...
✅ Model loaded on cuda

📸 Processing: cat.1.jpg

======================================================================
🎯 PREDICTION RESULT
======================================================================
Image: cat.1.jpg
Prediction: 🐱 CAT
Confidence: 95.67%

Probabilities:
  🐱 Cat: 95.67%
  🐶 Dog: 4.33%
======================================================================
```

### Full Test Suite:

```
🧪 Model Testing Suite
======================================================================

📥 Loading model from: models/classification/best_model.pth
✅ Model loaded successfully
   Device: cuda
   Best training accuracy: 97.50

📂 Loading test dataset from: data/classification
✅ Test dataset loaded
   Total samples: 2000
   Classes: ['cat', 'dog']

🔍 Evaluating on test set...
Testing: 100%|████████████████████████| 63/63 [00:15<00:00,  4.12it/s]

✅ Evaluation complete!
   Accuracy: 96.85%
   Correct: 1937/2000

📊 Classification Report:
======================================================================
              precision    recall  f1-score   support

         cat     0.9712    0.9650    0.9681      1000
         dog     0.9658    0.9720    0.9689      1000

    accuracy                         0.9685      2000
   macro avg     0.9685    0.9685    0.9685      2000
weighted avg     0.9685    0.9685    0.9685      2000
======================================================================
```

---

## 🔧 Advanced Options

### Use CPU instead of GPU

```powershell
python scripts/predict_image.py image.jpg --device cpu
```

### Specify custom model path

```powershell
python scripts/predict_image.py image.jpg --model path/to/model.pth
```

### Save prediction to specific location

```powershell
python scripts/predict_image.py image.jpg --output results/my_result.png
```

---

## 🐛 Troubleshooting

### Error: Model not found

**Solution:** Download `best_model.pth` from Kaggle and place in `models/classification/`

### Error: No test images found

**Solution:** Ensure images are in `data/classification/test/` or provide explicit paths

### Error: CUDA out of memory

**Solution:** Use CPU mode: `--device cpu`

### Error: Import errors

**Solution:** Install missing packages:

```powershell
pip install scikit-learn seaborn tqdm pillow matplotlib
```

---

## 📚 What's Next?

After testing your model:

1. **XAI Analysis** - Understand model decisions

   ```powershell
   python scripts/xai_analysis.py
   ```

2. **Error Analysis** - Study misclassifications
3. **Model Improvement** - Try different architectures
4. **Deployment** - Create API or web interface

---

## 💡 Tips

- Test on diverse images (different angles, lighting, breeds)
- Check confidence scores - low confidence might indicate edge cases
- Review misclassified images to understand model weaknesses
- Use full test suite for research reports/papers
- Use quick prediction for demos and presentations

---

## 🎓 Understanding the Results

### Confusion Matrix

- **Diagonal** = Correct predictions
- **Off-diagonal** = Misclassifications
- Helps identify which class is harder to predict

### ROC Curve (AUC)

- **AUC close to 1.0** = Excellent model
- **AUC = 0.5** = Random guessing
- Used for comparing models

### Classification Report

- **Precision** = Of all predicted cats, how many were actually cats?
- **Recall** = Of all actual cats, how many did we catch?
- **F1-Score** = Harmonic mean of precision and recall

---

## 📞 Need Help?

If you encounter issues:

1. Check that model file exists in correct location
2. Verify test images are accessible
3. Ensure all dependencies are installed
4. Check error messages for specific issues

Happy Testing! 🎉
