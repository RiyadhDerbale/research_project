# Setting Up Your Cat/Dog Dataset

## ✅ Changes Made

The training script has been updated to use real image data instead of dummy data:

### Modified Files:

1. **`scripts/train_classification.py`**

   - ✅ Replaced `DummyClassificationDataset` with `ImageFolderClassification`
   - ✅ Now loads real images from disk

2. **`configs/classification.yaml`**
   - ✅ Updated `data.root` to point to `data/classification`
   - ✅ Changed `image_size` from 32 to 224 (standard for ResNet)
   - ✅ Set `num_classes` to 2 (cat/dog)

---

## 📁 Required Folder Structure

Organize your cat/dog images like this:

```
data/classification/
├── train/
│   ├── cat/
│   │   ├── cat_001.jpg
│   │   ├── cat_002.jpg
│   │   ├── cat_003.jpg
│   │   └── ... (more cat images)
│   └── dog/
│       ├── dog_001.jpg
│       ├── dog_002.jpg
│       ├── dog_003.jpg
│       └── ... (more dog images)
└── val/
    ├── cat/
    │   ├── cat_val_001.jpg
    │   ├── cat_val_002.jpg
    │   └── ... (validation cat images)
    └── dog/
        ├── dog_val_001.jpg
        ├── dog_val_002.jpg
        └── ... (validation dog images)
```

### Recommended Split:

- **Training**: 80% of your images
- **Validation**: 20% of your images

Example: If you have 1000 cat images and 1000 dog images:

- `train/cat/`: 800 images
- `val/cat/`: 200 images
- `train/dog/`: 800 images
- `val/dog/`: 200 images

---

## 🚀 How to Run Training

### Step 1: Create the directory structure (if it doesn't exist)

```powershell
# Create directories
New-Item -ItemType Directory -Force -Path "data\classification\train\cat"
New-Item -ItemType Directory -Force -Path "data\classification\train\dog"
New-Item -ItemType Directory -Force -Path "data\classification\val\cat"
New-Item -ItemType Directory -Force -Path "data\classification\val\dog"
```

### Step 2: Copy your images into the folders

Move/copy your cat and dog images into the appropriate folders.

### Step 3: Run the training script

```powershell
# Basic training with default settings
python scripts/train_classification.py

# With custom settings
python scripts/train_classification.py `
    train.epochs=100 `
    train.batch_size=16 `
    train.learning_rate=0.0001

# Use ResNet-18 instead of simple CNN
python scripts/train_classification.py `
    model=resnet18 `
    train.epochs=50

# Enable Weights & Biases logging
python scripts/train_classification.py `
    wandb.enabled=true `
    wandb.project=cat_dog_classification
```

---

## 🔧 Configuration Options

Edit `configs/classification.yaml` to change:

```yaml
data:
  root: data/classification # Change if your data is elsewhere
  image_size: 224 # Image size (224 for ResNet)

train:
  epochs: 50 # Number of training epochs
  batch_size: 32 # Batch size (reduce if out of memory)
  learning_rate: 0.001 # Learning rate
  num_workers: 4 # Data loading workers

model:
  name: simple_cnn # or 'resnet18', 'resnet34', 'resnet50'
```

---

## 📊 What Happens During Training

1. **Model loads** your images from `data/classification/train/` and `data/classification/val/`
2. **Classes are auto-detected** from folder names (`cat`, `dog`)
3. **Training loop** runs for specified epochs
4. **Checkpoints saved** to `experiments/classification_exp_TIMESTAMP/checkpoints/`
5. **Logs saved** to `experiments/classification_exp_TIMESTAMP/train.log`

### Output Structure:

```
experiments/
└── classification_exp_20251217_143022/
    ├── checkpoints/
    │   ├── best_model.pth        # Best model (highest val accuracy)
    │   └── last_model.pth         # Last epoch model
    ├── train.log                  # Training logs
    └── .hydra/
        └── config.yaml            # Full config used
```

---

## 🐛 Troubleshooting

### Error: "Dataset path does not exist"

```powershell
# Make sure the directories exist
Test-Path "data\classification\train"
Test-Path "data\classification\val"
```

### Error: "Found 0 files"

- Ensure images are in the correct folders
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Check folder structure matches the expected layout

### CUDA Out of Memory

```powershell
# Reduce batch size
python scripts/train_classification.py train.batch_size=8

# Or use CPU
python scripts/train_classification.py train.device=cpu
```

### Slow Training

```powershell
# Increase workers (careful not to exceed CPU cores)
python scripts/train_classification.py train.num_workers=8
```

---

## 📈 After Training

Once training is complete, you can:

1. **Use the trained model for XAI analysis:**

   ```powershell
   python scripts/run_xai.py `
       task=classification `
       model_path=experiments/classification_exp_TIMESTAMP/checkpoints/best_model.pth
   ```

2. **Run manifold analysis:**

   ```powershell
   python scripts/run_manifold.py `
       model_path=experiments/classification_exp_TIMESTAMP/checkpoints/best_model.pth
   ```

3. **Create concept datasets** from your cat/dog images for TCAV analysis

---

## 💡 Tips

- **Start small**: Test with a few images first to ensure everything works
- **Image quality**: Use consistent image sizes/quality for best results
- **Data augmentation**: Training transforms include random flips, crops, and color jitter
- **Validation**: Keep validation set separate from training (never use val images in training)
- **Monitor training**: Check `train.log` or use `wandb.enabled=true` for real-time monitoring

---

## Next Steps

1. ✅ Organize your cat/dog images into the folder structure
2. ✅ Run training with `python scripts/train_classification.py`
3. ✅ Use the trained model for XAI, manifold, and concept analysis
4. ✅ Experiment with different models (ResNet-18, ResNet-50)
5. ✅ Fine-tune hyperparameters for better accuracy
