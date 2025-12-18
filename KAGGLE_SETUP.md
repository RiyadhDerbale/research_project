# Kaggle Setup Guide

This guide explains how to run the research project on Kaggle.

## 📋 Quick Start

# --- CLONE ---

!git clone https://github.com/RiyadhDerbale/research_project.git
%cd research_project
!pip install -r requirements-kaggle.txt

### 1. Create Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Select "Notebook" (not Script)
4. Turn on GPU: Settings → Accelerator → GPU T4 x2

### 2. Upload Your Code

Option A: **Upload as Dataset**

```bash
# On your local machine, create a zip of the project
zip -r research_project.zip src/ configs/ scripts/
```

- Upload to Kaggle as a dataset
- Add it as input to your notebook

Option B: **Copy Files Directly**

- Create cells and copy your code files
- Recreate directory structure

### 3. Add Your Dataset

1. Find "Dogs vs Cats" dataset on Kaggle
2. Click "Add Data" in your notebook
3. Search for "dogs-vs-cats"
4. Add it to your notebook

### 4. Install Dependencies

In the first cell of your notebook:

```python
# FIX numpy compatibility issues FIRST (very important!)
!pip install --upgrade numpy

# Then install required packages
!pip install -q hydra-core omegaconf captum grad-cam plotly wandb

# Verify installations
import torch
import numpy as np
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

**Important**: Always upgrade numpy first to avoid binary incompatibility errors!

### 5. Setup Project Structure

```python
import sys
from pathlib import Path

# If you uploaded code as a dataset
sys.path.append('/kaggle/input/research-project-code/')

# If you copied files directly
sys.path.append('/kaggle/working/')
```

### 6. Update Configuration

The code automatically detects Kaggle, but verify the dataset path:

```python
# In train_classification.py, line 34:
cfg.data.root = '/kaggle/input/dogs-vs-cats'  # Update if different
```

### 7. Run Training

```python
# Navigate to scripts directory
%cd /kaggle/working/

# Run training
!python scripts/train_classification.py
```

## 🔧 Kaggle-Specific Settings

The code automatically applies these optimizations on Kaggle:

| Setting               | Local                   | Kaggle                       | Reason                       |
| --------------------- | ----------------------- | ---------------------------- | ---------------------------- |
| `data.root`           | `./data/classification` | `/kaggle/input/dogs-vs-cats` | Kaggle dataset location      |
| `experiment.base_dir` | `./experiments`         | `/kaggle/working`            | Kaggle output location       |
| `num_workers`         | 4                       | 2                            | Kaggle multiprocessing limit |
| `pin_memory`          | True                    | False                        | Reduce memory usage          |
| `wandb.mode`          | online                  | offline                      | No API key needed            |

## 📦 Pre-installed Packages on Kaggle

Kaggle comes with many packages pre-installed. You DON'T need to install:

- ✅ numpy, pandas, matplotlib, seaborn
- ✅ torch, torchvision
- ✅ scikit-learn
- ✅ opencv-python
- ✅ pillow, tqdm

Only install what's missing (see `requirements-kaggle.txt`)

## 💾 Dataset Structure on Kaggle

Expected structure for Dogs vs Cats:

```
/kaggle/input/dogs-vs-cats/
├── train/
│   ├── cats/
│   │   ├── cat.0.jpg
│   │   ├── cat.1.jpg
│   │   └── ...
│   └── dogs/
│       ├── dog.0.jpg
│       ├── dog.1.jpg
│       └── ...
└── test/
    ├── cats/
    └── dogs/
```

If your dataset has a different structure, update the path in the code.

## 🐛 Common Kaggle Issues & Solutions

### Issue 1: "numpy.dtype size changed" or Binary Incompatibility Error

```python
# Solution: Upgrade numpy FIRST before installing other packages
!pip install --upgrade numpy
!pip install -q hydra-core omegaconf captum grad-cam plotly wandb

# Then restart the kernel (Runtime → Restart runtime)
```

**Why this happens**: Kaggle's pre-installed packages were compiled against an older numpy version.

### Issue 2: "Module not found" error

```python
# Solution: Add to sys.path
import sys
sys.path.append('/kaggle/input/your-code-dataset/')
```

### Issue 3: "Permission denied" when saving

```python
# Solution: Save to /kaggle/working/ only
exp_dir = Path('/kaggle/working/experiments')
```

### Issue 4: "Out of memory" error

```python
# Solution: Reduce batch size in config
cfg.train.batch_size = 16  # Instead of 32
```

### Issue 5: Wandb login prompt

```python
# Solution: Already handled - runs in offline mode
# No API key needed!
```

### Issue 6: Hydra config not found

```python
# Solution: Use absolute path
@hydra.main(config_path="/kaggle/input/your-dataset/configs",
            config_name="classification")
```

### Issue 7: DataLoader workers crash

```python
# Solution: Already handled - uses 2 workers on Kaggle
cfg.train.num_workers = 2
```

## 📊 Monitoring Training

### View Logs in Real-time

```python
# In a new cell
!tail -f /kaggle/working/experiment/train.log
```

### Check Wandb Logs (Offline)

```python
# Logs saved to:
/kaggle/working/experiment/wandb/
```

### View Saved Models

```python
!ls -lh /kaggle/working/experiment/checkpoints/
```

## 💡 Tips for Kaggle

1. **Fix numpy first**: Always run `!pip install --upgrade numpy` before other packages
2. **Use GPU wisely**: Kaggle gives 30 hours/week of GPU
3. **Save frequently**: Enable auto-save in notebook settings
4. **Use version control**: Save versions of your notebook
5. **Download results**: Download models before closing
6. **Keep output small**: Delete large files you don't need
7. **Restart kernel**: If you get import errors, restart the kernel after installing packages

## 🚀 Running Full Pipeline

```python
# Cell 1: Fix numpy compatibility FIRST
!pip install --upgrade numpy

# Cell 2: Restart kernel (Runtime → Restart runtime in Kaggle)
# Then continue with remaining cells...

# Cell 3: Install other dependencies
!pip install -q hydra-core omegaconf captum grad-cam plotly wandb

# Cell 4: Setup paths
import sys
sys.path.append('/kaggle/working/')

# Cell 5: Copy your code (if not using dataset)
!mkdir -p scripts configs src
# ... copy files ...

# Cell 6: Run training
!python scripts/train_classification.py

# Cell 7: Check results
!ls -lh /kaggle/working/experiment/checkpoints/
```

## 📥 Download Results

```python
# Download trained model
from IPython.display import FileLink
FileLink('/kaggle/working/experiment/checkpoints/best_model.pth')
```

Or use Kaggle's output tab to download files.

## ✅ Checklist Before Running

- [ ] GPU enabled in settings
- [ ] Dataset added as input
- [ ] Dependencies installed
- [ ] Code uploaded (as dataset or files)
- [ ] Paths updated in code
- [ ] Enough GPU hours remaining
- [ ] Auto-save enabled

## 🔗 Useful Links

- [Kaggle Docs](https://www.kaggle.com/docs/notebooks)
- [Kaggle GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)

---

Happy training on Kaggle! 🎉
