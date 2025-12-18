# Kaggle Quick Start - 5 Minutes ⚡

Run your training on Kaggle in 5 simple steps!

## 📝 Copy-Paste This Into Kaggle Notebook

### Cell 1: Fix NumPy

```python
!pip install --upgrade numpy
```

**⚠️ STOP! Restart kernel now**: Runtime → Restart runtime

---

### Cell 2: Install Packages

```python
!pip install -q omegaconf wandb
```

---

### Cell 3: Upload/Clone Your Code

```python
# Option A: If your repo is public
!git clone https://github.com/YOUR_USERNAME/research_project.git
%cd research_project

# Option B: Upload files manually to /kaggle/working/
```

---

### Cell 4: Run Training (Simple Version - No Config Files)

```python
!python scripts/train_classification_kaggle.py
```

**OR** if you want to use Hydra configs:

```python
!python scripts/train_classification.py
```

---

### Cell 5: Check Results

```python
!ls -lh /kaggle/working/experiment/checkpoints/
```

---

## 🎯 What You Need on Kaggle

### Required Files:

```
/kaggle/working/
├── src/              # Your source code
│   ├── models/
│   ├── datasets/
│   ├── training/
│   └── utils/
└── scripts/
    └── train_classification_kaggle.py  # Simple version (recommended)
    # OR
    └── train_classification.py         # Hydra version (needs configs/)
```

### Required Dataset:

- Add "Dogs vs Cats" dataset as input

---

## ⚙️ Configuration (Kaggle Defaults)

The `train_classification_kaggle.py` uses these settings:

```python
{
    'model': 'resnet18',
    'data_root': '/kaggle/input/dogs-vs-cats',
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'num_workers': 2,
    'wandb': 'offline mode'
}
```

To change settings, edit the `config_dict` in `train_classification_kaggle.py`.

---

## 🐛 Troubleshooting

| Error                        | Solution                                     |
| ---------------------------- | -------------------------------------------- |
| `numpy.dtype size changed`   | Upgrade numpy, restart kernel                |
| `Module not found`           | Check `sys.path` includes your code          |
| `Config directory not found` | Use `train_classification_kaggle.py` instead |
| `Out of memory`              | Reduce batch_size to 16                      |
| `Hydra error: -f argument`   | Use `train_classification_kaggle.py`         |

---

## 📊 Monitor Training

```python
# View logs
!tail -f /kaggle/working/experiment/train.log

# Check wandb logs (offline)
!ls /kaggle/working/experiment/wandb/
```

---

## 💾 Download Results

```python
from IPython.display import FileLink

# Download best model
FileLink('/kaggle/working/experiment/checkpoints/best_model.pth')
```

---

## ✅ Complete Example

```python
# ============== CELL 1 ==============
!pip install --upgrade numpy
# RESTART KERNEL NOW!

# ============== CELL 2 ==============
!pip install -q omegaconf wandb

# ============== CELL 3 ==============
!git clone https://github.com/YOUR_USERNAME/research_project.git
%cd research_project

# ============== CELL 4 ==============
# Update dataset path if needed
import sys
sys.path.append('/kaggle/working/research_project')

# ============== CELL 5 ==============
!python scripts/train_classification_kaggle.py

# ============== CELL 6 ==============
# Check results
!ls -lh /kaggle/working/experiment/checkpoints/
print("Training complete! 🎉")
```

---

## 🎓 Tips

1. ✅ Always upgrade numpy FIRST, then restart kernel
2. ✅ Use `train_classification_kaggle.py` for notebooks (easier)
3. ✅ Use `train_classification.py` for scripts (needs configs/)
4. ✅ Kaggle gives 30 hours/week GPU - use wisely!
5. ✅ Download your models before closing notebook

---

Happy training! 🚀
