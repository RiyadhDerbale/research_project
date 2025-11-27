# Quick Start Guide

## Installation

### Option 1: Using Conda (Recommended)

```bash
cd research_project
conda env create -f env.yaml
conda activate research_project
```

### Option 2: Using pip

```bash
cd research_project
pip install -r requirements.txt
```

## Running Experiments

### 1. Classification Training

Train a simple CNN on dummy data:

```bash
python scripts/train_classification.py
```

Train with custom config:

```bash
python scripts/train_classification.py \
    model.name=resnet18 \
    train.epochs=100 \
    train.batch_size=64 \
    wandb.enabled=true
```

### 2. Segmentation Training

Train U-Net:

```bash
python scripts/train_segmentation.py
```

With custom settings:

```bash
python scripts/train_segmentation.py \
    model.name=unet_mini \
    train.epochs=150 \
    data.num_classes=5
```

### 3. Generate XAI Attribution Maps

After training, generate explanations:

```bash
python scripts/run_xai.py \
    task=classification \
    model_path=experiments/YOUR_EXP_DIR/checkpoints/best_model.pth \
    xai.methods=[integrated_gradients,gradcam,saliency]
```

For segmentation:

```bash
python scripts/run_xai.py \
    task=segmentation \
    model_path=experiments/YOUR_EXP_DIR/checkpoints/best_model.pth
```

### 4. Manifold Analysis

Visualize latent space with UMAP:

```bash
python scripts/run_manifold.py \
    model_path=experiments/YOUR_EXP_DIR/checkpoints/best_model.pth \
    manifold.method=umap \
    manifold.n_components=2 \
    num_samples=1000
```

With PCA:

```bash
python scripts/run_manifold.py \
    manifold.method=pca \
    manifold.build_index=true
```

### 5. LLM-based Explanations

Generate natural language explanations (requires OpenAI API key):

```bash
export OPENAI_API_KEY=your_key_here

python scripts/run_llm_explanations.py \
    model_path=experiments/YOUR_EXP_DIR/checkpoints/best_model.pth \
    llm.provider=openai \
    llm.model=gpt-4 \
    num_samples=5
```

### 6. Concept Analysis (TCAV)

```bash
python scripts/run_concepts.py \
    model_path=experiments/YOUR_EXP_DIR/checkpoints/best_model.pth \
    concepts.concept_dirs=[data/concepts/concept1,data/concepts/concept2]
```

## Using Your Own Data

### Classification

Organize your data like this:

```
data/classification/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    └── ...
```

Update `src/datasets/classification.py` to use `ImageFolderClassification` instead of dummy data.

### Segmentation

Organize your data:

```
data/segmentation/
├── images/
│   ├── train/
│   └── val/
└── masks/
    ├── train/
    └── val/
```

Update `src/datasets/segmentation.py` to use `SegmentationDataset`.

## Jupyter Notebooks

Start Jupyter:

```bash
jupyter notebook notebooks/
```

Explore the tutorial notebooks:

- `01_classification_tutorial.ipynb` - Classification walkthrough
- `02_segmentation_tutorial.ipynb` - Segmentation walkthrough
- `03_xai_visualization.ipynb` - XAI visualization examples
- `04_concept_analysis.ipynb` - TCAV concept analysis
- `05_manifold_exploration.ipynb` - Latent space exploration

## Weights & Biases Integration

Enable experiment tracking:

```bash
wandb login

python scripts/train_classification.py \
    wandb.enabled=true \
    wandb.project=my_research_project
```

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size: `train.batch_size=16`
- Use smaller model: `model.name=simple_cnn`

### Slow Training

- Increase `train.num_workers=8`
- Enable mixed precision (TODO: implement AMP)

### Import Errors

- Ensure you're in the project root directory
- Check that all dependencies are installed
- Activate the conda environment

## Next Steps

1. Replace dummy datasets with your real data
2. Tune hyperparameters for your task
3. Add custom models in `src/models/`
4. Extend XAI methods in `src/xai/`
5. Create custom LLM prompts in `src/llm/`

## Support

For issues or questions:

- Check STRUCTURE.md for code organization
- Review example notebooks
- Open an issue on GitHub
