# PyTorch Research Project: Classification & Segmentation with XAI

A production-ready research framework for image classification and segmentation with comprehensive explainability tools.

## Features

- **Multi-task Support**: Classification (CNN/ResNet) and Segmentation (U-Net)
- **XAI Methods**: Integrated Gradients, Grad-CAM, DeepLift, LRP
- **Concept-level Explanations**: TCAV/ACE-style concept extraction and CAV computation
- **Manifold Analysis**: UMAP, PCA, FAISS indexing on embeddings
- **Counterfactual Generation**: Input perturbations and latent-based edits
- **LLM Explanations**: Natural language explanations via OpenAI/local LLMs
- **Uncertainty Estimation**: MC-Dropout and Deep Ensembles
- **Experiment Management**: Hydra configs + Weights & Biases tracking

## Project Structure

```
research_project/
├── src/                    # Main source code
│   ├── models/            # Model architectures
│   │   ├── classification/
│   │   └── segmentation/
│   ├── datasets/          # Dataset classes
│   ├── training/          # Training loops
│   ├── evaluation/        # Metrics and evaluation
│   ├── xai/              # Attribution methods
│   ├── concepts/         # Concept-based explanations
│   ├── manifold/         # Latent space analysis
│   ├── counterfactuals/  # Counterfactual generation
│   ├── llm/              # LLM-based explanations
│   └── utils/            # Utilities
├── configs/              # Hydra configuration files
├── scripts/              # CLI entry points
├── notebooks/            # Jupyter notebooks
├── experiments/          # Experiment outputs
└── data/                 # Dataset storage
```

## Installation

### Using Conda (Recommended)

```bash
conda env create -f env.yaml
conda activate research_project
```

### Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Classification Training

Train a CNN classifier on your dataset:

```bash
python scripts/train_classification.py \
    data.dataset_name=cifar10 \
    model.name=simple_cnn \
    train.epochs=50 \
    train.batch_size=32
```

With Weights & Biases logging:

```bash
python scripts/train_classification.py \
    task=classification \
    wandb.project=my_research \
    wandb.enabled=true
```

### 2. Segmentation Training

Train a U-Net segmentation model:

```bash
python scripts/train_segmentation.py \
    data.dataset_name=custom_segmentation \
    model.name=unet_mini \
    train.epochs=100 \
    train.batch_size=8
```

### 3. XAI - Generate Attribution Maps

Generate Integrated Gradients, Grad-CAM, and other attribution maps:

```bash
python scripts/run_xai.py \
    task=classification \
    model_path=experiments/checkpoints/best_model.pth \
    xai.methods=[integrated_gradients,gradcam,deeplift] \
    data.split=test
```

For segmentation:

```bash
python scripts/run_xai.py \
    task=segmentation \
    model_path=experiments/checkpoints/best_unet.pth \
    xai.methods=[integrated_gradients,gradcam]
```

### 4. Concept-based Explanations (TCAV/ACE)

Extract concepts and compute Concept Activation Vectors:

```bash
python scripts/run_concepts.py \
    task=classification \
    model_path=experiments/checkpoints/best_model.pth \
    concepts.concept_dirs=[concepts/stripes,concepts/dots] \
    concepts.layer=layer3
```

### 5. Manifold Analysis

Perform UMAP/PCA on latent representations and build FAISS index:

```bash
python scripts/run_manifold.py \
    task=classification \
    model_path=experiments/checkpoints/best_model.pth \
    manifold.method=umap \
    manifold.n_components=2 \
    manifold.build_index=true
```

### 6. LLM-based Explanations

Generate natural language explanations for predictions:

```bash
python scripts/run_llm_explanations.py \
    task=classification \
    model_path=experiments/checkpoints/best_model.pth \
    llm.provider=openai \
    llm.model=gpt-4 \
    llm.api_key=your_key_here
```

For local LLM:

```bash
python scripts/run_llm_explanations.py \
    llm.provider=local \
    llm.model_path=path/to/llama2
```

## Configuration

All configurations are managed via Hydra. Edit files in `configs/` to customize:

- `configs/model/`: Model architectures and hyperparameters
- `configs/data/`: Dataset paths and preprocessing
- `configs/train/`: Training hyperparameters
- `configs/xai/`: XAI method settings
- `configs/llm/`: LLM provider and prompt templates
- `configs/task/`: Task-specific settings (classification/segmentation)

Example config override:

```bash
python scripts/train_classification.py \
    model.backbone=resnet18 \
    train.optimizer.lr=0.001 \
    train.scheduler.name=cosine
```

## Dataset Format

### Classification

```
data/classification/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
└── val/
```

### Segmentation

```
data/segmentation/
├── images/
│   ├── train/
│   └── val/
└── masks/
    ├── train/
    └── val/
```

## Development

### Adding a New Model

1. Create model class in `src/models/classification/` or `src/models/segmentation/`
2. Add config in `configs/model/`
3. Register in model factory

### Adding a New XAI Method

1. Implement in `src/xai/attribution.py`
2. Add to method registry
3. Update configs if needed

### Adding a New Concept

1. Prepare concept image directory
2. Add to `configs/concepts/`
3. Run concept extraction

## Notebooks

Explore the following notebooks for examples:

- `notebooks/01_classification_tutorial.ipynb`: Classification walkthrough
- `notebooks/02_segmentation_tutorial.ipynb`: Segmentation walkthrough
- `notebooks/03_xai_visualization.ipynb`: Visualizing attribution maps
- `notebooks/04_concept_analysis.ipynb`: TCAV/ACE analysis
- `notebooks/05_manifold_exploration.ipynb`: Latent space exploration

## Experiments

All experiment outputs are saved to `experiments/`:

```
experiments/
├── logs/              # Training logs
├── checkpoints/       # Model checkpoints
├── xai_outputs/       # Attribution maps
├── concept_results/   # CAVs and concept scores
├── manifold_plots/    # UMAP/PCA visualizations
└── llm_explanations/  # Generated text explanations
```

## Citation

If you use this codebase, please cite:

```bibtex
@software{research_project_2025,
  author = {Your Name},
  title = {PyTorch Research Project: Classification & Segmentation with XAI},
  year = {2025},
  url = {https://github.com/yourusername/research_project}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or issues, please open a GitHub issue or contact: your.email@domain.com
