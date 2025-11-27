# Project Structure

```
research_project/
│
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── env.yaml                          # Conda environment
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── models/                       # Model architectures
│   │   ├── __init__.py
│   │   ├── classification/
│   │   │   ├── __init__.py
│   │   │   ├── simple_cnn.py        # Simple CNN classifier
│   │   │   └── resnet.py            # ResNet classifier
│   │   └── segmentation/
│   │       ├── __init__.py
│   │       ├── unet.py              # U-Net architecture
│   │       └── simple_segmenter.py  # Simple FCN segmenter
│   │
│   ├── datasets/                     # Dataset classes
│   │   ├── __init__.py
│   │   ├── classification.py        # Classification datasets
│   │   └── segmentation.py          # Segmentation datasets
│   │
│   ├── training/                     # Training loops
│   │   ├── __init__.py
│   │   ├── classification_trainer.py
│   │   └── segmentation_trainer.py
│   │
│   ├── evaluation/                   # Metrics & evaluation
│   │   ├── __init__.py
│   │   └── metrics.py               # Classification & segmentation metrics
│   │
│   ├── xai/                         # XAI methods
│   │   ├── __init__.py
│   │   ├── attribution.py           # IG, Grad-CAM, DeepLift, LRP
│   │   └── visualization.py         # Heatmap visualization
│   │
│   ├── concepts/                     # Concept-based explanations
│   │   ├── __init__.py
│   │   └── tcav.py                  # TCAV/ACE implementation
│   │
│   ├── manifold/                     # Latent space analysis
│   │   ├── __init__.py
│   │   └── analysis.py              # UMAP, PCA, FAISS
│   │
│   ├── counterfactuals/             # Counterfactual generation
│   │   ├── __init__.py
│   │   └── generation.py            # Input & latent perturbations
│   │
│   ├── llm/                         # LLM explanations
│   │   ├── __init__.py
│   │   └── explainer.py             # OpenAI/local LLM interface
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── config.py                # Hydra config management
│       ├── logging.py               # Logging utilities
│       └── general.py               # General helpers
│
├── configs/                          # Hydra configurations
│   ├── classification.yaml          # Classification config
│   ├── segmentation.yaml            # Segmentation config
│   ├── xai.yaml                     # XAI config
│   └── model/
│       ├── simple_cnn.yaml
│       └── unet_mini.yaml
│
├── scripts/                          # CLI entry points
│   ├── train_classification.py      # Train classifier
│   ├── train_segmentation.py        # Train segmenter
│   ├── run_xai.py                   # Generate XAI maps
│   ├── run_concepts.py              # Run TCAV analysis
│   ├── run_manifold.py              # Manifold analysis
│   └── run_llm_explanations.py      # Generate LLM explanations
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_classification_tutorial.ipynb
│   ├── 02_segmentation_tutorial.ipynb
│   ├── 03_xai_visualization.ipynb
│   ├── 04_concept_analysis.ipynb
│   └── 05_manifold_exploration.ipynb
│
├── experiments/                      # Experiment outputs
│   ├── logs/
│   ├── checkpoints/
│   ├── xai_outputs/
│   ├── concept_results/
│   └── manifold_plots/
│
└── data/                            # Dataset storage
    ├── classification/
    │   ├── train/
    │   └── val/
    └── segmentation/
        ├── images/
        └── masks/
```

## Key Components

### Models (`src/models/`)

- **Classification**: SimpleCNN, ResNet (18/34/50)
- **Segmentation**: UNetMini, SimpleCNNSegmenter
- All models support `return_features=True` for XAI
- All models have `get_embedding()` method for manifold analysis

### Datasets (`src/datasets/`)

- ImageFolderClassification: Standard image folder structure
- SegmentationDataset: Images + masks structure
- Dummy datasets for testing
- Transform utilities for augmentation

### Training (`src/training/`)

- Task-specific trainers with built-in metrics
- WandB integration
- Automatic checkpointing
- Learning rate scheduling

### XAI (`src/xai/`)

- Attribution methods: Integrated Gradients, Grad-CAM, DeepLift, Saliency
- Visualization utilities
- Support for both classification and segmentation

### Concepts (`src/concepts/`)

- TCAV implementation
- CAV training and testing
- Concept extraction utilities

### Manifold (`src/manifold/`)

- UMAP and PCA dimensionality reduction
- FAISS indexing for nearest neighbor search
- Feature extraction utilities

### Counterfactuals (`src/counterfactuals/`)

- Input perturbation-based CF generation
- Latent space CF stubs (requires generative model)

### LLM (`src/llm/`)

- OpenAI API integration
- Prompt templates for classification and segmentation
- Local LLM support (TODO)

### Utils (`src/utils/`)

- Hydra configuration management
- Logging setup
- Seed setting for reproducibility
- Model checkpoint utilities
