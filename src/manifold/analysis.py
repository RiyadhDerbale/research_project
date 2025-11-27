"""
Manifold analysis using UMAP, PCA, and FAISS indexing
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import faiss

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ManifoldAnalyzer:
    """
    Analyze latent space manifold structure
    
    Args:
        n_components: Number of dimensions for dimensionality reduction
        method: Method to use ('umap' or 'pca')
    """
    
    def __init__(self, n_components: int = 2, method: str = 'umap'):
        self.n_components = n_components
        self.method = method
        self.reducer = None
        self.scaler = StandardScaler()
        self.embeddings = None
        self.faiss_index = None
        
    def fit(self, features: np.ndarray):
        """
        Fit dimensionality reduction on features
        
        Args:
            features: High-dimensional features [N, D]
        """
        # Standardize
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit reducer
        if self.method == 'umap':
            self.reducer = umap.UMAP(
                n_components=self.n_components,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1
            )
        elif self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.embeddings = self.reducer.fit_transform(features_scaled)
        
        logger.info(
            f"Fitted {self.method.upper()} reduction: "
            f"{features.shape[1]} -> {self.n_components} dimensions"
        )
        
        return self.embeddings
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features to low-dimensional embeddings"""
        if self.reducer is None:
            raise ValueError("Must call fit() first")
        
        features_scaled = self.scaler.transform(features)
        return self.reducer.transform(features_scaled)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(features)
    
    def build_faiss_index(self, features: np.ndarray, use_gpu: bool = False):
        """
        Build FAISS index for fast nearest neighbor search
        
        Args:
            features: Features to index [N, D]
            use_gpu: Whether to use GPU for indexing
        """
        dimension = features.shape[1]
        
        # Normalize features for cosine similarity
        features = features.astype('float32')
        faiss.normalize_L2(features)
        
        # Create index
        if use_gpu and faiss.get_num_gpus() > 0:
            # GPU index
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(dimension)
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            # CPU index
            self.faiss_index = faiss.IndexFlatL2(dimension)
        
        # Add features
        self.faiss_index.add(features)
        
        logger.info(f"Built FAISS index with {features.shape[0]} vectors")
    
    def search_neighbors(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search k nearest neighbors
        
        Args:
            query: Query feature [D] or [1, D]
            k: Number of neighbors
            
        Returns:
            distances: Distances to neighbors [k]
            indices: Indices of neighbors [k]
        """
        if self.faiss_index is None:
            raise ValueError("Must call build_faiss_index() first")
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype('float32')
        faiss.normalize_L2(query)
        
        distances, indices = self.faiss_index.search(query, k)
        
        return distances[0], indices[0]
    
    def get_embedding_statistics(self) -> dict:
        """Compute statistics on embeddings"""
        if self.embeddings is None:
            raise ValueError("No embeddings computed")
        
        stats = {
            'mean': self.embeddings.mean(axis=0),
            'std': self.embeddings.std(axis=0),
            'min': self.embeddings.min(axis=0),
            'max': self.embeddings.max(axis=0),
        }
        
        return stats


def extract_features_from_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_embeddings: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from model for manifold analysis
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device
        use_embeddings: Use model.get_embedding() if available
        
    Returns:
        features: Extracted features [N, D]
        labels: Labels [N]
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Extract features
            if use_embeddings and hasattr(model, 'get_embedding'):
                feats = model.get_embedding(images)
            else:
                feats = model(images)
            
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
    
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    
    logger.info(f"Extracted features: {features.shape}")
    
    return features, labels


# TODO: Add t-SNE support
# TODO: Add cluster analysis (K-means, DBSCAN)
# TODO: Add manifold quality metrics
# TODO: Add interactive 3D visualization
