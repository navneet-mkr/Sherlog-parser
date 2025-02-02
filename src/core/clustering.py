"""Module for incremental clustering of log lines with persistence."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from src.models import ClusterInfo, Settings, LogLine
from src.models.clustering import ClusteringState, ClusteringParams, ClusterInfo as PydanticClusterInfo
from src.core.constants import (
    DEFAULT_N_CLUSTERS,
    MIN_CLUSTER_SIZE,
    MAX_CLUSTER_SIZE
)

logger = logging.getLogger(__name__)

class LogClusterer(BaseEstimator):
    """Handles incremental clustering of log lines using MiniBatchKMeans with persistence."""
    
    def __init__(
        self,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        batch_size: Optional[int] = None,
        init_size: Optional[int] = None,
        random_state: int = 42,
        settings: Optional[Settings] = None
    ):
        """Initialize the incremental clusterer.
        
        Args:
            n_clusters: Number of clusters to maintain
            batch_size: Optional override for batch size
            init_size: Optional override for initialization size
            random_state: Random seed for reproducibility
            settings: Optional Settings instance for configuration
        """
        self.settings = settings or Settings()
        self.params = ClusteringParams(
            n_clusters=n_clusters,
            batch_size=batch_size or self.settings.batch_size,
            init_size=init_size or self.settings.init_size,
            random_state=random_state,
            max_samples_per_cluster=MAX_CLUSTER_SIZE,
            min_cluster_size=MIN_CLUSTER_SIZE
        )
        
        # Initialize models
        self.scaler = StandardScaler()
        self.clusterer = MiniBatchKMeans(
            n_clusters=self.params.n_clusters,
            batch_size=self.params.batch_size,
            init_size=self.params.init_size,
            random_state=self.params.random_state
        )
        
        self.is_fitted = False
        self.state = ClusteringState(
            n_clusters=self.params.n_clusters,
            n_samples=0,
            clusters={},
            last_update=datetime.now()
        )
    
    def partial_fit(
        self,
        embeddings: np.ndarray,
        log_lines: Optional[List[LogLine]] = None
    ) -> 'LogClusterer':
        """Partially fit the clusterer on a batch of embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            log_lines: Optional list of corresponding LogLine objects
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If embeddings is empty or invalid
        """
        if embeddings.size == 0:
            raise ValueError("Empty embeddings array provided")
            
        try:
            # Update scaler incrementally
            if not self.is_fitted:
                self.scaler.partial_fit(embeddings)
            
            # Scale the embeddings
            scaled_embeddings = self.scaler.transform(embeddings)
            
            # Update clusters
            self.clusterer.partial_fit(scaled_embeddings)
            self.is_fitted = True
            self.state.last_update = datetime.now()
            self.state.n_samples += len(embeddings)
            
            if log_lines is not None:
                self._update_cluster_info(scaled_embeddings, log_lines)
                
            return self
            
        except Exception as e:
            logger.error(f"Error in partial_fit: {str(e)}")
            raise
    
    def _update_cluster_info(
        self,
        embeddings: np.ndarray,
        log_lines: List[LogLine]
    ) -> None:
        """Update cluster information with new samples.
        
        Args:
            embeddings: Scaled embeddings array
            log_lines: List of LogLine objects
        """
        labels = self.clusterer.predict(embeddings)
        centers = self.clusterer.cluster_centers_
        
        for label, line in zip(labels, log_lines):
            if label not in self.state.clusters:
                self.state.clusters[label] = PydanticClusterInfo(
                    cluster_id=label,
                    size=0,
                    sample_lines=[],
                    center=centers[label].tolist()
                )
            
            cluster = self.state.clusters[label]
            if len(cluster.sample_lines) < self.params.max_samples_per_cluster:
                cluster.sample_lines.append(line.raw_text)
                cluster.size += 1
                cluster.center = centers[label].tolist()
                cluster.last_update = datetime.now()
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings.
        
        Args:
            embeddings: Numpy array of embeddings to cluster
            
        Returns:
            Array of cluster labels
            
        Raises:
            ValueError: If clusterer is not fitted or embeddings is invalid
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted before predicting")
            
        scaled_embeddings = self.scaler.transform(embeddings)
        return self.clusterer.predict(scaled_embeddings)
    
    def get_cluster_info(self, cluster_id: int) -> Optional[PydanticClusterInfo]:
        """Get information about a cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            ClusterInfo object if it exists, None otherwise
        """
        return self.state.clusters.get(cluster_id)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get the current cluster centers.
        
        Returns:
            Numpy array of cluster centers in original space
            
        Raises:
            ValueError: If clusterer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted before getting centers")
            
        return self.scaler.inverse_transform(self.clusterer.cluster_centers_)
    
    def get_cluster_statistics(self) -> Dict[str, Dict]:
        """Get statistics about all clusters.
        
        Returns:
            Dictionary with cluster statistics
        """
        stats = {}
        for cluster_id, info in self.state.clusters.items():
            stats[cluster_id] = {
                'size': info.size,
                'has_pattern': info.pattern is not None,
                'center_norm': float(np.linalg.norm(info.center)) if info.center is not None else None
            }
        return stats
    
    def save_model(self, directory: Union[str, Path]) -> None:
        """Save the clustering model and its state.
        
        Args:
            directory: Directory to save model files
            
        Raises:
            IOError: If saving fails
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save sklearn models
            joblib.dump(self.scaler, directory / 'scaler.joblib')
            joblib.dump(self.clusterer, directory / 'clusterer.joblib')
            
            # Save state and parameters
            joblib.dump(self.state.dict(), directory / 'state.joblib')
            joblib.dump(self.params.dict(), directory / 'params.joblib')
            
            logger.info(f"Model saved to {directory}")
            
        except Exception as e:
            raise IOError(f"Failed to save model: {str(e)}")
    
    @classmethod
    def load_model(
        cls,
        directory: Union[str, Path],
        settings: Optional[Settings] = None
    ) -> 'LogClusterer':
        """Load a saved clustering model.
        
        Args:
            directory: Directory containing model files
            settings: Optional Settings instance
            
        Returns:
            Loaded LogClusterer instance
            
        Raises:
            IOError: If loading fails
        """
        directory = Path(directory)
        if not directory.exists():
            raise IOError(f"Model directory {directory} does not exist")
            
        try:
            # Load parameters and create instance
            params = ClusteringParams(**joblib.load(directory / 'params.joblib'))
            instance = cls(
                n_clusters=params.n_clusters,
                batch_size=params.batch_size,
                random_state=params.random_state,
                settings=settings
            )
            
            # Load sklearn models
            instance.scaler = joblib.load(directory / 'scaler.joblib')
            instance.clusterer = joblib.load(directory / 'clusterer.joblib')
            
            # Load state
            state_dict = joblib.load(directory / 'state.joblib')
            instance.state = ClusteringState(**state_dict)
            instance.is_fitted = True
            
            logger.info(f"Model loaded from {directory}")
            return instance
            
        except Exception as e:
            raise IOError(f"Failed to load model: {str(e)}")
    
    def __getstate__(self) -> Dict:
        """Get state for pickling."""
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state: Dict) -> None:
        """Set state for unpickling."""
        self.__dict__.update(state) 