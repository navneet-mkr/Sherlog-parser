"""Module for generating and managing text embeddings with caching support."""

import logging
from typing import List, Union, Optional
import numpy as np
from pathlib import Path
import hashlib
from diskcache import Cache

from sentence_transformers import SentenceTransformer
from src.core.constants import DEFAULT_EMBEDDING_MODEL
from src.models import Settings

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles the generation of text embeddings using SentenceTransformers with caching."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_size_limit: int = 2**30,  # 1GB
        settings: Optional[Settings] = None
    ):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            cache_dir: Optional directory for caching embeddings
            cache_size_limit: Maximum cache size in bytes (default: 1GB)
            settings: Optional Settings instance for configuration
        """
        self.settings = settings or Settings()
        self.model_name = model_name
        
        logger.info(f"Initializing embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize cache
        cache_dir = cache_dir or Path("cache/embeddings")
        self.cache = Cache(
            directory=str(cache_dir),
            size_limit=cache_size_limit,
            eviction_policy='least-recently-used'
        )
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Cache key string
        """
        return hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        use_cache: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            use_cache: Whether to use caching
        
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
            
        Raises:
            ValueError: If texts is empty
            RuntimeError: If embedding generation fails
        """
        if not texts:
            raise ValueError("No texts provided for embedding generation")
            
        try:
            if not use_cache:
                return self._generate_embeddings_direct(texts, batch_size)
            
            # Try to get embeddings from cache
            embeddings = []
            texts_to_embed = []
            indices_to_embed = []
            
            # Use multi-key get for better performance
            cache_keys = [self._get_cache_key(text) for text in texts]
            cached_results = self.cache.mget(cache_keys)
            
            for i, (text, cached) in enumerate(zip(texts, cached_results)):
                if cached is not None:
                    embeddings.append(cached)
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
            
            # Generate embeddings for texts not in cache
            if texts_to_embed:
                new_embeddings = self._generate_embeddings_direct(texts_to_embed, batch_size)
                
                # Cache new embeddings using multi-key set
                cache_items = {
                    self._get_cache_key(text): embedding
                    for text, embedding in zip(texts_to_embed, new_embeddings)
                }
                self.cache.mset(cache_items)
                
                # Insert new embeddings in correct positions
                for idx, embedding in zip(indices_to_embed, new_embeddings):
                    embeddings.insert(idx, embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    def _generate_embeddings_direct(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Generate embeddings directly without using cache.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
        
        Returns:
            numpy array of embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings.
        
        Returns:
            Integer dimension of the embedding vectors
        """
        return self.model.get_sentence_embedding_dimension()
    
    def normalize_embeddings(
        self, 
        embeddings: Union[np.ndarray, List[List[float]]]
    ) -> np.ndarray:
        """Normalize embedding vectors to unit length.
        
        Args:
            embeddings: Array or list of embedding vectors
        
        Returns:
            Normalized embeddings as numpy array
            
        Raises:
            ValueError: If embeddings is empty or contains invalid values
        """
        if isinstance(embeddings, list):
            if not embeddings:
                raise ValueError("Empty embeddings list provided")
            embeddings = np.array(embeddings)
            
        if embeddings.size == 0:
            raise ValueError("Empty embeddings array provided")
            
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero-length embedding vector detected")
            
        return embeddings / norms
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
    
    def get_cache_info(self) -> dict:
        """Get information about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'size': self.cache.size,
            'volume': len(self.cache),
            'directory': self.cache.directory
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.cache.close()
        return None 