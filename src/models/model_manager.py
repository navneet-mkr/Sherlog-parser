import os
import logging
import hashlib
import requests
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
from .llm_config import ModelInfo, ModelConfig

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model downloading, caching, and verification."""
    
    def __init__(self, models_dir: str = "/data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.models_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_model_path(self, model_key: str) -> Optional[Path]:
        """Get the path to a cached model."""
        model_info = ModelConfig.get_available_models().get(model_key)
        if not model_info:
            return None
            
        model_hash = self._get_model_hash(model_info.download_url)
        model_path = self.cache_dir / f"{model_key}-{model_hash}.gguf"
        
        if model_path.exists():
            return model_path
        return None
        
    def download_model(self, model_key: str, force: bool = False) -> Tuple[bool, str]:
        """Download and cache a model.
        
        Args:
            model_key: Key of the model to download
            force: Whether to force re-download even if cached
            
        Returns:
            Tuple of (success, message)
        """
        model_info = ModelConfig.get_available_models().get(model_key)
        if not model_info:
            return False, f"Model {model_key} not found in available models"
            
        if not model_info.download_url:
            return False, f"No download URL available for model {model_key}"
            
        model_hash = self._get_model_hash(model_info.download_url)
        model_path = self.cache_dir / f"{model_key}-{model_hash}.gguf"
        
        # Check if already downloaded
        if model_path.exists() and not force:
            return True, f"Model already cached at {model_path}"
            
        try:
            # Download with progress bar
            response = requests.get(model_info.download_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f, tqdm(
                desc=f"Downloading {model_info.name}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    
            return True, f"Successfully downloaded model to {model_path}"
            
        except Exception as e:
            if model_path.exists():
                model_path.unlink()
            return False, f"Failed to download model: {str(e)}"
            
    def verify_model(self, model_path: Path) -> bool:
        """Verify that a model file is valid."""
        try:
            # Basic verification - check file exists and size
            if not model_path.exists():
                return False
                
            # Check file size (should be at least 3GB for 7B models)
            min_size = 3 * 1024 * 1024 * 1024  # 3GB
            if model_path.stat().st_size < min_size:
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error verifying model: {str(e)}")
            return False
            
    def _get_model_hash(self, url: str) -> str:
        """Generate a hash for the model URL."""
        return hashlib.md5(url.encode()).hexdigest()[:8]
        
    def cleanup_cache(self, max_size_gb: int = 50):
        """Clean up the cache directory if it exceeds max size."""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*'))
            max_size = max_size_gb * 1024 * 1024 * 1024
            
            if total_size > max_size:
                # Remove oldest files first
                files = [(f, f.stat().st_mtime) for f in self.cache_dir.glob('*')]
                files.sort(key=lambda x: x[1])
                
                for file_path, _ in files:
                    if total_size <= max_size:
                        break
                    size = file_path.stat().st_size
                    file_path.unlink()
                    total_size -= size
                    
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")
            
    def import_custom_model(self, file_path: Path, model_name: str) -> Tuple[bool, str]:
        """Import a custom model file.
        
        Args:
            file_path: Path to the model file
            model_name: Name to give the imported model
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not file_path.exists():
                return False, "Model file not found"
                
            # Copy to cache directory with a unique name
            model_hash = hashlib.md5(file_path.read_bytes()).hexdigest()[:8]
            cache_path = self.cache_dir / f"custom-{model_name}-{model_hash}.gguf"
            
            with open(file_path, 'rb') as src, open(cache_path, 'wb') as dst:
                dst.write(src.read())
                
            return True, f"Successfully imported model to {cache_path}"
            
        except Exception as e:
            return False, f"Failed to import model: {str(e)}" 