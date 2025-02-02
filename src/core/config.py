"""Module for configuration management."""

from typing import Dict, Optional, Union
from pathlib import Path
from dynaconf import Dynaconf, Validator, ValidationError as DynaconfValidationError
import logging.config

from src.core.errors import ConfigError, error_handler

@error_handler(
    reraise=True,
    exclude=[KeyboardInterrupt, SystemExit]
)
class Settings:
    """Application settings using dynaconf."""
    
    def __init__(self, settings_files: Optional[list] = None):
        """Initialize settings.
        
        Args:
            settings_files: Optional list of settings files to load
            
        Raises:
            ConfigError: If configuration initialization fails
        """
        try:
            self.settings = Dynaconf(
                envvar_prefix="SHERLOG",
                settings_files=settings_files or ['settings.yaml', '.secrets.yaml'],
                environments=True,
                load_dotenv=True,
                validators=[
                    # Logging validators
                    Validator('logging.level', default="INFO"),
                    Validator('logging.format', default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                    Validator('logging.file_path', default=None),
                    
                    # Cache validators
                    Validator('cache.enabled', default=True),
                    Validator('cache.directory', default="cache"),
                    Validator('cache.max_size_mb', default=1024),
                    Validator('cache.ttl_days', default=30),
                    
                    # Model validators
                    Validator('model.embedding_model', default="all-MiniLM-L6-v2"),
                    Validator('model.n_clusters', default=20),
                    Validator('model.batch_size', default=1000),
                    Validator('model.random_seed', default=42),
                    
                    # Monitoring validators
                    Validator('monitoring.enabled', default=True),
                    Validator('monitoring.prometheus_port', default=8000),
                    Validator('monitoring.update_interval', default=15.0),
                    
                    # App validators
                    Validator('app.env', default="development"),
                    Validator('app.debug', default=False),
                ]
            )
        except DynaconfValidationError as e:
            raise ConfigError("Configuration validation failed", details={"errors": str(e)})
        except Exception as e:
            raise ConfigError(
                "Failed to initialize configuration",
                details={"error": str(e)}
            )
        
        # Create necessary directories
        self._setup_directories()
        
        # Configure logging
        self.configure_logging()
    
    def _setup_directories(self) -> None:
        """Create necessary directories.
        
        Raises:
            ConfigError: If directory creation fails
        """
        try:
            cache_dir = Path(self.settings.cache.directory)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            if self.settings.logging.file_path:
                log_dir = Path(self.settings.logging.file_path).parent
                log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ConfigError(
                "Failed to create required directories",
                details={
                    "error": str(e),
                    "cache_dir": str(cache_dir),
                    "log_dir": str(self.settings.logging.file_path) if self.settings.logging.file_path else None
                }
            )
    
    def configure_logging(self) -> None:
        """Configure logging based on settings.
        
        Raises:
            ConfigError: If logging configuration fails
        """
        try:
            config = {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "standard": {
                        "format": self.settings.logging.format
                    }
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "formatter": "standard",
                        "level": self.settings.logging.level
                    }
                },
                "root": {
                    "handlers": ["console"],
                    "level": self.settings.logging.level
                }
            }
            
            if self.settings.logging.file_path:
                config["handlers"]["file"] = {
                    "class": "logging.FileHandler",
                    "filename": str(self.settings.logging.file_path),
                    "formatter": "standard",
                    "level": self.settings.logging.level
                }
                config["root"]["handlers"].append("file")
            
            logging.config.dictConfig(config)
            
            if self.settings.app.debug:
                logging.getLogger().setLevel(logging.DEBUG)
                
        except Exception as e:
            raise ConfigError(
                "Failed to configure logging",
                details={
                    "error": str(e),
                    "config": config
                }
            )
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to dynaconf settings.
        
        Args:
            name: Attribute name
            
        Returns:
            Setting value
            
        Raises:
            ConfigError: If the setting doesn't exist
        """
        try:
            return getattr(self.settings, name)
        except AttributeError:
            raise ConfigError(
                f"Setting '{name}' not found",
                details={"available_settings": list(self.settings.to_dict().keys())}
            )

@error_handler(
    reraise=True,
    exclude=[KeyboardInterrupt, SystemExit]
)
def load_config() -> Settings:
    """Load application configuration.
    
    Returns:
        Settings instance
        
    Raises:
        ConfigError: If configuration loading fails
    """
    try:
        # Try loading from YAML first
        config_path = Path("config.yaml")
        if config_path.exists():
            return Settings(settings_files=[str(config_path)])
        
        # Fall back to environment variables
        return Settings()
    except Exception as e:
        raise ConfigError(
            "Failed to load configuration",
            details={
                "error": str(e),
                "config_path": str(config_path)
            }
        ) 