"""Configuration management for log parser."""

from dynaconf import Dynaconf, Validator
from typing import Dict

settings = Dynaconf(
    envvar_prefix="LOGPARSER",
    settings_files=['settings.yaml', '.secrets.yaml'],
    environments=True,
    load_dotenv=True,
    validators=[
        # LLM Settings
        Validator('model_name', default="llama2"),
        Validator('temperature', default=0.1, is_type_of=float),
        Validator('max_tokens', default=100, is_type_of=int),
        
        # Parser Settings
        Validator('similarity_threshold', default=0.8, is_type_of=float),
        Validator('max_template_length', default=200, is_type_of=int),
        Validator('max_examples_per_template', default=5, is_type_of=int),
        
        # Cache Settings
        Validator('cache.enabled', default=True, is_type_of=bool),
        Validator('cache.type', default="memory", is_type_of=str),
        Validator('cache.redis_url', default=None),
        Validator('cache.ttl_seconds', default=3600, is_type_of=int),
        
        # Processing Settings
        Validator('processing.batch_size', default=32, is_type_of=int),
        Validator('processing.max_workers', default=4, is_type_of=int),
        Validator('processing.retry_attempts', default=3, is_type_of=int),
        Validator('processing.retry_delay_seconds', default=1.0, is_type_of=float),
        
        # Logging Settings
        Validator('logging.level', default="INFO", is_type_of=str),
        Validator('logging.enable_debug', default=False, is_type_of=bool),
    ]
)

def get_llm_params() -> Dict:
    """Get LLM-specific parameters."""
    return {
        "model": settings.model_name,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens
    } 