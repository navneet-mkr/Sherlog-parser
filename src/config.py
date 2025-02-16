"""Configuration management for log parser.

This module provides a centralized configuration system using Dynaconf.
It handles all configuration settings including model parameters, parser settings,
caching, and processing options.
"""

from dynaconf import Dynaconf, Validator
from typing import Dict

# Initialize Dynaconf with our configuration hierarchy
# - Environment variables (prefix: LOGPARSER_)
# - settings.yaml (main configuration)
# - .secrets.yaml (sensitive data)
settings = Dynaconf(
    envvar_prefix="LOGPARSER",
    settings_files=['settings.yaml', '.secrets.yaml'],
    environments=True,
    load_dotenv=True,
    validators=[
        # LLM Settings - Control the behavior of the language model
        Validator('model_name', default="llama2"),  # Which LLM to use
        Validator('temperature', default=0.1, is_type_of=float),  # Controls randomness in generation
        Validator('max_tokens', default=100, is_type_of=int),  # Maximum length of generated text
        
        # Parser Settings - Configure log parsing behavior
        Validator('similarity_threshold', default=0.8, is_type_of=float),  # Threshold for template matching
        Validator('max_template_length', default=200, is_type_of=int),  # Maximum length of log templates
        Validator('max_examples_per_template', default=5, is_type_of=int),  # Number of examples to store per template
        
        # Cache Settings - Control caching behavior for better performance
        Validator('cache.enabled', default=True, is_type_of=bool),  # Enable/disable caching
        Validator('cache.type', default="memory", is_type_of=str),  # Cache type: memory or redis
        Validator('cache.redis_url', default=None),  # Redis connection URL if using Redis cache
        Validator('cache.ttl_seconds', default=3600, is_type_of=int),  # Cache entry lifetime
        
        # Processing Settings - Configure batch processing parameters
        Validator('processing.batch_size', default=32, is_type_of=int),  # Number of logs to process at once
        Validator('processing.max_workers', default=4, is_type_of=int),  # Number of parallel workers
        Validator('processing.retry_attempts', default=3, is_type_of=int),  # Number of retry attempts
        Validator('processing.retry_delay_seconds', default=1.0, is_type_of=float),  # Delay between retries
        
        # Logging Settings - Control logging behavior
        Validator('logging.level', default="INFO", is_type_of=str),  # Logging level (DEBUG, INFO, etc.)
        Validator('logging.enable_debug', default=False, is_type_of=bool),  # Enable detailed debug logging
    ]
)

def get_llm_params() -> Dict:
    """Get LLM-specific parameters.
    
    Returns:
        Dict containing model name, temperature, and max tokens for the LLM.
        These parameters control the behavior of the language model during template generation.
    """
    return {
        "model": settings.model_name,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens
    } 