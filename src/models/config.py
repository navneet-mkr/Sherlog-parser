"""Configuration models for the application."""

from typing import Optional, Dict, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, HttpUrl
from pydantic_settings import BaseSettings
from pathlib import Path

class ModelInfo(BaseModel):
    """Information about an Ollama model."""
    name: str
    model_id: str
    description: str
    context_length: int = Field(default=4096)
    parameters: Dict[str, Union[str, int, float]] = Field(default_factory=dict)

class LLMSettings(BaseModel):
    """Settings for LLM configuration."""
    model_name: str = Field(default="mistral", validation_alias="OLLAMA_MODEL")
    temperature: float = Field(default=0.1, validation_alias="LLM_TEMPERATURE")
    top_k: int = Field(default=40, validation_alias="LLM_TOP_K")
    top_p: float = Field(default=0.9, validation_alias="LLM_TOP_P")
    repeat_penalty: float = Field(default=1.1, validation_alias="LLM_REPEAT_PENALTY")
    context_length: int = Field(default=4096, validation_alias="LLM_CONTEXT_LENGTH")

class PipelineConfig(BaseModel):
    """Configuration for log processing pipeline."""
    input_dir: Path = Field(default=Path("./data/logs"))
    output_dir: Path = Field(default=Path("./output"))
    cache_dir: Path = Field(default=Path("./cache"))
    encoding: str = "utf-8"
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    batch_size: int = Field(default=1000, gt=0)

# Available Ollama models with their configurations
AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    "mistral": ModelInfo(
        name="Mistral 7B",
        model_id="mistral",
        description="A powerful open-source language model with strong reasoning capabilities.",
        context_length=8192,
        parameters={
            "temperature": 0.1,
            "top_k": 40,
            "top_p": 0.9,
        }
    ),
    "llama2": ModelInfo(
        name="Llama 2",
        model_id="llama2",
        description="Meta's latest model optimized for chat and instruction following.",
        context_length=4096,
        parameters={
            "temperature": 0.1,
            "top_k": 40,
            "top_p": 0.9,
        }
    ),
    "codellama": ModelInfo(
        name="Code Llama",
        model_id="codellama",
        description="Specialized model for code understanding and generation.",
        context_length=16384,
        parameters={
            "temperature": 0.1,
            "top_k": 40,
            "top_p": 0.9,
        }
    )
}

class OllamaSettings(BaseModel):
    """Settings for Ollama configuration."""
    base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    timeout: int = Field(default=120, validation_alias="OLLAMA_TIMEOUT")
    model: str = Field(default="mistral", validation_alias="OLLAMA_MODEL")

class Settings(BaseModel):
    """Application settings."""
    
    # Directory settings
    upload_dir: str = Field(default="./data/uploads", description="Directory for uploaded files")
    output_dir: str = Field(default="./output", description="Directory for output files")
    cache_dir: str = Field(default="./cache", description="Directory for cache files")
    
    # LLM settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )
    model_name: str = Field(
        default="mistral",
        description="Name of the LLM model to use"
    )
    
    # Pipeline settings
    similarity_threshold: float = Field(
        default=0.8,
        description="Threshold for template similarity matching",
        ge=0.0,
        le=1.0
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for processing logs",
        gt=0
    )
    
    # File settings
    encoding: str = Field(
        default="utf-8",
        description="Encoding for reading log files"
    )
    
    # Database settings
    db_path: str = Field(
        default=":memory:",
        description="Path to the database file"
    )
    persist_db: bool = Field(
        default=True,
        description="Whether to persist the database"
    )
    
    class Config:
        """Pydantic config."""
        env_prefix = "LOGPARSE_"  # Environment variable prefix

    def get_llm_kwargs(self) -> dict:
        """Get kwargs for LLM provider initialization."""
        return {
            "model": self.ollama_base_url,
            "temperature": self.model_name,
            "top_k": self.batch_size,
            "top_p": self.similarity_threshold,
            "repeat_penalty": self.batch_size
        } 