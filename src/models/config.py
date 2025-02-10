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
    model_name: str = Field("mistral", env='OLLAMA_MODEL')
    temperature: float = Field(0.1, env='LLM_TEMPERATURE')
    top_k: int = Field(40, env='LLM_TOP_K')
    top_p: float = Field(0.9, env='LLM_TOP_P')
    repeat_penalty: float = Field(1.1, env='LLM_REPEAT_PENALTY')
    context_length: int = Field(4096, env='LLM_CONTEXT_LENGTH')

class PipelineConfig(BaseModel):
    """Configuration for log processing pipeline."""
    input_dir: Path = Field("./data/logs")
    output_dir: Path = Field("./output")
    cache_dir: Path = Field("./cache")
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
    base_url: str = Field("http://localhost:11434", env='OLLAMA_BASE_URL')
    timeout: int = Field(120, env='OLLAMA_TIMEOUT')
    model: str = Field("mistral", env='OLLAMA_MODEL')

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding_model: str = Field("all-MiniLM-L6-v2", env='EMBEDDING_MODEL')
    similarity_threshold: float = Field(0.8, env='SIMILARITY_THRESHOLD')
    batch_size: int = Field(1000, env='BATCH_SIZE')
    
    model_config = ConfigDict(env_file='.env', env_file_encoding='utf-8')
    
    def get_llm_kwargs(self) -> dict:
        """Get kwargs for LLM provider initialization."""
        return {
            "model": self.ollama.model,
            "temperature": self.llm.temperature,
            "top_k": self.llm.top_k,
            "top_p": self.llm.top_p,
            "repeat_penalty": self.llm.repeat_penalty
        } 