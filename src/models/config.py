"""Configuration models for the application."""

from typing import Optional, Dict, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, HttpUrl
from pydantic_settings import BaseSettings
from pathlib import Path
from langchain.schema.language_model import BaseLanguageModel

class ModelInfo(BaseModel):
    """Information about an Ollama model."""
    name: str
    model_id: str
    description: str
    context_length: int = Field(default=4096)
    parameters: Dict[str, Union[str, int, float]] = Field(default_factory=dict)

class LLMConfig(BaseModel):
    """Configuration for Ollama LLM usage."""
    model_id: str
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    num_predict: int = Field(default=2048, gt=0)  # max_tokens in Ollama
    top_k: int = Field(default=40, ge=0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repeat_penalty: float = Field(default=1.1, ge=0.0)
    seed: Optional[int] = None

class PipelineConfig(BaseModel):
    """Configuration for log processing pipeline."""
    file_path: Path
    encoding: str = "utf-8"
    n_clusters: int = Field(default=20, gt=0)
    batch_size: int = Field(default=1000, gt=0)
    llm_config: Optional[Dict] = None

class RunConfig(BaseModel):
    """Configuration for a pipeline run."""
    ops: Dict
    resources: Dict

# Available Ollama models with their configurations
AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    "mistral": ModelInfo(
        name="Mistral 7B",
        model_id="mistral",
        description="A powerful open-source language model with strong reasoning capabilities.",
        context_length=8192,
        parameters={
            "mirostat": 0,
            "mirostat_eta": 0.1,
            "mirostat_tau": 5.0,
        }
    ),
    "llama2": ModelInfo(
        name="Llama 2",
        model_id="llama2",
        description="Meta's latest model optimized for chat and instruction following.",
        context_length=4096,
        parameters={
            "mirostat": 0,
            "mirostat_eta": 0.1,
            "mirostat_tau": 5.0,
        }
    ),
    "codellama": ModelInfo(
        name="Code Llama",
        model_id="codellama",
        description="Specialized model for code understanding and generation.",
        context_length=16384,
        parameters={
            "mirostat": 0,
            "mirostat_eta": 0.1,
            "mirostat_tau": 5.0,
        }
    )
}

class OllamaSettings(BaseModel):
    """Settings for Ollama configuration."""
    host: str = Field("http://ollama", env='OLLAMA_HOST')
    port: int = Field(11434, env='OLLAMA_PORT')
    timeout: int = Field(120, env='OLLAMA_TIMEOUT')

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    embedding_model: str = Field("all-MiniLM-L6-v2", env='EMBEDDING_MODEL')
    chunk_size: int = Field(10000, env='CHUNK_SIZE')
    n_clusters: int = Field(20, env='N_CLUSTERS')
    batch_size: int = Field(1000, env='BATCH_SIZE')
    init_size: int = Field(3000, env='INIT_SIZE')
    db_path: str = Field("logs.duckdb", env='DB_PATH')
    
    model_config = ConfigDict(env_file='.env', env_file_encoding='utf-8')
    
    def get_llm_kwargs(self) -> dict:
        """Get kwargs for LLM provider initialization."""
        return {
            "model_path": self.llm.model_path,
            "max_tokens": self.llm.max_tokens,
            "context_length": self.llm.context_length
        } 