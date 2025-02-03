"""Configuration models for the application."""

from typing import Optional, Dict, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, HttpUrl
from pydantic_settings import BaseSettings
from pathlib import Path
from langchain.schema.language_model import BaseLanguageModel

class ModelInfo(BaseModel):
    """Information about a model."""
    name: str
    repo_id: str
    filename: str
    description: str
    context_length: int
    memory_required: str

class LLMConfig(BaseModel):
    """Configuration for LLM usage."""
    model_type: Literal["local", "api"]
    model_name: str
    api_key: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, gt=0)

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

# Model configurations
AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    "mistral-7b-instruct": ModelInfo(
        name="Mistral 7B Instruct",
        repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        description="Instruction-tuned version of Mistral 7B. Better for following specific instructions.",
        context_length=8192,
        memory_required="8GB"
    ),
    "llama2-7b": ModelInfo(
        name="Llama 2 7B",
        repo_id="TheBloke/Llama-2-7B-GGUF",
        filename="llama-2-7b.Q4_K_M.gguf",
        description="Meta's latest model. Good all-round performance.",
        context_length=4096,
        memory_required="8GB"
    )
}

class LLMSettings(BaseModel):
    """Settings for LLM configuration."""
    
    provider_type: str = Field("openai", env='LLM_PROVIDER')
    model_name: str = Field("gpt-4", env='LLM_MODEL_NAME')
    model_path: Optional[str] = Field(None, env='LLM_MODEL_PATH')
    openai_api_key: Optional[str] = Field(None, env='OPENAI_API_KEY')
    max_tokens: int = Field(2000, env='LLM_MAX_TOKENS')
    context_length: int = Field(8192, env='LLM_CONTEXT_LENGTH')
    temperature: float = Field(0.1, env='LLM_TEMPERATURE')

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    llm: LLMSettings = Field(default_factory=LLMSettings)
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