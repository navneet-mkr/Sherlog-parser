"""Configuration models for the application."""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings

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