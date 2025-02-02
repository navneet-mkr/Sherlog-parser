"""Module containing Pydantic models for data validation and serialization."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
import numpy as np
from datetime import datetime

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
    
    def get_llm_kwargs(self) -> Dict:
        """Get kwargs for LLM provider initialization.
        
        Returns:
            Dictionary of LLM configuration parameters
        """
        return {
            "model_path": self.llm.model_path,
            "max_tokens": self.llm.max_tokens,
            "context_length": self.llm.context_length
        }

class LogLine(BaseModel):
    """Model representing a single log line with its metadata."""
    
    raw_text: str
    cluster_id: Optional[int] = None
    parsed_fields: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ClusterInfo(BaseModel):
    """Model representing cluster information and its regex pattern."""
    
    cluster_id: int
    regex_pattern: Optional[str] = None
    sample_lines: List[str] = Field(default_factory=list)
    center: Optional[np.ndarray] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class RegexGenerationPrompt(BaseModel):
    """Model for structuring regex generation prompts."""
    
    sample_lines: List[str]
    field_requirements: List[str] = Field(
        default=[
            "timestamp",
            "log_level",
            "message"
        ]
    )
    
    def format_prompt(self) -> str:
        """Format the prompt for the LLM.
        
        Returns:
            Formatted prompt string
        """
        lines_str = "\n".join(f"{i+1}. {line}" for i, line in enumerate(self.sample_lines))
        fields_str = "\n".join(f"- {field}" for field in self.field_requirements)
        
        return f"""Generate a Python regex pattern that matches these log lines:

{lines_str}

The pattern should:
1. Use named capture groups (e.g., (?P<timestamp>...)) for variable parts
2. Be returned as a raw string (r"...")
3. Include these required fields if present: 
{fields_str}
4. Add any additional meaningful field names that describe the captured data
5. Be as specific as possible while matching all sample lines

Respond with ONLY a JSON object containing:
{{
    "pattern": "the raw regex pattern",
    "field_descriptions": {{
        "field_name": "description of what this field captures"
    }}
}}"""

class RegexGenerationResponse(BaseModel):
    """Model for LLM regex generation responses."""
    
    pattern: str
    field_descriptions: Dict[str, str]

class LogBatch(BaseModel):
    """Model representing a batch of log lines for processing."""
    
    lines: List[LogLine]
    embeddings: Optional[np.ndarray] = None
    cluster_assignments: Optional[List[int]] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert the batch to a pandas DataFrame.
        
        Returns:
            DataFrame with log data
        """
        import pandas as pd
        
        # Create base DataFrame
        data = {
            'raw_log': [line.raw_text for line in self.lines],
            'cluster_id': [line.cluster_id for line in self.lines],
            'timestamp': [line.timestamp for line in self.lines]
        }
        
        # Add parsed fields
        all_fields = set()
        for line in self.lines:
            all_fields.update(line.parsed_fields.keys())
            
        for field in all_fields:
            data[field] = [line.parsed_fields.get(field, None) for line in self.lines]
            
        return pd.DataFrame(data) 