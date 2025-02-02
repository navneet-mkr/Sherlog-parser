"""Pydantic models for clustering."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
import numpy as np
import re

class ClusterPattern(BaseModel):
    """Regex pattern extracted from cluster samples."""
    
    pattern: str = Field(..., description="Regular expression pattern")
    confidence: float = Field(..., ge=0, le=1, description="Pattern confidence score")
    sample_count: int = Field(..., ge=0, description="Number of samples used to generate pattern")
    created_at: datetime = Field(default_factory=datetime.now, description="Pattern creation timestamp")
    
    @validator("pattern")
    def validate_pattern(cls, v):
        """Validate that pattern is a valid regex."""
        try:
            re.compile(v)
            return v
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {str(e)}")

class ClusterInfo(BaseModel):
    """Information about a cluster."""
    
    cluster_id: int = Field(..., description="Unique cluster identifier")
    size: int = Field(..., ge=0, description="Number of samples in cluster")
    center: Optional[List[float]] = Field(None, description="Cluster center coordinates")
    pattern: Optional[ClusterPattern] = Field(None, description="Extracted pattern")
    sample_lines: List[str] = Field(default_factory=list, description="Sample log lines")
    last_update: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            datetime: lambda x: x.isoformat()
        }
    
    @validator("center")
    def validate_center(cls, v):
        """Convert numpy array to list if needed."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

class ClusteringState(BaseModel):
    """State of the clustering model."""
    
    n_clusters: int = Field(..., ge=1, description="Number of clusters")
    n_samples: int = Field(0, ge=0, description="Total number of samples processed")
    clusters: Dict[int, ClusterInfo] = Field(default_factory=dict, description="Cluster information")
    last_update: datetime = Field(default_factory=datetime.now, description="Last state update")
    version: str = Field("1.0.0", description="Model version")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda x: x.isoformat()
        }

class ClusteringParams(BaseModel):
    """Parameters for clustering."""
    
    n_clusters: int = Field(20, ge=1, description="Number of clusters")
    batch_size: int = Field(1000, ge=1, description="Batch size for incremental updates")
    random_state: Optional[int] = Field(None, description="Random seed for reproducibility")
    max_samples_per_cluster: int = Field(1000, ge=1, description="Maximum samples to store per cluster")
    min_cluster_size: int = Field(10, ge=1, description="Minimum cluster size for pattern extraction")
    
    @validator("n_clusters")
    def validate_n_clusters(cls, v, values):
        """Validate number of clusters."""
        if v < 1:
            raise ValueError("Number of clusters must be positive")
        return v
    
    @validator("batch_size")
    def validate_batch_size(cls, v, values):
        """Validate batch size."""
        if v < 1:
            raise ValueError("Batch size must be positive")
        if v < values.get("n_clusters", 20):
            raise ValueError("Batch size should be at least as large as number of clusters")
        return v

class ClusterPrediction(BaseModel):
    """Cluster prediction result."""
    
    cluster_id: int = Field(..., description="Assigned cluster ID")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    matches_pattern: Optional[bool] = Field(None, description="Whether sample matches cluster pattern")
    distance_to_center: Optional[float] = Field(None, ge=0, description="Distance to cluster center")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp") 