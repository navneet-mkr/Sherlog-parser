"""Module for handling the complete log processing pipeline."""

import logging
from pathlib import Path
from typing import List, Optional, Union, BinaryIO, Dict, Any

from dagster import (
    job, op, In, Out, Nothing,
    DagsterType, TypeCheck,
    Field as DagsterField, String, Int,
    AssetMaterialization, MetadataValue,
    ResourceDefinition
)
import numpy as np
from pydantic import BaseModel, Field

from src.models import Settings, LogLine
from src.models.clustering import ClusteringState, ClusteringParams
from src.core.embeddings import EmbeddingGenerator
from src.core.clustering import LogClusterer
from src.core.errors import FileError, ParsingError, ClusteringError

logger = logging.getLogger(__name__)

# Pydantic models for operation configurations
class ReadLogConfig(BaseModel):
    """Configuration for log file reading operation."""
    encoding: str = Field(default="utf-8", description="File encoding")

class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation operation."""
    batch_size: int = Field(default=1000, description="Number of lines to process in each batch")

class ClusteringConfig(BaseModel):
    """Configuration for log clustering operation."""
    n_clusters: int = Field(default=20, description="Number of clusters to generate")

class AnalysisConfig(BaseModel):
    """Configuration for pattern analysis operation."""
    max_samples: int = Field(default=5, description="Maximum number of sample lines to show per cluster")

# Pydantic models for intermediate data
class EmbeddingData(BaseModel):
    """Data structure for embedding generation results."""
    embeddings: np.ndarray
    original_lines: List[str]
    
    class Config:
        arbitrary_types_allowed = True

class ClusteringData(BaseModel):
    """Data structure for clustering results."""
    clustering_state: ClusteringState
    original_lines: List[str]
    
    class Config:
        arbitrary_types_allowed = True

# Resource definitions
class ChunkCounter:
    def __init__(self):
        self.current_chunk = 0
    
    def get_next_chunk(self) -> int:
        chunk = self.current_chunk
        self.current_chunk += 1
        return chunk

class ClustererResource:
    def __init__(self):
        self.clusterer = None
    
    def get_clusterer(self) -> Optional[LogClusterer]:
        return self.clusterer
    
    def set_clusterer(self, clusterer: LogClusterer):
        self.clusterer = clusterer

chunk_counter = ResourceDefinition.hardcoded_resource(ChunkCounter())
clusterer_resource = ResourceDefinition.hardcoded_resource(ClustererResource())

# Custom Dagster types
def is_valid_log_lines(_, value: List[str]) -> TypeCheck:
    """Validate log lines."""
    if not isinstance(value, list):
        return TypeCheck(
            success=False,
            description="Value should be a list of strings"
        )
    if not all(isinstance(line, str) for line in value):
        return TypeCheck(
            success=False,
            description="All elements should be strings"
        )
    return TypeCheck(success=True)

LogLines = DagsterType(
    name="LogLines",
    type_check_fn=is_valid_log_lines,
    description="A list of log lines"
)

# Pipeline operations
@op(
    ins={"file_path": In(String)},
    out=Out(LogLines),
    config_schema={
        "encoding": DagsterField(String, default_value="utf-8", is_required=False, description="File encoding to use")
    }
)
def read_log_file(context, file_path: str) -> List[str]:
    """Read log lines from file.
    
    Args:
        context: Dagster execution context
        file_path: Path to the log file
        
    Returns:
        List of log lines
    """
    try:
        encoding = context.op_config["encoding"]
        path = Path(file_path)
        if not path.exists():
            raise FileError(f"File not found: {file_path}")
            
        lines = path.read_text(encoding=encoding).splitlines()
        
        context.log.info(f"Read {len(lines)} lines from {file_path}")
        
        # Record asset materialization
        context.log_event(
            AssetMaterialization(
                asset_key=f"log_file_{path.stem}",
                description=f"Log file: {file_path}",
                metadata={
                    "num_lines": MetadataValue.int(len(lines)),
                    "file_size": MetadataValue.int(path.stat().st_size)
                }
            )
        )
        
        return lines
    except Exception as e:
        raise FileError(f"Failed to read file: {str(e)}")

@op(
    ins={"log_lines": In(LogLines)},
    out=Out(dict),
    config_schema={
        "batch_size": DagsterField(Int, default_value=1000, is_required=False, description="Number of lines to process in each batch")
    }
)
def generate_embeddings(context, log_lines: List[str]) -> Dict[str, Any]:
    """Generate embeddings for log lines.
    
    Args:
        context: Dagster execution context
        log_lines: List of log lines
        
    Returns:
        Dictionary with embeddings and metadata
    """
    try:
        batch_size = context.op_config["batch_size"]
        settings = Settings()
        embedding_gen = EmbeddingGenerator(settings)
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(log_lines), batch_size):
            batch = log_lines[i:i + batch_size]
            batch_embeddings = embedding_gen.generate_embeddings(batch)
            embeddings.extend(batch_embeddings)
            
            context.log.info(f"Generated embeddings for batch {i//batch_size + 1}")
        
        embeddings_array = np.array(embeddings)
        
        # Record asset materialization
        context.log_event(
            AssetMaterialization(
                asset_key="embeddings",
                description="Generated embeddings",
                metadata={
                    "shape": MetadataValue.text(str(embeddings_array.shape)),
                    "memory_size": MetadataValue.int(embeddings_array.nbytes)
                }
            )
        )
        
        return {
            "embeddings": embeddings_array,
            "original_lines": log_lines
        }
    except Exception as e:
        raise ParsingError(f"Failed to generate embeddings: {str(e)}")

@op(
    ins={"embedding_data": In(dict)},
    out=Out(dict),
    config_schema={
        "n_clusters": DagsterField(Int, default_value=20, is_required=False, description="Number of clusters to generate")
    }
)
def cluster_logs(context, embedding_data: Dict[str, Any]) -> Dict[str, Any]:
    """Cluster log lines based on embeddings.
    
    Args:
        context: Dagster execution context
        embedding_data: Dictionary with embeddings and original lines
        
    Returns:
        Dictionary with clustering results
    """
    try:
        n_clusters = context.op_config["n_clusters"]
        settings = Settings()
        settings.model.n_clusters = n_clusters
        clusterer = LogClusterer(settings)
        
        # Update clustering
        clustering_state = clusterer.process_batch(
            embedding_data["original_lines"],
            embedding_data["embeddings"]
        )
        
        # Record asset materialization
        context.log_event(
            AssetMaterialization(
                asset_key="clusters",
                description="Generated clusters",
                metadata={
                    "num_clusters": MetadataValue.int(
                        len(clustering_state.clusters)
                    ),
                    "num_samples": MetadataValue.int(
                        clustering_state.n_samples
                    )
                }
            )
        )
        
        return {
            "clustering_state": clustering_state,
            "original_lines": embedding_data["original_lines"]
        }
    except Exception as e:
        raise ClusteringError(f"Failed to cluster logs: {str(e)}")

@op(
    ins={"cluster_data": In(dict)},
    out=Out(Nothing),
    config_schema={
        "max_samples": DagsterField(Int, default_value=5, is_required=False, description="Maximum number of sample lines to show per cluster")
    }
)
def analyze_patterns(context, cluster_data: Dict[str, Any]) -> None:
    """Analyze patterns in clusters.
    
    Args:
        context: Dagster execution context
        cluster_data: Dictionary with clustering results
    """
    try:
        max_samples = context.op_config["max_samples"]
        clustering_state = cluster_data["clustering_state"]
        
        for cluster_id, cluster_info in clustering_state.clusters.items():
            if cluster_info.pattern:
                context.log.info(f"\nCluster {cluster_id}:")
                context.log.info(f"Pattern: {cluster_info.pattern.pattern}")
                context.log.info(f"Confidence: {cluster_info.pattern.confidence:.2%}")
                context.log.info("Sample lines:")
                for line in cluster_info.sample_lines[:max_samples]:
                    context.log.info(f"  {line}")
    except Exception as e:
        raise ParsingError(f"Failed to analyze patterns: {str(e)}")

# Define the pipeline
@job(
    description="Process log files to extract patterns and clusters"
)
def log_processing_pipeline():
    """Define the complete log processing pipeline."""
    # Read log file
    log_lines = read_log_file()
    
    # Generate embeddings
    embedding_data = generate_embeddings(log_lines)
    
    # Cluster logs
    cluster_data = cluster_logs(embedding_data)
    
    # Analyze patterns
    analyze_patterns(cluster_data)

class LogPipeline:
    """Handles the complete log processing pipeline."""
    
    def __init__(
        self,
        settings: Optional[Settings] = None
    ):
        """Initialize the pipeline.
        
        Args:
            settings: Optional Settings instance
        """
        self.settings = settings or Settings()
    
    def process_file(self, file: Union[str, Path, BinaryIO]) -> Dict[str, Any]:
        """Process a log file through the complete pipeline.
        
        Args:
            file: Path to log file or file-like object
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            FileError: If file cannot be processed
        """
        try:
            # Handle file-like objects
            if not isinstance(file, (str, Path)):
                content = file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                temp_path = Path('temp_upload.log')
                temp_path.write_text(content)
                file_path = str(temp_path)
            else:
                file_path = str(file)
            
            # Execute the pipeline
            result = log_processing_pipeline.execute_in_process(
                run_config={
                    "ops": {
                        "read_log_file": {
                            "config": {"encoding": "utf-8"},
                            "inputs": {"file_path": file_path}
                        },
                        "generate_embeddings": {
                            "config": {"batch_size": self.settings.model.batch_size}
                        },
                        "cluster_logs": {
                            "config": {"n_clusters": self.settings.model.n_clusters}
                        },
                        "analyze_patterns": {
                            "config": {"max_samples": 5}
                        }
                    }
                }
            )
            
            # Clean up temporary file
            if not isinstance(file, (str, Path)):
                temp_path.unlink()
            
            if result.success:
                # Get the clustering state from the result
                cluster_data = result.output_for_node("cluster_logs")
                return cluster_data["clustering_state"]
            else:
                raise FileError(
                    "Pipeline execution failed",
                    details={"errors": str(result.failure_data)}
                )
                
        except Exception as e:
            raise FileError(f"Failed to process file: {str(e)}") 