"""Log Analysis Pipeline Assets.

This module defines the core data processing pipeline for log analysis using Dagster assets.
The pipeline consists of several stages:

1. Log Ingestion:
   - Reads and validates log files
   - Performs basic preprocessing

2. Embedding Generation:
   - Uses SentenceTransformers (all-MiniLM-L6-v2) for semantic embeddings
   - Processes logs in batches for memory efficiency
   - Caches embeddings for reuse

3. Clustering:
   - Groups similar log patterns using MiniBatchKMeans
   - Identifies core patterns in the data
   - Optimized for large-scale processing

4. Pattern Analysis:
   - Basic statistical analysis of clusters
   - Pattern frequency and distribution
   - Sample extraction for each cluster

5. LLM Analysis:
   - Deep semantic analysis using Ollama
   - Pattern extraction and categorization
   - Anomaly detection and recommendations

The pipeline is designed to be:
- Memory efficient (batch processing)
- Scalable (incremental processing)
- Maintainable (modular design)
- Observable (comprehensive logging and metadata)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from dagster import asset, Output, AssetIn, MetadataValue, Config
from diskcache import Cache
from typing import List, Dict, Any, Tuple
import logging
import time
from src.models.config import OllamaSettings
from src.models.ollama import create_ollama_analyzer

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize cache for embeddings
cache = Cache("/data/cache")

# Asset configurations
class ReadLogFileConfig(Config):
    """Configuration for log file reading operation.
    
    Attributes:
        file_path: Path to the log file to process
        encoding: Character encoding of the log file (default: utf-8)
    """
    file_path: str
    encoding: str = "utf-8"

class GenerateEmbeddingsConfig(Config):
    """Configuration for embedding generation.
    
    Attributes:
        batch_size: Number of log lines to process in each batch
                   Larger batches = faster processing but more memory usage
    """
    batch_size: int = 1000

class ClusterLogsConfig(Config):
    """Configuration for log clustering.
    
    Attributes:
        n_clusters: Number of clusters to generate
                   More clusters = finer granularity but potential overfitting
    """
    n_clusters: int = 20

class LLMConfig(Config):
    """Configuration for LLM-based analysis.
    
    Attributes:
        model_id: Identifier of the Ollama model to use
        temperature: Sampling temperature (0.0 to 1.0)
                    Higher = more creative, lower = more focused
        top_p: Nucleus sampling parameter (0.0 to 1.0)
              Controls diversity of generated text
        top_k: Top-k sampling parameter
              Limits token selection to k most likely tokens
    """
    model_id: str
    temperature: float
    top_p: float
    top_k: int

@asset(
    description="Reads and validates the input log file",
    metadata={
        "source": "log_file",
        "schema": "single column text logs"
    }
)
def read_log_file(context, config: ReadLogFileConfig) -> pd.DataFrame:
    """Read and validate the input log file.
    
    This asset represents the first stage of the pipeline where raw log data
    is ingested and validated. It performs basic checks like file existence
    and non-emptiness.
    
    Args:
        context: Dagster execution context for logging and metadata
        config: Configuration containing file path and encoding
        
    Returns:
        DataFrame with columns: ['log_line']
        Each row contains a single log line as text
        
    Raises:
        FileNotFoundError: If the specified log file doesn't exist
        pd.errors.EmptyDataError: If the log file is empty
        
    Metadata:
        - num_logs: Total number of log lines
        - sample_logs: First few lines of the file
    """
    if not Path(config.file_path).exists():
        raise FileNotFoundError(f"Log file not found: {config.file_path}")
        
    context.log.info(f"Reading log file: {config.file_path}")
    df = pd.read_csv(config.file_path, header=None, names=['log_line'], encoding=config.encoding)
    
    if df.empty:
        raise pd.errors.EmptyDataError("Log file is empty")
    
    # Add metadata about the dataset
    context.add_output_metadata({
        "num_logs": MetadataValue.int(len(df)),
        "sample_logs": MetadataValue.text("\n".join(df['log_line'].head().tolist()))
    })
    
    return df

@asset(
    deps=[read_log_file],
    description="Generates embeddings for log lines using sentence transformers",
    metadata={
        "model": "all-MiniLM-L6-v2",
        "type": "embeddings"
    }
)
def generate_embeddings(
    context,
    read_log_file: pd.DataFrame,
    config: GenerateEmbeddingsConfig
) -> Dict[str, Any]:
    """Generate semantic embeddings for log lines.
    
    This asset uses SentenceTransformers to convert log lines into vector
    embeddings that capture their semantic meaning. The embeddings are used
    for clustering similar logs together.
    
    The process is batched to handle large log files efficiently, and
    embeddings are cached to avoid recomputation.
    
    Args:
        context: Dagster execution context for logging and metadata
        read_log_file: DataFrame containing log lines from previous asset
        config: Configuration for batch processing
        
    Returns:
        Dictionary containing:
            - embeddings: numpy array of shape (n_logs, embedding_dim)
            - log_lines: original log lines for reference
            
    Metadata:
        - embedding_shape: Shape of the embeddings array
        - embedding_dtype: Data type of embeddings
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    log_lines = read_log_file['log_line'].tolist()
    total_batches = (len(log_lines) + config.batch_size - 1) // config.batch_size
    
    # Process in batches
    embeddings = []
    for i in range(0, len(log_lines), config.batch_size):
        batch = log_lines[i:i + config.batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
        context.log.info(f"Processed batch {i//config.batch_size + 1}/{total_batches}")
    
    embeddings_array = np.array(embeddings)
    
    # Add metadata about embeddings
    context.add_output_metadata({
        "embedding_shape": MetadataValue.text(str(embeddings_array.shape)),
        "embedding_dtype": MetadataValue.text(str(embeddings_array.dtype))
    })
    
    return {
        'embeddings': embeddings_array,
        'log_lines': log_lines
    }

@asset(
    deps=[generate_embeddings],
    description="Clusters log lines based on their embeddings",
    metadata={
        "algorithm": "MiniBatchKMeans",
        "type": "clustering"
    }
)
def cluster_logs(
    context,
    generate_embeddings: Dict[str, Any],
    config: ClusterLogsConfig
) -> Dict[str, Any]:
    """Cluster log lines based on their semantic embeddings.
    
    This asset groups similar log lines together using MiniBatchKMeans
    clustering on their embeddings. The clustering helps identify common
    patterns and anomalies in the logs.
    
    The clustering is performed incrementally to handle large datasets
    efficiently.
    
    Args:
        context: Dagster execution context for logging and metadata
        generate_embeddings: Dictionary containing embeddings and log lines
        config: Configuration for clustering parameters
        
    Returns:
        Dictionary containing:
            - clusters: Dict mapping cluster IDs to log lines
            - centroids: Cluster center embeddings
            - assignments: Cluster ID for each log line
            
    Metadata:
        - num_clusters: Number of clusters generated
        - cluster_sizes: Distribution of log lines across clusters
    """
    embeddings = generate_embeddings['embeddings']
    log_lines = generate_embeddings['log_lines']
    
    # Cluster embeddings
    kmeans = MiniBatchKMeans(
        n_clusters=config.n_clusters,
        random_state=42,
        batch_size=1000
    )
    cluster_assignments = kmeans.fit_predict(embeddings)
    
    # Group log lines by cluster
    clusters = {}
    for i, (line, cluster) in enumerate(zip(log_lines, cluster_assignments)):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(line)
    
    # Add metadata about clustering
    cluster_sizes = {k: len(v) for k, v in clusters.items()}
    context.add_output_metadata({
        "num_clusters": MetadataValue.int(config.n_clusters),
        "cluster_sizes": MetadataValue.json(cluster_sizes)
    })
    
    return {
        'clusters': clusters,
        'centroids': kmeans.cluster_centers_,
        'assignments': cluster_assignments
    }

@asset(
    deps=[cluster_logs],
    description="Analyzes patterns in each cluster and extracts key information",
    metadata={
        "type": "analysis"
    }
)
def analyze_patterns(context, cluster_logs: Dict[str, Any]) -> pd.DataFrame:
    """Perform basic statistical analysis of log clusters.
    
    This asset analyzes each cluster to extract basic statistics and
    patterns. It serves as a preliminary analysis before the more
    detailed LLM-based analysis.
    
    Args:
        context: Dagster execution context for logging and metadata
        cluster_logs: Dictionary containing cluster information
        
    Returns:
        DataFrame containing per-cluster statistics:
            - cluster_id: Unique identifier for each cluster
            - size: Number of logs in the cluster
            - sample_lines: Representative log lines
            - unique_patterns: Number of distinct patterns
            - avg_line_length: Average length of log lines
            
    Metadata:
        - total_clusters: Number of clusters analyzed
        - avg_cluster_size: Average number of logs per cluster
        - largest_cluster: Size of the largest cluster
    """
    clusters = cluster_logs['clusters']
    
    results = []
    for cluster_id, lines in clusters.items():
        # Calculate statistics
        result = {
            'cluster_id': cluster_id,
            'size': len(lines),
            'sample_lines': lines[:5],
            'unique_patterns': len(set(lines)),
            'avg_line_length': np.mean([len(line) for line in lines])
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Add metadata about analysis
    context.add_output_metadata({
        "total_clusters": MetadataValue.int(len(df)),
        "avg_cluster_size": MetadataValue.float(df['size'].mean()),
        "largest_cluster": MetadataValue.int(df['size'].max())
    })
    
    context.log.info("\nCluster Analysis Results:")
    context.log.info(df)
    
    return df

@asset(
    deps=[analyze_patterns],
    description="Uses Ollama to analyze patterns and extract insights",
    metadata={
        "type": "llm_analysis"
    }
)
def analyze_patterns_with_ollama(
    context,
    analyze_patterns: pd.DataFrame,
    config: LLMConfig
) -> Dict[str, Any]:
    """Perform deep semantic analysis of log patterns using Ollama LLM.
    
    This asset represents the final stage of the pipeline where an LLM
    is used to:
    1. Extract and explain log patterns
    2. Identify severity levels and categories
    3. Detect potential anomalies
    4. Provide actionable recommendations
    
    The analysis is performed per cluster, with results aggregated and
    enriched with usage statistics and metadata.
    
    Args:
        context: Dagster execution context for logging and metadata
        analyze_patterns: DataFrame with basic pattern analysis
        config: LLM configuration for Ollama
        
    Returns:
        Dictionary containing:
            - results: List of per-cluster analyses including:
                * patterns: Extracted regex patterns with metadata
                * summary: Cluster summary
                * anomalies: Detected anomalies
                * recommendations: Suggested actions
            - metadata: Analysis metadata including:
                * model configuration
                * token usage statistics
                * timestamp
                
    Metadata:
        - clusters_analyzed: Number of clusters processed
        - total_patterns: Total number of patterns extracted
        - total_anomalies: Total number of anomalies detected
        - total_tokens_used: Total tokens consumed by LLM
        - model_info: LLM configuration details
    
    Notes:
        - The LLM analysis is performed incrementally per cluster
        - Token usage is tracked for monitoring and optimization
        - Results are structured for easy visualization and export
    """
    try:
        # Get Ollama settings from environment
        settings = OllamaSettings()
        
        # Initialize Ollama analyzer
        analyzer = create_ollama_analyzer(
            base_url=f"{settings.host}:{settings.port}",
            model_id=config.model_id,
            config={
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k
            }
        )
        
        results = []
        total_clusters = len(analyze_patterns)
        total_tokens = 0
        
        for idx, row in analyze_patterns.iterrows():
            context.log.info(f"Processing cluster {idx + 1}/{total_clusters}")
            
            # Get sample logs from cluster
            sample_lines = row['sample_lines']
            if not isinstance(sample_lines, list):
                sample_lines = [sample_lines]
            
            # Analyze cluster logs
            analysis, usage = analyzer.analyze_logs(sample_lines)
            total_tokens += usage.get("total_tokens", 0)
            
            # Structure the results
            result = {
                'cluster_id': row['cluster_id'],
                'size': row['size'],
                'patterns': [p.model_dump() for p in analysis.patterns],
                'summary': analysis.summary,
                'anomalies': analysis.anomalies,
                'recommendations': analysis.recommendations,
                'usage': usage
            }
            results.append(result)
            
            # Log progress with pattern information
            context.log.info(
                f"Analyzed cluster {row['cluster_id']}: "
                f"Found {len(result['patterns'])} patterns, "
                f"{len(result['anomalies'])} anomalies. "
                f"Tokens used: {usage.get('total_tokens', 0)}"
            )
        
        # Add detailed metadata about analysis
        context.add_output_metadata({
            "clusters_analyzed": MetadataValue.int(len(results)),
            "total_patterns": MetadataValue.int(sum(len(r['patterns']) for r in results)),
            "total_anomalies": MetadataValue.int(sum(len(r['anomalies']) for r in results)),
            "total_tokens_used": MetadataValue.int(total_tokens),
            "model_info": MetadataValue.json({
                'model': config.model_id,
                'temperature': config.temperature,
                'top_p': config.top_p,
                'top_k': config.top_k
            })
        })
        
        return {
            'results': results,
            'metadata': {
                'model': config.model_id,
                'temperature': config.temperature,
                'timestamp': time.time(),
                'ollama_settings': settings.model_dump(),
                'total_tokens': total_tokens
            }
        }
            
    except Exception as e:
        context.log.error(f"Error in Ollama analysis: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to analyze patterns with Ollama: {str(e)}") 