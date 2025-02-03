import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from dagster import asset, Output, AssetIn, MetadataValue, Config
from diskcache import Cache
from typing import List, Dict, Any, Tuple
import logging

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize cache for embeddings
cache = Cache("/data/cache")

class ReadLogFileConfig(Config):
    file_path: str
    encoding: str = "utf-8"

class GenerateEmbeddingsConfig(Config):
    batch_size: int = 1000

class ClusterLogsConfig(Config):
    n_clusters: int = 20

@asset(
    description="Reads and validates the input log file",
    metadata={
        "source": "log_file",
        "schema": "single column text logs"
    }
)
def read_log_file(context, config: ReadLogFileConfig) -> pd.DataFrame:
    """Read and validate the log file.
    
    Args:
        context: Dagster execution context
        config: Configuration containing file path and encoding
        
    Returns:
        DataFrame with columns: ['log_line']
        
    Raises:
        FileNotFoundError: If log file doesn't exist
        pd.errors.EmptyDataError: If log file is empty
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
    """Generate embeddings for log lines using sentence transformers.
    
    Args:
        context: Dagster execution context
        read_log_file: DataFrame containing log lines
        config: Configuration containing batch size
        
    Returns:
        Dictionary containing:
            - embeddings: numpy array of embeddings
            - log_lines: original log lines
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
    """Cluster log lines based on their embeddings.
    
    Args:
        context: Dagster execution context
        generate_embeddings: Dictionary containing embeddings and log lines
        config: Configuration containing number of clusters
        
    Returns:
        Dictionary containing:
            - clusters: Dict mapping cluster IDs to log lines
            - centroids: Cluster centroids
            - assignments: Cluster assignments for each log line
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
    """Analyze patterns in each cluster and extract key information.
    
    Args:
        context: Dagster execution context
        cluster_logs: Dictionary containing cluster information
        
    Returns:
        DataFrame containing:
            - cluster_id: Cluster identifier
            - size: Number of logs in cluster
            - sample_lines: Sample log lines from cluster
            - unique_patterns: Number of unique patterns
            - avg_line_length: Average length of log lines
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