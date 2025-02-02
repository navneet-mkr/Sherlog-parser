# Pipeline Module

The pipeline module is the core component of Sherlog-parser that handles the complete log processing workflow. It provides both high-level interfaces and granular control over the log analysis process.

## LogPipeline Class

::: src.core.pipeline.LogPipeline
    options:
      show_root_heading: true
      show_source: true

## Pipeline Operations

### Read Log File Operation

::: src.core.pipeline.read_log_file
    options:
      show_source: true

### Generate Embeddings Operation

::: src.core.pipeline.generate_embeddings
    options:
      show_source: true

### Cluster Logs Operation

::: src.core.pipeline.cluster_logs
    options:
      show_source: true

## Configuration Models

### ReadLogConfig

::: src.core.pipeline.ReadLogConfig
    options:
      show_source: true

### EmbeddingConfig

::: src.core.pipeline.EmbeddingConfig
    options:
      show_source: true

### ClusteringConfig

::: src.core.pipeline.ClusteringConfig
    options:
      show_source: true

## Data Models

### EmbeddingData

::: src.core.pipeline.EmbeddingData
    options:
      show_source: true

### ClusteringData

::: src.core.pipeline.ClusteringData
    options:
      show_source: true 