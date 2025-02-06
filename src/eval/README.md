# LogParser-LLM Evaluation Framework

This directory contains the evaluation framework for LogParser-LLM, designed to assess the performance of our log parsing implementation against benchmark datasets from the LogParser-LLM paper.

## Overview

The evaluation framework:
- Uses benchmark datasets: Loghub-2k and LogPub
- Calculates metrics from the LogParser-LLM paper
- Integrates with Dagster for pipeline orchestration
- Supports multiple Ollama models
- Caches results for faster re-evaluation

## Directory Structure

```
src/eval/
├── README.md           # This file
├── datasets.py         # Dataset loading functionality
├── metrics.py          # Metric calculation implementations
└── eval_pipeline.py    # Dagster evaluation pipeline
```

## Metrics Implemented

1. **Grouping Accuracy (GA)**: Measures how well the parser groups similar log messages
2. **Parsing Accuracy (PA)**: Measures template extraction accuracy
3. **F1-score of Grouping Accuracy (FGA)**: F1 score for log grouping
4. **F1-score of Template Accuracy (FTA)**: F1 score for template matching
5. **Grouping Granularity Distance (GGD)**: Measures grouping granularity compared to ground truth
6. **Parsing Granularity Distance (PGD)**: Measures parsing granularity compared to ground truth
7. **Inference Time**: Average time per log message

## Dataset Setup

### Downloading Datasets

#### Loghub-2k Dataset
The Loghub-2k dataset is available from the LogPAI team's repository:

1. Using the download script (Recommended):
```bash
./download_datasets.sh
```
This script will automatically download and organize all required datasets in the correct directory structure.

2. Manual download from LogHub:
```bash
# Create dataset directory
mkdir -p data/eval_datasets/loghub_2k

# Download and extract datasets
wget https://github.com/logpai/loghub/raw/master/Apache/Apache_2k.log_structured.csv -O data/eval_datasets/loghub_2k/Apache/Apache.log_structured.csv
wget https://github.com/logpai/loghub/raw/master/Apache/Apache_2k.log_templates.csv -O data/eval_datasets/loghub_2k/Apache/Apache.log_templates.csv

wget https://github.com/logpai/loghub/raw/master/Hadoop/Hadoop_2k.log_structured.csv -O data/eval_datasets/loghub_2k/Hadoop/Hadoop.log_structured.csv
wget https://github.com/logpai/loghub/raw/master/Hadoop/Hadoop_2k.log_templates.csv -O data/eval_datasets/loghub_2k/Hadoop/Hadoop.log_templates.csv

# Repeat for other systems (Linux, Zookeeper, etc.)
```

Alternatively, you can manually download from:
- LogHub Repository: https://github.com/logpai/loghub
- Direct Download: https://zenodo.org/record/3227177

#### LogPub Dataset
The LogPub dataset is available from the LogPAI team's LogPub repository:

1. Download from LogPub:
```bash
# Create dataset directory
mkdir -p data/eval_datasets/logpub

# Download and extract datasets
# Note: LogPub requires registration. Visit https://github.com/logpai/LogPub
# After registration, download the datasets and place them in the logpub directory
```

Manual download steps for LogPub:
1. Visit https://github.com/logpai/LogPub
2. Fill out the registration form
3. Download the benchmark datasets
4. Extract and place in `data/eval_datasets/logpub/`

### Dataset Structure

After downloading, organize the datasets in the following structure:
```
data/eval_datasets/
├── loghub_2k/
│   ├── Apache/
│   │   ├── Apache.log_structured.csv
│   │   └── Apache.log_templates.csv
│   ├── Hadoop/
│   │   ├── Hadoop.log_structured.csv
│   │   └── Hadoop.log_templates.csv
│   ├── Linux/
│   │   ├── Linux.log_structured.csv
│   │   └── Linux.log_templates.csv
│   └── Zookeeper/
│       ├── Zookeeper.log_structured.csv
│       └── Zookeeper.log_templates.csv
└── logpub/
    ├── System1/
    │   ├── System1.log_structured.csv
    │   └── System1.log_templates.csv
    └── ...
```

### File Format Requirements

1. **Structured Log Files** (*.log_structured.csv):
   - Must contain 'Content' column with raw log messages
   - Must contain 'ParameterList' columns for parameter values
   - CSV format with proper headers

2. **Template Files** (*.log_templates.csv):
   - Must contain 'EventTemplate' column with ground truth templates
   - CSV format with proper headers

### Verifying Dataset Integrity

You can verify your dataset setup using the built-in validation:

```python
from src.eval.datasets import DatasetLoader

# Initialize loader
loader = DatasetLoader()

# List available datasets
available = loader.list_available_datasets()
print("Available datasets:", available)

# Try loading a dataset to verify
dataset = loader.load_dataset("Apache", "loghub_2k")
print(f"Loaded {dataset.name} with {dataset.size} logs")
```

### Dataset Statistics

Default test datasets from Loghub-2k:
- Apache: ~2,000 logs
- Hadoop: ~2,000 logs
- Linux: ~2,000 logs
- Zookeeper: ~2,000 logs

## Running Evaluations

### Option 1: Using Docker (Recommended)

1. Start the evaluation container:
```bash
docker compose --profile eval up eval
```

This will:
- Start required services (Dagster, Ollama)
- Mount datasets and cache volumes
- Run the evaluation pipeline
- Show results in Dagster UI

### Option 2: Running Locally

1. Ensure Ollama is running and accessible at `http://localhost:11434`

2. Run the evaluation script:
```bash
./evaluate.sh
```

## Configuration

### Environment Variables

- `OLLAMA_HOST`: Ollama API host (default: http://localhost)
- `OLLAMA_PORT`: Ollama API port (default: 11434)
- `DAGSTER_GRPC_HOST`: Dagster gRPC host (default: localhost)
- `DAGSTER_PORT`: Dagster UI port (default: 3000)

### Pipeline Configuration

The evaluation pipeline can be configured through Dagster. Key parameters:

```python
config = {
    "base_dir": "/app/data/eval_datasets",  # Dataset directory
    "cache_dir": "/app/data/eval_cache",    # Cache directory
    "ollama_base_url": "http://localhost:11434",
    "model_name": "mistral",                # Ollama model to use
    "similarity_threshold": 0.8,            # Template matching threshold
    "batch_size": 1000                      # Batch size for processing
}
```

## Viewing Results

1. Access the Dagster UI at `http://localhost:3000`
2. Navigate to the `evaluate_logparser_llm` job
3. View results in:
   - Job logs (real-time progress)
   - Asset materializations (metrics tables)
   - Markdown reports (comparison tables)

## Caching

Results are cached to speed up re-evaluation:
- Location: `data/eval_cache/`
- Format: JSON files with parsed templates and inference times
- Naming: `{dataset_name}_{model_name}_results.json`

## Development

To modify the evaluation framework:

1. Dataset Loading (`datasets.py`):
   - Add new dataset types
   - Modify parsing logic
   - Add data validation

2. Metrics (`metrics.py`):
   - Add new metrics
   - Modify calculation methods
   - Add validation

3. Pipeline (`eval_pipeline.py`):
   - Modify Dagster ops
   - Add new pipeline steps
   - Configure reporting

## Troubleshooting

1. **Dataset Not Found**:
   - Check directory structure
   - Verify file names match expected format
   - Ensure read permissions

2. **Ollama Connection**:
   - Verify Ollama is running
   - Check URL/port configuration
   - Ensure model is downloaded

3. **Dagster Issues**:
   - Check Dagster UI is accessible
   - Verify gRPC connection
   - Check pipeline logs

## Contributing

When adding features:
1. Follow existing code structure
2. Add appropriate tests
3. Update this documentation
4. Test both local and Docker execution 