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

1. Create the dataset directory:
```bash
mkdir -p data/eval_datasets/{loghub_2k,logpub}
```

2. Download and place datasets in the following structure:
```
data/eval_datasets/
├── loghub_2k/
│   ├── Apache/
│   │   ├── Apache.log_structured.csv
│   │   └── Apache.log_templates.csv
│   ├── Hadoop/
│   │   ├── Hadoop.log_structured.csv
│   │   └── Hadoop.log_templates.csv
│   └── ...
└── logpub/
    └── ...
```

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