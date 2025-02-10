# Evaluation Framework

This directory contains the evaluation framework for assessing LogParser-LLM's performance against benchmark datasets.

## 📁 Directory Structure

```
eval/
├── datasets/
│   ├── loghub_2k/        # Loghub-2k benchmark datasets
│   └── logpub/           # LogPub benchmark datasets (manual download)
├── metrics/
│   └── metrics.py        # Evaluation metrics implementation
├── pipeline/
│   └── eval_pipeline.py  # Pathway evaluation pipeline
└── README.md            # This file
```

## 🚀 Quick Start

1. Download datasets:
```bash
# Loghub-2k datasets (automatic)
./download_datasets.sh

# LogPub datasets (manual)
# 1. Register at https://zenodo.org/record/3227177
# 2. Download and extract to data/eval_datasets/logpub/
```

2. Run evaluation:
```bash
# Automatic Ollama configuration (recommended)
./evaluate.sh

# Use local Ollama installation
./evaluate.sh --use-local-ollama

# Use remote Ollama instance
./evaluate.sh --ollama-host http://your-ollama-server --ollama-port 11434
```

3. View results at http://localhost:8502

## 📊 Features

- Real-time streaming processing with Pathway
- Support for multiple benchmark datasets
- Comprehensive metrics calculation
- Interactive results visualization
- Result caching for faster re-runs
- Flexible Ollama integration options

## ⚙️ Configuration

### Command Line Options

The evaluation script supports the following options:
```bash
Options:
  --use-local-ollama     Use local Ollama instance instead of container
  --ollama-host HOST     Specify custom Ollama host (default: http://localhost)
  --ollama-port PORT     Specify custom Ollama port (default: 11434)
  -h, --help            Show help message
```

### Environment Variables

The evaluation pipeline can be configured through environment variables:

```bash
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434  # Automatically set by script
OLLAMA_MODEL=mistral

# Pipeline settings
SIMILARITY_THRESHOLD=0.8
BATCH_SIZE=1000
CACHE_DIR=./cache/eval
```

### Ollama Integration

The framework supports three modes of Ollama integration:

1. **Automatic Detection** (Default):
   - Detects if local Ollama is running
   - Prompts for user preference
   - Handles model management

2. **Local Installation**:
   - Uses existing Ollama setup
   - Shares models with other applications
   - Faster startup time

3. **Remote Instance**:
   - Connects to remote Ollama server
   - Supports custom host/port
   - Enables distributed setup

## 📈 Metrics

The framework calculates several metrics:

1. **Accuracy Metrics**:
   - Grouping Accuracy (GA)
   - Parsing Accuracy (PA)
   - F1 Score

2. **Performance Metrics**:
   - Processing Time
   - Memory Usage
   - Template Coverage

## 🔍 Results

Results are saved in multiple formats:

1. **CSV Files**:
   - `parsed_logs.csv`: Parsed log entries
   - `templates.csv`: Extracted templates
   - `metrics.csv`: Evaluation metrics

2. **Visualizations**:
   - Template matching accuracy
   - Processing time distribution
   - Memory usage over time

## 🔧 Troubleshooting

1. **Ollama Issues**:
   - Check if Ollama is running (`curl http://localhost:11434`)
   - Verify model availability (`curl http://localhost:11434/api/tags`)
   - Check GPU access if using NVIDIA

2. **Dataset Issues**:
   - Verify dataset format
   - Check file permissions
   - Ensure correct directory structure

3. **Performance Issues**:
   - Adjust batch size
   - Monitor memory usage
   - Check system resources

4. **Common Solutions**:
   - Clear cache: `rm -rf cache/eval/*`
   - Restart services: `docker compose restart evaluation`
   - Check logs: `docker compose logs evaluation`

## 📚 References

1. He, P., et al. (2016). An evaluation study on log parsing and its use in log mining. In DSN 2016.
2. Du, M., et al. (2016). DeepLog: Anomaly Detection and Diagnosis from System Logs. In CCS 2016.
3. Zhu, J., et al. (2019). Tools and Benchmarks for Automated Log Parsing. In ICSE 2019. 