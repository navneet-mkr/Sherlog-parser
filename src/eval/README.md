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
./evaluate.sh
```

3. View results at http://localhost:8502

## 📊 Features

- Real-time streaming processing with Pathway
- Support for multiple benchmark datasets
- Comprehensive metrics calculation
- Interactive results visualization
- Result caching for faster re-runs

## ⚙️ Configuration

The evaluation pipeline can be configured through environment variables:

```bash
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Pipeline settings
SIMILARITY_THRESHOLD=0.8
BATCH_SIZE=1000
CACHE_DIR=./cache/eval
```

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

1. **Dataset Issues**:
   - Verify dataset format
   - Check file permissions
   - Ensure correct directory structure

2. **Performance Issues**:
   - Adjust batch size
   - Monitor memory usage
   - Check system resources

3. **Model Issues**:
   - Verify Ollama connection
   - Check model availability
   - Monitor GPU utilization

## 📚 References

1. He, P., et al. (2016). An evaluation study on log parsing and its use in log mining. In DSN 2016.
2. Du, M., et al. (2016). DeepLog: Anomaly Detection and Diagnosis from System Logs. In CCS 2016.
3. Zhu, J., et al. (2019). Tools and Benchmarks for Automated Log Parsing. In ICSE 2019. 