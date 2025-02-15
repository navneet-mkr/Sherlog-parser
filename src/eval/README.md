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

## 📋 Example Scenarios

### 1. Basic Evaluation
```bash
# Simple evaluation with default settings
./evaluate.sh

# With specific dataset
./evaluate.sh --dataset loghub_2k --system Apache

# With custom thresholds
SIMILARITY_THRESHOLD=0.85 BATCH_SIZE=500 ./evaluate.sh
```

### 2. Model Comparison
```bash
# Evaluate different models
export DATASET_TYPE=loghub_2k
export SYSTEM=Hadoop

# Mistral model
OLLAMA_MODEL=mistral ./evaluate.sh

# Llama2 model
OLLAMA_MODEL=llama2 ./evaluate.sh

# Compare results
python src/eval/compare_models.py \
  output/eval/mistral_metrics.json \
  output/eval/llama2_metrics.json
```

### 3. Distributed Evaluation
```bash
# Split evaluation across multiple instances
./evaluate.sh --ollama-host http://ollama1:11434 --dataset-split 0-500
./evaluate.sh --ollama-host http://ollama2:11434 --dataset-split 501-1000

# Merge results
python src/eval/merge_results.py output/eval/split_*_metrics.json
```

### 4. Performance Testing

1. **Memory Usage Analysis**:
```bash
# Test with different batch sizes
for size in 100 500 1000 2000; do
  BATCH_SIZE=$size ./evaluate.sh
done

# Generate memory report
python src/eval/analyze_memory.py output/eval/*_metrics.json
```

2. **GPU vs CPU Comparison**:
```bash
# CPU-only evaluation
CUDA_VISIBLE_DEVICES="" ./evaluate.sh

# GPU evaluation
CUDA_VISIBLE_DEVICES=0 ./evaluate.sh

# Compare performance
python src/eval/compare_performance.py \
  output/eval/cpu_metrics.json \
  output/eval/gpu_metrics.json
```

3. **Throughput Testing**:
```bash
# Test with different concurrency levels
for level in 1 2 4 8; do
  CONCURRENCY=$level ./evaluate.sh
done

# Generate throughput report
python src/eval/analyze_throughput.py output/eval/*_metrics.json
```

### 5. Custom Evaluation Scenarios

1. **Custom Dataset Evaluation**:
```bash
# Prepare custom dataset
mkdir -p data/eval_datasets/custom
cp your-logs.txt data/eval_datasets/custom/

# Run evaluation
./evaluate.sh --dataset custom --system YourSystem
```

2. **Template Extraction Analysis**:
```bash
# Focus on template quality
export TEMPLATE_METRICS_ONLY=true
export MIN_TEMPLATE_COVERAGE=0.95
./evaluate.sh

# Analyze template quality
python src/eval/analyze_templates.py output/eval/templates.csv
```

3. **Error Analysis**:
```bash
# Generate detailed error reports
export DETAILED_ERRORS=true
./evaluate.sh

# Analyze error patterns
python src/eval/analyze_errors.py output/eval/error_report.json
```

### 6. Integration Examples

1. **With CI/CD Pipeline**:
```bash
# In GitHub Actions workflow
steps:
  - uses: actions/checkout@v2
  - name: Run Evaluation
    run: |
      ./evaluate.sh --use-local-ollama
      python src/eval/check_metrics.py \
        --min-accuracy 0.95 \
        output/eval/*_metrics.json
```

2. **With Monitoring**:
```bash
# Export metrics to Prometheus
export EXPORT_METRICS=true
./evaluate.sh

# In Grafana dashboard
# - Plot accuracy trends
# - Monitor processing time
# - Track memory usage
```

3. **With Automated Reports**:
```bash
# Generate PDF report
./evaluate.sh
python src/eval/generate_report.py \
  --format pdf \
  --output report.pdf \
  output/eval/*_metrics.json

# Send report by email
python src/eval/send_report.py report.pdf
```

### 7. Advanced Configuration Examples

1. **Custom Model Settings**:
```bash
# Fine-tuned evaluation
export OLLAMA_MODEL=mistral
export TEMPERATURE=0.1
export TOP_P=0.9
export TOP_K=40
./evaluate.sh
```

2. **Custom Metrics**:
```bash
# Add custom metrics
export CUSTOM_METRICS_MODULE=src/eval/custom_metrics.py
export CUSTOM_METRICS=timing,memory,accuracy
./evaluate.sh
```

3. **Resource Management**:
```bash
# Memory-optimized
export MAX_MEMORY=8G
export BATCH_SIZE=100
export CACHE_SIZE=2G
./evaluate.sh

# GPU-optimized
export CUDA_VISIBLE_DEVICES=0,1
export BATCH_SIZE=2000
export NUM_THREADS=8
./evaluate.sh
```

## 📚 References

1. He, P., et al. (2016). An evaluation study on log parsing and its use in log mining. In DSN 2016.
2. Du, M., et al. (2016). DeepLog: Anomaly Detection and Diagnosis from System Logs. In CCS 2016.
3. Zhu, J., et al. (2019). Tools and Benchmarks for Automated Log Parsing. In ICSE 2019. 