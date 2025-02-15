# Evaluation Framework Documentation

## Overview

The evaluation framework is designed to assess the performance of the log parsing system against benchmark datasets. It provides a comprehensive suite of tools for measuring accuracy, performance, and template quality.

## Architecture

```
evaluation/
├── UI Layer (src/eval/ui.py)
│   ├── Dataset Selection
│   ├── Model Configuration
│   └── Results Visualization
│
├── Pipeline Layer (src/pathway_pipeline/eval_pipeline.py)
│   ├── Template Matching
│   ├── LLM Processing
│   └── Metrics Calculation
│
└── Data Layer (src/eval/datasets/)
    ├── Dataset Loading
    ├── Ground Truth Management
    └── Results Storage
```

## Components

### 1. UI Layer (`src/eval/ui.py`)

The UI layer provides an interactive dashboard for running evaluations:

```python
# Key Components
- Dataset Selector: Choose system and dataset type
- Model Configuration: Adjust similarity threshold and batch size
- Results Display: Show metrics and visualizations
```

#### Features:
- Real-time metric updates
- Interactive parameter adjustment
- Template distribution visualization
- Detailed results exploration

### 2. Pipeline Layer (`src/pathway_pipeline/eval_pipeline.py`)

The core evaluation pipeline implements a two-stage approach:

1. **Template Matching Stage**:
```python
# Using vector similarity search
logs_with_matches = self.logs.join(
    self.template_index.query(
        self.logs.content,
        k=1,
        distance_threshold=self.similarity_threshold
    ),
    pw.left.content == pw.right.query
)
```

2. **LLM Processing Stage**:
```python
# For unmatched logs, use LLM
logs_without_matches = (
    self.logs
    .filter(lambda t: t.id not in logs_with_matches.id)
    .select(
        prompt=self._build_template_prompt(pw.this.content)
    )
    .select(
        llm_response=self.model(pw.this.prompt)
    )
)
```

### 3. Data Layer

#### Dataset Structure:
```
data/eval_datasets/
├── loghub_2k/
│   ├── Apache/
│   ├── Hadoop/
│   ├── Linux/
│   └── Zookeeper/
└── logpub/
    └── [similar structure]
```

#### Ground Truth Format:
```json
{
    "log_id": {
        "template": "<timestamp> ERROR Connection failed from <ip>",
        "variables": [
            {"position": 0, "type": "timestamp"},
            {"position": 4, "type": "ip"}
        ]
    }
}
```

## Workflow

1. **Initialization**:
   ```python
   pipeline = EvaluationPipeline(
       base_dir="./data/eval_datasets",
       dataset_type=dataset_type,
       system=system,
       similarity_threshold=0.8,
       batch_size=1000
   )
   ```

2. **Pipeline Setup**:
   ```python
   pipeline.setup_pipeline()
   # - Loads dataset
   # - Creates template index
   # - Initializes LLM
   ```

3. **Evaluation Process**:
   ```python
   metrics = pipeline.evaluate()
   # - Processes logs
   # - Calculates metrics
   # - Generates reports
   ```

## Metrics

The framework calculates several key metrics:

1. **Accuracy Metrics**:
   - Grouping Accuracy (GA): Measures template clustering quality
   - Parsing Accuracy (PA): Measures variable extraction accuracy
   - F1 Score: Combined measure of precision and recall

2. **Performance Metrics**:
   - Processing Time: Average time per log entry
   - Template Coverage: Percentage of logs matched to templates
   - Memory Usage: Resource utilization statistics

## Configuration

### Environment Variables:
```bash
OLLAMA_BASE_URL=http://localhost:11434  # Ollama API endpoint
OLLAMA_MODEL=mistral                    # Model to use
LOG_LEVEL=INFO                          # Logging verbosity
```

### Pipeline Configuration:
```yaml
pipeline:
  batch_size: 1000                # Logs per batch
  similarity_threshold: 0.8        # Template matching threshold
  cache_dir: "./cache/eval"       # Cache location
  output_dir: "./output/eval"     # Results location
```

## Output Structure

```
output/eval/
├── {system}_{dataset}_metrics.json     # Evaluation metrics
├── {system}_{dataset}_templates.csv    # Extracted templates
└── {system}_{dataset}_report.md        # Detailed report
```

## Usage Examples

1. **Basic Evaluation**:
   ```bash
   ./evaluate.sh --dataset loghub_2k --system Apache
   ```

2. **Custom Configuration**:
   ```bash
   SIMILARITY_THRESHOLD=0.85 BATCH_SIZE=500 ./evaluate.sh
   ```

3. **Distributed Evaluation**:
   ```bash
   ./evaluate.sh --ollama-host http://ollama1:11434
   ./evaluate.sh --ollama-host http://ollama2:11434
   ```

## Integration Points

1. **With CI/CD**:
   ```yaml
   - name: Run Evaluation
     run: |
       ./evaluate.sh
       python src/eval/check_metrics.py --min-accuracy 0.95
   ```

2. **With Monitoring**:
   ```python
   # Metrics are exposed for Prometheus
   eval_accuracy = Gauge('eval_accuracy', 'Evaluation accuracy')
   eval_processing_time = Histogram('eval_time', 'Processing time')
   ```

## Best Practices

1. **Dataset Management**:
   - Keep datasets versioned
   - Validate ground truth before evaluation
   - Use consistent naming conventions

2. **Performance Optimization**:
   - Adjust batch size based on available memory
   - Use GPU acceleration when available
   - Cache intermediate results

3. **Result Analysis**:
   - Compare results across different models
   - Track metrics over time
   - Investigate template mismatches

## Troubleshooting

1. **Common Issues**:
   - Dataset loading failures
   - Memory constraints
   - Model availability

2. **Solutions**:
   - Verify dataset structure
   - Adjust batch size
   - Check Ollama connection

## Future Improvements

1. **Planned Features**:
   - Multi-model comparison
   - Custom metric definitions
   - Automated regression testing

2. **Performance Enhancements**:
   - Parallel processing
   - Improved caching
   - Optimized template matching

## References

1. Benchmark Datasets:
   - LogHub: https://github.com/logpai/loghub
   - LogPub: https://zenodo.org/record/3227177

2. Evaluation Metrics:
   - He, P., et al. (2016). An evaluation study on log parsing
   - Du, M., et al. (2016). DeepLog: Anomaly Detection 