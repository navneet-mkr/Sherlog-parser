# Quick Start Guide

This guide will help you get started with Sherlog-parser quickly.

## Basic Usage

### 1. Python API

```python
from sherlog_parser import LogParser

# Initialize the parser
parser = LogParser()

# Parse a log file
results = parser.parse_file("app.log")

# Get cluster information
clusters = parser.get_clusters()

# Print extracted patterns
for cluster in clusters:
    print(f"Pattern: {cluster.pattern}")
    print(f"Sample lines: {cluster.sample_lines[:3]}")
```

### 2. Command Line Interface

```bash
# Analyze a single log file
sherlog-parser analyze app.log

# Analyze multiple log files
sherlog-parser analyze app1.log app2.log

# Save results to a file
sherlog-parser analyze app.log --output results.json
```

### 3. Web Interface

1. Start the Streamlit UI:
   ```bash
   streamlit run src/services/ui/app.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Upload your log file and click "Analyze"

## Common Use Cases

### 1. Basic Log Analysis

```python
from sherlog_parser import LogParser

parser = LogParser()
results = parser.parse_file("app.log")

# Print basic statistics
print(f"Total lines: {results.total_lines}")
print(f"Number of clusters: {results.n_clusters}")
print(f"Processing time: {results.processing_time}s")
```

### 2. Pattern Extraction

```python
from sherlog_parser import LogParser

parser = LogParser()
patterns = parser.extract_patterns("app.log")

for pattern in patterns:
    print(f"Pattern: {pattern.regex}")
    print(f"Confidence: {pattern.confidence}")
    print(f"Examples: {pattern.examples[:3]}\n")
```

### 3. Incremental Processing

```python
from sherlog_parser import LogParser

parser = LogParser()

# Process logs in chunks
with open("large.log", "r") as f:
    while True:
        chunk = list(itertools.islice(f, 1000))
        if not chunk:
            break
        parser.process_chunk(chunk)

# Get final results
results = parser.get_results()
```

## Working with Results

### 1. Export Results

```python
from sherlog_parser import LogParser

parser = LogParser()
results = parser.parse_file("app.log")

# Export to JSON
results.to_json("results.json")

# Export to CSV
results.to_csv("patterns.csv")
```

### 2. Visualize Patterns

```python
from sherlog_parser import LogParser, Visualizer

parser = LogParser()
results = parser.parse_file("app.log")

# Create visualizations
viz = Visualizer(results)
viz.plot_cluster_sizes()
viz.plot_pattern_confidence()
viz.save("analysis.html")
```

### 3. Filter Results

```python
from sherlog_parser import LogParser

parser = LogParser()
results = parser.parse_file("app.log")

# Filter by confidence
high_confidence = results.filter(min_confidence=0.9)

# Filter by cluster size
large_clusters = results.filter(min_cluster_size=100)
```

## Next Steps

- Read about [Configuration Options](configuration.md) for customization
- Learn about [Docker Setup](docker.md) for containerized deployment
- Check the [API Reference](../api/core/pipeline.md) for detailed documentation 