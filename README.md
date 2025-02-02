# 🔍 Sherlog-parser

A powerful, intelligent log parsing and analysis tool that leverages Large Language Models (LLMs) and machine learning to automatically cluster, analyze, and extract patterns from log files. Built with state-of-the-art LLMs at its core, it provides deep semantic understanding of your logs.

## ✨ Features

- 🤖 **LLM-Powered Analysis**:
  - Deep semantic understanding using state-of-the-art LLMs
  - Support for multiple LLM providers (OpenAI, local LLaMA, etc.)
  - Intelligent context extraction and summarization
- 🧠 **Intelligent Log Clustering**: Uses embeddings and incremental clustering to group similar log messages
- 🎯 **Pattern Extraction**: Automatically extracts regex patterns from log clusters
- ⚡ **High Performance**:
  - 💾 Efficient caching of embeddings using `diskcache`
  - 📊 Incremental clustering with scikit-learn
  - 📦 Batch processing for large log files
- 🚀 **Production Ready**:
  - 🛡️ Type-safe with Pydantic models
  - ⚙️ Configurable via YAML and environment variables
  - ✅ Extensive test coverage
  - 🔄 Proper error handling and logging
- 🔄 **Advanced Pipeline**:
  - 🏗️ Built with Dagster for robust pipeline orchestration
  - 📊 Visual pipeline UI for debugging and monitoring
  - 📦 Automatic dependency management
  - 📝 Asset tracking and materialization

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sherlog-parser.git
cd sherlog-parser

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📚 Documentation

The complete documentation is available in two formats:

### 🌐 Online Documentation

Visit our [Documentation Site](https://yourusername.github.io/sherlog-parser) for:
- 📖 Detailed API reference
- 👥 User guides
- ⚙️ Configuration options
- 🏗️ Development guidelines
- 🔍 Architecture overview

### Local Documentation

To build and view the documentation locally:

```bash
# Install documentation dependencies
pip install -r requirements.txt

# Build the documentation
mkdocs build

# Serve the documentation locally
mkdocs serve
```

Then visit `http://127.0.0.1:8000` in your web browser.

## 🚀 Quick Start

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

## 🐳 Quick Start with Docker

The easiest way to get started is using Docker Compose:

```bash
# Start all services with a single command
docker-compose -f src/config/docker-compose.yml up --build
```

This will:
1. Build and start all necessary services
2. Launch the Streamlit UI on http://localhost:8501
3. Start the Dagster pipeline service on http://localhost:3000

### 🎮 Using the Application

1. **🖥️ Access the Streamlit UI**:
   - Open your browser and go to `http://localhost:8501`
   - Upload your log file using the file uploader
   - Adjust the number of clusters and batch size if needed
   - Click "Analyze Logs" to start processing

2. **📊 Monitor Pipeline Progress**:
   - The Dagster UI will be available at `http://localhost:3000`
   - You'll get a Run ID after starting the analysis
   - Use the Dagster UI to:
     - Monitor pipeline execution
     - View detailed logs and progress
     - Access results and visualizations
     - Debug any issues if they occur

3. **📈 View Results**:
   - After processing completes, you'll see:
     - Extracted patterns from your logs
     - Cluster information and statistics
     - Sample log lines for each pattern
     - Pattern confidence scores

### 🔧 Docker Services

The application runs three main services:
1. **Base**: Contains core dependencies and shared code
2. **Dagster**: Handles pipeline orchestration and monitoring
3. **Streamlit**: Provides the user interface for log analysis

All services are configured to work together out of the box, with:
- Shared volumes for log data and cache
- Automatic service discovery
- Persistent storage for results
- Environment variable configuration

## ⚙️ Configuration

Configuration can be provided via `config.yaml` or environment variables:

```yaml
# Application configuration
env: production
debug: false

# Logging configuration
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: logs/app.log

# Cache configuration
cache:
  enabled: true
  directory: cache
  max_size_mb: 1024
  ttl_days: 30

# Model configuration
model:
  embedding_model: all-MiniLM-L6-v2
  n_clusters: 20
  batch_size: 1000
  random_seed: 42
```

## 🏗️ Architecture

The project follows a modular architecture with clear separation of concerns:

- **🔧 Core Components**:
  - `embeddings.py`: Handles text embedding generation and caching
  - `clustering.py`: Implements incremental clustering and pattern extraction
  - `config.py`: Manages configuration and settings

- **📊 Models**:
  - `clustering.py`: Models for clustering state and predictions

## 📖 Usage Guide

### 🖥️ Using the Streamlit UI

The Streamlit UI provides an interactive interface for analyzing log files. To start the UI:

```bash
# Start the Streamlit application
streamlit run src/services/ui/app.py
```

This will open a web browser window with the Sherlog-parser interface. From here you can:

1. **Upload Log Files**:
   - Click the "Choose a log file" button
   - Select a `.log` or `.txt` file from your computer
   - The file will be automatically processed and analyzed

2. **View Cluster Information**:
   - After processing, you'll see a dropdown to select different clusters
   - Each cluster represents a group of similar log messages
   - For each cluster, you can view:
     - Number of log lines in the cluster
     - Extracted pattern confidence
     - The regex pattern that matches the logs
     - Sample log lines from the cluster

### Using the API

For programmatic access, you can use the HTTP API:

1. **Start the API Server**:
```bash
uvicorn src.services.api.main:app --host 0.0.0.0 --port 8000
```

2. **API Endpoints**:
```bash
# Parse a single log line
curl -X POST "http://localhost:8000/parse" \
  -H "Content-Type: application/json" \
  -d '{"log_line": "2024-02-14 10:15:30 INFO [main] Starting application"}'

# Process a batch of log lines
curl -X POST "http://localhost:8000/parse/batch" \
  -H "Content-Type: application/json" \
  -d '{"log_lines": ["line1", "line2"]}'

# Get cluster information
curl "http://localhost:8000/clusters/1"
```

### Using Docker

For containerized deployment:

1. **Build and Start Services**:
```bash
# Build images
docker-compose -f src/config/docker-compose.yml build

# Start services
docker-compose -f src/config/docker-compose.yml up -d
```

2. **Access Services**:
- Streamlit UI: http://localhost:8501
- API: http://localhost:8000

### Configuration Options

1. **Environment Variables**:
```bash
# LLM Configuration
export LLM_PROVIDER=local-llama-cpp  # or 'openai'
export LLM_MODEL_FILE=mistral-7b.gguf
export OPENAI_API_KEY=your_key_here  # if using OpenAI

# Processing Configuration
export CHUNK_SIZE=1000
export N_CLUSTERS=50
export BATCH_SIZE=100
```

2. **YAML Configuration** (`config.yaml`):
```yaml
# Override default settings
model:
  n_clusters: 30
  batch_size: 2000
```

### Error Codes

When encountering errors, you'll receive specific error codes:

- 1000: Configuration errors
- 2000: File handling errors
- 3000: Parsing errors
- 4000: Clustering errors
- 5000: Database errors
- 6000: API errors
- 7000: Validation errors

Each error includes:
- Error code
- Descriptive message
- Detailed context for debugging

### Best Practices

1. **Log File Preparation**:
   - Ensure logs are in text format
   - Remove any binary or non-text content
   - Split large log files into manageable chunks

2. **Performance Optimization**:
   - Use appropriate batch sizes for your data
   - Enable caching for better performance
   - Monitor resource usage

3. **Error Handling**:
   - Check error messages and codes
   - Review error details for debugging
   - Monitor logs for recurring issues

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
black .
isort .
pylint src/

# Generate documentation
mkdocs serve
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details on how to get involved.

## License

This project is licensed under the terms of the MIT license.

## Acknowledgments

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) for text embeddings
- [scikit-learn](https://scikit-learn.org/) for clustering algorithms
- [diskcache](https://github.com/grantjenks/python-diskcache) for efficient caching
- [pydantic](https://pydantic-docs.helpmanual.io/) for data validation

## Pipeline Architecture

The log processing pipeline is built using Dagster and consists of the following operations:

1. **Read Log File** (`read_log_file`):
   - Reads and validates input log files
   - Supports various encodings
   - Tracks file statistics as assets

2. **Generate Embeddings** (`generate_embeddings`):
   - Converts log lines to embeddings
   - Processes in configurable batches
   - Monitors memory usage and performance

3. **Cluster Logs** (`cluster_logs`):
   - Groups similar log lines
   - Configurable number of clusters
   - Tracks clustering statistics

4. **Analyze Patterns** (`analyze_patterns`):
   - Extracts regex patterns from clusters
   - Calculates pattern confidence
   - Provides sample matches

### Pipeline Visualization

To view the pipeline UI and monitor execution:

```bash
# Start the Dagster UI
dagster dev

# Visit http://localhost:3000 in your browser
```

### Pipeline Configuration

The pipeline can be configured through YAML or environment variables:

```yaml
ops:
  read_log_file:
    config:
      encoding: utf-8
  generate_embeddings:
    config:
      batch_size: 1000
  cluster_logs:
    config:
      n_clusters: 20
  analyze_patterns:
    config:
      max_samples: 5
```

### Pipeline Benefits

1. **Debugging**:
   - Visual representation of the pipeline
   - Detailed execution logs
   - Step-by-step inspection
   - Asset materialization tracking

2. **Performance**:
   - Efficient batch processing
   - Resource monitoring
   - Caching and memoization
   - Parallel execution support

3. **Maintenance**:
   - Clear operation dependencies
   - Modular and testable components
   - Configuration management
   - Error handling and retries 