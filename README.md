# 🔍 Sherlog-parser

A powerful, intelligent log parsing and analysis tool that leverages Large Language Models (LLMs) and machine learning to automatically cluster, analyze, and extract patterns from log files. Built with Ollama integration for efficient local inference.

## ✨ Features

- 🤖 **LLM-Powered Analysis**:
  - Deep semantic understanding using Ollama models
  - Support for multiple models (Mistral, Llama 2, CodeLlama)
  - Efficient local inference with GPU acceleration
- 🧠 **Intelligent Log Clustering**: Uses embeddings and incremental clustering to group similar log messages
- 🎯 **Pattern Extraction**: Automatically extracts regex patterns from log clusters
- ⚡ **High Performance**:
  - 🚀 Fast local inference with Ollama
  - 📊 Incremental clustering with scikit-learn
  - 📦 Batch processing for large log files
- 🚀 **Production Ready**:
  - 🛡️ Type-safe with Pydantic models
  - ⚙️ Configurable via environment variables
  - ✅ Extensive test coverage
  - 🔄 Proper error handling and logging
- 🔄 **Advanced Pipeline**:
  - 🏗️ Built with Dagster for robust pipeline orchestration
  - 📊 Visual pipeline UI for debugging and monitoring
  - 🤖 Integrated model management
  - 📝 Asset tracking and materialization

## 🛠️ Quick Start with Docker

The easiest way to get started is using Docker Compose:

```bash
# Start all services
docker compose up -d

# Access the interfaces:
- Streamlit UI: http://localhost:8501
- Dagster UI: http://localhost:3000
- Ollama API: http://localhost:11434
```

### 🎮 Using the Application

1. **🖥️ Access the Streamlit UI**:
   - Open your browser and go to `http://localhost:8501`
   - Select a model from the available options
   - Upload your log file using the file uploader
   - Adjust analysis parameters if needed
   - Click "Analyze Logs" to start processing

2. **📊 Monitor Pipeline Progress**:
   - The Dagster UI will be available at `http://localhost:3000`
   - Monitor pipeline execution and progress
   - View detailed logs and results
   - Access visualizations and insights

### 🤖 Model Management

The application includes a built-in model management interface:

1. **Available Models**:
   - Mistral: Powerful open-source model for general tasks
   - Llama 2: Meta's latest model optimized for chat
   - CodeLlama: Specialized for code understanding

2. **Model Operations**:
   - Pull new models from Ollama
   - View model details and parameters
   - Delete unused models
   - Monitor model status

## ⚙️ Configuration

Configuration is managed through environment variables in `.env`:

```bash
# Ollama Settings
OLLAMA_HOST=http://ollama
OLLAMA_PORT=11434
OLLAMA_TIMEOUT=120

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=10000
N_CLUSTERS=20
BATCH_SIZE=1000
```

## 🏗️ Architecture

The project follows a microservices architecture:

1. **Streamlit UI**:
   - Web interface for log analysis
   - Model management
   - Result visualization

2. **Dagster Service**:
   - Pipeline orchestration
   - Job monitoring
   - Asset management

3. **Ollama Service**:
   - LLM inference
   - Model management
   - GPU acceleration

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
docker compose --profile test up test

# Run specific tests
docker compose run --rm test pytest tests/test_ollama_integration.py -v

# Run with coverage
docker compose run --rm test pytest --cov=src --cov-report=term-missing tests/
```

## 📚 Documentation

For more detailed information:
- [Docker Setup](docs/user-guide/docker.md)
- [Model Management](docs/user-guide/models.md)
- [Pipeline Configuration](docs/user-guide/pipeline.md)
- [API Reference](docs/api/index.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details. 