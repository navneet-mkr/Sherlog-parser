# 🔍 Sherlog-parser

A powerful, intelligent log parsing and analysis tool that leverages Large Language Models (LLMs) and machine learning to automatically cluster, analyze, and extract patterns from log files.

## 🚀 Quick Start (2 minutes)

```bash
# Clone and start with default settings (using Ollama container)
git clone https://github.com/yourusername/log-parse-ai.git
cd log-parse-ai
./start.sh

# OR use your local Ollama installation
./start.sh --use-local-ollama

# OR connect to remote Ollama instance
./start.sh --ollama-host http://your-ollama-server --ollama-port 11434
```

That's it! Access the web interface at http://localhost:8501 🎉

## 📊 Evaluation Framework

We provide a comprehensive evaluation framework to assess LogParser-LLM's performance against benchmark datasets:

### Quick Start

1. Download evaluation datasets:
```bash
# Download Loghub-2k datasets automatically
./download_datasets.sh

# LogPub datasets require manual download after registration
# See src/eval/README.md for details
```

2. Run evaluation:
```bash
# Run evaluation in Docker (recommended)
docker compose --profile eval up eval

# OR run locally
./evaluate.sh
```

The framework:
- Uses Loghub-2k and LogPub benchmark datasets
- Calculates metrics from the LogParser-LLM paper
- Integrates with Dagster for pipeline orchestration
- Supports multiple Ollama models
- Caches results for faster re-evaluation

For detailed setup and usage instructions, see [Evaluation Framework Documentation](src/eval/README.md).

## 🔄 Flexible Ollama Integration

Choose how you want to use Ollama:

1. **Containerized** (Default): 
   - Zero setup required
   - Automatically managed by Docker
   - Isolated environment
   ```bash
   ./start.sh
   ```

2. **Local Installation**:
   - Use your existing Ollama setup
   - Share models with other applications
   - Faster startup time
   ```bash
   ./start.sh --use-local-ollama
   ```

3. **Remote Instance**:
   - Connect to any Ollama server
   - Share resources across network
   - Custom configuration
   ```bash
   ./start.sh --ollama-host http://your-server --ollama-port 11434
   ```

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

## 📋 Prerequisites

- Docker and Docker Compose
- 16GB+ RAM recommended
- (Optional) Local Ollama installation if not using containerized version

## 🛠️ Deployment Options

### Using the Startup Script

The `start.sh` script provides flexible deployment options:

```bash
# Default setup (using containerized Ollama)
./start.sh

# Use local Ollama instance
./start.sh --use-local-ollama

# Use custom Ollama host and port
./start.sh --ollama-host http://my-ollama-server --ollama-port 12345
```

Available options:
- `--ollama-host HOST`: Specify custom Ollama host (default: http://ollama)
- `--ollama-port PORT`: Specify custom Ollama port (default: 11434)
- `--use-local-ollama`: Use local Ollama instance instead of container
- `-h, --help`: Show help message

### Manual Configuration

If you prefer manual configuration:

1. Set environment variables:
```bash
export OLLAMA_HOST=http://your-ollama-host
export OLLAMA_PORT=your-ollama-port
```

2. Start services:
```bash
# With containerized Ollama
docker compose --profile with-ollama up -d

# Without Ollama container (using external instance)
docker compose up -d dagster streamlit
```

## 📝 Usage

1. Access the web interface at http://localhost:8501
2. Upload a log file (.log or .txt)
3. Select processing parameters:
   - Number of clusters
   - Batch size
   - Model settings
4. Start analysis
5. View results in the Dagster UI (http://localhost:3000)

## 🔍 Monitoring & Management

- View application logs:
```bash
docker compose logs -f
```

- Stop all services:
```bash
docker compose down
```

- Manage models through the UI:
  - Pull new models
  - Remove unused models
  - View model details

## 🔧 Troubleshooting

1. **Ollama Connection Issues**
   - Check if Ollama is running at the specified host/port
   - Verify network connectivity
   - Check system resources

2. **Service Health**
   - Monitor service status in the UI
   - Check Docker logs for errors
   - Verify port availability

3. **Performance Issues**
   - Adjust batch size based on available memory
   - Monitor resource usage
   - Consider using a more powerful machine for large logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

[Your License Here] 