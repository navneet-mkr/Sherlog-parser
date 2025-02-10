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
# Run evaluation dashboard
./evaluate.sh

# Access results at http://localhost:8502
```

### Important Update (March 2024)

The evaluation framework has been updated to use the latest Pathway version (>=0.7.0). Key changes include:
- Integration with Pathway's core package (no separate xpacks required)
- Improved vector similarity search
- Enhanced streaming capabilities
- Better memory management

If you're upgrading from an older version:
1. Update your dependencies: `pip install -r requirements.txt`
2. Clear the cache: `rm -rf cache/eval/*`
3. Restart the evaluation service: `docker compose restart evaluation`

The framework:
- Uses Loghub-2k and LogPub benchmark datasets
- Calculates metrics from the LogParser-LLM paper
- Real-time streaming processing with Pathway
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
  - 📊 Real-time streaming with Pathway
  - 📦 Efficient vector similarity search
- 🚀 **Production Ready**:
  - 🛡️ Type-safe with Pydantic models
  - ⚙️ Configurable via environment variables
  - ✅ Extensive test coverage
  - 🔄 Proper error handling and logging
- 🔄 **Advanced Pipeline**:
  - 🏗️ Built with Pathway for real-time processing
  - 📊 Streaming architecture for scalability
  - 🤖 Integrated model management
  - 📝 Automatic template extraction

## 🛠️ Quick Start with Docker

The easiest way to get started is using Docker Compose:

```bash
# Start all services
docker compose up -d

# Access the interfaces:
- Web UI: http://localhost:8501
- Evaluation UI: http://localhost:8502
- Ollama API: http://localhost:11434
```

### 🎮 Using the Application

1. **🖥️ Access the Web UI**:
   - Open your browser and go to `http://localhost:8501`
   - Select a model from the available options
   - Upload your log file using the file uploader
   - Adjust analysis parameters if needed
   - Click "Analyze Logs" to start processing

2. **📊 Monitor Progress**:
   - Watch real-time processing in the UI
   - View extracted templates and patterns
   - Export results and insights
   - Access detailed metrics

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
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_TIMEOUT=120

# Pipeline Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.8
BATCH_SIZE=1000
```

## 🏗️ Architecture

The project follows a microservices architecture:

1. **Web Interface**:
   - Streamlit-based UI
   - Real-time visualization
   - Interactive analysis

2. **Processing Pipeline**:
   - Pathway streaming engine
   - Vector similarity search
   - Template extraction

3. **Inference Service**:
   - Ollama LLM integration
   - Model management
   - GPU acceleration

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_ollama_integration.py -v

# Run with coverage
pytest --cov=src --cov-report=term-missing tests/
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
- 8GB+ RAM recommended
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
- `--ollama-host HOST`: Specify custom Ollama host (default: http://localhost)
- `--ollama-port PORT`: Specify custom Ollama port (default: 11434)
- `--use-local-ollama`: Use local Ollama instance instead of container
- `-h, --help`: Show help message

### Manual Configuration

If you prefer manual configuration:

1. Set environment variables:
```bash
export OLLAMA_BASE_URL=http://your-ollama-host:11434
export OLLAMA_MODEL=mistral
```

2. Start services:
```bash
# With containerized Ollama
docker compose --profile with-ollama up -d

# Without Ollama container (using external instance)
docker compose up -d streamlit
```

## 📝 Usage

1. Access the web interface at http://localhost:8501
2. Upload a log file (.log or .txt)
3. Select processing parameters:
   - Similarity threshold
   - Batch size
   - Model settings
4. Start analysis
5. View results in real-time

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