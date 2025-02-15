# 🔍 Sherlog-parser

A powerful, intelligent log parsing and analysis tool that leverages Large Language Models (LLMs) and machine learning to automatically extract patterns and templates from log files.

## 🚀 Quick Start (2 minutes)

```bash
# Start with automatic Ollama configuration
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
# Run evaluation with automatic Ollama configuration
./evaluate.sh

# OR use local Ollama installation
./evaluate.sh --use-local-ollama

# OR connect to remote Ollama instance
./evaluate.sh --ollama-host http://your-ollama-server --ollama-port 11434

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

## 🏗️ Architecture

The project follows a microservices architecture:

```
src/
├── pathway_pipeline/    # Main pipeline implementation
│   ├── pipeline.py     # Log parsing pipeline
│   ├── eval_pipeline.py # Evaluation pipeline
│   └── schema.py       # Data schemas
│
├── models/             # ML models and configuration
├── services/           # Web services and API
└── ui/                 # User interface
```

## ✨ Features

- 🤖 **LLM-Powered Analysis**:
  - Deep semantic understanding using Ollama models
  - Support for multiple models (Mistral, Llama 2, CodeLlama)
  - Efficient local inference with GPU acceleration
- 🧠 **Intelligent Template Extraction**: 
  - Automatic pattern recognition
  - Variable identification
  - Semantic similarity matching
- ⚡ **High Performance**:
  - 🚀 Fast local inference with Ollama
  - 📊 Real-time streaming with Pathway
  - 📦 Efficient vector similarity search
- 🚀 **Production Ready**:
  - 🛡️ Type-safe with Pydantic models
  - ⚙️ Configurable via environment variables
  - ✅ Extensive test coverage
  - 🔄 Proper error handling and logging

## ⚙️ Configuration

Configuration is managed through environment variables in `.env`:

```bash
# Ollama Settings
LOGPARSE_OLLAMA_BASE_URL=http://localhost:11434
LOGPARSE_MODEL_NAME=mistral
LOGPARSE_SIMILARITY_THRESHOLD=0.8
LOGPARSE_BATCH_SIZE=32
```

See `src/models/config.py` for all available configuration options.

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

## 📝 Usage

1. Access the web interface at http://localhost:8501
2. Upload a log file (.log or .txt)
3. Select processing parameters:
   - Similarity threshold
   - Batch size
   - Model settings
4. Start analysis
5. View results in real-time

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

## 📄 License

[Your License Here] 