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

## 🏗️ Architecture

The project follows a simple, streamlined architecture:

```
src/
├── core/               # Core functionality
├── eval/              # Evaluation framework
├── models/            # Model implementations
├── ui/                # UI components
└── utils/             # Utility functions
```

## ✨ Features

- 🤖 LLM-based log parsing
- 🔍 Semantic template matching
- 📊 Comprehensive evaluation metrics
- 🎯 High accuracy and performance
- 🔄 Efficient batch processing

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

## 📝 Usage

1. Access the web interface at http://localhost:8501
2. Upload a log file (.log or .txt)
3. Select processing parameters:
   - Similarity threshold
   - Batch size
   - Model settings
4. Start analysis
5. View results in real-time

### API Access

The system also provides a REST API for programmatic access:

```bash
# Get all templates
GET /templates

# Get logs for a specific template
GET /logs/{template_id}
```

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

## 📋 Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended
- (Optional) Local Ollama installation if not using containerized version

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

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License

Copyright (c) 2024 Sherlog-parser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 