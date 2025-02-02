# Sherlog-parser Documentation

Welcome to the Sherlog-parser documentation! This powerful tool uses machine learning to automatically cluster and extract patterns from log files, making log analysis more efficient and insightful.

## Overview

Sherlog-parser is designed to handle large-scale log analysis with intelligent pattern recognition. It combines modern machine learning techniques with efficient data processing to provide meaningful insights from your log files.

## Key Features

### 🧠 Intelligent Log Clustering
- Uses state-of-the-art embeddings for semantic understanding
- Implements incremental clustering for efficient processing
- Automatically groups similar log messages

### 🎯 Pattern Extraction
- Automatically extracts regex patterns from log clusters
- Identifies common log patterns
- Helps in log standardization

### ⚡ High Performance
- Efficient caching of embeddings using `diskcache`
- Incremental clustering with scikit-learn
- Batch processing for large log files
- Optimized for production workloads

### 🏭 Production Ready
- Type-safe with Pydantic models
- Configurable via YAML and environment variables
- Extensive test coverage
- Proper error handling and logging

### 🔄 Advanced Pipeline
- Built with Dagster for robust pipeline orchestration
- Visual pipeline UI for debugging and monitoring
- Automatic dependency management
- Asset tracking and materialization

## Getting Started

Check out our [Quick Start Guide](user-guide/quickstart.md) to begin using Sherlog-parser, or dive into the [Installation Instructions](user-guide/installation.md) for detailed setup steps.

## API Documentation

For detailed API documentation, visit the following sections:

- [Core Components](api/core/pipeline.md)
- [Data Models](api/models/data.md)
- [Error Handling](api/core/errors.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on how to get involved. 