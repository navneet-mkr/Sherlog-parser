# Installation Guide

This guide will help you install and set up Sherlog-parser on your system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- Docker (optional, for containerized deployment)

## Installation Methods

### Method 1: Direct Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sherlog-parser.git
   cd sherlog-parser
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sherlog-parser.git
   cd sherlog-parser
   ```

2. Build and start services:
   ```bash
   docker-compose -f src/config/docker-compose.yml up --build
   ```

## Verifying Installation

After installation, verify that everything is working:

1. Run the test suite:
   ```bash
   pytest
   ```

2. Try a simple example:
   ```python
   from sherlog_parser import LogParser
   
   parser = LogParser()
   results = parser.parse_file("sample.log")
   print(results)
   ```

## Configuration

1. Copy the example configuration:
   ```bash
   cp config.example.yaml config.yaml
   ```

2. Edit `config.yaml` with your settings:
   ```yaml
   model:
     embedding_model: all-MiniLM-L6-v2
     n_clusters: 20
     batch_size: 1000
   ```

## Next Steps

- Check out the [Quick Start Guide](quickstart.md) for basic usage
- Read about [Configuration Options](configuration.md)
- Learn about [Docker Setup](docker.md) 