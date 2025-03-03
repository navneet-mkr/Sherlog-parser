# 🔍 Sherlog-parser

A powerful, intelligent log parsing and analysis tool that leverages Large Language Models (LLMs) and machine learning to automatically extract patterns and templates from log files.

## 🚀 Quick Start (2 minutes)

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/sherlog-parser.git
cd sherlog-parser

# Run the local setup script (sets up everything automatically)
./run_log_analyzer.sh
```

This will:
- Set up Python virtual environment
- Install dependencies
- Start TimescaleDB in Docker
- Install and start Ollama
- Launch the Streamlit dashboard

Access the web interface at http://localhost:8501 🎉

### Docker Deployment

For containerized deployment:
```bash
# Start all services using Docker Compose
docker-compose up -d
```

## 🏗️ Project Structure

The project follows a clean, modular architecture:

```
sherlog-parser/
├── src/                    # Source code
│   ├── core/              # Core functionality and common utilities
│   │   ├── pipeline.py    # Log processing pipeline
│   │   └── timeseries.py  # TimescaleDB integration
│   ├── models/            # Model implementations
│   ├── eval/              # Evaluation framework
│   ├── ui/                # UI components
│   │   └── log_analyzer.py # Streamlit dashboard
│   ├── prompts/           # LLM prompt templates
│   ├── config/            # Configuration management
│   └── services/          # Service layer implementations
├── tests/                 # Test suite
├── docs/                  # Documentation
├── docker/               # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── run_log_analyzer.sh   # Local development setup script
├── config.yaml           # Main configuration file
├── .env                  # Environment variables
└── requirements.txt      # Python dependencies
```

### Key Components

- **Core Module**: Common utilities and core functionality
  - **Pipeline**: End-to-end log processing pipeline
  - **TimescaleDB Integration**: Time series database integration
- **Models**: Implementations of different log parsing models
- **Evaluation**: Comprehensive evaluation framework with metrics
- **UI**: Streamlit-based interactive dashboard
- **Configuration**: Centralized configuration management
- **Services**: Business logic and service implementations

## ✨ Features

- 🤖 LLM-based log parsing
- 🔍 Semantic template matching
- 📊 Comprehensive evaluation metrics
- 🎯 High accuracy and performance
- 🔄 Efficient batch processing
- 📈 Time series analysis with TimescaleDB
- 🎨 Interactive Streamlit dashboard
- 🔧 Modular and extensible architecture
- 📝 Type-safe configuration management

## 🗄️ TimescaleDB Integration

The system now includes TimescaleDB integration for efficient time series analysis:

- **Structured Storage**: Logs are stored with:
  - Timestamp
  - Log Level
  - Component
  - Template
  - Parameters (as JSONB)
  - Raw Message

- **Time Series Queries**: Analyze logs over time:
  ```sql
  -- Error trends over time
  SELECT time_bucket('1 hour', timestamp) AS hour,
         level,
         count(*) as count
  FROM logs
  WHERE level = 'ERROR'
  GROUP BY hour
  ORDER BY hour;
  ```

- **Component Analysis**:
  ```sql
  -- Most active components
  SELECT component,
         count(*) as count
  FROM logs
  GROUP BY component
  ORDER BY count DESC
  LIMIT 10;
  ```

- **Template Pattern Analysis**:
  ```sql
  -- Common log patterns
  SELECT template,
         count(*) as occurrence_count
  FROM logs
  GROUP BY template
  ORDER BY occurrence_count DESC;
  ```

## 🖥️ Interactive Dashboard

The Streamlit dashboard provides:

1. **Log Upload**:
   - Upload log files
   - Process in batches
   - Real-time progress tracking

2. **Analysis Features**:
   - Error trends visualization
   - Component activity analysis
   - Template pattern discovery
   - Custom SQL queries

3. **Time Series Analysis**:
   - Time-based filtering
   - Aggregation by time buckets
   - Component-wise trends

## 📋 Prerequisites

- Python 3.9+
- Docker (for TimescaleDB)
- 8GB+ RAM recommended
- Local Ollama installation (automatic with setup script)

## ⚙️ Configuration

Configuration is managed through a hierarchical system:

1. **Environment Variables** (`.env`):
   ```bash
   ENVIRONMENT=development
   DEBUG=false
   OPENAI_API_KEY=your-key
   ANTHROPIC_API_KEY=your-key
   ```

2. **Application Config** (`config.yaml`):
   ```yaml
   model:
     name: qwen2.5-coder
     temperature: 0.1
     max_tokens: 100
   
   parser:
     similarity_threshold: 0.8
     max_template_length: 200
   
   processing:
     batch_size: 32
     max_workers: 4
   ```

3. **Development Settings**:
   ```bash
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Install documentation tools
   pip install -e ".[docs]"
   ```

## 🧪 Testing and Quality Assurance

```bash
# Run all tests with coverage
pytest --cov=src tests/

# Run type checking
mypy src/

# Format code
black src/ tests/
isort src/ tests/

# Lint code
pylint src/ tests/
```

## 📝 Usage

### Web Interface

1. Access the web interface at http://localhost:8501
2. Upload a log file (.log or .txt)
3. Select processing parameters:
   - Model configuration
   - Parser settings
   - Processing options
4. Start analysis
5. View results in real-time

### API Access

The system provides a REST API for programmatic access:

```python
from sherlog_parser.client import LogParserClient

# Initialize client
client = LogParserClient()

# Parse logs
templates = client.parse_logs("path/to/logfile.log")

# Get template details
template_logs = client.get_logs(template_id="template_123")
```

For API documentation, see [API Reference](docs/api.md).

## 📊 Evaluation

The project includes a comprehensive evaluation framework:

### Docker-based Evaluation
```bash
# Run evaluation with default settings
./evaluate.sh

# Use local Ollama instance
./evaluate.sh --use-local-ollama

# Custom Ollama host/port
./evaluate.sh --ollama-host http://your-ollama-server --ollama-port 11434
```

### Local Evaluation (No Docker)
You can run evaluations directly on your local machine without Docker using the `evaluate_local.py` script:

```bash
# Run evaluation with default settings (Apache system, loghub_2k dataset)
python evaluate_local.py

# Evaluate specific system and dataset
python evaluate_local.py --system Hadoop --dataset-type loghub_all

# Launch the evaluation UI
python evaluate_local.py --ui

# Use custom Ollama port
python evaluate_local.py --ollama-port 11435
```

The evaluation framework provides:
- 📊 Comprehensive metrics calculation
- 📈 Performance analysis
- 🎯 Template matching accuracy
- 🔍 Detailed error analysis
- 📋 Automated report generation

## 🔧 Troubleshooting

1. **Configuration Issues**
   - Verify environment variables in `.env`
   - Check `config.yaml` settings
   - Ensure proper API keys are set

2. **Model Problems**
   - Verify model availability in Ollama
   - Check model configuration
   - Monitor resource usage

3. **Performance Issues**
   - Adjust batch size and worker count
   - Monitor memory usage
   - Consider GPU acceleration

4. **Development Setup**
   - Install development dependencies: `pip install -e ".[dev]"`
   - Run linting and type checking
   - Verify test coverage

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -e ".[dev]"`
4. Make your changes
5. Run tests and linting
6. Submit a pull request

See [Contributing Guide](CONTRIBUTING.md) for detailed guidelines.

## 📚 Documentation

- [Algorithm Details](docs/algorithm.md)

Build documentation locally:
```bash
# Install documentation tools
pip install -e ".[docs]"

# Build docs
mkdocs build

# Serve docs locally
mkdocs serve
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔍 Anomaly Detection

The system now includes powerful real-time anomaly detection capabilities:

### Command-line Interface

Quick anomaly detection using the standalone script:
```bash
# Basic usage - analyze last 4 hours
./src/analyze_anomalies.py --table logs

# Analyze specific time window with filters
./src/analyze_anomalies.py --hours 12 --level ERROR --component api-server

# Adjust detection sensitivity
./src/analyze_anomalies.py --eps 0.2 --min-samples 2 --table logs
```

### Features

1. **Real-time Detection**:
   - Embedding-based clustering (DBSCAN)
   - Numeric anomaly detection
   - Configurable sensitivity parameters
   - Component-level analysis

2. **Historical Analysis**:
   ```bash
   # Run with historical comparison (default)
   ./src/analyze_anomalies.py --table logs
   
   # Skip historical analysis
   ./src/analyze_anomalies.py --table logs --no-history
   ```

3. **Visualization & Reporting**:
   - Interactive HTML timelines
   - Historical trend plots
   - Statistical comparisons
   - CSV exports
   ```bash
   # Custom output directory
   ./src/analyze_anomalies.py --output-dir "/path/to/reports"
   ```

4. **Filtering Options**:
   ```bash
   # Log level filtering
   --level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
   
   # Component filtering
   --component your-component-name
   ```

5. **Integration Options**:
   ```bash
   # Custom database connection
   --db-url "postgresql://user:pass@host:5432/db"
   
   # Custom Ollama endpoint
   --ollama-url "http://your-ollama-host:11434"
   ```

### Web Interface

The Streamlit dashboard includes anomaly detection features:

1. **Real-time Monitoring**:
   - Live anomaly detection
   - Interactive visualizations
   - Historical comparisons

2. **Configuration**:
   - Time window selection
   - Sensitivity parameters
   - Component filters
   - Log level filters

3. **Analysis Views**:
   - Anomaly timeline
   - Cluster distribution
   - Error ratio trends
   - Statistical metrics

### Output Format

The anomaly detection generates comprehensive reports:

1. **CSV Export**:
   ```csv
   timestamp,level,component,message,is_embedding_anomaly,is_numeric_anomaly,cluster_label
   2024-02-17T10:00:00,ERROR,api-server,"Connection timeout",true,false,1
   ```

2. **Visualization Files**:
   - `anomaly_timeline_YYYYMMDD_HHMMSS.html`
   - `historical_trends_YYYYMMDD_HHMMSS.html`

3. **Console Output**:
   ```
   Anomaly Summary
   ---------------
   Total Anomalies: 42
   Embedding Anomalies: 35
   Numeric Anomalies: 7
   Error Ratio: 15.2%

   Historical Comparison
   --------------------
   Historical Mean: 25.3
   Current vs Mean: +2.31σ
   95th Percentile: 45.6
   ```

### Integration Examples

1. **Automated Monitoring**:
   ```bash
   # Run every hour using cron
   0 * * * * /path/to/analyze_anomalies.py --hours 1 --table logs
   ```

2. **Custom Analysis Pipeline**:
   ```python
   from src.core.pipeline import LogProcessingPipeline
   from src.core.anomaly_incidents import IncidentAnomalyDetector

   # Initialize
   pipeline = LogProcessingPipeline(db_url="your-db-url")
   detector = IncidentAnomalyDetector(pipeline=pipeline)

   # Detect anomalies
   anomalies = detector.detect_anomalies(
       table_name="logs",
       hours=4,
       additional_filters={"level": "ERROR"}
   )
   ```

For detailed API documentation, see [Anomaly Detection API](docs/anomaly_detection.md). 