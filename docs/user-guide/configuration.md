# Configuration Guide

Sherlog-parser provides multiple ways to configure its behavior. This guide explains all available configuration options and methods.

## Configuration Methods

### 1. YAML Configuration

Create a `config.yaml` file in your project root:

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

### 2. Environment Variables

Set configuration via environment variables:

```bash
# Application settings
export SHERLOG_ENV=production
export SHERLOG_DEBUG=false

# Logging settings
export SHERLOG_LOG_LEVEL=INFO
export SHERLOG_LOG_FILE=logs/app.log

# Cache settings
export SHERLOG_CACHE_ENABLED=true
export SHERLOG_CACHE_DIR=cache
export SHERLOG_CACHE_SIZE_MB=1024

# Model settings
export SHERLOG_MODEL_NAME=all-MiniLM-L6-v2
export SHERLOG_N_CLUSTERS=20
export SHERLOG_BATCH_SIZE=1000
```

### 3. Programmatic Configuration

Configure settings in your Python code:

```python
from sherlog_parser import LogParser, Settings

settings = Settings(
    env="production",
    model=ModelSettings(
        embedding_model="all-MiniLM-L6-v2",
        n_clusters=20
    ),
    cache=CacheSettings(
        enabled=True,
        max_size_mb=1024
    )
)

parser = LogParser(settings)
```

## Configuration Options

### Application Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `env` | str | "development" | Environment (development/production) |
| `debug` | bool | false | Enable debug mode |

### Logging Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | str | "INFO" | Log level (DEBUG/INFO/WARNING/ERROR) |
| `format` | str | See above | Log message format |
| `file_path` | str | "logs/app.log" | Log file location |

### Cache Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable caching |
| `directory` | str | "cache" | Cache directory |
| `max_size_mb` | int | 1024 | Maximum cache size |
| `ttl_days` | int | 30 | Cache entry lifetime |

### Model Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `embedding_model` | str | "all-MiniLM-L6-v2" | Embedding model name |
| `n_clusters` | int | 20 | Number of clusters |
| `batch_size` | int | 1000 | Processing batch size |
| `random_seed` | int | 42 | Random seed for reproducibility |

## Advanced Configuration

### Custom Embedding Models

```python
from sherlog_parser import LogParser, Settings
from sentence_transformers import SentenceTransformer

settings = Settings(
    model=ModelSettings(
        embedding_model=SentenceTransformer("custom-model")
    )
)

parser = LogParser(settings)
```

### Custom Cache Backend

```python
from sherlog_parser import LogParser, Settings
from mycache import CustomCache

settings = Settings(
    cache=CacheSettings(
        backend=CustomCache(),
        enabled=True
    )
)

parser = LogParser(settings)
```

### Custom Logger

```python
import logging
from sherlog_parser import LogParser, Settings

# Configure custom logger
logger = logging.getLogger("custom")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

settings = Settings(
    logging=LoggingSettings(
        logger=logger
    )
)

parser = LogParser(settings)
```

## Configuration Best Practices

1. **Environment-specific Configuration**
   - Use different config files for development and production
   - Override sensitive values with environment variables
   - Keep secrets out of version control

2. **Performance Optimization**
   - Adjust batch size based on available memory
   - Enable caching for better performance
   - Monitor and adjust cache size as needed

3. **Error Handling**
   - Set appropriate log levels
   - Configure error notifications
   - Implement proper fallbacks

## Troubleshooting

### Common Issues

1. **Cache Issues**
   ```yaml
   cache:
     enabled: false  # Disable cache temporarily
   ```

2. **Memory Issues**
   ```yaml
   model:
     batch_size: 500  # Reduce batch size
   ```

3. **Performance Issues**
   ```yaml
   cache:
     max_size_mb: 2048  # Increase cache size
   ```

## Next Steps

- Learn about [Docker Setup](docker.md) for containerized deployment
- Check the [API Reference](../api/core/pipeline.md) for implementation details
- Read the [Contributing Guide](../development/contributing.md) for development setup 