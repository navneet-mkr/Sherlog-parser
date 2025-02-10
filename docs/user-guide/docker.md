# Docker Setup Guide

This guide explains how to set up and run the log parsing application using Docker.

## Services

The application consists of three main services:

1. **Streamlit** (`streamlit`):
   - Web interface for log analysis
   - Real-time visualization
   - Interactive configuration

2. **Evaluation** (`evaluation`):
   - Benchmark evaluation dashboard
   - Performance metrics
   - Result visualization

3. **Ollama** (`ollama`):
   - LLM inference service
   - Model management
   - GPU acceleration

## Quick Start

The easiest way to get started is using Docker Compose:

```bash
# Start all services
docker compose up -d

# Access the interfaces:
- Web UI: http://localhost:8501
- Evaluation UI: http://localhost:8502
- Ollama API: http://localhost:11434
```

## Configuration

### Environment Variables

The services can be configured using environment variables:

```bash
# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_TIMEOUT=120

# Pipeline Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.8
BATCH_SIZE=1000

# Logging
LOG_LEVEL=INFO
```

### Volume Mounts

The services use the following volume mounts:

```yaml
volumes:
  # Data directories
  - ./data:/data        # Log files and datasets
  - ./output:/output    # Generated results
  - ./cache:/cache      # Cache files

  # Ollama model storage
  - ollama_data:/root/.ollama
```

## Development Setup

For development, you can use the provided development configuration:

```bash
# Start services in development mode
docker compose -f docker-compose.dev.yml up -d

# Run tests
docker compose -f docker-compose.dev.yml run --rm test

# View logs
docker compose -f docker-compose.dev.yml logs -f
```

## GPU Support

To enable GPU support for Ollama:

1. Install NVIDIA Container Toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

2. Configure Docker:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

3. The `docker-compose.yml` already includes GPU configuration:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Troubleshooting

1. **Service Health Checks**
   - All services include health checks
   - Monitor status with `docker compose ps`
   - Check logs with `docker compose logs [service]`

2. **Common Issues**
   - Port conflicts: Change ports in docker-compose.yml
   - Memory issues: Adjust container memory limits
   - GPU access: Verify NVIDIA drivers and toolkit

3. **Resource Usage**
   - Monitor with `docker stats`
   - Check GPU usage with `nvidia-smi`
   - Adjust resource limits as needed

## Security

1. **Network Security**
   - Services use internal Docker network
   - Only necessary ports exposed
   - No direct database access

2. **File Permissions**
   - Mounted volumes use appropriate permissions
   - Containers run as non-root when possible
   - Sensitive files not exposed

3. **Model Security**
   - Models stored in secure volume
   - API access controlled
   - No external model downloads

## Maintenance

1. **Updates**
   ```bash
   # Pull latest images
   docker compose pull

   # Rebuild services
   docker compose build --no-cache

   # Restart with updates
   docker compose up -d
   ```

2. **Cleanup**
   ```bash
   # Remove stopped containers
   docker compose down

   # Clean up unused volumes
   docker volume prune

   # Remove all containers and volumes
   docker compose down -v
   ```

3. **Backups**
   ```bash
   # Backup data directory
   tar -czf backup.tar.gz data/

   # Backup Ollama models
   docker run --rm -v ollama_data:/data -v $(pwd):/backup alpine tar -czf /backup/ollama_backup.tar.gz /data
   ``` 