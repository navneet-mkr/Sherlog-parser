# Docker Setup Guide

This guide explains how to deploy Sherlog-parser using Docker containers.

## Prerequisites

- Docker Engine 20.10.0 or later
- Docker Compose v2.0.0 or later
- Git (for cloning the repository)

## Quick Start

The fastest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone https://github.com/yourusername/sherlog-parser.git
cd sherlog-parser

# Start all services
docker-compose -f src/config/docker-compose.yml up --build
```

This will:
1. Build all necessary containers
2. Start the Streamlit UI on `http://localhost:8501`
3. Launch the Dagster pipeline service on `http://localhost:3000`

## Docker Services

The application consists of three main services:

### 1. Base Service

```yaml
base:
  build:
    context: .
    dockerfile: src/docker/Dockerfile.base
  volumes:
    - ./src:/app/src
    - ./cache:/app/cache
```

This service:
- Contains core dependencies
- Provides shared code
- Manages cache storage

### 2. Dagster Service

```yaml
dagster:
  build:
    context: .
    dockerfile: src/docker/Dockerfile.dagster
  ports:
    - "3000:3000"
  volumes:
    - ./src:/app/src
    - dagster_home:/opt/dagster/dagster_home
```

This service:
- Handles pipeline orchestration
- Provides pipeline monitoring UI
- Manages workflow execution

### 3. Streamlit Service

```yaml
streamlit:
  build:
    context: .
    dockerfile: src/docker/Dockerfile.streamlit
  ports:
    - "8501:8501"
  volumes:
    - ./src:/app/src
```

This service:
- Provides the web interface
- Handles file uploads
- Displays analysis results

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Application settings
SHERLOG_ENV=production
SHERLOG_DEBUG=false

# Service ports
STREAMLIT_PORT=8501
DAGSTER_PORT=3000

# Resource limits
MEMORY_LIMIT=4g
CPU_LIMIT=2
```

### Docker Compose Configuration

Customize `docker-compose.yml`:

```yaml
version: '3.8'

services:
  base:
    environment:
      - SHERLOG_ENV=${SHERLOG_ENV}
    deploy:
      resources:
        limits:
          memory: ${MEMORY_LIMIT}
          cpus: ${CPU_LIMIT}

  dagster:
    environment:
      - DAGSTER_HOME=/opt/dagster/dagster_home
    ports:
      - "${DAGSTER_PORT}:3000"

  streamlit:
    environment:
      - STREAMLIT_SERVER_PORT=${STREAMLIT_PORT}
    ports:
      - "${STREAMLIT_PORT}:8501"
```

## Volume Management

The application uses several Docker volumes:

```yaml
volumes:
  # Persistent cache storage
  cache_data:
    driver: local

  # Dagster home directory
  dagster_home:
    driver: local

  # Log storage
  log_data:
    driver: local
```

## Production Deployment

For production deployment:

1. **Use Production Configuration**:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

2. **Enable Health Checks**:
   ```yaml
   services:
     streamlit:
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8501"]
         interval: 30s
         timeout: 10s
         retries: 3
   ```

3. **Configure Logging**:
   ```yaml
   services:
     base:
       logging:
         driver: "json-file"
         options:
           max-size: "10m"
           max-file: "3"
   ```

## Resource Management

### Memory Management

```yaml
services:
  base:
    deploy:
      resources:
        limits:
          memory: 4g
        reservations:
          memory: 2g
```

### CPU Management

```yaml
services:
  dagster:
    deploy:
      resources:
        limits:
          cpus: '2'
        reservations:
          cpus: '1'
```

## Troubleshooting

### Common Issues

1. **Container Startup Issues**:
   ```bash
   # Check container logs
   docker-compose logs -f

   # Restart specific service
   docker-compose restart dagster
   ```

2. **Volume Permission Issues**:
   ```bash
   # Fix permissions
   sudo chown -R 1000:1000 ./cache
   ```

3. **Resource Constraints**:
   ```bash
   # Monitor resource usage
   docker stats
   ```

## Best Practices

1. **Security**:
   - Use non-root users in containers
   - Implement proper access controls
   - Keep dependencies updated

2. **Performance**:
   - Use volume mounts for cache
   - Implement proper resource limits
   - Monitor container health

3. **Maintenance**:
   - Regular backup of volumes
   - Monitor log sizes
   - Update base images

## Next Steps

- Read about [Configuration Options](configuration.md)
- Check the [API Reference](../api/core/pipeline.md)
- Learn about [Contributing](../development/contributing.md) 