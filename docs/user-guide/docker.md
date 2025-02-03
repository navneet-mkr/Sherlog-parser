# Docker Setup

## Services

The application consists of the following services:

1. **Streamlit UI** (`streamlit`):
   - Web interface for log analysis
   - Built from `src/docker/Dockerfile.streamlit`
   - Runs on port 8501

2. **Dagster** (`dagster`):
   - Pipeline orchestration
   - Built from `src/docker/Dockerfile.dagster`
   - UI runs on port 3000
   - gRPC server on port 4000

3. **Ollama** (`ollama`):
   - LLM service for inference
   - Uses official `ollama/ollama` image
   - Runs on port 11434
   - Supports GPU acceleration

4. **Test** (`test`):
   - Test environment
   - Built from `src/docker/Dockerfile.test`
   - Runs tests in isolation
   - Available with `--profile test`

## Running the Application

1. Start all services:
```bash
docker compose up -d
```

2. Run tests:
```bash
docker compose --profile test up test
```

3. Access services:
   - Streamlit UI: http://localhost:8501
   - Dagster UI: http://localhost:3000
   - Ollama API: http://localhost:11434

## Development

1. Build images:
```bash
docker compose build
```

2. View logs:
```bash
docker compose logs -f
```

3. Stop services:
```bash
docker compose down
```

## Volumes

The application uses the following volumes:

1. `dagster_data`: Persistent storage for Dagster
2. `dagster_home`: Dagster home directory
3. `dagster_tmp`: Temporary storage
4. `ollama_models`: Model storage for Ollama

## Environment Variables

Key environment variables are configured in `.env`:

1. Ollama settings:
   - `OLLAMA_HOST`
   - `OLLAMA_PORT`
   - `OLLAMA_TIMEOUT`

2. Model settings:
   - `EMBEDDING_MODEL`
   - `CHUNK_SIZE`
   - `N_CLUSTERS`
   - `BATCH_SIZE`

3. Database settings:
   - `DB_PATH` 