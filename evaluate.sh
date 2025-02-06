#!/bin/bash

# Show configuration
echo "=== Configuration ==="
echo "Ollama URL: ${OLLAMA_HOST:-http://localhost}:${OLLAMA_PORT:-11434}"
echo "===================="
echo

# Function to wait for Ollama
wait_for_ollama() {
    local ollama_url="${OLLAMA_HOST:-http://localhost}:${OLLAMA_PORT:-11434}"
    echo "Waiting for Ollama to be ready at: ${ollama_url}"
    local timeout=60  # timeout after 60 seconds
    local start_time=$(date +%s)
    
    while true; do
        if curl -s "${ollama_url}/api/version" > /dev/null; then
            echo "Ollama is ready!"
            return 0
        fi
        
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            echo "Error: Timed out waiting for Ollama after ${timeout} seconds"
            echo "Debug steps:"
            echo "1. Check if Ollama is running locally: curl ${ollama_url}/api/version"
            echo "2. Verify OLLAMA_HOST and OLLAMA_PORT environment variables:"
            echo "   Current OLLAMA_HOST: ${OLLAMA_HOST:-http://localhost}"
            echo "   Current OLLAMA_PORT: ${OLLAMA_PORT:-11434}"
            echo "   Trying to connect to: ${ollama_url}"
            echo "3. Make sure Ollama is installed and running: ollama serve"
            echo "4. Check if there are any firewall issues blocking port 11434"
            echo "5. If running in Docker, ensure the container has network access to Ollama"
            exit 1
        fi
        
        echo "Ollama is not ready - sleeping 5s (${elapsed}s elapsed)"
        sleep 5
    done
}

# Function to wait for Dagster
wait_for_dagster() {
    echo "Waiting for Dagster to be ready..."
    local timeout=60  # timeout after 60 seconds
    local start_time=$(date +%s)
    
    while true; do
        if curl -s "localhost:3001" > /dev/null; then
            echo "Dagster is ready!"
            return 0
        fi
        
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            echo "Error: Timed out waiting for Dagster after ${timeout} seconds"
            echo "Debug steps:"
            echo "1. Check if evaluation container is running: docker compose ps eval"
            echo "2. Check evaluation container logs: docker compose logs eval"
            echo "3. Try restarting the evaluation container: docker compose restart eval"
            exit 1
        fi
        
        echo "Dagster is not ready - sleeping 5s (${elapsed}s elapsed)"
        sleep 5
    done
}

# Create necessary directories
mkdir -p data/eval_datasets/{loghub_2k,logpub} data/eval_cache

# Check if datasets exist
if [ ! -d "data/eval_datasets/loghub_2k" ] || [ ! "$(ls -A data/eval_datasets/loghub_2k)" ]; then
    echo "Error: Loghub-2k datasets not found in data/eval_datasets/loghub_2k"
    echo "Please download and place the datasets in the correct directory structure:"
    echo "  data/eval_datasets/loghub_2k/"
    echo "    ├── Apache/"
    echo "    │   ├── Apache.log_structured.csv"
    echo "    │   └── Apache.log_templates.csv"
    echo "    ├── Hadoop/"
    echo "    │   ├── Hadoop.log_structured.csv"
    echo "    │   └── Hadoop.log_templates.csv"
    echo "    └── ..."
    exit 1
fi

# Start the evaluation container
echo "Starting evaluation container..."
docker compose --profile eval --profile with-ollama up -d

# Wait for required services
wait_for_ollama
wait_for_dagster

# Run the evaluation pipeline
echo "Starting LogParser-LLM evaluation..."
echo "Results will be available in the Dagster UI at http://localhost:3001"

# Run evaluation using the Dagster container
docker compose exec eval dagster job execute \
    -m src.eval.eval_pipeline \
    -j evaluate_logparser_llm \
    -c src/eval/run_config.yaml

echo "Evaluation complete! View results in the Dagster UI at http://localhost:3001" 