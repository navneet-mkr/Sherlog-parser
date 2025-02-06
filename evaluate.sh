#!/bin/bash

# Function to wait for Ollama
wait_for_ollama() {
    echo "Waiting for Ollama to be ready..."
    until curl -s "${OLLAMA_HOST:-http://localhost}:${OLLAMA_PORT:-11434}/api/version" > /dev/null; do
        echo "Ollama is not ready - sleeping 5s"
        sleep 5
    done
    echo "Ollama is ready!"
}

# Function to wait for Dagster
wait_for_dagster() {
    echo "Waiting for Dagster to be ready..."
    until curl -s "${DAGSTER_GRPC_HOST:-localhost}:${DAGSTER_PORT:-3000}" > /dev/null; do
        echo "Dagster is not ready - sleeping 5s"
        sleep 5
    done
    echo "Dagster is ready!"
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

# Wait for required services
wait_for_ollama
wait_for_dagster

# Run the evaluation pipeline
echo "Starting LogParser-LLM evaluation..."
echo "Results will be available in the Dagster UI at http://localhost:3000"

# Run evaluation with configured environment
python -m dagster job execute \
    -m src.eval.eval_pipeline \
    -j evaluate_logparser_llm \
    --config '{
        "base_dir": "/app/data/eval_datasets",
        "cache_dir": "/app/data/eval_cache",
        "ollama_base_url": "'"${OLLAMA_HOST:-http://localhost}"':'"${OLLAMA_PORT:-11434}"'",
        "model_name": "mistral",
        "similarity_threshold": 0.8,
        "batch_size": 1000
    }'

echo "Evaluation complete! View results in the Dagster UI at http://localhost:3000" 