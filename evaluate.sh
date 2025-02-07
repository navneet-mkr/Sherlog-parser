#!/bin/bash

# Check if mode is provided
if [ $# -lt 1 ]; then
    echo "Usage: ./evaluate.sh <mode>"
    echo "Modes: local, remote"
    exit 1
fi

MODE=$1

# Set Ollama URL based on mode
if [ "$MODE" = "local" ]; then
    OLLAMA_URL="http://localhost:11434"
    echo "Running in local mode with Ollama at $OLLAMA_URL"
elif [ "$MODE" = "remote" ]; then
    OLLAMA_URL="http://ollama:11434"
    echo "Running in remote mode with Ollama at $OLLAMA_URL"
else
    echo "Invalid mode: $MODE"
    echo "Valid modes: local, remote"
    exit 1
fi

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
while ! curl -s "$OLLAMA_URL/api/tags" > /dev/null; do
    sleep 1
done
echo "Ollama is ready"

# Run evaluation
PYTHONPATH=. dagster job execute \
    -f src/eval/eval_pipeline.py \
    -j evaluate_logparser_llm \
    --config '{
        "load_dataset": {
            "base_dir": "data",
            "dataset_type": "Apache_loghub_2k",
            "system": "Apache"
        },
        "parse_dataset": {
            "ollama_base_url": "'$OLLAMA_URL'",
            "model_name": "llama2",
            "similarity_threshold": 0.8,
            "batch_size": 10,
            "cache_dir": "cache"
        },
        "evaluate_results": {
            "model_name": "llama2"
        },
        "generate_template_file": {
            "output_dir": "eval_results",
            "model_name": "llama2"
        }
    }' 