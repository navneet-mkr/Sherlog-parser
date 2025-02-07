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

# Create necessary directories
mkdir -p cache eval_results

# Run evaluation
PYTHONPATH=. dagster job execute \
    -f src/eval/eval_pipeline.py \
    -j evaluate_logparser_llm \
    -c src/eval/run_config.yaml 