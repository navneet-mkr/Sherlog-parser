#!/bin/bash
set -e

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
until curl -s -f "http://localhost:11434/api/tags" > /dev/null 2>&1; do
    sleep 2
done

echo "Ollama is ready. Pulling required models..."

# Pull required models
models=("mistral" "llama2" "codellama")

for model in "${models[@]}"; do
    echo "Pulling $model..."
    curl -s -X POST "http://localhost:11434/api/pull" -d "{\"name\": \"$model\"}"
    echo "Finished pulling $model"
done

echo "Model initialization complete!" 