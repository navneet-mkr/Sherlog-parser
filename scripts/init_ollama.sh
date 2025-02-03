#!/bin/bash

# Wait for Ollama to be ready
echo "Waiting for Ollama service..."
until curl -s -f "http://ollama:11434/api/tags" > /dev/null; do
    sleep 2
done
echo "Ollama service is ready!"

# Pull default models
MODELS=("mistral" "llama2" "codellama")

for model in "${MODELS[@]}"; do
    echo "Pulling $model..."
    curl -X POST "http://ollama:11434/api/pull" \
         -H "Content-Type: application/json" \
         -d "{\"name\": \"$model\"}"
    echo "Finished pulling $model"
done

echo "All models pulled successfully!" 