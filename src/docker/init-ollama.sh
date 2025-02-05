#!/bin/bash
set -e

echo "Starting Ollama service..."
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama service to start..."
until curl -s -f "http://localhost:11434/api/tags" > /dev/null 2>&1; do
    sleep 2
done

echo "Ollama service is ready!"

# Keep container running
wait 