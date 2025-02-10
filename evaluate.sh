#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create required directories
mkdir -p data/eval_datasets output/eval cache/eval

# Check if Ollama is running
if ! curl -s -f "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo -e "${RED}Error: Ollama is not running${NC}"
    echo "Please start Ollama first:"
    echo "ollama serve"
    exit 1
fi

# Check if model is available
if ! curl -s -f "http://localhost:11434/api/tags" | grep -q "mistral"; then
    echo "Pulling Mistral model..."
    curl -X POST http://localhost:11434/api/pull -d '{"name": "mistral"}'
fi

# Start evaluation service
echo "Starting evaluation service..."
docker compose up -d evaluation

# Wait for service to be ready
echo -n "Waiting for evaluation service..."
max_retries=30
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if curl -s -f "http://localhost:8502" > /dev/null 2>&1; then
        echo -e "\n${GREEN}Evaluation service is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 2
    retry_count=$((retry_count + 1))
done

if [ $retry_count -eq $max_retries ]; then
    echo -e "\n${RED}Failed to start evaluation service${NC}"
    exit 1
fi

echo -e "\n${GREEN}Evaluation pipeline is ready!${NC}"
echo -e "\nAccess the evaluation dashboard at:"
echo -e "http://localhost:8502" 