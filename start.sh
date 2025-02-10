#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check if a service is running
check_service() {
    local service_name=$1
    local url=$2
    local max_retries=30
    local retry_count=0

    echo -n "Checking $service_name... "
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo -n "."
            sleep 2
        fi
    done
    
    echo -e "\n${RED}Failed to connect to $service_name at $url${NC}"
    return 1
}

# Create required directories
mkdir -p data/logs output cache

# Start services
echo "Starting services..."

# Start Ollama
if ! curl -s -f "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo "Starting Ollama..."
    docker compose up -d ollama
    
    # Wait for Ollama to be ready
    check_service "Ollama" "http://localhost:11434" || exit 1
    
    # Pull required models
    echo "Pulling Mistral model..."
    curl -X POST http://localhost:11434/api/pull -d '{"name": "mistral"}'
fi

# Start Streamlit
echo "Starting Streamlit..."
docker compose up -d streamlit

# Wait for services
check_service "Streamlit" "http://localhost:8501" || exit 1

echo -e "\n${GREEN}All services are running!${NC}"
echo -e "\nAccess the application at:"
echo -e "- Web UI: http://localhost:8501"
echo -e "- Ollama API: http://localhost:11434" 