#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
USE_LOCAL_OLLAMA=false
OLLAMA_HOST="http://localhost"
OLLAMA_PORT="11434"

# Function to show help message
show_help() {
    echo "Usage: ./start.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --use-local-ollama     Use local Ollama instance instead of container"
    echo "  --ollama-host HOST     Specify custom Ollama host (default: http://localhost)"
    echo "  --ollama-port PORT     Specify custom Ollama port (default: 11434)"
    echo "  -h, --help             Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-local-ollama)
            USE_LOCAL_OLLAMA=true
            shift
            ;;
        --ollama-host)
            OLLAMA_HOST="$2"
            shift 2
            ;;
        --ollama-port)
            OLLAMA_PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

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

# Check if Ollama is running locally
if curl -s -f "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo -e "${YELLOW}Detected local Ollama instance running at http://localhost:11434${NC}"
    
    # Ask user for preference if not specified via command line
    if [ "$USE_LOCAL_OLLAMA" = false ]; then
        read -p "Would you like to use the local Ollama instance? (y/N) " choice
        case "$choice" in 
            y|Y )
                USE_LOCAL_OLLAMA=true
                ;;
            * )
                echo "Will use containerized Ollama instead."
                ;;
        esac
    fi
fi

# Start services based on user choice
if [ "$USE_LOCAL_OLLAMA" = true ]; then
    echo "Using local Ollama instance..."
    export OLLAMA_BASE_URL="http://localhost:11434"
    
    # Start only the Streamlit service
    docker compose up -d streamlit
else
    echo "Starting containerized Ollama..."
    docker compose up -d ollama
    
    # Wait for Ollama to be ready
    check_service "Ollama" "http://localhost:11434" || exit 1
    
    # Check if Mistral model is available
    if ! curl -s -f "http://localhost:11434/api/tags" | grep -q "mistral"; then
        echo "Pulling Mistral model..."
        curl -X POST http://localhost:11434/api/pull -d '{"name": "mistral"}'
    fi
    
    # Start Streamlit with containerized Ollama
    docker compose up -d streamlit
fi

# Wait for Streamlit
check_service "Streamlit" "http://localhost:8501" || exit 1

echo -e "\n${GREEN}All services are running!${NC}"
echo -e "\nAccess the application at:"
echo -e "- Web UI: http://localhost:8501"
echo -e "- Ollama API: http://localhost:11434"

# Show which Ollama instance is being used
if [ "$USE_LOCAL_OLLAMA" = true ]; then
    echo -e "\n${YELLOW}Using local Ollama instance${NC}"
else
    echo -e "\n${YELLOW}Using containerized Ollama instance${NC}" 