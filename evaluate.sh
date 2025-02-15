#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
BOLD='\033[1m'

# Default values
USE_LOCAL_OLLAMA=false
OLLAMA_HOST="http://localhost"
OLLAMA_PORT="11434"
CONTAINERS_STARTED=false

# Function to handle cleanup on script exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ "$CONTAINERS_STARTED" = true ]; then
        echo "Stopping Docker containers..."
        docker compose down
    fi
    echo -e "${GREEN}Cleanup complete${NC}"
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Function to show help message
show_help() {
    echo "Usage: ./evaluate.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --use-local-ollama     Use local Ollama instance instead of container"
    echo "  --ollama-host HOST     Specify custom Ollama host (default: http://localhost)"
    echo "  --ollama-port PORT     Specify custom Ollama port (default: 11434)"
    echo "  -h, --help             Show this help message"
    exit 0
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
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create required directories
mkdir -p data/eval_datasets output/eval cache/eval

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

# Function to display results
show_results() {
    echo -e "\n${BOLD}Evaluation Results:${NC}"
    
    # Check if results file exists
    RESULTS_FILE="./output/eval/results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo -e "\n${GREEN}Results from $RESULTS_FILE:${NC}"
        # Pretty print the JSON results
        cat "$RESULTS_FILE" | jq '.'
    else
        echo -e "${YELLOW}No results file found at $RESULTS_FILE${NC}"
    fi
    
    # Show any error logs
    ERROR_LOG="./output/eval/error.log"
    if [ -f "$ERROR_LOG" ]; then
        if [ -s "$ERROR_LOG" ]; then
            echo -e "\n${RED}Errors encountered:${NC}"
            cat "$ERROR_LOG"
        fi
    fi
}

# Function to check container status
check_container_status() {
    local container_name=$1
    local status=$(docker inspect -f '{{.State.Status}}' "$container_name" 2>/dev/null)
    
    if [ "$status" != "running" ]; then
        echo -e "\n${RED}Container $container_name is not running (status: $status)${NC}"
        return 1
    fi
    return 0
}

# Function to monitor container logs for errors
monitor_container_logs() {
    local container_name=$1
    local error_pattern="error|Error|ERROR|Exception|EXCEPTION|Failed|FAILED"
    
    if docker logs "$container_name" 2>&1 | grep -iE "$error_pattern" > /dev/null; then
        echo -e "\n${RED}Errors detected in $container_name logs:${NC}"
        docker logs "$container_name" 2>&1 | grep -iE "$error_pattern"
        return 1
    fi
    return 0
}

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

# Configure Ollama based on user choice
if [ "$USE_LOCAL_OLLAMA" = true ]; then
    echo "Using local Ollama instance..."
    export OLLAMA_BASE_URL="http://localhost:11434"
    
    # Check if model is available locally
    if ! curl -s -f "http://localhost:11434/api/tags" | grep -q "mistral"; then
        echo -e "${YELLOW}Mistral model not found in local Ollama instance${NC}"
        read -p "Would you like to pull the Mistral model? (Y/n) " choice
        case "$choice" in 
            n|N )
                echo -e "${RED}Mistral model is required for evaluation${NC}"
                exit 1
                ;;
            * )
                echo "Pulling Mistral model..."
                curl -X POST http://localhost:11434/api/pull -d '{"name": "mistral"}'
                ;;
        esac
    fi
    
    # Start evaluation service with local Ollama
    docker compose up -d evaluation
    CONTAINERS_STARTED=true
else
    echo "Starting containerized Ollama..."
    # Start Ollama container with profile
    docker compose --profile with-ollama up -d ollama
    CONTAINERS_STARTED=true
    
    # Wait for Ollama to be ready
    check_service "Ollama" "http://localhost:11434" || cleanup
    
    # Check if model is available
    if ! curl -s -f "http://localhost:11434/api/tags" | grep -q "mistral"; then
        echo "Pulling Mistral model..."
        curl -X POST http://localhost:11434/api/pull -d '{"name": "mistral"}'
    fi
    
    # Start evaluation service
    docker compose up -d evaluation
fi

# Wait for evaluation service
echo -n "Waiting for evaluation service..."
max_retries=30
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if curl -s -f "http://localhost:8502" > /dev/null 2>&1; then
        echo -e "\n${GREEN}Evaluation service is ready!${NC}"
        break
    fi
    
    retry_count=$((retry_count + 1))
    if [ $retry_count -lt $max_retries ]; then
        echo -n "."
        sleep 2
    fi
done

if [ $retry_count -eq $max_retries ]; then
    echo -e "\n${RED}Failed to start evaluation service${NC}"
    cleanup
    exit 1
fi

echo -e "\n${GREEN}Evaluation pipeline is ready!${NC}"
echo -e "\nAccess the evaluation dashboard at:"
echo -e "http://localhost:8502"

# Show which Ollama instance is being used
if [ "$USE_LOCAL_OLLAMA" = true ]; then
    echo -e "\n${YELLOW}Using local Ollama instance${NC}"
else
    echo -e "\n${YELLOW}Using containerized Ollama instance${NC}"
fi

# Start evaluation
echo -e "\n${BOLD}Running evaluation...${NC}"
echo -e "${YELLOW}This may take a while depending on the dataset size${NC}"

# Monitor evaluation progress
while true; do
    # Check if evaluation container is still running
    if ! check_container_status "log-parse-ai-evaluation-1"; then
        echo -e "\n${RED}Evaluation container stopped unexpectedly${NC}"
        show_results
        cleanup
        exit 1
    fi
    
    # Check container logs for errors
    if ! monitor_container_logs "log-parse-ai-evaluation-1"; then
        echo -e "\n${RED}Evaluation failed due to errors in container logs${NC}"
        show_results
        cleanup
        exit 1
    fi
    
    if [ -f "$RESULTS_FILE" ]; then
        show_results
        echo -e "\n${GREEN}Evaluation completed successfully!${NC}"
        break
    fi
    
    # Check for errors
    if [ -f "$ERROR_LOG" ] && [ -s "$ERROR_LOG" ]; then
        echo -e "\n${RED}Evaluation failed with errors:${NC}"
        cat "$ERROR_LOG"
        break
    fi
    
    echo -n "."
    sleep 5
done

# Keep containers running but allow for graceful shutdown
echo -e "\n${YELLOW}Evaluation complete. Press Ctrl+C to stop and cleanup.${NC}"
while true; do
    sleep 1
done 