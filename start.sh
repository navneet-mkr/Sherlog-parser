#!/bin/bash

# Default values
DEFAULT_OLLAMA_HOST="http://ollama"
DEFAULT_OLLAMA_PORT="11434"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Help message
show_help() {
    echo "Usage: ./start.sh [OPTIONS]"
    echo
    echo "Start the log parsing application with optional Ollama configuration."
    echo
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  --ollama-host HOST         Specify Ollama host (default: $DEFAULT_OLLAMA_HOST)"
    echo "  --ollama-port PORT         Specify Ollama port (default: $DEFAULT_OLLAMA_PORT)"
    echo "  --use-local-ollama         Use local Ollama instance instead of container"
    echo
    echo "Examples:"
    echo "  ./start.sh                                    # Use default Ollama container"
    echo "  ./start.sh --use-local-ollama                # Use local Ollama instance"
    echo "  ./start.sh --ollama-host http://localhost    # Use specific Ollama host"
}

# Parse command line arguments
OLLAMA_HOST=$DEFAULT_OLLAMA_HOST
OLLAMA_PORT=$DEFAULT_OLLAMA_PORT
USE_LOCAL_OLLAMA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --ollama-host)
            OLLAMA_HOST="$2"
            shift 2
            ;;
        --ollama-port)
            OLLAMA_PORT="$2"
            shift 2
            ;;
        --use-local-ollama)
            USE_LOCAL_OLLAMA=true
            OLLAMA_HOST="http://localhost"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Function to get the correct Ollama host for containers
get_container_ollama_host() {
    if [ "$USE_LOCAL_OLLAMA" = true ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "host.docker.internal"  # For macOS
        else
            echo "172.17.0.1"  # For Linux (docker0 interface)
        fi
    else
        echo "ollama"  # Default container name
    fi
}

# Export environment variables
export OLLAMA_HOST=$OLLAMA_HOST
export OLLAMA_PORT=$OLLAMA_PORT

# Set container-specific Ollama host
if [ "$USE_LOCAL_OLLAMA" = true ] || [ "$OLLAMA_HOST" != "$DEFAULT_OLLAMA_HOST" ]; then
    CONTAINER_OLLAMA_HOST=$(get_container_ollama_host)
    export CONTAINER_OLLAMA_URL="http://${CONTAINER_OLLAMA_HOST}"
else
    CONTAINER_OLLAMA_URL="http://ollama"
fi

# Function to check if a URL is accessible
check_url() {
    local url="$1"
    local max_attempts=5
    local attempt=1
    local wait_time=2

    echo -n "Checking connection to Ollama..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url/api/tags" > /dev/null 2>&1; then
            echo -e "\r${GREEN}✓ Connected to Ollama"
            return 0
        fi
        echo -n "."
        sleep $wait_time
        attempt=$((attempt + 1))
    done
    echo -e "\n❌ Failed to connect to Ollama at $url"
    return 1
}

# Function to check if service is ready
check_service() {
    local service="$1"
    local url="$2"
    local max_attempts=10
    local attempt=1
    
    echo -n "Starting $service..."
    while [ $attempt -le $max_attempts ]; do
        if [ "$service" = "Streamlit" ]; then
            # For Streamlit, check if the process is running and port is listening
            if docker compose ps streamlit | grep -q "running" && \
               curl -s "$url" | grep -q "Streamlit"; then
                echo -e "\r${GREEN}✓ $service is ready${NC}"
                return 0
            fi
        else
            # For other services, use standard HTTP check
            if curl -s -f "$url" > /dev/null 2>&1; then
                echo -e "\r${GREEN}✓ $service is ready${NC}"
                return 0
            fi
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo -e "\n${YELLOW}⚠️  $service health check timed out, but service might still be starting...${NC}"
        # Don't exit if it's Streamlit, as it might still be starting up
        if [ "$service" = "Streamlit" ]; then
            return 0
        fi
    else
        echo -e "\n❌ $service failed to start"
    fi
    return 1
}

# Function to cleanup on exit
cleanup() {
    echo -e "\n\n${YELLOW}Shutting down services...${NC}"
    docker compose down > /dev/null 2>&1
    echo -e "${GREEN}✓ All services stopped${NC}"
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT TERM

# Function to check if rebuild is needed
needs_rebuild() {
    local service="$1"
    # Check if image exists
    if ! docker compose images "$service" | grep -q "$service"; then
        return 0
    fi
    return 1
}

# Start services
if [ "$USE_LOCAL_OLLAMA" = true ] || [ "$OLLAMA_HOST" != "$DEFAULT_OLLAMA_HOST" ]; then
    echo -e "${BLUE}🔍 Using external Ollama at $OLLAMA_HOST:$OLLAMA_PORT${NC}"
    echo -e "${BLUE}📦 Container Ollama URL: $CONTAINER_OLLAMA_URL:$OLLAMA_PORT${NC}"
    
    if ! check_url "$OLLAMA_HOST:$OLLAMA_PORT"; then
        echo "Please ensure:"
        echo "1. Ollama is running at the specified host and port"
        echo "2. The host is accessible from this machine"
        echo "3. Ollama API is responding correctly"
        exit 1
    fi
    
    echo -e "${BLUE}🚀 Starting services...${NC}"
    
    # Check if we need to rebuild
    if needs_rebuild "streamlit"; then
        echo -e "${BLUE}📦 Building containers...${NC}"
        docker compose build streamlit > /dev/null 2>&1
    fi
    
    OLLAMA_HOST=$CONTAINER_OLLAMA_URL docker compose up -d dagster streamlit > /dev/null 2>&1
else
    echo -e "${BLUE}🚀 Starting all services...${NC}"
    
    # Check if we need to rebuild
    if needs_rebuild "streamlit"; then
        echo -e "${BLUE}📦 Building containers...${NC}"
        docker compose build streamlit > /dev/null 2>&1
    fi
    
    docker compose --profile with-ollama up -d > /dev/null 2>&1
fi

# Check if services are running
check_service "Dagster" "http://localhost:3000" || exit 1
check_service "Streamlit" "http://localhost:8501" || exit 1

# Show success message and URLs
echo -e "\n${GREEN}✨ All services are running!${NC}\n"
echo -e "${YELLOW}🌟 Open Sherlog Parser:${NC}"
echo -e "${BLUE}➜ http://localhost:8501${NC}"
echo
echo -e "${YELLOW}Additional URLs:${NC}"
echo -e "- Dagster Dashboard: http://localhost:3000"
echo -e "- Ollama API: $OLLAMA_HOST:$OLLAMA_PORT"
echo
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Keep the script running and show Streamlit logs
docker compose logs -f streamlit | grep -v "Examining" | grep -v "RuntimeError" | grep -v "Traceback" 