#!/bin/bash

# Default values
DEFAULT_OLLAMA_HOST="http://ollama"
DEFAULT_OLLAMA_PORT="11434"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local width=40
    local percentage=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    
    printf "\rProgress: ["
    printf "%${filled}s" '' | tr ' ' '='
    printf "%${empty}s" '' | tr ' ' ' '
    printf "] %d%%" $percentage
}

# Status update function
show_status() {
    local message="$1"
    local status="${2:-...}" # Default to ... if no status provided
    printf "\r%-50s %s" "$message" "$status"
}

# Initialization progress
show_initialization() {
    local steps=(
        "🔍 Checking environment"
        "📦 Verifying Docker"
        "🔌 Checking network"
        "🚀 Preparing services"
    )
    
    echo -e "\n${BLUE}Initializing Sherlog Parser${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for i in "${!steps[@]}"; do
        show_status "${steps[$i]}"
        sleep 0.5
        echo -e "\r${steps[$i]} ${GREEN}✓${NC}"
    done
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

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

# Show initialization progress
show_initialization

# Function to check if a URL is accessible with progress
check_url() {
    local url="$1"
    local max_attempts=5
    local attempt=1
    local wait_time=2

    echo -e "\n${BLUE}Connecting to Ollama${NC}"
    while [ $attempt -le $max_attempts ]; do
        show_status "Attempting connection" "attempt $attempt of $max_attempts"
        if curl -s -f "$url/api/tags" > /dev/null 2>&1; then
            echo -e "\r${GREEN}✓ Successfully connected to Ollama${NC}"
            return 0
        fi
        sleep $wait_time
        attempt=$((attempt + 1))
    done
    echo -e "\r${RED}✗ Failed to connect to Ollama at $url${NC}"
    return 1
}

# Function to check if service is ready with progress
check_service() {
    local service="$1"
    local url="$2"
    local max_attempts=10
    local attempt=1
    
    echo -e "\n${BLUE}Starting $service${NC}"
    while [ $attempt -le $max_attempts ]; do
        show_progress $attempt $max_attempts
        
        if [ "$service" = "Streamlit" ]; then
            if docker compose ps streamlit | grep -q "running" && \
               curl -s "$url" | grep -q "Streamlit"; then
                echo -e "\n${GREEN}✓ $service is ready${NC}"
                return 0
            fi
        else
            if curl -s -f "$url" > /dev/null 2>&1; then
                echo -e "\n${GREEN}✓ $service is ready${NC}"
                return 0
            fi
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo -e "\n${YELLOW}⚠️  $service health check timed out, but service might still be starting...${NC}"
        if [ "$service" = "Streamlit" ]; then
            return 0
        fi
    else
        echo -e "\n${RED}✗ $service failed to start${NC}"
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

# Start services with progress
start_services() {
    echo -e "\n${BLUE}Starting Services${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if needs_rebuild "streamlit"; then
        show_status "Building Streamlit container"
        docker compose build streamlit > /dev/null 2>&1
        echo -e "\r${GREEN}✓ Built Streamlit container${NC}"
    fi
    
    show_status "Starting containers"
    if [ "$USE_LOCAL_OLLAMA" = true ] || [ "$OLLAMA_HOST" != "$DEFAULT_OLLAMA_HOST" ]; then
        OLLAMA_HOST=$CONTAINER_OLLAMA_URL docker compose up -d dagster streamlit > /dev/null 2>&1
    else
        docker compose --profile with-ollama up -d > /dev/null 2>&1
    fi
    echo -e "\r${GREEN}✓ Started containers${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Main execution
if [ "$USE_LOCAL_OLLAMA" = true ] || [ "$OLLAMA_HOST" != "$DEFAULT_OLLAMA_HOST" ]; then
    echo -e "\n${BLUE}Configuration${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "🔍 External Ollama: $OLLAMA_HOST:$OLLAMA_PORT"
    echo -e "📦 Container URL: $CONTAINER_OLLAMA_URL:$OLLAMA_PORT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if ! check_url "$OLLAMA_HOST:$OLLAMA_PORT"; then
        echo -e "\n${RED}Error: Cannot connect to Ollama${NC}"
        echo "Please ensure:"
        echo "1. Ollama is running at the specified host and port"
        echo "2. The host is accessible from this machine"
        echo "3. Ollama API is responding correctly"
        exit 1
    fi
fi

# Start services
start_services

# Check services
check_service "Dagster" "http://localhost:3000" || exit 1
check_service "Streamlit" "http://localhost:8501" || exit 1

# Show success message and URLs
echo -e "\n${GREEN}✨ Sherlog Parser is Ready!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}🌟 Open in your browser:${NC}"
echo -e "${BLUE}➜ http://localhost:8501${NC}"
echo
echo -e "${YELLOW}Additional URLs:${NC}"
echo -e "- Dagster Dashboard: http://localhost:3000"
echo -e "- Ollama API: $OLLAMA_HOST:$OLLAMA_PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Show filtered logs with timestamp prefix
docker compose logs -f streamlit | grep -v "Examining" | grep -v "RuntimeError" | grep -v "Traceback" | while read -r line; do
    timestamp=$(date '+%H:%M:%S')
    echo "[$timestamp] $line"
done 