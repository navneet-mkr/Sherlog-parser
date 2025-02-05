#!/bin/bash

# Default values
DEFAULT_OLLAMA_HOST="http://ollama"
DEFAULT_OLLAMA_PORT="11434"

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
    echo "  ./start.sh --ollama-host http://my-ollama --ollama-port 12345  # Custom host and port"
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

# Export environment variables
export OLLAMA_HOST=$OLLAMA_HOST
export OLLAMA_PORT=$OLLAMA_PORT

# Function to check if a URL is accessible
check_url() {
    local url="$1"
    local max_attempts=5
    local attempt=1
    local wait_time=2

    echo "Checking connection to $url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url/api/tags" > /dev/null 2>&1; then
            echo "✅ Successfully connected to Ollama at $url"
            return 0
        fi
        echo "Attempt $attempt of $max_attempts: Cannot connect to $url, waiting ${wait_time}s..."
        sleep $wait_time
        attempt=$((attempt + 1))
        wait_time=$((wait_time * 2))
    done
    
    return 1
}

# Determine if we should start the Ollama container
if [ "$USE_LOCAL_OLLAMA" = true ] || [ "$OLLAMA_HOST" != "$DEFAULT_OLLAMA_HOST" ]; then
    echo "🔍 Using external Ollama instance at $OLLAMA_HOST:$OLLAMA_PORT"
    
    # Check if the external Ollama instance is accessible
    if ! check_url "$OLLAMA_HOST:$OLLAMA_PORT"; then
        echo "❌ Cannot connect to Ollama at $OLLAMA_HOST:$OLLAMA_PORT"
        echo "Please ensure:"
        echo "1. Ollama is running at the specified host and port"
        echo "2. The host is accessible from this machine"
        echo "3. Ollama API is responding correctly"
        exit 1
    fi
    
    # Start without Ollama container
    echo "🚀 Starting services without Ollama container..."
    docker compose up -d dagster streamlit
else
    echo "🐳 Starting all services including Ollama container..."
    docker compose --profile with-ollama up -d
fi

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 5

# Check if services are running
if ! docker compose ps | grep -q "running"; then
    echo "❌ Services failed to start properly"
    echo "Logs from services:"
    docker compose logs
    exit 1
fi

echo "✅ All services are running!"
echo
echo "🌐 Access the application:"
echo "- Web Interface: http://localhost:8501"
echo "- Dagster Dashboard: http://localhost:3000"
echo
echo "📝 Logs can be viewed with: docker compose logs -f"
echo "⏹️ To stop all services: docker compose down" 