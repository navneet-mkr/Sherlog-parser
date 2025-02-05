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

# Function to get condensed error logs
get_container_errors() {
    local service="$1"
    local lines=10  # Number of error lines to show
    local errors
    
    # Get recent logs with errors
    errors=$(docker compose logs --tail=50 $service 2>&1 | grep -i -E "error|exception|fatal|failed|traceback" | tail -n $lines)
    
    if [ ! -z "$errors" ]; then
        echo "$errors"
        return 0
    fi
    return 1
}

# Function to show debugging guidance
show_debug_guidance() {
    local service="$1"
    local error_type="$2"
    
    echo -e "\n${YELLOW}Debugging Guidance${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}Issue detected in $service service${NC}"
    echo
    echo "To investigate:"
    echo "1. View full logs:"
    echo "   docker compose logs $service"
    echo
    echo "2. Access container shell:"
    echo "   docker compose exec $service bash"
    echo
    echo "3. Check container status:"
    echo "   docker compose ps $service"
    echo
    echo -e "${YELLOW}Need help?${NC}"
    echo "- Report this issue: https://github.com/yourusername/log-parse-ai/issues/new"
    echo "- Include the error message and your configuration"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

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

# Enhanced service check with error reporting
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
        
        # Check for errors after a few attempts
        if [ $attempt -eq 5 ]; then
            echo -e "\n${YELLOW}Checking for errors...${NC}"
            if errors=$(get_container_errors $service); then
                echo -e "\n${RED}Found errors in $service:${NC}"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "$errors"
                show_debug_guidance $service "startup"
                
                # For non-Streamlit services, exit on error
                if [ "$service" != "Streamlit" ]; then
                    return 1
                fi
            fi
        fi
        
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo -e "\n${YELLOW}⚠️  $service health check timed out${NC}"
        
        # Check for errors one last time
        if errors=$(get_container_errors $service); then
            echo -e "\n${RED}Found errors in $service:${NC}"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "$errors"
            show_debug_guidance $service "timeout"
        fi
        
        if [ "$service" = "Streamlit" ]; then
            return 0
        fi
        return 1
    fi
    
    return 1
}

# Enhanced cleanup with error checking
cleanup() {
    echo -e "\n\n${YELLOW}Shutting Down Services${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check for errors before shutdown
    local services=$(docker compose ps --services --filter "status=running")
    
    if [ ! -z "$services" ]; then
        echo -e "${BLUE}Checking for errors before shutdown...${NC}"
        for service in $services; do
            if errors=$(get_container_errors $service); then
                echo -e "\n${YELLOW}Recent errors in $service:${NC}"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "$errors"
                echo
            fi
        done
    fi
    
    # Proceed with shutdown
    local total_services=$(echo "$services" | wc -l)
    local current=0
    
    if [ -z "$services" ]; then
        echo -e "${YELLOW}No running services found${NC}"
    else
        for service in $services; do
            current=$((current + 1))
            show_progress $current $total_services
            echo -e "\n${BLUE}Stopping $service...${NC}"
            
            # Stop individual service
            if docker compose stop $service > /dev/null 2>&1; then
                echo -e "${GREEN}✓ Stopped $service${NC}"
            else
                echo -e "${RED}✗ Failed to stop $service${NC}"
                # Try to get any error information
                if errors=$(get_container_errors $service); then
                    echo -e "Last errors from $service:"
                    echo "$errors"
                fi
            fi
        done
    fi
    
    echo -e "\n${BLUE}Cleaning up resources...${NC}"
    
    # Remove containers
    show_status "Removing containers"
    if docker compose rm -f > /dev/null 2>&1; then
        echo -e "\r${GREEN}✓ Removed containers${NC}"
    else
        echo -e "\r${RED}✗ Failed to remove containers${NC}"
    fi
    
    # Remove networks
    show_status "Removing networks"
    if docker network prune -f > /dev/null 2>&1; then
        echo -e "\r${GREEN}✓ Removed networks${NC}"
    else
        echo -e "\r${RED}✗ Failed to remove networks${NC}"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}✨ Cleanup completed${NC}"
    echo -e "${BLUE}Thank you for using Sherlog Parser!${NC}"
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

# Function to show build progress
show_build_progress() {
    local service="$1"
    local build_output
    
    echo -e "\n${BLUE}Building $service container${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run docker compose build with progress output
    build_output=$(docker compose build $service 2>&1)
    
    # Process the build output line by line
    while IFS= read -r line; do
        if [[ $line == *"Step "* ]] && [[ $line == *"/"* ]]; then
            # Extract step number and total steps
            step=$(echo $line | grep -o 'Step [0-9]*/[0-9]*' | cut -d'/' -f1 | cut -d' ' -f2)
            total=$(echo $line | grep -o 'Step [0-9]*/[0-9]*' | cut -d'/' -f2)
            show_progress $step $total
            
            # Extract and show the actual command being run
            command=$(echo $line | cut -d' ' -f4-)
            echo -e "\n${YELLOW}▶ $command${NC}"
        elif [[ $line == *"-->"* ]]; then
            # Show layer caching information
            echo -e "${GREEN}✓${NC} Using cache"
        elif [[ $line == "Successfully tagged"* ]]; then
            echo -e "\n${GREEN}✓ Build completed successfully${NC}"
        elif [[ $line == *"error"* ]] || [[ $line == *"Error"* ]]; then
            echo -e "\n${RED}✗ Error: $line${NC}"
        fi
    done <<< "$build_output"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Start services with progress
start_services() {
    echo -e "\n${BLUE}Starting Services${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if needs_rebuild "streamlit"; then
        show_build_progress "streamlit"
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