#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Log Parser AI Setup...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}pip3 is not installed. Please install pip3 first.${NC}"
    exit 1
fi

# Check if Docker is installed (needed for TimescaleDB)
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Create and activate virtual environment
echo -e "${BLUE}Setting up Python virtual environment...${NC}"
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Start TimescaleDB using Docker
echo -e "${BLUE}Setting up TimescaleDB...${NC}"
if ! docker ps | grep -q timescaledb; then
    echo "Starting TimescaleDB container..."
    docker run -d --name timescaledb \
        -p 5432:5432 \
        -e POSTGRES_PASSWORD=password \
        -e POSTGRES_USER=postgres \
        -e POSTGRES_DB=logs \
        timescale/timescaledb:latest-pg14
    
    # Wait for database to be ready
    echo -n "Waiting for TimescaleDB..."
    until docker exec timescaledb pg_isready -U postgres > /dev/null 2>&1; do
        echo -n "."
        sleep 1
    done
    echo -e "${GREEN}ready!${NC}"
fi

# Create .streamlit directory and config if they don't exist
mkdir -p .streamlit
if [ ! -f .streamlit/secrets.toml ]; then
    echo -e "${BLUE}Creating Streamlit configuration...${NC}"
    cat > .streamlit/secrets.toml << EOL
db_url = "postgresql://postgres:password@localhost:5432/logs"
EOL
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${BLUE}Installing Ollama...${NC}"
    curl https://ollama.ai/install.sh | sh
fi

# Start Ollama service
echo -e "${BLUE}Starting Ollama service...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &
    sleep 2  # Give it a moment to start
fi

# Pull the Mistral model
echo -e "${BLUE}Pulling Mistral model...${NC}"
ollama pull mistral

# Create cache directory
mkdir -p cache

# Start the Streamlit app
echo -e "${GREEN}Setup complete! Starting Log Analysis Dashboard...${NC}"
echo -e "${BLUE}You can access the dashboard at:${NC} http://localhost:8501"
streamlit run src/ui/log_analyzer.py

# Cleanup function
cleanup() {
    echo -e "${BLUE}Cleaning up...${NC}"
    docker stop timescaledb
    docker rm timescaledb
    pkill -f "ollama serve"
    deactivate
}

# Set up cleanup on script exit
trap cleanup EXIT 