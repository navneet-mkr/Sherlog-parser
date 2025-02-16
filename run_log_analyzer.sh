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

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create and activate virtual environment
echo -e "${BLUE}Setting up Python virtual environment...${NC}"
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Start TimescaleDB using Docker Compose
echo -e "${BLUE}Setting up TimescaleDB...${NC}"
docker-compose up -d timescaledb

# Wait for TimescaleDB to be healthy
echo "Waiting for TimescaleDB to be ready..."
while ! docker-compose exec timescaledb pg_isready -U postgres > /dev/null 2>&1; do
    echo -n "."
    sleep 1
done
echo -e "\nTimescaleDB is ready!"

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
ollama serve &

# Pull the Mistral model
echo -e "${BLUE}Pulling Mistral model...${NC}"
ollama pull mistral

# Create cache directory
mkdir -p cache

# Start the Streamlit app
echo -e "${GREEN}Setup complete! Starting Log Analysis Dashboard...${NC}"
streamlit run src/ui/log_analyzer.py

# Cleanup function
cleanup() {
    echo -e "${BLUE}Cleaning up...${NC}"
    docker-compose down
    pkill -f "ollama serve"
    deactivate
}

# Set up cleanup on script exit
trap cleanup EXIT 