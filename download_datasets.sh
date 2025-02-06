#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to create directory if it doesn't exist
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo -e "${GREEN}Created directory: $1${NC}"
    fi
}

# Function to download a file if it doesn't exist
download_file() {
    local url=$1
    local output=$2
    local dir=$(dirname "$output")
    
    create_dir "$dir"
    
    if [ ! -f "$output" ]; then
        echo -e "${YELLOW}Downloading: $(basename "$output")${NC}"
        wget -q --show-progress "$url" -O "$output"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully downloaded: $(basename "$output")${NC}"
        else
            echo -e "${RED}Failed to download: $(basename "$output")${NC}"
            rm -f "$output"
            return 1
        fi
    else
        echo -e "${GREEN}File already exists: $(basename "$output")${NC}"
    fi
    return 0
}

# Create base directories
create_dir "data/eval_datasets/loghub_2k"
create_dir "data/eval_datasets/logpub"

# Download Loghub-2k datasets
echo -e "\n${GREEN}Downloading Loghub-2k datasets...${NC}"

# Apache
download_file "https://github.com/logpai/loghub/raw/master/Apache/Apache_2k.log_structured.csv" \
    "data/eval_datasets/loghub_2k/Apache/Apache.log_structured.csv"
download_file "https://github.com/logpai/loghub/raw/master/Apache/Apache_2k.log_templates.csv" \
    "data/eval_datasets/loghub_2k/Apache/Apache.log_templates.csv"

# Hadoop
download_file "https://github.com/logpai/loghub/raw/master/Hadoop/Hadoop_2k.log_structured.csv" \
    "data/eval_datasets/loghub_2k/Hadoop/Hadoop.log_structured.csv"
download_file "https://github.com/logpai/loghub/raw/master/Hadoop/Hadoop_2k.log_templates.csv" \
    "data/eval_datasets/loghub_2k/Hadoop/Hadoop.log_templates.csv"

# Linux
download_file "https://github.com/logpai/loghub/raw/master/Linux/Linux_2k.log_structured.csv" \
    "data/eval_datasets/loghub_2k/Linux/Linux.log_structured.csv"
download_file "https://github.com/logpai/loghub/raw/master/Linux/Linux_2k.log_templates.csv" \
    "data/eval_datasets/loghub_2k/Linux/Linux.log_templates.csv"

# Zookeeper
download_file "https://github.com/logpai/loghub/raw/master/Zookeeper/Zookeeper_2k.log_structured.csv" \
    "data/eval_datasets/loghub_2k/Zookeeper/Zookeeper.log_structured.csv"
download_file "https://github.com/logpai/loghub/raw/master/Zookeeper/Zookeeper_2k.log_templates.csv" \
    "data/eval_datasets/loghub_2k/Zookeeper/Zookeeper.log_templates.csv"

# LogPub notice
echo -e "\n${YELLOW}Note about LogPub datasets:${NC}"
echo -e "LogPub datasets require registration. Please:"
echo -e "1. Visit ${GREEN}https://github.com/logpai/LogPub${NC}"
echo -e "2. Fill out the registration form"
echo -e "3. Download the benchmark datasets"
echo -e "4. Extract them to: ${GREEN}data/eval_datasets/logpub/${NC}"

# Verify downloads
echo -e "\n${GREEN}Verifying downloads...${NC}"
python3 -c "
from src.eval.datasets import DatasetLoader
try:
    loader = DatasetLoader()
    available = loader.list_available_datasets()
    print('\nAvailable datasets:')
    for dataset_type, systems in available.items():
        print(f'\n{dataset_type}:')
        for system in systems:
            try:
                dataset = loader.load_dataset(system, dataset_type)
                print(f'  ✓ {system}: {dataset.size} logs, {len(set(dataset.ground_truth_templates.values()))} templates')
            except Exception as e:
                print(f'  ✗ {system}: Error - {str(e)}')
except Exception as e:
    print(f'\nError during verification: {str(e)}')"

echo -e "\n${GREEN}Download process complete!${NC}"
echo -e "You can now run evaluations using: ${YELLOW}./evaluate.sh${NC}" 