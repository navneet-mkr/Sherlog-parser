#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check and install wget
check_wget() {
    if ! command -v wget &> /dev/null; then
        echo -e "${YELLOW}wget is not installed. Attempting to install...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install wget
            else
                echo -e "${RED}Homebrew is not installed. Please install wget manually:${NC}"
                echo "1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                echo "2. Then run: brew install wget"
                exit 1
            fi
        elif command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y wget
        elif command -v yum &> /dev/null; then
            sudo yum install -y wget
        else
            echo -e "${RED}Could not install wget automatically. Please install wget manually.${NC}"
            exit 1
        fi
        
        if ! command -v wget &> /dev/null; then
            echo -e "${RED}Failed to install wget. Please install it manually.${NC}"
            exit 1
        fi
        echo -e "${GREEN}wget installed successfully!${NC}"
    fi
}

# Check for wget before proceeding
check_wget

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
    "data/eval_datasets/loghub_2k/Apache/Apache_2k.log_structured.csv"
download_file "https://github.com/logpai/loghub/raw/master/Apache/Apache_2k.log_templates.csv" \
    "data/eval_datasets/loghub_2k/Apache/Apache_2k.log_templates.csv"

# Hadoop
download_file "https://github.com/logpai/loghub/raw/master/Hadoop/Hadoop_2k.log_structured.csv" \
    "data/eval_datasets/loghub_2k/Hadoop/Hadoop_2k.log_structured.csv"
download_file "https://github.com/logpai/loghub/raw/master/Hadoop/Hadoop_2k.log_templates.csv" \
    "data/eval_datasets/loghub_2k/Hadoop/Hadoop_2k.log_templates.csv"

# Linux
download_file "https://github.com/logpai/loghub/raw/master/Linux/Linux_2k.log_structured.csv" \
    "data/eval_datasets/loghub_2k/Linux/Linux_2k.log_structured.csv"
download_file "https://github.com/logpai/loghub/raw/master/Linux/Linux_2k.log_templates.csv" \
    "data/eval_datasets/loghub_2k/Linux/Linux_2k.log_templates.csv"

# Zookeeper
download_file "https://github.com/logpai/loghub/raw/master/Zookeeper/Zookeeper_2k.log_structured.csv" \
    "data/eval_datasets/loghub_2k/Zookeeper/Zookeeper_2k.log_structured.csv"
download_file "https://github.com/logpai/loghub/raw/master/Zookeeper/Zookeeper_2k.log_templates.csv" \
    "data/eval_datasets/loghub_2k/Zookeeper/Zookeeper_2k.log_templates.csv"

# LogPub notice
echo -e "\n${YELLOW}Note about LogPub datasets:${NC}"
echo -e "LogPub datasets require registration. Please:"
echo -e "1. Visit ${GREEN}https://github.com/logpai/LogPub${NC}"
echo -e "2. Fill out the registration form"
echo -e "3. Download the benchmark datasets"
echo -e "4. Extract them to: ${GREEN}data/eval_datasets/logpub/${NC}"

# Replace with enhanced instructions
echo -e "\n${YELLOW}=== LogPub Dataset Setup Instructions ===${NC}"
echo -e "\nLogPub datasets require manual setup due to registration requirements:"

echo -e "\n${GREEN}1. Registration Process:${NC}"
echo -e "   - Visit ${GREEN}https://github.com/logpai/LogPub${NC}"
echo -e "   - Click on 'Request for Access' or fill the Google Form"
echo -e "   - Provide your information (name, organization, purpose)"
echo -e "   - Wait for approval email (usually within 1-2 business days)"

echo -e "\n${GREEN}2. Downloading Datasets:${NC}"
echo -e "   - After approval, you'll receive download links"
echo -e "   - Download the benchmark datasets (.zip files)"
echo -e "   - Each system should have two files:"
echo -e "     * {System}_structured.csv - Contains the raw logs"
echo -e "     * {System}_templates.csv  - Contains the templates"

echo -e "\n${GREEN}3. Setup Instructions:${NC}"
echo -e "   - Extract the downloaded files to: ${GREEN}data/eval_datasets/logpub/${NC}"
echo -e "   - Ensure the following directory structure:"
echo -e "     data/eval_datasets/logpub/"
echo -e "     ├── System1/"
echo -e "     │   ├── System1_structured.csv"
echo -e "     │   └── System1_templates.csv"
echo -e "     ├── System2/"
echo -e "     │   ├── System2_structured.csv"
echo -e "     │   └── System2_templates.csv"
echo -e "     └── ..."

echo -e "\n${GREEN}4. Verification:${NC}"
echo -e "   - After setup, you can verify the datasets using:"
echo -e "     python3 -c 'from src.eval.datasets import DatasetLoader; loader = DatasetLoader(); print(loader.list_available_datasets())'"

echo -e "\n${YELLOW}Note:${NC} LogPub datasets are used in research. Please cite their paper:"
echo -e "   'LogPub: A Comprehensive Benchmark for Log Parsing Research'"
echo -e "   Available at: https://arxiv.org/abs/2308.02022"

# Verify downloads
echo -e "\n${GREEN}Verifying downloads...${NC}"

# Function to verify file
verify_file() {
    local file=$1
    if [ -f "$file" ]; then
        # Check if file is not empty and is a valid CSV
        if [ -s "$file" ] && head -n 1 "$file" | grep -q "," ; then
            echo -e "${GREEN}✓ $(basename "$file") exists and appears valid${NC}"
            return 0
        else
            echo -e "${RED}✗ $(basename "$file") exists but may be corrupted${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ $(basename "$file") is missing${NC}"
        return 1
    fi
}

# Verify Loghub-2k datasets
echo -e "\n${YELLOW}Verifying Loghub-2k datasets:${NC}"
datasets=("Apache" "Hadoop" "Linux" "Zookeeper")
all_files_ok=true

for dataset in "${datasets[@]}"; do
    echo -e "\n${YELLOW}Checking $dataset:${NC}"
    structured_file="data/eval_datasets/loghub_2k/$dataset/${dataset}_2k.log_structured.csv"
    templates_file="data/eval_datasets/loghub_2k/$dataset/${dataset}_2k.log_templates.csv"
    
    verify_file "$structured_file" || all_files_ok=false
    verify_file "$templates_file" || all_files_ok=false
done

if [ "$all_files_ok" = true ]; then
    echo -e "\n${GREEN}All Loghub-2k files were downloaded successfully!${NC}"
else
    echo -e "\n${RED}Some files are missing or corrupted. Please run the script again.${NC}"
    exit 1
fi

# Run the Python verification for additional dataset checks
echo -e "\n${YELLOW}Running detailed dataset verification:${NC}"
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