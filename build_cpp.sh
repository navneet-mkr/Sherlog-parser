#!/bin/bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Building Sherlog Parser C++ Implementation =====${NC}"

# Create build directory
mkdir -p cpp/build
cd cpp/build

# Configure CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo -e "${YELLOW}Building...${NC}"
make -j$(sysctl -n hw.ncpu)

echo -e "${GREEN}Build completed successfully!${NC}"
echo "Run the main executable with: ./sherlog_parser"
echo "Run benchmark with: ./benchmarks/numeric_analysis_benchmark"