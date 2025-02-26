#!/bin/bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Compiling Sherlog Parser C++ Implementation (Direct) =====${NC}"

# Create output directory
mkdir -p bin

# Get OS and architecture info
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo -e "${YELLOW}Detected Apple Silicon (ARM64) - using NEON optimizations${NC}"
    ARCH_FLAGS="-march=armv8-a+simd -DUSE_ARM_SIMD"
elif [[ "$(uname -m)" == "x86_64" ]]; then
    # For x86_64, try detecting AVX2 or SSE4.2
    echo -e "${YELLOW}Detected x86_64 - checking for SIMD support${NC}"
    if g++ -mavx2 -dM -E - < /dev/null 2>/dev/null | grep -q "__AVX2__"; then
        echo "AVX2 supported - using AVX2 optimizations"
        ARCH_FLAGS="-mavx2 -DUSE_X86_SIMD"
    elif g++ -msse4.2 -dM -E - < /dev/null 2>/dev/null | grep -q "__SSE4_2__"; then
        echo "SSE4.2 supported - using SSE4.2 optimizations"
        ARCH_FLAGS="-msse4.2 -DUSE_X86_SIMD"
    else
        echo "No SIMD support detected - using scalar code"
        ARCH_FLAGS=""
    fi
else
    echo -e "${YELLOW}Unknown architecture - using scalar code${NC}"
    ARCH_FLAGS=""
fi

# Configure compilation with C++17 and optimization flags
STD_FLAGS="-std=c++17 -Wall -Wextra -O3"
INCLUDE_FLAGS="-I./include"

# Compile numeric_analysis.cpp to object file
echo -e "${YELLOW}Compiling numeric_analysis.cpp...${NC}"
g++ $STD_FLAGS $ARCH_FLAGS $INCLUDE_FLAGS -c src/numeric_analysis.cpp -o bin/numeric_analysis.o

# Compile main.cpp to object file
echo -e "${YELLOW}Compiling main.cpp...${NC}"
g++ $STD_FLAGS $ARCH_FLAGS $INCLUDE_FLAGS -c src/main.cpp -o bin/main.o

# Link the objects to create the executable
echo -e "${YELLOW}Linking...${NC}"
g++ bin/numeric_analysis.o bin/main.o -o bin/sherlog_parser

echo -e "${GREEN}Build completed successfully!${NC}"
echo "Run the executable with: ./bin/sherlog_parser"