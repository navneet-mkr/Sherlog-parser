#!/bin/bash
set -e

# Create build directory
mkdir -p build
cd build

# Configure CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo "Build completed successfully!"
echo "Run the main executable with: ./sherlog_parser"
echo "Run benchmarks with: ./benchmarks/numeric_analysis_benchmark"