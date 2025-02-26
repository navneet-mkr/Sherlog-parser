#!/bin/bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Building Sherlog Parser C++ Implementation (Simple Build) =====${NC}"

# Create temporary CMakeLists.txt that only builds the core library and main application
cat > CMakeLists.txt.simple << 'EOF'
cmake_minimum_required(VERSION 3.14)
project(sherlog_parser_cpp VERSION 0.1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Architecture-specific flags
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    # Apple Silicon (ARM) - use NEON
    message(STATUS "Detected Apple Silicon (ARM64) - using NEON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    add_definitions(-DUSE_ARM_SIMD)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    # ARM64 - use NEON
    message(STATUS "Detected ARM64 - using NEON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    add_definitions(-DUSE_ARM_SIMD)
else()
    # x86 - check for AVX2/SSE4.2
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
    CHECK_CXX_COMPILER_FLAG("-msse4.2" COMPILER_SUPPORTS_SSE42)
    
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        add_definitions(-DUSE_X86_SIMD)
        message(STATUS "Building with AVX2 support")
    elseif(COMPILER_SUPPORTS_SSE42)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
        add_definitions(-DUSE_X86_SIMD)
        message(STATUS "Building with SSE4.2 support")
    else()
        message(STATUS "Building without SIMD optimizations")
    endif()
endif()

# Source files - only numeric_analysis for simplicity
set(SOURCES
    src/numeric_analysis.cpp
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Build library
add_library(sherlog_parser_lib ${SOURCES})

# Build executable
add_executable(sherlog_parser src/main.cpp)
target_link_libraries(sherlog_parser sherlog_parser_lib)
EOF

# Create build directory
mkdir -p build
cd build

# Configure CMake with the simple CMakeLists.txt
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -GNinja .. -DCMAKELIST_PATH=../CMakeLists.txt.simple

# Build
echo -e "${YELLOW}Building...${NC}"
ninja

echo -e "${GREEN}Build completed successfully!${NC}"
echo "Run the main executable with: ./sherlog_parser"

# Return to main directory and cleanup
cd ..
mv build/compile_commands.json . 2>/dev/null || true