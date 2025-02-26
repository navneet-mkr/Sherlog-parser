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

# Detect CPU architecture and enable SIMD instructions
include(CheckCXXCompilerFlag)

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

# Source files
set(SOURCES
    src/numeric_analysis.cpp
    src/log_prefilter.cpp
    src/anomaly_detector.cpp
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

# Tests
option(BUILD_TESTS "Build the test suite" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Benchmarks
option(BUILD_BENCHMARKS "Build the benchmark suite" ON)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install
install(TARGETS sherlog_parser
        RUNTIME DESTINATION bin)
        
install(TARGETS sherlog_parser_lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
        
install(DIRECTORY include/
        DESTINATION include/sherlog_parser
        FILES_MATCHING PATTERN "*.hpp")