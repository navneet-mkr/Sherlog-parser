CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic

# Detect architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Mac-specific flags
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        CXXFLAGS += -DUSE_ARM_SIMD
    else
        CXXFLAGS += -DUSE_X86_SIMD
    endif
else
    # Linux-specific flags
    CXXFLAGS += -DUSE_X86_SIMD
endif

INCLUDES = -I./include
LDFLAGS = -pthread

# Source directories
SRC_DIR = src
INCLUDE_DIR = include
BENCH_DIR = benchmarks
BUILD_DIR = build
BIN_DIR = bin

# Ensure build directories exist
$(shell mkdir -p $(BUILD_DIR) $(BIN_DIR))

# Main source files (excluding benchmarks)
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Benchmark files
NUMERIC_BENCH_SRC = $(BENCH_DIR)/numeric_analysis_benchmark.cpp
NUMERIC_BENCH_OBJ = $(BUILD_DIR)/numeric_analysis_benchmark.o

LOG_PARSER_BENCH_SRC = $(BENCH_DIR)/log_parser/log_parser_benchmark.cpp
LOG_PARSER_BENCH_OBJ = $(BUILD_DIR)/log_parser_benchmark.o

# External libraries
BENCHMARK_LIB = -lbenchmark

all: main numeric_benchmark log_parser_benchmark

# Main application
main: $(BIN_DIR)/main

$(BIN_DIR)/main: $(OBJS) $(BUILD_DIR)/main.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Object files for main application
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Numeric benchmark
numeric_benchmark: $(BIN_DIR)/numeric_benchmark

$(BIN_DIR)/numeric_benchmark: $(NUMERIC_BENCH_OBJ) $(filter-out $(BUILD_DIR)/main.o, $(OBJS))
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(BENCHMARK_LIB)

$(BUILD_DIR)/numeric_analysis_benchmark.o: $(NUMERIC_BENCH_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Log parser benchmark
log_parser_benchmark: $(BIN_DIR)/log_parser_benchmark

$(BIN_DIR)/log_parser_benchmark: $(LOG_PARSER_BENCH_OBJ) $(filter-out $(BUILD_DIR)/main.o, $(OBJS))
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(BENCHMARK_LIB)

$(BUILD_DIR)/log_parser_benchmark.o: $(LOG_PARSER_BENCH_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/* $(BIN_DIR)/*

.PHONY: all clean numeric_benchmark log_parser_benchmark