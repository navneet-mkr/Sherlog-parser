# Benchmark tools
cmake_minimum_required(VERSION 3.14)

# Find Google Benchmark
find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found - downloading and building from source")
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.7.0
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

# Build benchmarks
add_executable(numeric_analysis_benchmark numeric_analysis_benchmark.cpp)
target_link_libraries(numeric_analysis_benchmark sherlog_parser_lib benchmark::benchmark)

# Commented out until these files are created
# add_executable(log_prefilter_benchmark log_prefilter_benchmark.cpp)
# target_link_libraries(log_prefilter_benchmark sherlog_parser_lib benchmark::benchmark)

# add_executable(anomaly_detector_benchmark anomaly_detector_benchmark.cpp)
# target_link_libraries(anomaly_detector_benchmark sherlog_parser_lib benchmark::benchmark)

# Install
install(TARGETS 
    numeric_analysis_benchmark
    # log_prefilter_benchmark
    # anomaly_detector_benchmark
    RUNTIME DESTINATION bin/benchmarks
)