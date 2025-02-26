# Tests for the C++ implementation
cmake_minimum_required(VERSION 3.14)

# Find or download GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.13.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Test executables
add_executable(
    numeric_analysis_test
    numeric_analysis_test.cpp
)
target_link_libraries(
    numeric_analysis_test
    GTest::gtest_main
    sherlog_parser_lib
)

# Register tests
include(GoogleTest)
gtest_discover_tests(numeric_analysis_test)

# Commented out until these files are created
# add_executable(
#    log_prefilter_test
#    log_prefilter_test.cpp
# )
# target_link_libraries(
#    log_prefilter_test
#    GTest::gtest_main
#    sherlog_parser_lib
# )

# add_executable(
#    anomaly_detector_test
#    anomaly_detector_test.cpp
# )
# target_link_libraries(
#    anomaly_detector_test
#    GTest::gtest_main
#    sherlog_parser_lib
# )

# gtest_discover_tests(log_prefilter_test)
# gtest_discover_tests(anomaly_detector_test)