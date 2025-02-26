#include "template_matcher.hpp"
#include "log_parser.hpp"
#include "prefix_tree.hpp"
#include "numeric_analysis.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <memory>
#include <thread>

// For comparing with Python version
#include <cstdlib>

using namespace sherlog;
using namespace std::chrono;

// Simple benchmark result structure
struct BenchmarkResult {
    std::string name;
    double avg_time_ms;
    double throughput; // items/sec
    size_t iterations;
    
    void print() const {
        std::cout << std::setw(30) << std::left << name 
                  << std::setw(15) << std::right << std::fixed << std::setprecision(3) << avg_time_ms << " ms"
                  << std::setw(15) << std::right << std::fixed << std::setprecision(2) << throughput << " items/s" 
                  << std::setw(10) << std::right << iterations << " iterations" 
                  << std::endl;
    }
};

// Helper to create random log messages
std::vector<std::string> generate_random_logs(size_t count, size_t seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dist(1, 5); // Different log patterns
    
    std::vector<std::string> patterns = {
        "User <OID> logged in from <LOI>",
        "Failed to connect to server <OBN> after <OBA> attempts",
        "Database query took <OBA> ms for user <OID>",
        "Process <PID> started with arguments <OTP>",
        "HTTP request returned status <STC> for <LOI>"
    };
    
    std::vector<std::string> users = {"user123", "admin", "system", "root", "guest"};
    std::vector<std::string> ips = {"192.168.1.1", "10.0.0.5", "172.16.254.1", "127.0.0.1"};
    std::vector<std::string> servers = {"db01", "app02", "web03", "cache04"};
    std::vector<std::string> statuses = {"200", "404", "500", "403", "301"};
    
    std::vector<std::string> logs;
    logs.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        int pattern_idx = dist(gen) - 1;
        std::string log = patterns[pattern_idx];
        
        // Replace variables with values
        if (log.find("<OID>") != std::string::npos) {
            log.replace(log.find("<OID>"), 5, users[gen() % users.size()]);
        }
        if (log.find("<LOI>") != std::string::npos) {
            log.replace(log.find("<LOI>"), 5, ips[gen() % ips.size()]);
        }
        if (log.find("<OBN>") != std::string::npos) {
            log.replace(log.find("<OBN>"), 5, servers[gen() % servers.size()]);
        }
        if (log.find("<OBA>") != std::string::npos) {
            log.replace(log.find("<OBA>"), 5, std::to_string(gen() % 1000));
        }
        if (log.find("<PID>") != std::string::npos) {
            log.replace(log.find("<PID>"), 5, std::to_string(gen() % 10000));
        }
        if (log.find("<STC>") != std::string::npos) {
            log.replace(log.find("<STC>"), 5, statuses[gen() % statuses.size()]);
        }
        if (log.find("<OTP>") != std::string::npos) {
            log.replace(log.find("<OTP>"), 5, "--verbose --config=/etc/app.conf");
        }
        
        logs.push_back(log);
    }
    
    return logs;
}

// Helper to read logs from file
std::vector<std::string> read_logs_from_file(const std::string& filename, size_t limit = 0) {
    std::vector<std::string> logs;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return logs;
    }
    
    std::string line;
    while (std::getline(file, line) && (limit == 0 || logs.size() < limit)) {
        logs.push_back(line);
    }
    
    std::cout << "Read " << logs.size() << " lines from " << filename << std::endl;
    return logs;
}

// Template matcher benchmark
BenchmarkResult benchmark_template_matcher(size_t iterations) {
    // Create test templates
    std::vector<std::pair<std::string, std::string>> template_pairs = {
        {"User <OID> logged in from <LOI>", "User admin logged in from 192.168.1.1"},
        {"Failed to connect to server <OBN> after <OBA> attempts", "Failed to connect to server db01 after 3 attempts"},
        {"Process <PID> started with arguments <OTP>", "Process 1234 started with arguments --verbose"}
    };
    
    TemplateMatcher matcher(0.8);
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        for (const auto& pair : template_pairs) {
            auto result = matcher.match(pair.first, pair.second);
            // Prevent optimization
            if (result.similarity_score < 0) {
                std::cout << "Error" << std::endl;
            }
        }
    }
    
    auto end = high_resolution_clock::now();
    auto total_time = duration_cast<milliseconds>(end - start).count();
    
    return {
        "Template Matcher",
        static_cast<double>(total_time) / (iterations * template_pairs.size()),
        (iterations * template_pairs.size() * 1000.0) / total_time,
        iterations * template_pairs.size()
    };
}

// Numeric analysis benchmark
BenchmarkResult benchmark_numeric_analysis(size_t iterations) {
    // Create test data
    std::vector<std::vector<double>> data;
    std::vector<std::string> field_names = {"value1", "value2", "value3"};
    
    // Generate random data with some anomalies
    std::mt19937 gen(42);
    std::normal_distribution<> dist(100.0, 15.0);
    
    for (size_t i = 0; i < 10000; ++i) {
        std::vector<double> row;
        for (size_t j = 0; j < 3; ++j) {
            double value = dist(gen);
            if (i % 100 == 0) {
                // Add anomalies
                value *= 3;
            }
            row.push_back(value);
        }
        data.push_back(row);
    }
    
    NumericAnomalyDetector detector(3.0, 1.5, 10, true);
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        auto result = detector.detect_anomalies(data, field_names);
        // Prevent optimization
        if (result.anomaly_mask.empty()) {
            std::cout << "Error" << std::endl;
        }
    }
    
    auto end = high_resolution_clock::now();
    auto total_time = duration_cast<milliseconds>(end - start).count();
    
    return {
        "Numeric Analysis",
        static_cast<double>(total_time) / iterations,
        (iterations * 1000.0) / total_time,
        iterations
    };
}

// Helper for running Python benchmarks
BenchmarkResult run_python_benchmark(const std::string& script_name, const std::string& benchmark_name) {
    std::string command = "python3 benchmark_python.py " + script_name;
    
    // Run the Python benchmark
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error running Python benchmark" << std::endl;
        return {"Python " + benchmark_name, 0.0, 0.0, 0};
    }
    
    char buffer[128];
    std::string result = "";
    
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != nullptr) {
            result += buffer;
        }
    }
    pclose(pipe);
    
    // Parse the benchmark result
    std::istringstream iss(result);
    std::string line;
    double avg_time_ms = 0.0;
    double throughput = 0.0;
    size_t iterations = 0;
    
    while (std::getline(iss, line)) {
        if (line.find("avg_time_ms:") != std::string::npos) {
            avg_time_ms = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("throughput:") != std::string::npos) {
            throughput = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("iterations:") != std::string::npos) {
            iterations = std::stoul(line.substr(line.find(":") + 1));
        }
    }
    
    return {"Python " + benchmark_name, avg_time_ms, throughput, iterations};
}

int main(int argc, char** argv) {
    std::cout << "Sherlog C++ Implementation Benchmark" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Parse command line args
    size_t template_matcher_iterations = 1000;
    size_t numeric_analysis_iterations = 10;
    bool run_python_comparison = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--template-iterations" && i + 1 < argc) {
            template_matcher_iterations = std::stoul(argv[++i]);
        } else if (arg == "--numeric-iterations" && i + 1 < argc) {
            numeric_analysis_iterations = std::stoul(argv[++i]);
        } else if (arg == "--python-comparison") {
            run_python_comparison = true;
        }
    }
    
    // Print header
    std::cout << std::setw(30) << std::left << "Benchmark"
              << std::setw(15) << std::right << "Avg Time"
              << std::setw(15) << std::right << "Throughput"
              << std::setw(10) << std::right << "Iterations"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    // Run benchmarks
    auto template_result = benchmark_template_matcher(template_matcher_iterations);
    template_result.print();
    
    auto numeric_result = benchmark_numeric_analysis(numeric_analysis_iterations);
    numeric_result.print();
    
    // Python comparison if requested
    if (run_python_comparison) {
        std::cout << std::endl << "Python Comparison:" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        auto py_template_result = run_python_benchmark("template_matcher", "Template Matcher");
        py_template_result.print();
        
        auto py_numeric_result = run_python_benchmark("numeric_analysis", "Numeric Analysis");
        py_numeric_result.print();
        
        // Print speedup
        std::cout << std::endl << "C++ Speedup:" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        std::cout << std::setw(30) << std::left << "Template Matcher" 
                  << std::setw(15) << std::right << std::fixed << std::setprecision(2) 
                  << py_template_result.avg_time_ms / template_result.avg_time_ms << "x" << std::endl;
        std::cout << std::setw(30) << std::left << "Numeric Analysis" 
                  << std::setw(15) << std::right << std::fixed << std::setprecision(2)
                  << py_numeric_result.avg_time_ms / numeric_result.avg_time_ms << "x" << std::endl;
    }
    
    return 0;
}