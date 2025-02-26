#include <benchmark/benchmark.h>
#include <vector>
#include <string>
#include <random>
#include <sstream>
#include <algorithm>
#include <iomanip>

#include "../../include/log_parser.hpp"

// Generate random logs for benchmarking
std::vector<std::string> generateRandomLogs(size_t num_logs, unsigned int seed = 42) {
    std::vector<std::string> templates = {
        "Connection to {server} failed after {retry} attempts",
        "User {username} logged in from {ip_address}",
        "Database query took {time}ms to execute",
        "Memory usage at {percent}%, system load: {load}",
        "Failed to process request from {client}: {error_code}",
        "Thread {thread_id} waiting for lock on resource {resource_id}",
        "API request to {endpoint} returned status {status_code}",
        "File {filename} not found in directory {directory}",
        "Successfully processed {count} items in {duration}s",
        "Cache hit ratio: {ratio}%, misses: {misses}"
    };
    
    std::vector<std::string> log_messages;
    log_messages.reserve(num_logs);
    
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> template_dist(0, templates.size() - 1);
    std::uniform_int_distribution<int> small_int_dist(1, 100);
    std::uniform_int_distribution<int> medium_int_dist(1, 1000);
    std::uniform_int_distribution<int> large_int_dist(1000, 9999);
    std::uniform_int_distribution<int> ip_dist(1, 255);
    std::uniform_real_distribution<float> small_float_dist(0.1, 10.0);
    std::uniform_real_distribution<float> large_float_dist(10.0, 99.9);
    
    // Status codes distribution
    std::vector<int> status_codes = {200, 201, 400, 401, 403, 404, 500};
    std::uniform_int_distribution<size_t> status_idx_dist(0, status_codes.size() - 1);
    
    for (size_t i = 0; i < num_logs; ++i) {
        std::string templ = templates[template_dist(rng)];
        std::string log = templ;
        
        // Replace placeholders with random values
        size_t pos;
        
        // Server
        if ((pos = log.find("{server}")) != std::string::npos) {
            log.replace(pos, 8, "server-" + std::to_string(small_int_dist(rng)));
        }
        
        // Retry
        if ((pos = log.find("{retry}")) != std::string::npos) {
            log.replace(pos, 7, std::to_string(small_int_dist(rng) % 10 + 1));
        }
        
        // Username
        if ((pos = log.find("{username}")) != std::string::npos) {
            log.replace(pos, 10, "user" + std::to_string(large_int_dist(rng)));
        }
        
        // IP address
        if ((pos = log.find("{ip_address}")) != std::string::npos) {
            std::stringstream ip_ss;
            ip_ss << ip_dist(rng) << "." << ip_dist(rng) << "." 
                  << ip_dist(rng) << "." << ip_dist(rng);
            log.replace(pos, 12, ip_ss.str());
        }
        
        // Time
        if ((pos = log.find("{time}")) != std::string::npos) {
            log.replace(pos, 6, std::to_string(medium_int_dist(rng) * 5));
        }
        
        // Percent
        if ((pos = log.find("{percent}")) != std::string::npos) {
            log.replace(pos, 9, std::to_string(small_int_dist(rng)));
        }
        
        // Load
        if ((pos = log.find("{load}")) != std::string::npos) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << small_float_dist(rng);
            log.replace(pos, 6, ss.str());
        }
        
        // Client
        if ((pos = log.find("{client}")) != std::string::npos) {
            log.replace(pos, 8, "client-" + std::to_string(medium_int_dist(rng)));
        }
        
        // Error code
        if ((pos = log.find("{error_code}")) != std::string::npos) {
            log.replace(pos, 12, "ERR" + std::to_string(medium_int_dist(rng) % 900 + 100));
        }
        
        // Thread ID
        if ((pos = log.find("{thread_id}")) != std::string::npos) {
            log.replace(pos, 11, "thread-" + std::to_string(small_int_dist(rng)));
        }
        
        // Resource ID
        if ((pos = log.find("{resource_id}")) != std::string::npos) {
            log.replace(pos, 13, "res-" + std::to_string(small_int_dist(rng) % 50 + 1));
        }
        
        // Endpoint
        if ((pos = log.find("{endpoint}")) != std::string::npos) {
            std::stringstream ss;
            ss << "/api/v" << (small_int_dist(rng) % 3 + 1) 
               << "/resources/" << medium_int_dist(rng);
            log.replace(pos, 10, ss.str());
        }
        
        // Status code
        if ((pos = log.find("{status_code}")) != std::string::npos) {
            log.replace(pos, 13, std::to_string(status_codes[status_idx_dist(rng)]));
        }
        
        // Filename
        if ((pos = log.find("{filename}")) != std::string::npos) {
            log.replace(pos, 10, "file" + std::to_string(medium_int_dist(rng)) + ".txt");
        }
        
        // Directory
        if ((pos = log.find("{directory}")) != std::string::npos) {
            log.replace(pos, 11, "/var/data/dir" + std::to_string(small_int_dist(rng) % 50 + 1));
        }
        
        // Count
        if ((pos = log.find("{count}")) != std::string::npos) {
            log.replace(pos, 7, std::to_string(medium_int_dist(rng) * 10));
        }
        
        // Duration
        if ((pos = log.find("{duration}")) != std::string::npos) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << small_float_dist(rng) * 6.0;
            log.replace(pos, 10, ss.str());
        }
        
        // Ratio
        if ((pos = log.find("{ratio}")) != std::string::npos) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << large_float_dist(rng);
            log.replace(pos, 7, ss.str());
        }
        
        // Misses
        if ((pos = log.find("{misses}")) != std::string::npos) {
            log.replace(pos, 8, std::to_string(medium_int_dist(rng)));
        }
        
        log_messages.push_back(log);
    }
    
    return log_messages;
}

// Benchmark for parsing a single log
static void BM_ParseSingleLog(benchmark::State& state) {
    auto logs = generateRandomLogs(100);
    sherlog::LogParserLLM parser;
    
    for (auto _ : state) {
        // Use a different log for each iteration to prevent caching effects
        size_t i = state.iterations() % logs.size();
        benchmark::DoNotOptimize(parser.parse_log(logs[i], static_cast<int>(i)));
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Benchmark for batch parsing
static void BM_ParseLogBatch(benchmark::State& state) {
    const size_t batch_size = static_cast<size_t>(state.range(0));
    auto logs = generateRandomLogs(batch_size);
    
    // Convert to the format expected by parse_logs_batch
    std::vector<std::pair<std::string, int>> log_pairs;
    for (size_t i = 0; i < logs.size(); ++i) {
        log_pairs.emplace_back(logs[i], static_cast<int>(i));
    }
    
    sherlog::LogParserLLM parser;
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(parser.parse_logs_batch(log_pairs));
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

// Register benchmarks
BENCHMARK(BM_ParseSingleLog);
BENCHMARK(BM_ParseLogBatch)->RangeMultiplier(10)->Range(10, 10000);

BENCHMARK_MAIN();