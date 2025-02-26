#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>

#include "../include/numeric_analysis.hpp"

// Random data generator
class RandomDataGenerator {
public:
    RandomDataGenerator(size_t seed = 42) : rng_(seed) {}
    
    std::vector<float> generateVector(size_t size, float min = 0.0f, float max = 1.0f) {
        std::vector<float> vec(size);
        std::uniform_real_distribution<float> dist(min, max);
        
        for (auto& val : vec) {
            val = dist(rng_);
        }
        
        return vec;
    }
    
    std::vector<std::vector<float>> generateMatrix(size_t rows, size_t cols, float min = 0.0f, float max = 1.0f) {
        std::vector<std::vector<float>> matrix(rows);
        
        for (auto& row : matrix) {
            row = generateVector(cols, min, max);
        }
        
        return matrix;
    }
    
    std::vector<int> generateGroups(size_t size, int num_groups) {
        std::vector<int> groups(size);
        std::uniform_int_distribution<int> dist(0, num_groups - 1);
        
        for (auto& val : groups) {
            val = dist(rng_);
        }
        
        return groups;
    }
    
    std::vector<std::unordered_map<std::string, std::string>> generateMappedData(
        size_t size, 
        const std::vector<std::string>& field_names,
        float numeric_ratio = 0.8f
    ) {
        std::vector<std::unordered_map<std::string, std::string>> data(size);
        std::uniform_real_distribution<float> val_dist(0.0f, 100.0f);
        std::uniform_real_distribution<float> type_dist(0.0f, 1.0f);
        
        for (auto& entry : data) {
            for (const auto& field : field_names) {
                if (type_dist(rng_) < numeric_ratio) {
                    // Generate numeric value
                    entry[field] = std::to_string(val_dist(rng_));
                } else {
                    // Generate string value
                    entry[field] = "string-value-" + std::to_string(static_cast<int>(val_dist(rng_)));
                }
            }
        }
        
        return data;
    }
    
private:
    std::mt19937 rng_;
};

// Benchmark for detectFieldAnomalies
static void BM_DetectFieldAnomalies(benchmark::State& state) {
    const size_t num_samples = state.range(0);
    
    RandomDataGenerator generator;
    std::vector<float> values = generator.generateVector(num_samples, 0.0f, 100.0f);
    
    // Inject some anomalies (5%)
    const size_t num_anomalies = num_samples * 0.05;
    std::uniform_int_distribution<size_t> idx_dist(0, num_samples - 1);
    std::uniform_real_distribution<float> val_dist(200.0f, 500.0f);
    
    std::mt19937 rng(42);
    for (size_t i = 0; i < num_anomalies; ++i) {
        size_t idx = idx_dist(rng);
        values[idx] = val_dist(rng);
    }
    
    NumericAnomalyDetector detector;
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(detector.detectFieldAnomalies(values));
    }
    
    state.SetItemsProcessed(state.iterations() * num_samples);
    state.SetBytesProcessed(state.iterations() * num_samples * sizeof(float));
}

// Benchmark for detectAnomalies
static void BM_DetectAnomalies(benchmark::State& state) {
    const size_t num_samples = state.range(0);
    const size_t num_fields = state.range(1);
    
    RandomDataGenerator generator;
    auto values = generator.generateMatrix(num_samples, num_fields, 0.0f, 100.0f);
    
    // Generate field names
    std::vector<std::string> field_names;
    for (size_t i = 0; i < num_fields; ++i) {
        field_names.push_back("field_" + std::to_string(i));
    }
    
    // Generate group IDs (30% of samples are in a group)
    std::vector<int> group_ids(num_samples, -1);
    const size_t num_grouped = num_samples * 0.3;
    const int num_groups = 5;
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> idx_dist(0, num_samples - 1);
    std::uniform_int_distribution<int> group_dist(0, num_groups - 1);
    
    for (size_t i = 0; i < num_grouped; ++i) {
        size_t idx = idx_dist(rng);
        group_ids[idx] = group_dist(rng);
    }
    
    NumericAnomalyDetector detector;
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(detector.detectAnomalies(values, field_names, group_ids));
    }
    
    state.SetItemsProcessed(state.iterations() * num_samples * num_fields);
    state.SetBytesProcessed(state.iterations() * num_samples * num_fields * sizeof(float));
}

// Benchmark for extractNumericFields
static void BM_ExtractNumericFields(benchmark::State& state) {
    const size_t num_samples = state.range(0);
    const size_t num_fields = state.range(1);
    
    // Generate field names
    std::vector<std::string> field_names;
    for (size_t i = 0; i < num_fields; ++i) {
        field_names.push_back("field_" + std::to_string(i));
    }
    
    RandomDataGenerator generator;
    auto data = generator.generateMappedData(num_samples, field_names, 0.7f);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(NumericAnomalyDetector::extractNumericFields(data));
    }
    
    state.SetItemsProcessed(state.iterations() * num_samples * num_fields);
}

// Register benchmarks with various input sizes
BENCHMARK(BM_DetectFieldAnomalies)
    ->RangeMultiplier(10)
    ->Range(100, 1000000);

BENCHMARK(BM_DetectAnomalies)
    ->Args({1000, 10})
    ->Args({10000, 10})
    ->Args({100000, 10})
    ->Args({1000, 50})
    ->Args({10000, 50});

BENCHMARK(BM_ExtractNumericFields)
    ->Args({100, 10})
    ->Args({1000, 10})
    ->Args({10000, 10})
    ->Args({1000, 50});

BENCHMARK_MAIN();