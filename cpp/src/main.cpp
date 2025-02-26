#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>
#include <iomanip>

#include "../include/numeric_analysis.hpp"

int main() {
    std::cout << "Sherlog Parser C++ Implementation - Basic Demo" << std::endl;
    std::cout << "==============================================\n" << std::endl;
    
    // Generate random data
    std::cout << "Generating test data..." << std::endl;
    const size_t num_samples = 1000;
    const size_t num_fields = 5;
    
    // Create random number generator
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    
    // Generate multi-dimensional data
    std::vector<std::vector<float>> values(num_samples, std::vector<float>(num_fields));
    for (auto& row : values) {
        for (auto& val : row) {
            val = dist(rng);
        }
    }
    
    // Add some anomalies
    std::uniform_int_distribution<size_t> sample_dist(0, num_samples - 1);
    std::uniform_int_distribution<size_t> field_dist(0, num_fields - 1);
    
    for (int i = 0; i < 50; ++i) {
        size_t sample = sample_dist(rng);
        size_t field = field_dist(rng);
        // Make an extreme value (10x normal)
        values[sample][field] = dist(rng) * 10.0f;
    }
    
    // Field names
    std::vector<std::string> field_names = {
        "response_time_ms",
        "cpu_usage_percent",
        "memory_mb",
        "items_processed",
        "queue_depth"
    };
    
    // Create detector with default parameters
    std::cout << "Creating detector..." << std::endl;
    NumericAnomalyDetector detector;
    
    // Measure execution time
    std::cout << "Detecting anomalies..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto [combined_anomalies, field_anomalies] = detector.detectAnomalies(values, field_names);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Count anomalies per field
    std::cout << "\nAnomalies detected:" << std::endl;
    size_t total_anomalies = std::count(combined_anomalies.begin(), combined_anomalies.end(), true);
    
    std::cout << "  Total: " << total_anomalies << " out of " << num_samples << " samples" << std::endl;
    
    for (const auto& [field, anomalies] : field_anomalies) {
        size_t count = std::count(anomalies.begin(), anomalies.end(), true);
        std::cout << "  Field '" << field << "': " << count << " anomalies" << std::endl;
    }
    
    std::cout << "\nExecution time: " << duration.count() << " microseconds" << std::endl;
    
    // Demo of per-field anomaly detection
    std::cout << "\nDemonstrating single field anomaly detection:" << std::endl;
    
    // Extract values for a single field
    std::vector<float> single_field(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        single_field[i] = values[i][0];
    }
    
    // Detect anomalies
    start_time = std::chrono::high_resolution_clock::now();
    auto single_field_anomalies = detector.detectFieldAnomalies(single_field);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    size_t single_field_count = std::count(single_field_anomalies.begin(), single_field_anomalies.end(), true);
    
    std::cout << "  Detected " << single_field_count << " anomalies in field '" 
              << field_names[0] << "'" << std::endl;
    std::cout << "  Execution time: " << duration.count() << " microseconds" << std::endl;
    
    return 0;
}