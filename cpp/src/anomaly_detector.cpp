#include "../include/anomaly_detector.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <limits>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <sstream>  // Add this for std::stringstream
#include <iomanip>  // Add this for std::put_time
#include <ctime>    // Add this for time functions

// Simple DBSCAN implementation for high-dimensional embedding clustering
class DBSCAN {
public:
    DBSCAN(float eps, size_t min_samples, 
           std::function<float(const std::vector<float>&, const std::vector<float>&)> distance_func)
        : eps_(eps), min_samples_(min_samples), distance_func_(std::move(distance_func)) {}
    
    std::vector<int> fit_predict(const std::vector<std::vector<float>>& data) {
        if (data.empty()) return {};
        
        const size_t n_samples = data.size();
        std::vector<int> labels(n_samples, -1); // -1 means unclassified
        std::vector<bool> visited(n_samples, false);
        
        // Current cluster ID
        int cluster_id = 0;
        
        // Process each point
        for (size_t i = 0; i < n_samples; ++i) {
            if (visited[i]) continue;
            
            visited[i] = true;
            // Find neighbors
            std::vector<size_t> neighbors = rangeQuery(data, i);
            
            if (neighbors.size() < min_samples_) {
                // Mark as noise
                labels[i] = -1;
                continue;
            }
            
            // Start a new cluster
            labels[i] = cluster_id;
            
            // Process neighbors
            std::queue<size_t> queue;
            for (size_t neighbor : neighbors) {
                queue.push(neighbor);
            }
            
            while (!queue.empty()) {
                size_t current = queue.front();
                queue.pop();
                
                if (!visited[current]) {
                    visited[current] = true;
                    std::vector<size_t> current_neighbors = rangeQuery(data, current);
                    
                    if (current_neighbors.size() >= min_samples_) {
                        // Add new neighbors to the queue
                        for (size_t neighbor : current_neighbors) {
                            queue.push(neighbor);
                        }
                    }
                }
                
                if (labels[current] == -1) {
                    // Mark as part of the current cluster
                    labels[current] = cluster_id;
                }
            }
            
            // Move to the next cluster
            ++cluster_id;
        }
        
        return labels;
    }
    
private:
    float eps_;
    size_t min_samples_;
    std::function<float(const std::vector<float>&, const std::vector<float>&)> distance_func_;
    
    std::vector<size_t> rangeQuery(const std::vector<std::vector<float>>& data, size_t point_idx) {
        std::vector<size_t> neighbors;
        const auto& point = data[point_idx];
        
        for (size_t i = 0; i < data.size(); ++i) {
            if (distance_func_(point, data[i]) <= eps_) {
                neighbors.push_back(i);
            }
        }
        
        return neighbors;
    }
};

// Constructor
IncidentAnomalyDetector::IncidentAnomalyDetector(
    AnomalyDetectorConfig config,
    std::optional<ExplainerFunction> explainer_func
) : 
    config_(std::move(config)),
    explainer_func_(std::move(explainer_func)) {
    
    // Initialize numeric detector
    numeric_detector_ = std::make_unique<NumericAnomalyDetector>(
        config_.numeric_std_threshold, 
        1.5f,  // Default IQR threshold
        3,     // Min samples
        true   // Use robust stats
    );
    
    // Initialize pre-filter if enabled
    if (config_.enable_prefilter) {
        prefilter_ = std::make_unique<LogPreFilter>(config_.prefilter_config);
    }
}

// Compute cosine distance between two vectors
float IncidentAnomalyDetector::cosineDistance(
    const std::vector<float>& a, 
    const std::vector<float>& b
) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 1.0f; // Maximum distance for invalid vectors
    }
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a <= std::numeric_limits<float>::epsilon() || 
        norm_b <= std::numeric_limits<float>::epsilon()) {
        return 1.0f; // Maximum distance for zero vectors
    }
    
    float similarity = dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    
    // Clamp to [-1, 1] to handle floating point errors
    similarity = std::max(-1.0f, std::min(1.0f, similarity));
    
    // Convert to distance (1 - similarity)
    return 1.0f - similarity;
}

// Perform DBSCAN clustering on embeddings
std::vector<int> IncidentAnomalyDetector::performClustering(
    const std::vector<EmbeddingVector>& embeddings
) {
    if (embeddings.empty()) {
        return {};
    }
    
    // Prepare data for clustering
    std::vector<std::vector<float>> data;
    data.reserve(embeddings.size());
    
    for (const auto& embedding : embeddings) {
        data.push_back(embedding.data);
    }
    
    // Create DBSCAN instance
    DBSCAN dbscan(
        config_.eps,
        config_.min_samples,
        [](const std::vector<float>& a, const std::vector<float>& b) {
            return IncidentAnomalyDetector::cosineDistance(a, b);
        }
    );
    
    // Perform clustering
    return dbscan.fit_predict(data);
}

// Extract numeric fields from log entries
std::vector<std::string> IncidentAnomalyDetector::extractNumericFields(
    const std::vector<LogEntry>& logs
) {
    if (logs.empty()) {
        return {};
    }
    
    // Prepare data in the format expected by NumericAnomalyDetector
    std::vector<std::unordered_map<std::string, std::string>> data;
    
    for (const auto& log : logs) {
        std::unordered_map<std::string, std::string> entry;
        
        // Add standard fields
        entry["message"] = log.message;
        entry["level"] = log.level;
        entry["component"] = log.component;
        
        // Add timestamp
        auto time_t = std::chrono::system_clock::to_time_t(log.timestamp);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        entry["timestamp"] = ss.str();
        
        // Add parameters
        for (const auto& [key, value] : log.parameters) {
            entry[key] = value;
        }
        
        data.push_back(std::move(entry));
    }
    
    // Use NumericAnomalyDetector to extract numeric fields
    return NumericAnomalyDetector::extractNumericFields(data);
}

// Extract numeric values from log entries
std::vector<std::vector<float>> IncidentAnomalyDetector::extractNumericValues(
    const std::vector<LogEntry>& logs,
    const std::vector<std::string>& field_names
) {
    if (logs.empty() || field_names.empty()) {
        return {};
    }
    
    std::vector<std::vector<float>> result(logs.size());
    
    for (size_t i = 0; i < logs.size(); ++i) {
        const auto& log = logs[i];
        std::vector<float> values(field_names.size(), 0.0f);
        
        for (size_t j = 0; j < field_names.size(); ++j) {
            const auto& field = field_names[j];
            
            auto it = log.parameters.find(field);
            if (it != log.parameters.end()) {
                try {
                    values[j] = std::stof(it->second);
                } catch (...) {
                    // Not a number, leave as 0.0
                }
            }
        }
        
        result[i] = std::move(values);
    }
    
    return result;
}

// Detect anomalies in logs
std::vector<std::pair<LogEntry, AnomalyDetectionResult>> IncidentAnomalyDetector::detectAnomalies(
    const std::vector<LogEntry>& logs,
    const std::vector<EmbeddingVector>& embeddings,
    size_t time_window_hours
) {
    if (logs.empty() || embeddings.empty() || logs.size() != embeddings.size()) {
        std::cerr << "Invalid inputs to detectAnomalies" << std::endl;
        return {};
    }
    
    // Apply pre-filtering if enabled
    std::vector<LogEntry> filtered_logs;
    std::vector<EmbeddingVector> filtered_embeddings;
    std::vector<size_t> original_indices;
    
    if (config_.enable_prefilter && prefilter_) {
        filtered_logs = prefilter_->filterLogs(logs);
        
        // Map filtered logs back to original indices and create filtered embeddings
        std::unordered_map<std::string, size_t> message_to_index;
        for (size_t i = 0; i < logs.size(); ++i) {
            message_to_index[logs[i].message] = i;
        }
        
        for (const auto& log : filtered_logs) {
            auto it = message_to_index.find(log.message);
            if (it != message_to_index.end()) {
                size_t idx = it->second;
                original_indices.push_back(idx);
                filtered_embeddings.push_back(embeddings[idx]);
            }
        }
    } else {
        filtered_logs = logs;
        filtered_embeddings = embeddings;
        for (size_t i = 0; i < logs.size(); ++i) {
            original_indices.push_back(i);
        }
    }
    
    // Perform clustering
    std::vector<int> cluster_labels = performClustering(filtered_embeddings);
    
    // Find small clusters (potential group anomalies)
    std::unordered_map<int, size_t> cluster_counts;
    for (int label : cluster_labels) {
        if (label != -1) { // Skip noise points
            cluster_counts[label]++;
        }
    }
    
    std::unordered_set<int> small_clusters;
    for (const auto& [cluster, count] : cluster_counts) {
        if (count < config_.min_samples) {
            small_clusters.insert(cluster);
        }
    }
    
    // Detect numeric anomalies if present
    std::vector<bool> numeric_anomalies(filtered_logs.size(), false);
    std::unordered_map<std::string, std::vector<bool>> field_anomalies;
    
    std::vector<std::string> numeric_fields = extractNumericFields(filtered_logs);
    if (!numeric_fields.empty()) {
        std::vector<std::vector<float>> numeric_values = extractNumericValues(filtered_logs, numeric_fields);
        
        auto [combined, per_field] = numeric_detector_->detectAnomalies(
            numeric_values, 
            numeric_fields, 
            cluster_labels
        );
        
        numeric_anomalies = combined;
        field_anomalies = per_field;
    }
    
    // Prepare results
    std::vector<std::pair<LogEntry, AnomalyDetectionResult>> anomalies;
    
    for (size_t i = 0; i < filtered_logs.size(); ++i) {
        bool is_embedding_anomaly = false;
        
        // Check if this is an embedding anomaly
        if (cluster_labels[i] == -1) {
            // Noise point
            is_embedding_anomaly = true;
        } else if (small_clusters.find(cluster_labels[i]) != small_clusters.end()) {
            // Part of a small cluster
            is_embedding_anomaly = true;
        }
        
        bool is_numeric_anomaly = numeric_anomalies[i];
        
        // Only keep anomalies
        if (!is_embedding_anomaly && !is_numeric_anomaly) {
            continue;
        }
        
        // Create result
        AnomalyDetectionResult result;
        result.is_embedding_anomaly = is_embedding_anomaly;
        result.is_numeric_anomaly = is_numeric_anomaly;
        result.cluster_label = cluster_labels[i];
        result.detection_time = std::chrono::system_clock::now();
        
        // Collect numeric deviations if this is a numeric anomaly
        if (is_numeric_anomaly && !numeric_fields.empty()) {
            for (const auto& field : numeric_fields) {
                if (field_anomalies.count(field) > 0 && 
                    field_anomalies[field][i]) {
                    
                    // Calculate deviation if in a valid cluster
                    int cluster = cluster_labels[i];
                    if (cluster != -1) {
                        // Find all logs in this cluster
                        std::vector<float> cluster_values;
                        for (size_t j = 0; j < filtered_logs.size(); ++j) {
                            if (cluster_labels[j] == cluster) {
                                auto it = filtered_logs[j].parameters.find(field);
                                if (it != filtered_logs[j].parameters.end()) {
                                    try {
                                        cluster_values.push_back(std::stof(it->second));
                                    } catch (...) {
                                        // Not a number, ignore
                                    }
                                }
                            }
                        }
                        
                        if (cluster_values.size() >= 2) {
                            // Calculate mean and std
                            float sum = 0.0f;
                            for (float val : cluster_values) {
                                sum += val;
                            }
                            float mean = sum / cluster_values.size();
                            
                            float sum_sq_diff = 0.0f;
                            for (float val : cluster_values) {
                                float diff = val - mean;
                                sum_sq_diff += diff * diff;
                            }
                            float std_dev = std::sqrt(sum_sq_diff / cluster_values.size());
                            
                            if (std_dev > std::numeric_limits<float>::epsilon()) {
                                // Find current value
                                auto it = filtered_logs[i].parameters.find(field);
                                if (it != filtered_logs[i].parameters.end()) {
                                    try {
                                        float value = std::stof(it->second);
                                        float deviation = (value - mean) / std_dev;
                                        result.numeric_deviations.emplace_back(field, deviation);
                                    } catch (...) {
                                        // Not a number, ignore
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Add explanation if enabled
        if (config_.explain_anomalies && explainer_func_.has_value()) {
            result.explanation = explainer_func_.value()(filtered_logs[i], result);
        }
        
        anomalies.emplace_back(filtered_logs[i], result);
    }
    
    // Log results
    std::cout << "Found " << anomalies.size() << " anomalies in " << filtered_logs.size() 
              << " logs (" << (anomalies.size() * 100.0f / filtered_logs.size()) 
              << "% anomaly rate)" << std::endl;
    
    return anomalies;
}