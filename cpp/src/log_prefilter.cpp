#include "../include/log_prefilter.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <unordered_map>
#include <sstream>

LogPreFilter::LogPreFilter(PreFilterConfig config)
    : config_(std::move(config)),
      random_generator_(std::random_device{}()) {
}

std::vector<LogEntry> LogPreFilter::filterLogs(const std::vector<LogEntry>& logs) {
    if (logs.empty()) {
        return logs;
    }
    
    // Start with priority logs
    std::vector<LogEntry> priority_logs;
    std::unordered_map<std::string, std::vector<LogEntry>> level_logs;
    
    // Split logs by level
    for (const auto& log : logs) {
        if (config_.priority_levels.find(log.level) != config_.priority_levels.end()) {
            priority_logs.push_back(log);
        } else {
            level_logs[log.level].push_back(log);
        }
    }
    
    // Sample other levels
    std::vector<LogEntry> sampled_logs;
    for (const auto& [level, logs_for_level] : level_logs) {
        float ratio = config_.level_sample_ratios.count(level) > 0
            ? config_.level_sample_ratios.at(level)
            : config_.default_sample_ratio;
            
        std::vector<LogEntry> sampled = sampleByLevel(logs_for_level, level, ratio);
        sampled_logs.insert(sampled_logs.end(), sampled.begin(), sampled.end());
    }
    
    // Combine all logs
    std::vector<LogEntry> filtered_logs;
    filtered_logs.reserve(priority_logs.size() + sampled_logs.size());
    filtered_logs.insert(filtered_logs.end(), priority_logs.begin(), priority_logs.end());
    filtered_logs.insert(filtered_logs.end(), sampled_logs.begin(), sampled_logs.end());
    
    // Deduplicate if configured
    if (config_.max_duplicates.has_value()) {
        filtered_logs = deduplicateMessages(filtered_logs);
    }
    
    // Sort by timestamp
    std::sort(filtered_logs.begin(), filtered_logs.end(), 
        [](const LogEntry& a, const LogEntry& b) {
            return a.timestamp < b.timestamp;
        }
    );
    
    // Log reduction stats
    float reduction = (1.0f - static_cast<float>(filtered_logs.size()) / static_cast<float>(logs.size())) * 100.0f;
    std::cout << "Pre-filter reduced log volume by " << reduction << "% "
              << "(" << filtered_logs.size() << " / " << logs.size() << " logs kept)" << std::endl;
    
    return filtered_logs;
}

std::vector<LogEntry> LogPreFilter::sampleByLevel(
    const std::vector<LogEntry>& logs,
    const std::string& level,
    float ratio
) {
    if (logs.empty()) {
        return logs;
    }
    
    // Calculate sample size
    size_t sample_size = std::max(
        config_.min_logs_per_level,
        static_cast<size_t>(logs.size() * ratio)
    );
    
    // Ensure we don't try to sample more than we have
    sample_size = std::min(sample_size, logs.size());
    
    if (sample_size == logs.size()) {
        return logs; // No sampling needed
    }
    
    // Perform sampling
    std::vector<LogEntry> sampled;
    sampled.reserve(sample_size);
    
    if (sample_size <= 1) {
        // Just take the first and/or last entry
        sampled.push_back(logs.front());
        if (sample_size > 1 && logs.size() > 1) {
            sampled.push_back(logs.back());
        }
    } else {
        // Always include first and last
        sampled.push_back(logs.front());
        
        if (logs.size() > 2) {
            // Sample from the middle
            size_t middle_sample = sample_size - 2; // Excluding first and last
            
            // Create index vector
            std::vector<size_t> indices;
            indices.reserve(logs.size() - 2);
            for (size_t i = 1; i < logs.size() - 1; ++i) {
                indices.push_back(i);
            }
            
            // Random shuffle
            std::shuffle(indices.begin(), indices.end(), random_generator_);
            
            // Take first N indices
            indices.resize(middle_sample);
            
            // Sort indices to preserve time order
            std::sort(indices.begin(), indices.end());
            
            // Add sampled entries
            for (size_t idx : indices) {
                sampled.push_back(logs[idx]);
            }
        }
        
        // Add the last entry
        if (logs.size() > 1) {
            sampled.push_back(logs.back());
        }
    }
    
    return sampled;
}

std::vector<LogEntry> LogPreFilter::deduplicateMessages(const std::vector<LogEntry>& logs) {
    if (!config_.max_duplicates.has_value() || logs.empty()) {
        return logs;
    }
    
    // Count message occurrences
    std::unordered_map<std::string, std::vector<size_t>> message_indices;
    for (size_t i = 0; i < logs.size(); ++i) {
        message_indices[logs[i].message].push_back(i);
    }
    
    std::vector<LogEntry> deduplicated;
    deduplicated.reserve(logs.size());
    
    // Process each unique message
    for (const auto& [message, indices] : message_indices) {
        size_t max_dupes = config_.max_duplicates.value();
        
        if (indices.size() <= max_dupes) {
            // Keep all instances
            for (size_t idx : indices) {
                deduplicated.push_back(logs[idx]);
            }
        } else {
            // Keep first and last
            deduplicated.push_back(logs[indices.front()]);
            
            // Sample from middle if needed
            size_t middle_count = max_dupes - 2;
            if (middle_count > 0 && indices.size() > 2) {
                // Create temporary vector of middle indices
                std::vector<size_t> middle_indices(indices.begin() + 1, indices.end() - 1);
                
                // Shuffle and select
                std::shuffle(middle_indices.begin(), middle_indices.end(), random_generator_);
                middle_indices.resize(std::min(middle_count, middle_indices.size()));
                std::sort(middle_indices.begin(), middle_indices.end());
                
                // Add selected middle entries
                for (size_t idx : middle_indices) {
                    deduplicated.push_back(logs[idx]);
                }
            }
            
            // Add last instance
            deduplicated.push_back(logs[indices.back()]);
        }
    }
    
    return deduplicated;
}