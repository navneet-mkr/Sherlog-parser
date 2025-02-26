#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <random>
#include <algorithm>
#include <chrono>

/**
 * @struct PreFilterConfig
 * @brief Configuration for log pre-filtering.
 */
struct PreFilterConfig {
    /// Priority levels to always keep
    std::unordered_set<std::string> priority_levels{"ERROR", "CRITICAL"};
    
    /// Sampling ratios for different log levels
    std::unordered_map<std::string, float> level_sample_ratios{
        {"INFO", 0.1f},
        {"DEBUG", 0.05f},
        {"WARNING", 0.5f}
    };
    
    /// Default sampling ratio for unspecified levels
    float default_sample_ratio = 0.1f;
    
    /// Minimum number of logs to keep per level
    size_t min_logs_per_level = 10;
    
    /// Maximum number of duplicate messages to keep
    std::optional<size_t> max_duplicates = 5;
};

/**
 * @struct LogEntry
 * @brief Simplified representation of a log entry.
 */
struct LogEntry {
    std::string message;
    std::string level;
    std::string component;
    std::chrono::system_clock::time_point timestamp;
    
    // Additional fields could be added as needed
    std::unordered_map<std::string, std::string> parameters;
};

/**
 * @class LogPreFilter
 * @brief Pre-filters logs before expensive operations like embedding and clustering.
 */
class LogPreFilter {
public:
    /**
     * @brief Constructor
     * @param config Pre-filter configuration
     */
    explicit LogPreFilter(PreFilterConfig config = PreFilterConfig());
    
    /**
     * @brief Apply pre-filtering to reduce log volume
     * @param logs Vector of log entries
     * @return Filtered vector of log entries
     */
    std::vector<LogEntry> filterLogs(const std::vector<LogEntry>& logs);
    
private:
    PreFilterConfig config_;
    std::mt19937 random_generator_;
    
    /**
     * @brief Sample logs of a specific level
     * @param logs Vector of log entries
     * @param level Log level to sample
     * @param ratio Sampling ratio (0.0 to 1.0)
     * @return Sampled vector of log entries
     */
    std::vector<LogEntry> sampleByLevel(
        const std::vector<LogEntry>& logs,
        const std::string& level,
        float ratio
    );
    
    /**
     * @brief Reduce duplicate messages while preserving time distribution
     * @param logs Vector of log entries
     * @return Deduplicated vector of log entries
     */
    std::vector<LogEntry> deduplicateMessages(const std::vector<LogEntry>& logs);
};