#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <cmath>
#include <algorithm>
#include <limits>

/**
 * @class NumericAnomalyDetector
 * @brief Detector for anomalies in numeric log fields using statistical methods.
 */
class NumericAnomalyDetector {
public:
    /**
     * @brief Constructor
     * @param std_threshold Number of standard deviations for outlier detection
     * @param iqr_threshold IQR multiplier for outlier detection
     * @param min_samples Minimum samples needed for analysis
     * @param use_robust Whether to use robust statistics (median/IQR vs mean/std)
     */
    NumericAnomalyDetector(
        float std_threshold = 3.0f,
        float iqr_threshold = 1.5f,
        size_t min_samples = 10,
        bool use_robust = true
    );

    /**
     * @brief Detect anomalies in numeric fields
     * @param values Multi-dimensional array of values (each row is a sample, each column is a field)
     * @param field_names Names of the numeric fields (optional)
     * @param group_ids Optional grouping information for samples
     * @return Pair of (combined anomaly flags, per-field anomaly flags)
     */
    std::pair<std::vector<bool>, std::unordered_map<std::string, std::vector<bool>>> 
    detectAnomalies(
        const std::vector<std::vector<float>>& values,
        const std::vector<std::string>& field_names = {},
        const std::vector<int>& group_ids = {}
    );

    /**
     * @brief Extract numeric values from string data
     * @param data Map of column names to string values
     * @param exclude_fields Fields to exclude from numeric extraction
     * @return Vector of numeric values extracted
     */
    static std::vector<std::string> extractNumericFields(
        const std::vector<std::unordered_map<std::string, std::string>>& data,
        const std::vector<std::string>& exclude_fields = {"message", "component", "level", "timestamp"}
    );

    /**
     * @brief Detect anomalies in a single numeric field
     * @param values Vector of numeric values
     * @return Boolean vector marking anomalies
     */
    std::vector<bool> detectFieldAnomalies(const std::vector<float>& values);

private:
    float std_threshold_;
    float iqr_threshold_;
    size_t min_samples_;
    bool use_robust_;

    /**
     * @brief Calculate median of a vector
     * @param values Vector of values
     * @return Median value
     */
    static float median(std::vector<float> values);

    /**
     * @brief Calculate quantile of a vector
     * @param values Vector of values
     * @param q Quantile (0.0 to 1.0)
     * @return Quantile value
     */
    static float quantile(std::vector<float> values, float q);
};