#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <chrono>
#include <memory>
#include <functional>

#include "log_prefilter.hpp"
#include "numeric_analysis.hpp"

/**
 * @struct EmbeddingVector
 * @brief Vector representation of a log message.
 */
struct EmbeddingVector {
    std::vector<float> data;
};

/**
 * @struct AnomalyDetectionResult
 * @brief Result of anomaly detection for a log entry.
 */
struct AnomalyDetectionResult {
    bool is_embedding_anomaly = false;
    bool is_numeric_anomaly = false;
    int cluster_label = -1;
    std::chrono::system_clock::time_point detection_time;
    std::optional<std::string> explanation;
    std::vector<std::pair<std::string, float>> numeric_deviations;
};

/**
 * @struct AnomalyDetectorConfig
 * @brief Configuration for the anomaly detector.
 */
struct AnomalyDetectorConfig {
    float eps = 0.3f;                       ///< DBSCAN epsilon parameter (distance threshold)
    size_t min_samples = 3;                 ///< DBSCAN min samples for core points
    float numeric_std_threshold = 2.5f;     ///< Standard deviations for numeric outliers
    bool explain_anomalies = true;          ///< Whether to generate explanations
    size_t max_explanations = 100;          ///< Maximum number of anomalies to explain
    bool enable_prefilter = false;          ///< Whether to enable pre-filtering
    PreFilterConfig prefilter_config;       ///< Pre-filter configuration
};

/**
 * @typedef ExplainerFunction
 * @brief Function type for generating explanations.
 */
using ExplainerFunction = std::function<std::string(
    const LogEntry&,
    const AnomalyDetectionResult&
)>;

/**
 * @class IncidentAnomalyDetector
 * @brief Detector for anomalous logs during incidents using local clustering.
 */
class IncidentAnomalyDetector {
public:
    /**
     * @brief Constructor
     * @param config Detector configuration
     * @param explainer_func Optional function for generating explanations
     */
    explicit IncidentAnomalyDetector(
        AnomalyDetectorConfig config = AnomalyDetectorConfig(),
        std::optional<ExplainerFunction> explainer_func = std::nullopt
    );
    
    /**
     * @brief Detect anomalies in logs
     * @param logs Vector of log entries
     * @param embeddings Vector of embedding vectors corresponding to logs
     * @param time_window_hours Time window in hours (for context in results)
     * @return Vector of pairs (log entry, detection result)
     */
    std::vector<std::pair<LogEntry, AnomalyDetectionResult>> detectAnomalies(
        const std::vector<LogEntry>& logs,
        const std::vector<EmbeddingVector>& embeddings,
        size_t time_window_hours = 4
    );
    
private:
    AnomalyDetectorConfig config_;
    std::optional<ExplainerFunction> explainer_func_;
    std::unique_ptr<LogPreFilter> prefilter_;
    std::unique_ptr<NumericAnomalyDetector> numeric_detector_;
    
    /**
     * @brief Perform DBSCAN clustering on embeddings
     * @param embeddings Vector of embedding vectors
     * @return Vector of cluster labels
     */
    std::vector<int> performClustering(const std::vector<EmbeddingVector>& embeddings);
    
    /**
     * @brief Compute cosine distance between two vectors
     * @param a First vector
     * @param b Second vector
     * @return Cosine distance (1 - cosine similarity)
     */
    static float cosineDistance(const std::vector<float>& a, const std::vector<float>& b);
    
    /**
     * @brief Extract numeric fields from log entries
     * @param logs Vector of log entries
     * @return Vector of numeric field names
     */
    static std::vector<std::string> extractNumericFields(const std::vector<LogEntry>& logs);
    
    /**
     * @brief Extract numeric values from log entries
     * @param logs Vector of log entries
     * @param field_names Names of fields to extract
     * @return 2D vector of numeric values
     */
    static std::vector<std::vector<float>> extractNumericValues(
        const std::vector<LogEntry>& logs,
        const std::vector<std::string>& field_names
    );
};