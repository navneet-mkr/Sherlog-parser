#include "../include/numeric_analysis.hpp"
#include <array>
#include <cassert>
#include <limits>
#include <functional>
#include <numeric>
#include <unordered_set>

// SIMD detection and includes
#if defined(__x86_64__) || defined(_M_X64)
    #define USE_X86_SIMD
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    // Don't redefine since it's already defined in CMake
    #ifndef USE_ARM_SIMD
    #define USE_ARM_SIMD
    #endif
    #include <arm_neon.h>
#endif

// Constructor
NumericAnomalyDetector::NumericAnomalyDetector(
    float std_threshold,
    float iqr_threshold,
    size_t min_samples,
    bool use_robust
) : 
    std_threshold_(std_threshold),
    iqr_threshold_(iqr_threshold),
    min_samples_(min_samples),
    use_robust_(use_robust) {
}

// Calculate median of a vector
float NumericAnomalyDetector::median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0f;
    }
    
    std::sort(values.begin(), values.end());
    
    size_t size = values.size();
    if (size % 2 == 0) {
        return (values[size/2 - 1] + values[size/2]) / 2.0f;
    } else {
        return values[size/2];
    }
}

// Calculate quantile of a vector
float NumericAnomalyDetector::quantile(std::vector<float> values, float q) {
    if (values.empty()) {
        return 0.0f;
    }
    
    std::sort(values.begin(), values.end());
    
    float pos = q * (static_cast<float>(values.size()) - 1);
    size_t idx = static_cast<size_t>(pos);
    float frac = pos - static_cast<float>(idx);
    
    if (idx + 1 < values.size()) {
        return values[idx] * (1.0f - frac) + values[idx + 1] * frac;
    } else {
        return values[idx];
    }
}

// Detect anomalies in a single numeric field
std::vector<bool> NumericAnomalyDetector::detectFieldAnomalies(const std::vector<float>& values) {
    std::vector<bool> anomalies(values.size(), false);
    
    if (values.size() < 2) {
        return anomalies; // Not enough data
    }
    
    if (use_robust_) {
        // Use robust statistics (median/IQR)
        float q1 = quantile(values, 0.25f);
        float q3 = quantile(values, 0.75f);
        float iqr = q3 - q1;
        
        if (iqr <= std::numeric_limits<float>::epsilon()) {
            return anomalies; // IQR is zero, all points are the same
        }
        
        float lower = q1 - (iqr_threshold_ * iqr);
        float upper = q3 + (iqr_threshold_ * iqr);
        
        // Compute outliers - manually unrolled for performance
        for (size_t i = 0; i < values.size(); i += 4) {
            for (size_t j = 0; j < 4 && i + j < values.size(); ++j) {
                float val = values[i + j];
                anomalies[i + j] = (val < lower) || (val > upper);
            }
        }
    } else {
        // Use classical statistics (mean/std)
        
        // Calculate mean
        float sum = 0.0f;
        
#if defined(USE_X86_SIMD)
        // Use AVX for mean calculation if available
        if (values.size() >= 8) {
            __m256 sum_avx = _mm256_setzero_ps();
            size_t vec_end = values.size() - (values.size() % 8);
            
            for (size_t i = 0; i < vec_end; i += 8) {
                __m256 vec = _mm256_loadu_ps(&values[i]);
                sum_avx = _mm256_add_ps(sum_avx, vec);
            }
            
            // Horizontal sum
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_avx);
            sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                  sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
            
            // Handle remaining elements
            for (size_t i = vec_end; i < values.size(); ++i) {
                sum += values[i];
            }
        } else {
            // Fallback to scalar calculation for small vectors
            for (float val : values) {
                sum += val;
            }
        }
#elif defined(USE_ARM_SIMD)
        // Use ARM NEON for mean calculation if available
        if (values.size() >= 4) {
            float32x4_t sum_neon = vdupq_n_f32(0.0f);
            size_t vec_end = values.size() - (values.size() % 4);
            
            for (size_t i = 0; i < vec_end; i += 4) {
                float32x4_t vec = vld1q_f32(&values[i]);
                sum_neon = vaddq_f32(sum_neon, vec);
            }
            
            // Horizontal sum
            float sum_array[4];
            vst1q_f32(sum_array, sum_neon);
            sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
            
            // Handle remaining elements
            for (size_t i = vec_end; i < values.size(); ++i) {
                sum += values[i];
            }
        } else {
            // Fallback to scalar calculation for small vectors
            for (float val : values) {
                sum += val;
            }
        }
#else
        // Scalar implementation
        for (float val : values) {
            sum += val;
        }
#endif
        
        float mean = sum / static_cast<float>(values.size());
        
        // Calculate standard deviation
        float sum_sq_diff = 0.0f;
        
#if defined(USE_X86_SIMD)
        // Use AVX for variance calculation if available
        if (values.size() >= 8) {
            __m256 mean_avx = _mm256_set1_ps(mean);
            __m256 sum_sq_avx = _mm256_setzero_ps();
            size_t vec_end = values.size() - (values.size() % 8);
            
            for (size_t i = 0; i < vec_end; i += 8) {
                __m256 vec = _mm256_loadu_ps(&values[i]);
                __m256 diff = _mm256_sub_ps(vec, mean_avx);
                __m256 sq_diff = _mm256_mul_ps(diff, diff);
                sum_sq_avx = _mm256_add_ps(sum_sq_avx, sq_diff);
            }
            
            // Horizontal sum
            float sum_sq_array[8];
            _mm256_storeu_ps(sum_sq_array, sum_sq_avx);
            sum_sq_diff = sum_sq_array[0] + sum_sq_array[1] + sum_sq_array[2] + sum_sq_array[3] +
                          sum_sq_array[4] + sum_sq_array[5] + sum_sq_array[6] + sum_sq_array[7];
            
            // Handle remaining elements
            for (size_t i = vec_end; i < values.size(); ++i) {
                float diff = values[i] - mean;
                sum_sq_diff += diff * diff;
            }
        } else {
            // Fallback to scalar calculation for small vectors
            for (float val : values) {
                float diff = val - mean;
                sum_sq_diff += diff * diff;
            }
        }
#elif defined(USE_ARM_SIMD)
        // Use ARM NEON for variance calculation if available
        if (values.size() >= 4) {
            float32x4_t mean_neon = vdupq_n_f32(mean);
            float32x4_t sum_sq_neon = vdupq_n_f32(0.0f);
            size_t vec_end = values.size() - (values.size() % 4);
            
            for (size_t i = 0; i < vec_end; i += 4) {
                float32x4_t vec = vld1q_f32(&values[i]);
                float32x4_t diff = vsubq_f32(vec, mean_neon);
                float32x4_t sq_diff = vmulq_f32(diff, diff);
                sum_sq_neon = vaddq_f32(sum_sq_neon, sq_diff);
            }
            
            // Horizontal sum
            float sum_sq_array[4];
            vst1q_f32(sum_sq_array, sum_sq_neon);
            sum_sq_diff = sum_sq_array[0] + sum_sq_array[1] + sum_sq_array[2] + sum_sq_array[3];
            
            // Handle remaining elements
            for (size_t i = vec_end; i < values.size(); ++i) {
                float diff = values[i] - mean;
                sum_sq_diff += diff * diff;
            }
        } else {
            // Fallback to scalar calculation for small vectors
            for (float val : values) {
                float diff = val - mean;
                sum_sq_diff += diff * diff;
            }
        }
#else
        // Scalar implementation
        for (float val : values) {
            float diff = val - mean;
            sum_sq_diff += diff * diff;
        }
#endif
        
        float std_dev = std::sqrt(sum_sq_diff / static_cast<float>(values.size()));
        
        if (std_dev <= std::numeric_limits<float>::epsilon()) {
            return anomalies; // All points are the same, no anomalies
        }
        
        // Calculate z-scores and flag anomalies
#if defined(USE_X86_SIMD)
        // Use AVX for z-score calculation if available
        if (values.size() >= 8) {
            __m256 mean_avx = _mm256_set1_ps(mean);
            __m256 std_avx = _mm256_set1_ps(std_dev);
            __m256 threshold_avx = _mm256_set1_ps(std_threshold_);
            size_t vec_end = values.size() - (values.size() % 8);
            
            for (size_t i = 0; i < vec_end; i += 8) {
                __m256 vec = _mm256_loadu_ps(&values[i]);
                __m256 diff = _mm256_sub_ps(vec, mean_avx);
                __m256 abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff); // Absolute value
                __m256 z_score = _mm256_div_ps(abs_diff, std_avx);
                __m256 is_anomaly = _mm256_cmp_ps(z_score, threshold_avx, _CMP_GT_OQ);
                
                // Extract results
                int mask = _mm256_movemask_ps(is_anomaly);
                for (int j = 0; j < 8; ++j) {
                    if ((mask >> j) & 1) {
                        anomalies[i + j] = true;
                    }
                }
            }
            
            // Handle remaining elements
            for (size_t i = vec_end; i < values.size(); ++i) {
                float z_score = std::abs(values[i] - mean) / std_dev;
                anomalies[i] = (z_score > std_threshold_);
            }
        } else {
            // Fallback to scalar calculation for small vectors
            for (size_t i = 0; i < values.size(); ++i) {
                float z_score = std::abs(values[i] - mean) / std_dev;
                anomalies[i] = (z_score > std_threshold_);
            }
        }
#elif defined(USE_ARM_SIMD)
        // Use ARM NEON for z-score calculation if available
        if (values.size() >= 4) {
            float32x4_t mean_neon = vdupq_n_f32(mean);
            float32x4_t std_neon = vdupq_n_f32(std_dev);
            float32x4_t threshold_neon = vdupq_n_f32(std_threshold_);
            size_t vec_end = values.size() - (values.size() % 4);
            
            for (size_t i = 0; i < vec_end; i += 4) {
                float32x4_t vec = vld1q_f32(&values[i]);
                float32x4_t diff = vsubq_f32(vec, mean_neon);
                float32x4_t abs_diff = vabsq_f32(diff); // Absolute value
                float32x4_t z_score = vdivq_f32(abs_diff, std_neon);
                uint32x4_t is_anomaly = vcgtq_f32(z_score, threshold_neon);
                
                // Extract results
                uint32_t mask[4];
                vst1q_u32(mask, is_anomaly);
                for (int j = 0; j < 4; ++j) {
                    anomalies[i + j] = (mask[j] != 0);
                }
            }
            
            // Handle remaining elements
            for (size_t i = vec_end; i < values.size(); ++i) {
                float z_score = std::abs(values[i] - mean) / std_dev;
                anomalies[i] = (z_score > std_threshold_);
            }
        } else {
            // Fallback to scalar calculation for small vectors
            for (size_t i = 0; i < values.size(); ++i) {
                float z_score = std::abs(values[i] - mean) / std_dev;
                anomalies[i] = (z_score > std_threshold_);
            }
        }
#else
        // Scalar implementation with loop unrolling
        for (size_t i = 0; i < values.size(); i += 4) {
            for (size_t j = 0; j < 4 && i + j < values.size(); ++j) {
                float z_score = std::abs(values[i + j] - mean) / std_dev;
                anomalies[i + j] = (z_score > std_threshold_);
            }
        }
#endif
    }
    
    return anomalies;
}

// Detect anomalies in numeric fields
std::pair<std::vector<bool>, std::unordered_map<std::string, std::vector<bool>>> 
NumericAnomalyDetector::detectAnomalies(
    const std::vector<std::vector<float>>& values,
    const std::vector<std::string>& field_names,
    const std::vector<int>& group_ids
) {
    // Initialize result containers
    size_t num_samples = values.size();
    size_t num_fields = num_samples > 0 ? values[0].size() : 0;
    
    std::vector<bool> combined_anomalies(num_samples, false);
    std::unordered_map<std::string, std::vector<bool>> field_anomalies;
    
    // Early exit if no data
    if (num_samples == 0 || num_fields == 0) {
        return {combined_anomalies, field_anomalies};
    }
    
    // Initialize field names if not provided
    std::vector<std::string> field_names_internal;
    if (field_names.empty()) {
        for (size_t i = 0; i < num_fields; ++i) {
            field_names_internal.push_back("field_" + std::to_string(i));
        }
    } else {
        field_names_internal = field_names;
    }
    
    // Ensure we have enough field names
    if (field_names_internal.size() < num_fields) {
        for (size_t i = field_names_internal.size(); i < num_fields; ++i) {
            field_names_internal.push_back("field_" + std::to_string(i));
        }
    }
    
    // Process each field
    for (size_t field_idx = 0; field_idx < num_fields; ++field_idx) {
        std::string field_name = field_names_internal[field_idx];
        field_anomalies[field_name] = std::vector<bool>(num_samples, false);
        
        // Extract values for this field
        std::vector<float> field_values(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            field_values[i] = values[i][field_idx];
        }
        
        if (group_ids.empty()) {
            // Process all values together
            if (field_values.size() >= min_samples_) {
                std::vector<bool> anomaly_mask = detectFieldAnomalies(field_values);
                field_anomalies[field_name] = anomaly_mask;
                
                // Update combined anomalies
                for (size_t i = 0; i < num_samples; ++i) {
                    // Fix for |= operator in std::vector<bool>
                    combined_anomalies[i] = combined_anomalies[i] || anomaly_mask[i];
                }
            }
        } else {
            // Process by groups
            std::unordered_map<int, std::vector<size_t>> group_indices;
            std::unordered_map<int, std::vector<float>> group_values;
            
            // Group values by cluster
            for (size_t i = 0; i < num_samples; ++i) {
                int group = group_ids[i];
                if (group == -1) continue; // Skip noise points in clustering
                
                group_indices[group].push_back(i);
                group_values[group].push_back(field_values[i]);
            }
            
            // Process each group separately
            for (const auto& [group, indices] : group_indices) {
                if (indices.size() >= min_samples_) {
                    std::vector<bool> group_anomalies = detectFieldAnomalies(group_values[group]);
                    
                    // Map anomalies back to original indices
                    for (size_t i = 0; i < indices.size(); ++i) {
                        size_t orig_idx = indices[i];
                        field_anomalies[field_name][orig_idx] = group_anomalies[i];
                        // Fix for |= operator in std::vector<bool>
                        combined_anomalies[orig_idx] = combined_anomalies[orig_idx] || group_anomalies[i];
                    }
                }
            }
        }
    }
    
    return {combined_anomalies, field_anomalies};
}

// Extract numeric fields from string data
std::vector<std::string> NumericAnomalyDetector::extractNumericFields(
    const std::vector<std::unordered_map<std::string, std::string>>& data,
    const std::vector<std::string>& exclude_fields
) {
    if (data.empty()) {
        return {};
    }
    
    // Build set of excluded fields for faster lookup
    std::unordered_set<std::string> excluded_set(exclude_fields.begin(), exclude_fields.end());
    
    // First, collect all potential field names from the data
    std::unordered_set<std::string> all_fields;
    for (const auto& entry : data) {
        for (const auto& [field, _] : entry) {
            if (excluded_set.find(field) == excluded_set.end()) {
                all_fields.insert(field);
            }
        }
    }
    
    // Sample entries to identify numeric fields
    size_t sample_size = std::min(data.size(), static_cast<size_t>(100));
    
    std::unordered_map<std::string, size_t> numeric_counts;
    for (size_t i = 0; i < sample_size; ++i) {
        const auto& entry = data[i];
        
        for (const std::string& field : all_fields) {
            auto it = entry.find(field);
            if (it != entry.end()) {
                const std::string& value = it->second;
                
                // Try to convert to float
                try {
                    std::stof(value);
                    numeric_counts[field]++;
                } catch (...) {
                    // Not a number, ignore
                }
            }
        }
    }
    
    // Consider fields numeric if they are numeric in at least 80% of samples
    float threshold = static_cast<float>(sample_size) * 0.8f;
    
    std::vector<std::string> numeric_fields;
    for (const auto& [field, count] : numeric_counts) {
        if (static_cast<float>(count) >= threshold) {
            numeric_fields.push_back(field);
        }
    }
    
    return numeric_fields;
}