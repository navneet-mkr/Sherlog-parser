#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>
#include "../include/numeric_analysis.hpp"

// Test fixture for NumericAnomalyDetector
class NumericAnomalyDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a detector with default parameters
        detector = std::make_unique<NumericAnomalyDetector>();
        
        // Generate random data
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);
        
        values.resize(100);
        for (auto& val : values) {
            val = dist(rng);
        }
        
        // Add some anomalies
        values[10] = 500.0f;
        values[50] = 600.0f;
        values[90] = -200.0f;
    }
    
    std::unique_ptr<NumericAnomalyDetector> detector;
    std::vector<float> values;
};

// Test detection of anomalies in a single field using robust statistics
TEST_F(NumericAnomalyDetectorTest, DetectFieldAnomaliesRobust) {
    // Create detector with robust statistics
    NumericAnomalyDetector robust_detector(3.0f, 1.5f, 10, true);
    
    // Detect anomalies
    std::vector<bool> anomalies = robust_detector.detectFieldAnomalies(values);
    
    // Verify size
    ASSERT_EQ(anomalies.size(), values.size());
    
    // Verify anomalies are correctly identified
    EXPECT_TRUE(anomalies[10]);
    EXPECT_TRUE(anomalies[50]);
    EXPECT_TRUE(anomalies[90]);
    
    // Count total anomalies (should be exactly 3)
    size_t anomaly_count = std::count(anomalies.begin(), anomalies.end(), true);
    EXPECT_EQ(anomaly_count, 3);
}

// Test detection of anomalies in a single field using classic statistics
TEST_F(NumericAnomalyDetectorTest, DetectFieldAnomaliesClassic) {
    // Create detector with classic statistics
    NumericAnomalyDetector classic_detector(3.0f, 1.5f, 10, false);
    
    // Detect anomalies
    std::vector<bool> anomalies = classic_detector.detectFieldAnomalies(values);
    
    // Verify size
    ASSERT_EQ(anomalies.size(), values.size());
    
    // Verify anomalies are correctly identified
    EXPECT_TRUE(anomalies[10]);
    EXPECT_TRUE(anomalies[50]);
    EXPECT_TRUE(anomalies[90]);
    
    // Count total anomalies (should be exactly 3)
    size_t anomaly_count = std::count(anomalies.begin(), anomalies.end(), true);
    EXPECT_EQ(anomaly_count, 3);
}

// Test detection of anomalies across multiple fields
TEST_F(NumericAnomalyDetectorTest, DetectAnomaliesMultiField) {
    // Create multi-dimensional data
    std::vector<std::vector<float>> multi_values(100, std::vector<float>(2));
    
    // Copy values to first dimension
    for (size_t i = 0; i < values.size(); ++i) {
        multi_values[i][0] = values[i];
    }
    
    // Add different values to second dimension
    std::mt19937 rng(43); // Different seed
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    
    for (auto& row : multi_values) {
        row[1] = dist(rng);
    }
    
    // Add anomalies to second dimension
    multi_values[20][1] = 500.0f;
    multi_values[60][1] = -150.0f;
    
    // Field names
    std::vector<std::string> field_names = {"field1", "field2"};
    
    // Detect anomalies
    auto [combined, per_field] = detector->detectAnomalies(multi_values, field_names);
    
    // Verify sizes
    ASSERT_EQ(combined.size(), multi_values.size());
    ASSERT_EQ(per_field.size(), 2);
    ASSERT_EQ(per_field["field1"].size(), multi_values.size());
    ASSERT_EQ(per_field["field2"].size(), multi_values.size());
    
    // Verify anomalies in field1
    EXPECT_TRUE(per_field["field1"][10]);
    EXPECT_TRUE(per_field["field1"][50]);
    EXPECT_TRUE(per_field["field1"][90]);
    
    // Verify anomalies in field2
    EXPECT_TRUE(per_field["field2"][20]);
    EXPECT_TRUE(per_field["field2"][60]);
    
    // Verify combined anomalies
    EXPECT_TRUE(combined[10]);
    EXPECT_TRUE(combined[20]);
    EXPECT_TRUE(combined[50]);
    EXPECT_TRUE(combined[60]);
    EXPECT_TRUE(combined[90]);
    
    // Count total anomalies
    size_t field1_count = std::count(per_field["field1"].begin(), per_field["field1"].end(), true);
    size_t field2_count = std::count(per_field["field2"].begin(), per_field["field2"].end(), true);
    size_t combined_count = std::count(combined.begin(), combined.end(), true);
    
    EXPECT_EQ(field1_count, 3);
    EXPECT_EQ(field2_count, 2);
    EXPECT_EQ(combined_count, 5);
}

// Test detection of anomalies with grouping
TEST_F(NumericAnomalyDetectorTest, DetectAnomaliesWithGroups) {
    // Create multi-dimensional data
    std::vector<std::vector<float>> multi_values(100, std::vector<float>(1));
    
    // Copy values to first dimension
    for (size_t i = 0; i < values.size(); ++i) {
        multi_values[i][0] = values[i];
    }
    
    // Create group IDs (0, 1, or -1 for noise)
    std::vector<int> group_ids(100, -1);
    
    // Assign first 40 points to group 0
    for (size_t i = 0; i < 40; ++i) {
        group_ids[i] = 0;
    }
    
    // Assign next 40 points to group 1
    for (size_t i = 40; i < 80; ++i) {
        group_ids[i] = 1;
    }
    
    // Leave last 20 points as noise (-1)
    
    // Field names
    std::vector<std::string> field_names = {"field1"};
    
    // Detect anomalies
    auto [combined, per_field] = detector->detectAnomalies(multi_values, field_names, group_ids);
    
    // Verify sizes
    ASSERT_EQ(combined.size(), multi_values.size());
    ASSERT_EQ(per_field.size(), 1);
    ASSERT_EQ(per_field["field1"].size(), multi_values.size());
    
    // Verify anomalies - note that anomaly at index 90 is in noise group, so not detected
    EXPECT_TRUE(per_field["field1"][10]);
    EXPECT_TRUE(per_field["field1"][50]);
    EXPECT_FALSE(per_field["field1"][90]); // Noise point, not detected
    
    // Verify combined anomalies match field anomalies
    for (size_t i = 0; i < combined.size(); ++i) {
        EXPECT_EQ(combined[i], per_field["field1"][i]);
    }
}

// Test extract numeric fields
TEST_F(NumericAnomalyDetectorTest, ExtractNumericFields) {
    // Create test data
    std::vector<std::unordered_map<std::string, std::string>> data(10);
    
    // Add fields to data
    for (auto& entry : data) {
        entry["numeric1"] = "123.45";
        entry["numeric2"] = "678";
        entry["string1"] = "hello";
        entry["mixed"] = "world";
        entry["timestamp"] = "2023-01-01";
    }
    
    // Make some entries numeric in the mixed field
    data[0]["mixed"] = "42";
    data[1]["mixed"] = "43";
    data[2]["mixed"] = "44";
    
    // Extract numeric fields
    std::vector<std::string> numeric_fields = NumericAnomalyDetector::extractNumericFields(data);
    
    // Verify results
    EXPECT_EQ(numeric_fields.size(), 2);
    EXPECT_TRUE(std::find(numeric_fields.begin(), numeric_fields.end(), "numeric1") != numeric_fields.end());
    EXPECT_TRUE(std::find(numeric_fields.begin(), numeric_fields.end(), "numeric2") != numeric_fields.end());
    EXPECT_FALSE(std::find(numeric_fields.begin(), numeric_fields.end(), "string1") != numeric_fields.end());
    EXPECT_FALSE(std::find(numeric_fields.begin(), numeric_fields.end(), "mixed") != numeric_fields.end());
    EXPECT_FALSE(std::find(numeric_fields.begin(), numeric_fields.end(), "timestamp") != numeric_fields.end());
}

// Empty data tests
TEST_F(NumericAnomalyDetectorTest, EmptyData) {
    // Empty vector
    std::vector<float> empty_vec;
    std::vector<bool> anomalies = detector->detectFieldAnomalies(empty_vec);
    EXPECT_TRUE(anomalies.empty());
    
    // Empty matrix
    std::vector<std::vector<float>> empty_matrix;
    auto [combined, per_field] = detector->detectAnomalies(empty_matrix);
    EXPECT_TRUE(combined.empty());
    EXPECT_TRUE(per_field.empty());
    
    // Empty data for extract numeric fields
    std::vector<std::unordered_map<std::string, std::string>> empty_data;
    std::vector<std::string> numeric_fields = NumericAnomalyDetector::extractNumericFields(empty_data);
    EXPECT_TRUE(numeric_fields.empty());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}