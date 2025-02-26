#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include "template_matcher.hpp"
#include "log_parser.hpp"
#include "prefix_tree.hpp"
#include "numeric_analysis.hpp"

using namespace sherlog;

TEST_CASE("TemplateMatcher tests", "[template_matcher]") {
    TemplateMatcher matcher(0.8);
    
    SECTION("Exact matches") {
        std::string template1 = "User <OID> logged in from <LOI>";
        std::string template2 = "User <OID> logged in from <LOI>";
        
        auto result = matcher.match(template1, template2);
        
        REQUIRE(result.match_type == MatchType::EXACT);
        REQUIRE(result.similarity_score == 1.0);
        REQUIRE(result.matched_positions.size() == 5);
        REQUIRE(result.variable_positions.size() == 2);
    }
    
    SECTION("Similar matches") {
        std::string template1 = "User <OID> logged in from <LOI>";
        std::string template2 = "User <OID> authenticated from <LOI>";
        
        auto result = matcher.match(template1, template2);
        
        REQUIRE(result.match_type == MatchType::SIMILAR);
        REQUIRE(result.similarity_score >= 0.8);
    }
    
    SECTION("Non-matching templates") {
        std::string template1 = "User <OID> logged in from <LOI>";
        std::string template2 = "Failed to connect to database with error <STC>";
        
        auto result = matcher.match(template1, template2);
        
        REQUIRE(result.match_type == MatchType::NO_MATCH);
        REQUIRE(result.similarity_score < 0.8);
    }
    
    SECTION("Template merging") {
        std::string template1 = "User <OID> logged in from <LOI>";
        std::string template2 = "User <OID> logged in from <LOI> with role <SID>";
        
        auto merged = matcher.merge_templates(template1, template2);
        
        REQUIRE(merged.has_value());
        REQUIRE(*merged == "User <OID> logged in from <LOI> with role <SID>");
    }
}

TEST_CASE("LogParser tests", "[log_parser]") {
    LogParserLLM parser(0.8);
    
    SECTION("Simple log parsing") {
        std::string log_content = "User admin logged in from 192.168.1.1";
        int log_id = 1;
        
        auto [template_str, params] = parser.parse_log(log_content, log_id);
        
        REQUIRE(template_str.find("<OID>") != std::string::npos);
        REQUIRE(template_str.find("<LOI>") != std::string::npos);
        REQUIRE(params.size() == 2);
    }
    
    SECTION("Batch processing") {
        std::vector<std::pair<std::string, int>> logs = {
            {"User admin logged in from 192.168.1.1", 1},
            {"User root logged in from 10.0.0.1", 2},
            {"Failed to connect to server db01", 3}
        };
        
        auto results = parser.parse_logs_batch(logs);
        
        REQUIRE(results.size() == 3);
        REQUIRE(results[0].first != results[2].first); // Different templates
        REQUIRE(results[0].first == results[1].first); // Same template
    }
}

TEST_CASE("NumericAnalysis tests", "[numeric_analysis]") {
    NumericAnomalyDetector detector(3.0, 1.5, 10, true);
    
    SECTION("Statistical calculation") {
        std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        
        auto stats = detector.calculate_statistics(values);
        
        REQUIRE(stats.mean == Approx(5.5));
        REQUIRE(stats.median == Approx(5.5));
        REQUIRE(stats.q1 == Approx(3.25));
        REQUIRE(stats.q3 == Approx(7.75));
        REQUIRE(stats.iqr == Approx(4.5));
    }
    
    SECTION("Anomaly detection") {
        // Create test data with one anomaly
        std::vector<std::vector<double>> data(20, std::vector<double>(1, 0.0));
        for (size_t i = 0; i < 20; ++i) {
            data[i][0] = 10.0 + i % 5;
        }
        data[10][0] = 100.0; // One anomaly
        
        std::vector<std::string> field_names = {"value"};
        
        auto result = detector.detect_anomalies(data, field_names);
        
        REQUIRE(result.anomaly_mask.size() == 20);
        REQUIRE(result.anomaly_mask[10] == true);
        REQUIRE(std::count(result.anomaly_mask.begin(), result.anomaly_mask.end(), true) == 1);
    }
}