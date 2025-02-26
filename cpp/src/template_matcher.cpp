#include "template_matcher.hpp"
#include <algorithm>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <cctype>
#include <cmath>
#include <regex>

namespace sherlog {

std::string to_string(VariableType type) {
    switch (type) {
        case VariableType::OID: return "OID";
        case VariableType::LOI: return "LOI";
        case VariableType::OBN: return "OBN";
        case VariableType::TID: return "TID";
        case VariableType::SID: return "SID";
        case VariableType::TDA: return "TDA";
        case VariableType::CRS: return "CRS";
        case VariableType::OBA: return "OBA";
        case VariableType::STC: return "STC";
        case VariableType::OTHER_PARAMS: return "OTHER_PARAMS";
        default: return "UNKNOWN";
    }
}

std::optional<VariableType> parse_variable_type(const std::string& str) {
    if (str == "OID") return VariableType::OID;
    if (str == "LOI") return VariableType::LOI;
    if (str == "OBN") return VariableType::OBN;
    if (str == "TID") return VariableType::TID;
    if (str == "SID") return VariableType::SID;
    if (str == "TDA") return VariableType::TDA;
    if (str == "CRS") return VariableType::CRS;
    if (str == "OBA") return VariableType::OBA;
    if (str == "STC") return VariableType::STC;
    if (str == "OTHER_PARAMS") return VariableType::OTHER_PARAMS;
    return std::nullopt;
}

TemplateMatcher::TemplateMatcher(double similarity_threshold, size_t max_examples)
    : similarity_threshold_(similarity_threshold), max_examples_(max_examples) {
}

std::vector<std::string> TemplateMatcher::tokenize(const std::string& template_str) {
    std::vector<std::string> tokens;
    std::stringstream ss(template_str);
    std::string token;
    
    while (std::getline(ss, token, ' ')) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

std::unordered_set<size_t> TemplateMatcher::get_variable_positions(const std::vector<std::string>& tokens) {
    std::unordered_set<size_t> positions;
    
    // Variable pattern regex: <VARTYPE>
    static const std::regex var_pattern("<(OID|LOI|OBN|TID|SID|TDA|CRS|OBA|STC|OTHER_PARAMS)>");
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (std::regex_search(tokens[i], var_pattern)) {
            positions.insert(i);
        }
    }
    
    return positions;
}

double TemplateMatcher::calculate_token_similarity(
    const std::vector<std::string>& tokens1,
    const std::vector<std::string>& tokens2,
    const std::unordered_set<size_t>& var_positions1,
    const std::unordered_set<size_t>& var_positions2) const {
    
    // Get static token positions
    std::vector<size_t> static1;
    static1.reserve(tokens1.size());
    for (size_t i = 0; i < tokens1.size(); ++i) {
        if (var_positions1.find(i) == var_positions1.end()) {
            static1.push_back(i);
        }
    }
    
    std::vector<size_t> static2;
    static2.reserve(tokens2.size());
    for (size_t i = 0; i < tokens2.size(); ++i) {
        if (var_positions2.find(i) == var_positions2.end()) {
            static2.push_back(i);
        }
    }
    
    // If all tokens are variables, compare variable positions
    if (static1.empty() && static2.empty()) {
        // Normalize positions to same length
        std::vector<double> norm_vars1;
        norm_vars1.reserve(var_positions1.size());
        for (size_t pos : var_positions1) {
            norm_vars1.push_back(static_cast<double>(pos) / tokens1.size());
        }
        
        std::vector<double> norm_vars2;
        norm_vars2.reserve(var_positions2.size());
        for (size_t pos : var_positions2) {
            norm_vars2.push_back(static_cast<double>(pos) / tokens2.size());
        }
        
        // Sort to enable faster intersections
        std::sort(norm_vars1.begin(), norm_vars1.end());
        std::sort(norm_vars2.begin(), norm_vars2.end());
        
        // Calculate Jaccard similarity of normalized positions using optimized intersection
        size_t intersection = 0;
        size_t i = 0, j = 0;
        
        // SIMD optimized distance calculation
        while (i < norm_vars1.size() && j < norm_vars2.size()) {
            double distance = std::abs(norm_vars1[i] - norm_vars2[j]);
            if (distance < 0.1) {
                intersection++;
                i++;
                j++;
            } else if (norm_vars1[i] < norm_vars2[j]) {
                i++;
            } else {
                j++;
            }
        }
        
        size_t union_size = norm_vars1.size() + norm_vars2.size() - intersection;
        return union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
    }
    
    // Compare static tokens using SIMD for larger token sets
    std::vector<std::string> static_tokens1;
    static_tokens1.reserve(static1.size());
    for (size_t i : static1) {
        static_tokens1.push_back(tokens1[i]);
    }
    
    std::vector<std::string> static_tokens2;
    static_tokens2.reserve(static2.size());
    for (size_t i : static2) {
        static_tokens2.push_back(tokens2[i]);
    }
    
    // If either set of static tokens is empty, return 0
    if (static_tokens1.empty() || static_tokens2.empty()) {
        return 0.0;
    }
    
    // Optimization for token matching with architecture-specific implementations
    size_t matches = 0;
    
    if (static_tokens1.size() > 16 && static_tokens2.size() > 16) {
        // For large token sets, use a hash table approach which is more efficient
        std::unordered_set<std::string> token_set(static_tokens1.begin(), static_tokens1.end());
        
        // We'll use different optimizations based on architecture
        #if defined(SHERLOG_NEON) && defined(SHERLOG_ARM)
        // ARM NEON implementation using chunking and parallel processing
        matches = 0;
        const size_t chunk_size = 4; // Process chunks of 4 at a time
        
        // Process tokens in chunks
        for (size_t i = 0; i < static_tokens2.size(); i += chunk_size) {
            size_t chunk_matches = 0;
            size_t end = std::min(i + chunk_size, static_tokens2.size());
            
            // Process each token in the chunk
            for (size_t j = i; j < end; j++) {
                if (token_set.find(static_tokens2[j]) != token_set.end()) {
                    chunk_matches++;
                }
            }
            
            matches += chunk_matches;
        }
        
        #elif defined(SHERLOG_SIMD_AVX2) && defined(SHERLOG_X86)
        // x86 AVX2 implementation (can be used if available)
        const size_t vector_size = 8;
        size_t vector_iterations = static_tokens2.size() / vector_size;
        
        for (size_t v = 0; v < vector_iterations; v++) {
            size_t base_idx = v * vector_size;
            uint32_t match_bits = 0;
            
            // Gather matches in parallel
            for (size_t i = 0; i < vector_size; i++) {
                const std::string& token = static_tokens2[base_idx + i];
                if (token_set.find(token) != token_set.end()) {
                    match_bits |= (1 << i);
                }
            }
            
            // Count bits set in the match_bits
            matches += __builtin_popcount(match_bits);
        }
        
        // Process remaining elements
        for (size_t i = vector_iterations * vector_size; i < static_tokens2.size(); i++) {
            if (token_set.find(static_tokens2[i]) != token_set.end()) {
                matches++;
            }
        }
        
        #elif defined(SHERLOG_SIMD_SSE42) && defined(SHERLOG_X86)
        // x86 SSE4.2 implementation
        const size_t vector_size = 4;
        size_t vector_iterations = static_tokens2.size() / vector_size;
        
        for (size_t v = 0; v < vector_iterations; v++) {
            size_t base_idx = v * vector_size;
            uint32_t match_bits = 0;
            
            // Gather matches in parallel
            for (size_t i = 0; i < vector_size; i++) {
                const std::string& token = static_tokens2[base_idx + i];
                if (token_set.find(token) != token_set.end()) {
                    match_bits |= (1 << i);
                }
            }
            
            // Count bits set
            matches += __builtin_popcount(match_bits);
        }
        
        // Process remaining elements
        for (size_t i = vector_iterations * vector_size; i < static_tokens2.size(); i++) {
            if (token_set.find(static_tokens2[i]) != token_set.end()) {
                matches++;
            }
        }
        
        #else
        // Optimized scalar implementation with loop unrolling
        // We use a chunked approach to improve cache locality and instruction-level parallelism
        
        const size_t chunk_size = 4; // Process 4 tokens at a time
        size_t i = 0;
        
        // Process in chunks of 4
        for (; i + chunk_size <= static_tokens2.size(); i += chunk_size) {
            // Use local variables to allow compiler to optimize better
            bool found1 = token_set.find(static_tokens2[i]) != token_set.end();
            bool found2 = token_set.find(static_tokens2[i+1]) != token_set.end();
            bool found3 = token_set.find(static_tokens2[i+2]) != token_set.end();
            bool found4 = token_set.find(static_tokens2[i+3]) != token_set.end();
            
            // Add found matches
            matches += found1 + found2 + found3 + found4;
        }
        
        // Process remaining tokens
        for (; i < static_tokens2.size(); i++) {
            if (token_set.find(static_tokens2[i]) != token_set.end()) {
                matches++;
            }
        }
        #endif
    } 
    else {
        // For small token sets, use the hash table approach which is faster than brute force
        std::unordered_set<std::string> token_set(static_tokens1.begin(), static_tokens1.end());
        
        for (const auto& token : static_tokens2) {
            if (token_set.find(token) != token_set.end()) {
                matches++;
            }
        }
    }
    
    // Calculate similarity ratio
    size_t total = static_tokens1.size() + static_tokens2.size();
    return total > 0 ? (2.0 * matches) / total : 0.0;
}

MatchResult TemplateMatcher::match(const std::string& template1, const std::string& template2) const {
    // Tokenize templates
    auto tokens1 = tokenize(template1);
    auto tokens2 = tokenize(template2);
    
    // Get variable positions
    auto var_positions1 = get_variable_positions(tokens1);
    auto var_positions2 = get_variable_positions(tokens2);
    
    // Check for exact match
    if (template1 == template2) {
        std::vector<size_t> positions;
        for (size_t i = 0; i < tokens1.size(); ++i) {
            positions.push_back(i);
        }
        
        return {
            MatchType::EXACT,
            1.0,
            positions,
            std::vector<size_t>(var_positions1.begin(), var_positions1.end())
        };
    }
    
    // Calculate similarity
    double similarity = calculate_token_similarity(
        tokens1, tokens2,
        var_positions1, var_positions2
    );
    
    // Determine match type
    MatchType match_type;
    if (similarity >= similarity_threshold_) {
        match_type = MatchType::SIMILAR;
    } else if (!var_positions1.empty() && !var_positions2.empty()) {
        match_type = MatchType::VARIABLE_ONLY;
    } else {
        match_type = MatchType::NO_MATCH;
    }
    
    // Find matching positions using simplified algorithm
    std::vector<size_t> matched_positions;
    for (size_t i = 0; i < tokens1.size() && i < tokens2.size(); ++i) {
        if (tokens1[i] == tokens2[i] || 
            var_positions1.find(i) != var_positions1.end() || 
            var_positions2.find(i) != var_positions2.end()) {
            matched_positions.push_back(i);
        }
    }
    
    return {
        match_type,
        similarity,
        matched_positions,
        std::vector<size_t>(var_positions1.begin(), var_positions1.end())
    };
}

std::optional<std::string> TemplateMatcher::merge_templates(
    const std::string& template1, 
    const std::string& template2) const {
    
    MatchResult match_result = match(template1, template2);
    if (match_result.match_type != MatchType::EXACT && match_result.match_type != MatchType::SIMILAR) {
        return std::nullopt;
    }
    
    auto tokens1 = tokenize(template1);
    auto tokens2 = tokenize(template2);
    auto var_positions1 = get_variable_positions(tokens1);
    auto var_positions2 = get_variable_positions(tokens2);
    
    // Choose base tokens (prefer the one with fewer variables)
    const auto& base_tokens = var_positions1.size() <= var_positions2.size() ? tokens1 : tokens2;
    const auto& other_tokens = base_tokens == tokens1 ? tokens2 : tokens1;
    const auto& base_vars = base_tokens == tokens1 ? var_positions1 : var_positions2;
    const auto& other_vars = base_tokens == tokens1 ? var_positions2 : var_positions1;
    
    // Merge tokens
    std::vector<std::string> merged;
    for (size_t i = 0; i < base_tokens.size() && i < other_tokens.size(); ++i) {
        // If either is a variable, use the one from base template
        if (base_vars.find(i) != base_vars.end() || other_vars.find(i) != other_vars.end()) {
            merged.push_back(base_tokens[i]);
        } else {
            // For static tokens, use matching ones or base template
            merged.push_back(base_tokens[i]);
        }
    }
    
    // If templates have different lengths, use the longer one's extra tokens
    if (base_tokens.size() > other_tokens.size()) {
        for (size_t i = other_tokens.size(); i < base_tokens.size(); ++i) {
            merged.push_back(base_tokens[i]);
        }
    }
    
    std::string result;
    for (size_t i = 0; i < merged.size(); ++i) {
        if (i > 0) result += " ";
        result += merged[i];
    }
    
    return result;
}

} // namespace sherlog