#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <optional>
#include <functional>

// Architecture detection
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM)
    #define SHERLOG_ARM 1
    // ARM NEON support
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #define SHERLOG_NEON 1
        #include <arm_neon.h>
    #endif
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define SHERLOG_X86 1
    // x86 SIMD support
    #if defined(__AVX2__)
        #define SHERLOG_SIMD_AVX2 1
        #include <immintrin.h>
    #elif defined(__SSE4_2__)
        #define SHERLOG_SIMD_SSE42 1
        #include <nmmintrin.h>
    #elif defined(__SSE2__)
        #define SHERLOG_SIMD_SSE2 1
        #include <emmintrin.h>
    #endif
#endif

namespace sherlog {

/**
 * Enumeration for different match types.
 */
enum class MatchType {
    EXACT,
    SIMILAR,
    VARIABLE_ONLY,
    NO_MATCH
};

/**
 * Enumeration for variable types in log templates.
 */
enum class VariableType {
    OID,           // Object IDs (session IDs, user IDs)
    LOI,           // Location Info (paths, URIs, IPs)
    OBN,           // Object Names (domains, tasks, jobs)
    TID,           // Type Indicators
    SID,           // Switch Indicators (numerical)
    TDA,           // Time/Date Actions
    CRS,           // Computing Resources
    OBA,           // Object Amounts
    STC,           // Status Codes
    OTHER_PARAMS   // Other Parameters
};

/**
 * Convert variable type to string.
 */
std::string to_string(VariableType type);

/**
 * Parse variable type from string.
 */
std::optional<VariableType> parse_variable_type(const std::string& str);

/**
 * Result of template matching.
 */
struct MatchResult {
    MatchType match_type;
    double similarity_score;
    std::vector<size_t> matched_positions;
    std::vector<size_t> variable_positions;
};

/**
 * Class for handling template matching and similarity comparison.
 */
class TemplateMatcher {
public:
    /**
     * Initialize template matcher.
     * @param similarity_threshold Threshold for considering templates similar
     * @param max_examples Maximum number of examples to keep per template
     */
    explicit TemplateMatcher(double similarity_threshold = 0.8, size_t max_examples = 5);

    /**
     * Match two templates and determine their similarity.
     * @param template1 First template
     * @param template2 Second template
     * @return MatchResult with match type and similarity details
     */
    MatchResult match(const std::string& template1, const std::string& template2) const;
    
    /**
     * Merge two similar templates if possible.
     * @param template1 First template
     * @param template2 Second template
     * @return Merged template if templates are similar enough, std::nullopt otherwise
     */
    std::optional<std::string> merge_templates(const std::string& template1, 
                                             const std::string& template2) const;

    // Getters and setters
    double get_similarity_threshold() const { return similarity_threshold_; }
    void set_similarity_threshold(double threshold) { similarity_threshold_ = threshold; }
    
    size_t get_max_examples() const { return max_examples_; }
    void set_max_examples(size_t max) { max_examples_ = max; }

private:
    /**
     * Tokenize template into words (with optional caching).
     * @param template Template string to tokenize
     * @return Vector of tokens
     */
    static std::vector<std::string> tokenize(const std::string& template_str);
    
    /**
     * Get positions of variable tokens in template.
     * @param tokens List of template tokens
     * @return Set of variable token positions
     */
    static std::unordered_set<size_t> get_variable_positions(const std::vector<std::string>& tokens);
    
    /**
     * Calculate similarity between token sequences.
     * @param tokens1 First token sequence
     * @param tokens2 Second token sequence
     * @param var_positions1 Variable positions in first sequence
     * @param var_positions2 Variable positions in second sequence
     * @return Similarity score between 0 and 1
     */
    double calculate_token_similarity(
        const std::vector<std::string>& tokens1,
        const std::vector<std::string>& tokens2,
        const std::unordered_set<size_t>& var_positions1,
        const std::unordered_set<size_t>& var_positions2) const;

    double similarity_threshold_;
    size_t max_examples_;
    
    // Consider adding token caching for performance if needed
};

} // namespace sherlog