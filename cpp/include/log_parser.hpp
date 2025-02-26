#pragma once

#include "template_matcher.hpp"
#include "prefix_tree.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <optional>
#include <list>

namespace sherlog {

/**
 * Represents an extracted log template with parameters.
 */
class LogTemplate {
public:
    LogTemplate(
        const std::string& template_str,
        const std::unordered_map<std::string, VariableType>& variable_types,
        const std::vector<std::string>& examples = {}
    );
    
    /**
     * Check if this template is similar enough to another template for merging.
     * @param other Template to compare with
     * @param similarity_threshold Threshold for considering templates similar
     * @return True if templates are similar enough for merging
     */
    bool matches(const LogTemplate& other, double similarity_threshold = 0.8) const;
    
    /**
     * Merge this template with another similar template.
     * @param other Template to merge with
     * @return New merged template
     */
    std::shared_ptr<LogTemplate> merge_with(const LogTemplate& other) const;
    
    // Getters
    const std::string& get_template() const { return template_; }
    const std::unordered_map<std::string, VariableType>& get_variable_types() const { return variable_types_; }
    const std::unordered_set<std::string>& get_syntax_templates() const { return syntax_templates_; }
    size_t get_frequency() const { return frequency_; }
    const std::vector<std::string>& get_examples() const { return examples_; }
    
    // Modifiers
    void add_syntax_template(const std::string& syntax);
    void increment_frequency();
    void add_example(const std::string& example, size_t max_examples = 5);

private:
    std::string template_;
    std::unordered_map<std::string, VariableType> variable_types_;
    std::unordered_set<std::string> syntax_templates_;
    size_t frequency_;
    std::vector<std::string> examples_;
    
    // Shared template matcher
    static TemplateMatcher matcher_;
};

/**
 * Represents a cluster of similar log messages.
 */
class LogCluster {
public:
    explicit LogCluster(std::shared_ptr<LogTemplate> template_ptr);
    
    // Getters
    std::shared_ptr<LogTemplate> get_template() const { return template_; }
    const std::unordered_set<int>& get_log_ids() const { return log_ids_; }
    const std::vector<std::string>& get_examples() const { return examples_; }
    size_t get_frequency() const { return frequency_; }
    
    // Modifiers
    void add_log_id(int log_id);
    void add_example(const std::string& example);
    void increment_frequency();
    void add_syntax_template(const std::vector<std::string>& tokens);
    void set_template(std::shared_ptr<LogTemplate> new_template);

private:
    std::shared_ptr<LogTemplate> template_;
    std::unordered_set<int> log_ids_;
    std::vector<std::string> examples_;
    size_t frequency_;
};

/**
 * Class for caching in the log parser.
 */
template<typename K, typename V>
class LRUCache {
public:
    explicit LRUCache(size_t max_size = 1000);
    
    std::optional<V> get(const K& key) const;
    void put(const K& key, const V& value);
    size_t size() const;
    void clear();

private:
    struct CacheEntry {
        V value;
        typename std::list<K>::iterator list_it;
    };
    
    std::unordered_map<K, CacheEntry> cache_;
    std::list<K> lru_list_;
    size_t max_size_;
    mutable std::mutex mutex_;
};

/**
 * Main implementation of the LogParser-LLM algorithm.
 */
class LogParserLLM {
public:
    LogParserLLM(
        double similarity_threshold = 0.8,
        size_t lru_cache_size = 1000
    );
    
    /**
     * Parse a single log message.
     * @param content Log message content
     * @param log_id Unique identifier for the log
     * @return Pair of (template, parameters)
     */
    std::pair<std::string, std::unordered_map<std::string, std::string>> 
    parse_log(const std::string& content, int log_id);
    
    /**
     * Parse a batch of logs in parallel.
     * @param logs Vector of pairs (log_message, log_id)
     * @param batch_size Size of processing batches
     * @param max_workers Number of worker threads
     * @return Vector of pairs (template, parameters) for each log
     */
    std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> 
    parse_logs_batch(
        const std::vector<std::pair<std::string, int>>& logs,
        size_t batch_size = 32,
        size_t max_workers = 4
    );
    
    /**
     * Get parser statistics for monitoring.
     * @return Map of statistics
     */
    std::unordered_map<std::string, size_t> get_statistics() const;
    
    // External template extraction interface (should be implemented by users)
    using TemplateExtractor = std::function<
        std::pair<std::string, std::unordered_map<std::string, VariableType>>(const std::string&)
    >;
    
    void set_template_extractor(TemplateExtractor extractor);

private:
    /**
     * Tokenize a log message into words.
     * @param log_message Log message to tokenize
     * @return Vector of tokens
     */
    std::vector<std::string> tokenize(const std::string& log_message) const;
    
    /**
     * Calculate similarity between two token sequences.
     * @param tokens1 First token sequence
     * @param tokens2 Second token sequence
     * @return Similarity score between 0 and 1
     */
    double calculate_token_similarity(
        const std::vector<std::string>& tokens1,
        const std::vector<std::string>& tokens2
    ) const;
    
    /**
     * Extract parameters from a log message using its template.
     * @param log_message Log message
     * @param template Template string
     * @param variable_types Map of variable types
     * @return Map of parameter names to values
     */
    std::unordered_map<std::string, std::string> extract_parameters(
        const std::string& log_message,
        const std::string& template_str,
        const std::unordered_map<std::string, VariableType>& variable_types
    ) const;
    
    /**
     * Extract template using the configured extractor or fallback.
     * @param log_message Log message to extract template from
     * @return Pair of (template, variable_types)
     */
    std::pair<std::string, std::unordered_map<std::string, VariableType>> 
    extract_template(const std::string& log_message) const;
    
    /**
     * Handle case where a strict match is found in prefix tree.
     * @param cluster Matching cluster
     * @param log_message Original log message
     * @param log_id Log identifier
     * @return Pair of (template, parameters)
     */
    std::pair<std::string, std::unordered_map<std::string, std::string>> 
    handle_strict_match(
        std::shared_ptr<LogCluster> cluster,
        const std::string& log_message,
        int log_id
    );
    
    /**
     * Handle case where a similar template is found in pool.
     * @param matched_template Existing template
     * @param new_template New template
     * @param log_tokens Tokenized log message
     * @param log_message Original log message
     * @param log_id Log identifier
     * @return Pair of (template, parameters)
     */
    std::pair<std::string, std::unordered_map<std::string, std::string>> 
    handle_template_match(
        std::shared_ptr<LogTemplate> matched_template,
        std::shared_ptr<LogTemplate> new_template,
        const std::vector<std::string>& log_tokens,
        const std::string& log_message,
        int log_id
    );
    
    /**
     * Handle case where no similar template exists.
     * @param new_template New template
     * @param log_tokens Tokenized log message
     * @param log_message Original log message
     * @param log_id Log identifier
     * @return Pair of (template, parameters)
     */
    std::pair<std::string, std::unordered_map<std::string, std::string>> 
    handle_new_template(
        std::shared_ptr<LogTemplate> new_template,
        const std::vector<std::string>& log_tokens,
        const std::string& log_message,
        int log_id
    );
    
    /**
     * Generate a cache key from tokens.
     * @param tokens Token sequence
     * @return Cache key string
     */
    std::string get_cache_key(const std::vector<std::string>& tokens) const;
    
    /**
     * Find a similar entry in the LLM cache.
     * @param tokens Token sequence
     * @return Optional pair of (template, variable_types)
     */
    std::optional<std::pair<std::string, std::unordered_map<std::string, VariableType>>> 
    find_similar_cache_entry(const std::vector<std::string>& tokens) const;
    
    /**
     * Update the cache with new result.
     * @param tokens Token sequence
     * @param template Extracted template
     * @param variable_types Map of variable types
     */
    void update_cache(
        const std::vector<std::string>& tokens,
        const std::string& template_str,
        const std::unordered_map<std::string, VariableType>& variable_types
    );

    // Member variables
    PrefixTree trie_;
    double similarity_threshold_;
    std::unordered_map<std::string, std::shared_ptr<LogTemplate>> template_pool_;
    mutable std::shared_mutex template_pool_mutex_;
    
    // LLM cache
    using CacheValue = std::pair<std::string, std::unordered_map<std::string, VariableType>>;
    LRUCache<std::string, CacheValue> llm_cache_;
    
    // Template extraction function (to be set by user)
    TemplateExtractor template_extractor_;
};

} // namespace sherlog