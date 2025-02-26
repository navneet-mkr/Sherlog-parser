#include "log_parser.hpp"
#include <sstream>
#include <regex>
#include <algorithm>
#include <thread>
#include <future>

namespace sherlog {

// Static initialization of shared matcher
TemplateMatcher LogTemplate::matcher_(0.8);

// LogTemplate implementation
LogTemplate::LogTemplate(
    const std::string& template_str,
    const std::unordered_map<std::string, VariableType>& variable_types,
    const std::vector<std::string>& examples)
    : template_(template_str),
      variable_types_(variable_types),
      frequency_(1) {
    
    // Initialize with provided examples
    for (const auto& example : examples) {
        add_example(example);
    }
}

bool LogTemplate::matches(const LogTemplate& other, double similarity_threshold) const {
    matcher_.set_similarity_threshold(similarity_threshold);
    auto result = matcher_.match(template_, other.template_);
    return result.match_type == MatchType::EXACT || result.match_type == MatchType::SIMILAR;
}

std::shared_ptr<LogTemplate> LogTemplate::merge_with(const LogTemplate& other) const {
    auto merged_template = matcher_.merge_templates(template_, other.template_);
    if (!merged_template) {
        return std::make_shared<LogTemplate>(*this);
    }
    
    // Merge variable types (keep most specific type for each variable)
    auto merged_var_types = variable_types_;
    for (const auto& [var, var_type] : other.variable_types_) {
        // Simple merge strategy: just add any new variables
        if (merged_var_types.find(var) == merged_var_types.end()) {
            merged_var_types[var] = var_type;
        }
    }
    
    // Create new template with merged data
    auto result = std::make_shared<LogTemplate>(
        *merged_template,
        merged_var_types
    );
    
    // Merge other data
    result->syntax_templates_ = syntax_templates_;
    for (const auto& syntax : other.syntax_templates_) {
        result->syntax_templates_.insert(syntax);
    }
    
    result->frequency_ = frequency_ + other.frequency_;
    
    // Merge examples (up to max_examples_)
    result->examples_ = examples_;
    for (const auto& example : other.examples_) {
        if (result->examples_.size() < matcher_.get_max_examples()) {
            result->examples_.push_back(example);
        } else {
            break;
        }
    }
    
    return result;
}

void LogTemplate::add_syntax_template(const std::string& syntax) {
    syntax_templates_.insert(syntax);
}

void LogTemplate::increment_frequency() {
    frequency_++;
}

void LogTemplate::add_example(const std::string& example, size_t max_examples) {
    if (examples_.size() < max_examples) {
        examples_.push_back(example);
    }
}

// LogCluster implementation
LogCluster::LogCluster(std::shared_ptr<LogTemplate> template_ptr)
    : template_(template_ptr), frequency_(1) {
}

void LogCluster::add_log_id(int log_id) {
    log_ids_.insert(log_id);
}

void LogCluster::add_example(const std::string& example) {
    if (examples_.size() < 5) {
        examples_.push_back(example);
    }
}

void LogCluster::increment_frequency() {
    frequency_++;
}

void LogCluster::add_syntax_template(const std::vector<std::string>& tokens) {
    std::string syntax;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) syntax += " ";
        syntax += tokens[i];
    }
    template_->add_syntax_template(syntax);
}

void LogCluster::set_template(std::shared_ptr<LogTemplate> new_template) {
    template_ = new_template;
}

// LRUCache implementation
template<typename K, typename V>
LRUCache<K, V>::LRUCache(size_t max_size)
    : max_size_(max_size) {
}

template<typename K, typename V>
std::optional<V> LRUCache<K, V>::get(const K& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return std::nullopt;
    }
    
    // For const method, we can't modify the LRU list, but we still need to return the value
    return it->second.value;
}

template<typename K, typename V>
void LRUCache<K, V>::put(const K& key, const V& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Update existing entry
        it->second.value = value;
        
        // Move to front of LRU list
        lru_list_.erase(it->second.list_it);
        lru_list_.push_front(key);
        it->second.list_it = lru_list_.begin();
    } else {
        // Add new entry
        lru_list_.push_front(key);
        cache_[key] = {value, lru_list_.begin()};
        
        // Evict if over capacity
        if (cache_.size() > max_size_) {
            auto last = lru_list_.back();
            cache_.erase(last);
            lru_list_.pop_back();
        }
    }
}

template<typename K, typename V>
size_t LRUCache<K, V>::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

template<typename K, typename V>
void LRUCache<K, V>::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_list_.clear();
}

// Explicit template instantiation
template class LRUCache<std::string, std::pair<std::string, std::unordered_map<std::string, VariableType>>>;

// LogParserLLM implementation
LogParserLLM::LogParserLLM(
    double similarity_threshold,
    size_t lru_cache_size)
    : similarity_threshold_(similarity_threshold),
      llm_cache_(lru_cache_size) {
    
    // Set default template extractor (simple fallback)
    template_extractor_ = [](const std::string& log_message) -> std::pair<std::string, std::unordered_map<std::string, VariableType>> {
        // Basic variable detection using heuristics
        std::vector<std::string> tokens;
        std::stringstream ss(log_message);
        std::string token;
        
        while (ss >> token) {
            tokens.push_back(token);
        }
        
        std::string template_str;
        std::unordered_map<std::string, VariableType> variable_types;
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) template_str += " ";
            
            // Check if token looks like a variable (simple heuristics)
            bool is_ip = std::regex_match(tokens[i], std::regex("\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"));
            bool is_number = std::regex_match(tokens[i], std::regex("\\d+"));
            bool is_hex = std::regex_match(tokens[i], std::regex("0x[0-9a-fA-F]+"));
            bool is_id = std::regex_match(tokens[i], std::regex("[a-zA-Z0-9_-]+\\d+"));
            
            if (is_ip) {
                template_str += "<LOI>";
                variable_types["param_" + std::to_string(i) + "_LOI"] = VariableType::LOI;
            } else if (is_number) {
                template_str += "<OBA>";
                variable_types["param_" + std::to_string(i) + "_OBA"] = VariableType::OBA;
            } else if (is_hex) {
                template_str += "<OID>";
                variable_types["param_" + std::to_string(i) + "_OID"] = VariableType::OID;
            } else if (is_id) {
                template_str += "<OID>";
                variable_types["param_" + std::to_string(i) + "_OID"] = VariableType::OID;
            } else {
                template_str += tokens[i];
            }
        }
        
        return {template_str, variable_types};
    };
}

std::vector<std::string> LogParserLLM::tokenize(const std::string& log_message) const {
    std::vector<std::string> tokens;
    std::stringstream ss(log_message);
    std::string token;
    
    while (ss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

double LogParserLLM::calculate_token_similarity(
    const std::vector<std::string>& tokens1,
    const std::vector<std::string>& tokens2) const {
    
    if (tokens1.empty() || tokens2.empty()) {
        return 0.0;
    }
    
    // Calculate suffix match length
    size_t suffix_match_length = 0;
    for (size_t i = 0; i < std::min(tokens1.size(), tokens2.size()); ++i) {
        size_t idx1 = tokens1.size() - 1 - i;
        size_t idx2 = tokens2.size() - 1 - i;
        
        if (tokens1[idx1] != tokens2[idx2]) {
            break;
        }
        
        suffix_match_length++;
    }
    
    // Calculate suffix similarity score (weighted more heavily)
    size_t max_length = std::max(tokens1.size(), tokens2.size());
    double suffix_similarity = static_cast<double>(suffix_match_length) / max_length;
    
    // Calculate overall token similarity (Jaccard)
    std::unordered_set<std::string> set1(tokens1.begin(), tokens1.end());
    std::unordered_set<std::string> set2(tokens2.begin(), tokens2.end());
    
    std::unordered_set<std::string> intersection;
    for (const auto& token : set1) {
        if (set2.find(token) != set2.end()) {
            intersection.insert(token);
        }
    }
    
    size_t union_size = set1.size() + set2.size() - intersection.size();
    double jaccard_similarity = static_cast<double>(intersection.size()) / union_size;
    
    // Combine with heavy weight (0.7) on suffix matching
    return 0.7 * suffix_similarity + 0.3 * jaccard_similarity;
}

std::unordered_map<std::string, std::string> LogParserLLM::extract_parameters(
    const std::string& log_message,
    const std::string& template_str,
    const std::unordered_map<std::string, VariableType>& variable_types) const {
    
    std::unordered_map<std::string, std::string> parameters;
    
    auto log_tokens = tokenize(log_message);
    auto template_tokens = tokenize(template_str);
    
    // Handle case where token counts don't match
    if (log_tokens.size() != template_tokens.size()) {
        return parameters;
    }
    
    // Extract parameters by comparing tokens
    for (size_t i = 0; i < log_tokens.size(); ++i) {
        // If template token is a variable placeholder
        static const std::regex var_pattern("<(OID|LOI|OBN|TID|SID|TDA|CRS|OBA|STC|OTHER_PARAMS)>");
        if (std::regex_search(template_tokens[i], var_pattern)) {
            // Find the variable type from the token
            std::string var_type;
            std::smatch match;
            if (std::regex_search(template_tokens[i], match, var_pattern)) {
                var_type = match[1].str();
            } else {
                var_type = "OTHER_PARAMS";
            }
            
            std::string param_name = "param_" + std::to_string(i) + "_" + var_type;
            parameters[param_name] = log_tokens[i];
        }
    }
    
    return parameters;
}

std::pair<std::string, std::unordered_map<std::string, VariableType>> 
LogParserLLM::extract_template(const std::string& log_message) const {
    try {
        // Use provided template extractor
        return template_extractor_(log_message);
    } catch (const std::exception&) {
        // Fallback to simple template
        auto tokens = tokenize(log_message);
        std::string template_str;
        std::unordered_map<std::string, VariableType> variable_types;
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) template_str += " ";
            
            if (!std::isalpha(tokens[i][0])) {
                template_str += "<OTHER_PARAMS>";
                variable_types["param_" + std::to_string(i) + "_OTHER_PARAMS"] = VariableType::OTHER_PARAMS;
            } else {
                template_str += tokens[i];
            }
        }
        
        return {template_str, variable_types};
    }
}

std::pair<std::string, std::unordered_map<std::string, std::string>> 
LogParserLLM::handle_strict_match(
    std::shared_ptr<LogCluster> cluster,
    const std::string& log_message,
    int log_id) {
    
    cluster->add_log_id(log_id);
    cluster->add_example(log_message);
    cluster->increment_frequency();
    
    return {
        cluster->get_template()->get_template(),
        extract_parameters(
            log_message,
            cluster->get_template()->get_template(),
            cluster->get_template()->get_variable_types()
        )
    };
}

std::pair<std::string, std::unordered_map<std::string, std::string>> 
LogParserLLM::handle_template_match(
    std::shared_ptr<LogTemplate> matched_template,
    std::shared_ptr<LogTemplate> new_template,
    const std::vector<std::string>& log_tokens,
    const std::string& log_message,
    int log_id) {
    
    // Merge templates
    auto merged_template = matched_template->merge_with(*new_template);
    
    // Update template pool
    {
        std::unique_lock<std::shared_mutex> lock(template_pool_mutex_);
        template_pool_[merged_template->get_template()] = merged_template;
    }
    
    // Convert tokens to string
    std::string syntax;
    for (size_t i = 0; i < log_tokens.size(); ++i) {
        if (i > 0) syntax += " ";
        syntax += log_tokens[i];
    }
    
    // Update tree and add syntax template
    auto cluster = std::make_shared<LogCluster>(merged_template);
    cluster->add_log_id(log_id);
    cluster->add_example(log_message);
    trie_.insert(syntax, cluster);
    merged_template->add_syntax_template(syntax);
    
    return {
        merged_template->get_template(),
        extract_parameters(
            log_message,
            merged_template->get_template(),
            merged_template->get_variable_types()
        )
    };
}

std::pair<std::string, std::unordered_map<std::string, std::string>> 
LogParserLLM::handle_new_template(
    std::shared_ptr<LogTemplate> new_template,
    const std::vector<std::string>& log_tokens,
    const std::string& log_message,
    int log_id) {
    
    // Add to template pool
    {
        std::unique_lock<std::shared_mutex> lock(template_pool_mutex_);
        template_pool_[new_template->get_template()] = new_template;
    }
    
    // Convert tokens to string
    std::string syntax;
    for (size_t i = 0; i < log_tokens.size(); ++i) {
        if (i > 0) syntax += " ";
        syntax += log_tokens[i];
    }
    
    // Update tree and add syntax template
    auto cluster = std::make_shared<LogCluster>(new_template);
    cluster->add_log_id(log_id);
    cluster->add_example(log_message);
    trie_.insert(syntax, cluster);
    new_template->add_syntax_template(syntax);
    
    return {
        new_template->get_template(),
        extract_parameters(
            log_message,
            new_template->get_template(),
            new_template->get_variable_types()
        )
    };
}

std::string LogParserLLM::get_cache_key(const std::vector<std::string>& tokens) const {
    // Keep only tokens that are likely part of the template structure
    std::vector<std::string> structural_tokens;
    
    for (const auto& token : tokens) {
        // Check for structural characters
        bool has_structural_chars = false;
        for (char c : "[]{}()=:/") {
            if (token.find(c) != std::string::npos) {
                has_structural_chars = true;
                break;
            }
        }
        
        // Check for all caps words
        bool is_all_caps = !token.empty() && 
                          std::all_of(token.begin(), token.end(), [](char c) {
                              return std::isupper(c) || !std::isalpha(c);
                          });
        
        // Check for common log keywords
        bool is_common_keyword = (
            token == "error" || token == "warning" || token == "info" || 
            token == "debug" || token == "failed" || token == "succeeded"
        );
        
        if (has_structural_chars || is_all_caps || is_common_keyword) {
            structural_tokens.push_back(token);
        }
    }
    
    // Join tokens into a key
    std::string key;
    if (!structural_tokens.empty()) {
        for (size_t i = 0; i < structural_tokens.size(); ++i) {
            if (i > 0) key += " ";
            key += structural_tokens[i];
        }
    } else if (!tokens.empty()) {
        // Use first few tokens as fallback
        for (size_t i = 0; i < std::min(size_t(3), tokens.size()); ++i) {
            if (i > 0) key += " ";
            key += tokens[i];
        }
    }
    
    return key;
}

std::optional<std::pair<std::string, std::unordered_map<std::string, VariableType>>> 
LogParserLLM::find_similar_cache_entry(const std::vector<std::string>& tokens) const {
    std::string cache_key = get_cache_key(tokens);
    return llm_cache_.get(cache_key);
}

void LogParserLLM::update_cache(
    const std::vector<std::string>& tokens,
    const std::string& template_str,
    const std::unordered_map<std::string, VariableType>& variable_types) {
    
    std::string cache_key = get_cache_key(tokens);
    llm_cache_.put(cache_key, {template_str, variable_types});
}

std::pair<std::string, std::unordered_map<std::string, std::string>> 
LogParserLLM::parse_log(const std::string& content, int log_id) {
    // 1. Preprocessing
    auto tokens = tokenize(content);
    
    // 2. Prefix Tree Traversal
    auto traversal_result = trie_.traverse(tokens);
    
    // 3. Process strict match
    if (traversal_result.strict_matched_cluster) {
        return handle_strict_match(traversal_result.strict_matched_cluster, content, log_id);
    }
    
    // 4. Check LLM cache before calling LLM
    auto cached_result = find_similar_cache_entry(tokens);
    std::string template_str;
    std::unordered_map<std::string, VariableType> variable_types;
    
    if (cached_result) {
        // Use cached result
        std::tie(template_str, variable_types) = *cached_result;
    } else {
        // No cache hit - Extract template
        std::tie(template_str, variable_types) = extract_template(content);
        // Cache the result
        update_cache(tokens, template_str, variable_types);
    }
    
    auto new_template = std::make_shared<LogTemplate>(
        template_str,
        variable_types,
        std::vector<std::string>{content}
    );
    
    // 5. Check template pool for similar templates
    {
        std::shared_lock<std::shared_mutex> lock(template_pool_mutex_);
        for (const auto& [temp_str, existing_template] : template_pool_) {
            if (existing_template->matches(*new_template, similarity_threshold_)) {
                return handle_template_match(
                    existing_template, new_template, tokens, content, log_id
                );
            }
        }
    }
    
    // 6. Check loose matches
    for (const auto& cluster : traversal_result.loose_matched_clusters) {
        if (cluster->get_template()->matches(*new_template, similarity_threshold_)) {
            auto merged = cluster->get_template()->merge_with(*new_template);
            cluster->set_template(merged);
            cluster->add_example(content);
            
            return {
                merged->get_template(),
                extract_parameters(
                    content,
                    merged->get_template(),
                    merged->get_variable_types()
                )
            };
        }
    }
    
    // 7. Create new template
    return handle_new_template(new_template, tokens, content, log_id);
}

std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> 
LogParserLLM::parse_logs_batch(
    const std::vector<std::pair<std::string, int>>& logs,
    size_t batch_size,
    size_t max_workers) {
    
    std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> results;
    results.reserve(logs.size());
    
    if (logs.empty()) {
        return results;
    }
    
    // Determine number of threads to use
    size_t hardware_threads = std::thread::hardware_concurrency();
    size_t min_threads = std::max<size_t>(1, hardware_threads);
    size_t num_workers = std::min(max_workers, min_threads);
    
    if (num_workers <= 1 || logs.size() <= batch_size) {
        // Process sequentially if only one worker or small batch
        for (const auto& log : logs) {
            try {
                results.push_back(parse_log(log.first, log.second));
            } catch (const std::exception&) {
                // Fallback to empty template on error
                results.push_back({log.first, {}});
            }
        }
    } else {
        // Process in parallel batches
        std::vector<std::future<std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>>>> futures;
        
        for (size_t i = 0; i < logs.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, logs.size());
            std::vector<std::pair<std::string, int>> batch(logs.begin() + i, logs.begin() + end);
            
            futures.push_back(std::async(std::launch::async, [this, batch]() {
                std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> batch_results;
                batch_results.reserve(batch.size());
                
                for (const auto& log : batch) {
                    try {
                        batch_results.push_back(parse_log(log.first, log.second));
                    } catch (const std::exception&) {
                        // Fallback to empty template on error
                        batch_results.push_back({log.first, {}});
                    }
                }
                
                return batch_results;
            }));
        }
        
        // Collect results
        for (auto& future : futures) {
            auto batch_results = future.get();
            results.insert(results.end(), batch_results.begin(), batch_results.end());
        }
    }
    
    return results;
}

std::unordered_map<std::string, size_t> LogParserLLM::get_statistics() const {
    std::shared_lock<std::shared_mutex> lock(template_pool_mutex_);
    
    return {
        {"total_templates", template_pool_.size()},
        {"total_clusters", trie_.size()},
        {"lru_cache_info", llm_cache_.size()}
    };
}

void LogParserLLM::set_template_extractor(TemplateExtractor extractor) {
    template_extractor_ = extractor;
}

} // namespace sherlog