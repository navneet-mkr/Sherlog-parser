#include "prefix_tree.hpp"
#include "log_parser.hpp"
#include <sstream>
#include <algorithm>

namespace sherlog {

PrefixTreeNode::PrefixTreeNode()
    : is_variable(false), frequency(0) {
}

PrefixTree::PrefixTree()
    : root_(std::make_shared<PrefixTreeNode>()), size_(0) {
}

void PrefixTree::insert(const std::string& key, std::shared_ptr<LogCluster> cluster) {
    // Acquire write lock
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    std::shared_ptr<PrefixTreeNode> current = root_;
    std::stringstream ss(key);
    std::string token;
    
    while (std::getline(ss, token, ' ')) {
        if (token.empty()) {
            continue;
        }
        
        if (current->children.find(token) == current->children.end()) {
            current->children[token] = std::make_shared<PrefixTreeNode>();
        }
        
        current = current->children[token];
        
        // Check if this is a variable token
        if (token.find("<") != std::string::npos && token.find(">") != std::string::npos) {
            current->is_variable = true;
        }
    }
    
    // Increment frequency
    current->frequency++;
    
    // Add cluster to node
    current->clusters.push_back(cluster);
    
    // Update size
    size_++;
}

std::shared_ptr<LogCluster> PrefixTree::find(const std::string& key) const {
    // Acquire read lock
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::shared_ptr<PrefixTreeNode> current = root_;
    std::stringstream ss(key);
    std::string token;
    
    while (std::getline(ss, token, ' ')) {
        if (token.empty()) {
            continue;
        }
        
        if (current->children.find(token) == current->children.end()) {
            return nullptr;
        }
        
        current = current->children[token];
    }
    
    // Check if this node has clusters
    if (!current->clusters.empty()) {
        return current->clusters.front();
    }
    
    return nullptr;
}

std::vector<std::shared_ptr<LogCluster>> PrefixTree::find_prefixes(const std::string& key) const {
    // Acquire read lock
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<std::shared_ptr<LogCluster>> matches;
    std::shared_ptr<PrefixTreeNode> current = root_;
    std::stringstream ss(key);
    std::string token;
    
    while (std::getline(ss, token, ' ')) {
        if (token.empty()) {
            continue;
        }
        
        if (current->children.find(token) == current->children.end()) {
            break;
        }
        
        current = current->children[token];
        
        // Add clusters at this node
        matches.insert(matches.end(), current->clusters.begin(), current->clusters.end());
    }
    
    return matches;
}

TraversalResult PrefixTree::traverse(const std::vector<std::string>& tokens) const {
    // Acquire read lock
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    TraversalResult result;
    
    // Convert tokens to key
    std::string key;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) key += " ";
        key += tokens[i];
    }
    
    // Try exact match first
    auto exact_match = find(key);
    if (exact_match) {
        result.strict_matched_cluster = exact_match;
        return result;
    }
    
    // Try prefix matches
    auto prefix_matches = find_prefixes(key);
    if (!prefix_matches.empty()) {
        // Find the longest match
        std::shared_ptr<LogCluster> longest_match = nullptr;
        size_t longest_match_length = 0;
        
        for (const auto& match : prefix_matches) {
            const auto& template_tokens = match->get_template()->get_syntax_templates();
            for (const auto& syntax : template_tokens) {
                size_t length = std::count(syntax.begin(), syntax.end(), ' ') + 1;
                if (length > longest_match_length) {
                    longest_match_length = length;
                    longest_match = match;
                }
            }
        }
        
        if (longest_match) {
            result.loose_matched_clusters.push_back(longest_match);
        }
    }
    
    return result;
}

size_t PrefixTree::size() const {
    // Acquire read lock
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return size_;
}

void PrefixTree::clear() {
    // Acquire write lock
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    root_ = std::make_shared<PrefixTreeNode>();
    size_ = 0;
}

} // namespace sherlog