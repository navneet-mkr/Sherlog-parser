#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>
#include <mutex>
#include <shared_mutex>

namespace sherlog {

// Forward declarations
class LogCluster;
class LogTemplate;

/**
 * Result of prefix tree traversal.
 */
struct TraversalResult {
    std::shared_ptr<LogCluster> strict_matched_cluster;
    std::vector<std::shared_ptr<LogCluster>> loose_matched_clusters;
};

/**
 * Node in the prefix tree (trie).
 */
class PrefixTreeNode {
public:
    PrefixTreeNode();
    
    // Children nodes
    std::unordered_map<std::string, std::shared_ptr<PrefixTreeNode>> children;
    
    // Is this node a variable placeholder?
    bool is_variable;
    
    // Clusters at this node
    std::vector<std::shared_ptr<LogCluster>> clusters;
    
    // Frequency counter
    size_t frequency;
};

/**
 * Prefix tree (trie) implementation for efficient template lookup.
 * Thread-safe with read-write locks for concurrent access.
 */
class PrefixTree {
public:
    PrefixTree();
    
    /**
     * Insert a key-value pair into the trie.
     * @param key Tokenized key string (space-separated)
     * @param cluster Log cluster to associate with the key
     */
    void insert(const std::string& key, std::shared_ptr<LogCluster> cluster);
    
    /**
     * Find exact match for a key.
     * @param key Tokenized key string (space-separated)
     * @return Associated cluster if found, nullptr otherwise
     */
    std::shared_ptr<LogCluster> find(const std::string& key) const;
    
    /**
     * Find prefix matches for a key.
     * @param key Tokenized key string (space-separated)
     * @return Vector of clusters with matching prefixes
     */
    std::vector<std::shared_ptr<LogCluster>> find_prefixes(const std::string& key) const;
    
    /**
     * Traverse the trie to find matching clusters.
     * @param tokens Vector of tokens to find
     * @return TraversalResult with matching clusters
     */
    TraversalResult traverse(const std::vector<std::string>& tokens) const;
    
    /**
     * Get the number of keys in the trie.
     * @return Number of keys
     */
    size_t size() const;
    
    /**
     * Clear the trie.
     */
    void clear();

private:
    std::shared_ptr<PrefixTreeNode> root_;
    size_t size_;
    
    // Thread synchronization
    mutable std::shared_mutex mutex_;
};

} // namespace sherlog