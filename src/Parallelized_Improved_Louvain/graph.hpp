#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <cassert>
#include<stdio.h>

/**
 * Graph representation optimized for Louvain algorithm
 * Uses Compressed Sparse Row (CSR) format for memory efficiency and cache performance
 * 
 * Key design decisions:
 * 1. CSR format: Best for iterating over neighbors (main operation in Louvain)
 * 2. Separate edge weights: Supports weighted graphs efficiently
 * 3. Pre-computed degrees: Avoids recomputation in modularity calculations
 * 4. Undirected storage: Each edge stored once to save memory
 */

class Graph {
public:
    using NodeID = uint32_t;
    using EdgeID = uint64_t;
    using Weight = float;
    
private:
    // CSR (Compressed Sparse Row) representation
    // For node i: neighbors are col_idx[row_ptr[i] ... row_ptr[i+1]-1]
    std::vector<EdgeID> row_ptr_;      // Size: num_nodes + 1
    std::vector<NodeID> col_idx_;      // Size: num_edges (or 2*num_edges for undirected)
    std::vector<Weight> weights_;      // Size: num_edges (or 2*num_edges for undirected)
    
    // Graph metadata
    NodeID num_nodes_;
    EdgeID num_edges_;  // Number of unique edges (not counting both directions)
    Weight total_weight_;
    
    // Pre-computed node properties
    std::vector<Weight> node_degrees_;  // Weighted degree of each node
    
    bool is_directed_;
    
public:
    // Constructor
    Graph(NodeID num_nodes, bool directed = false) 
        : num_nodes_(num_nodes)
        , num_edges_(0)
        , total_weight_(0.0)
        , is_directed_(directed) {
        row_ptr_.resize(num_nodes + 1, 0);
        node_degrees_.resize(num_nodes, 0.0);
    }
    
    // Build graph from edge list
    void build_from_edges(const std::vector<std::tuple<NodeID, NodeID, Weight>>& edges) {
        num_edges_ = edges.size();
        
        // Count edges per node (for undirected: count both directions)
        std::vector<EdgeID> edge_counts(num_nodes_, 0);
        
        for (const auto& [u, v, w] : edges) {
            assert(u < num_nodes_ && v < num_nodes_);
            edge_counts[u]++;
            if (!is_directed_ && u != v) {
                edge_counts[v]++;
            }
        }
        
        // Build row_ptr using prefix sum
        row_ptr_[0] = 0;
        for (NodeID i = 0; i < num_nodes_; ++i) {
            row_ptr_[i + 1] = row_ptr_[i] + edge_counts[i];
        }
        
        // Allocate space for edges
        EdgeID total_entries = row_ptr_[num_nodes_];
        col_idx_.resize(total_entries);
        weights_.resize(total_entries);
        
        // Fill edges (use edge_counts as write positions)
        std::fill(edge_counts.begin(), edge_counts.end(), 0);
        
        for (const auto& [u, v, w] : edges) {
            // Add edge u -> v
            EdgeID pos_u = row_ptr_[u] + edge_counts[u]++;
            col_idx_[pos_u] = v;
            weights_[pos_u] = w;
            
            // For undirected, add v -> u (if not self-loop)
            if (!is_directed_ && u != v) {
                EdgeID pos_v = row_ptr_[v] + edge_counts[v]++;
                col_idx_[pos_v] = u;
                weights_[pos_v] = w;
            }
        }
        
        // Sort neighbors for each node (improves cache performance)
        for (NodeID i = 0; i < num_nodes_; ++i) {
            sort_neighbors(i);
        }
        
        // Pre-compute degrees and total weight
        compute_degrees();
    }
    
    // Accessors
    NodeID num_nodes() const { return num_nodes_; }
    EdgeID num_edges() const { return num_edges_; }
    Weight total_weight() const { return total_weight_; }
    Weight degree(NodeID node) const { return node_degrees_[node]; }
    
    // Neighbor iteration (most important operation for Louvain)
    EdgeID neighbor_start(NodeID node) const { return row_ptr_[node]; }
    EdgeID neighbor_end(NodeID node) const { return row_ptr_[node + 1]; }
    EdgeID num_neighbors(NodeID node) const { return neighbor_end(node) - neighbor_start(node); }
    
    NodeID neighbor(EdgeID edge_idx) const { return col_idx_[edge_idx]; }
    Weight weight(EdgeID edge_idx) const { return weights_[edge_idx]; }
    
    // Range-based for loop support
    struct NeighborIterator {
        const Graph* graph;
        EdgeID current;
        
        struct NeighborData {
            NodeID node;
            Weight weight;
        };
        
        NeighborData operator*() const {
            return {graph->col_idx_[current], graph->weights_[current]};
        }
        
        NeighborIterator& operator++() { ++current; return *this; }
        bool operator!=(const NeighborIterator& other) const { 
            return current != other.current; 
        }
    };
    
    struct NeighborRange {
        const Graph* graph;
        EdgeID start, end_;
        
        NeighborIterator begin() const { return {graph, start}; }
        NeighborIterator end() const { return {graph, end_}; }
    };
    
    NeighborRange neighbors(NodeID node) const {
        return {this, row_ptr_[node], row_ptr_[node + 1]};
    }
    
    // Check if edge exists (binary search since neighbors are sorted)
    bool has_edge(NodeID u, NodeID v) const {
        EdgeID start = row_ptr_[u];
        EdgeID end = row_ptr_[u + 1];
        auto it = std::lower_bound(col_idx_.begin() + start, 
                                   col_idx_.begin() + end, v);
        return it != col_idx_.begin() + end && *it == v;
    }
    
    // Get edge weight (returns 0 if edge doesn't exist)
    Weight get_edge_weight(NodeID u, NodeID v) const {
        EdgeID start = row_ptr_[u];
        EdgeID end = row_ptr_[u + 1];
        auto it = std::lower_bound(col_idx_.begin() + start, 
                                   col_idx_.begin() + end, v);
        if (it != col_idx_.begin() + end && *it == v) {
            EdgeID idx = std::distance(col_idx_.begin(), it);
            return weights_[idx];
        }
        return 0.0;
    }
    
    // Memory usage
    size_t memory_bytes() const {
        return sizeof(*this) +
               row_ptr_.size() * sizeof(EdgeID) +
               col_idx_.size() * sizeof(NodeID) +
               weights_.size() * sizeof(Weight) +
               node_degrees_.size() * sizeof(Weight);
    }
    
    void print_stats() const {
        double avg_degree = 0.0;
        if (num_nodes_ > 0) {
            avg_degree = static_cast<double>(col_idx_.size()) / num_nodes_;
        }
        
        printf("Graph Statistics:\n");
        printf("  Nodes: %u\n", num_nodes_);
        printf("  Edges: %lu\n", num_edges_);
        printf("  Average degree: %.2f\n", avg_degree);
        printf("  Total weight: %.2f\n", total_weight_);
        printf("  Memory: %.2f MB\n", memory_bytes() / (1024.0 * 1024.0));
    }
    
private:
    void sort_neighbors(NodeID node) {
        EdgeID start = row_ptr_[node];
        EdgeID end = row_ptr_[node + 1];
        
        // Create indices for sorting
        std::vector<EdgeID> indices(end - start);
        std::iota(indices.begin(), indices.end(), 0);
        
        // Sort by neighbor ID
        std::sort(indices.begin(), indices.end(), [&](EdgeID a, EdgeID b) {
            return col_idx_[start + a] < col_idx_[start + b];
        });
        
        // Reorder using indices
        std::vector<NodeID> sorted_col(end - start);
        std::vector<Weight> sorted_weights(end - start);
        
        for (size_t i = 0; i < indices.size(); ++i) {
            sorted_col[i] = col_idx_[start + indices[i]];
            sorted_weights[i] = weights_[start + indices[i]];
        }
        
        // Copy back
        std::copy(sorted_col.begin(), sorted_col.end(), col_idx_.begin() + start);
        std::copy(sorted_weights.begin(), sorted_weights.end(), weights_.begin() + start);
    }
    
    void compute_degrees() {
        total_weight_ = 0.0;
        
        for (NodeID i = 0; i < num_nodes_; ++i) {
            Weight degree = 0.0;
            for (EdgeID e = row_ptr_[i]; e < row_ptr_[i + 1]; ++e) {
                degree += weights_[e];
            }
            node_degrees_[i] = degree;
            total_weight_ += degree;
        }
        
        // For undirected graphs, each edge is counted twice
        if (!is_directed_) {
            total_weight_ /= 2.0;
        }
    }
};


/**
 * Utility class for building graphs incrementally
 * More convenient for some use cases than providing all edges at once
 */
class GraphBuilder {
private:
    struct Edge {
        Graph::NodeID u, v;
        Graph::Weight w;
    };
    
    std::vector<Edge> edges_;
    Graph::NodeID num_nodes_;
    bool directed_;
    
public:
    GraphBuilder(Graph::NodeID num_nodes, bool directed = false)
        : num_nodes_(num_nodes), directed_(directed) {}
    
    void add_edge(Graph::NodeID u, Graph::NodeID v, Graph::Weight w = 1.0) {
        edges_.push_back({u, v, w});
    }
    
    Graph build() {
        Graph g(num_nodes_, directed_);
        
        std::vector<std::tuple<Graph::NodeID, Graph::NodeID, Graph::Weight>> edge_list;
        edge_list.reserve(edges_.size());
        
        for (const auto& e : edges_) {
            edge_list.emplace_back(e.u, e.v, e.w);
        }
        
        g.build_from_edges(edge_list);
        return g;
    }
};


#endif // GRAPH_HPP