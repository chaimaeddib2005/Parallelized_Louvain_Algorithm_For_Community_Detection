#ifndef LOUVAIN_HPP
#define LOUVAIN_HPP

#include "graph.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>

/**
 * Traditional Louvain Algorithm for Community Detection
 * Based on: Blondel et al. (2008) "Fast unfolding of communities in large networks"
 * 
 * Algorithm:
 * 1. Phase 1: Iteratively move nodes to neighboring communities to maximize modularity
 * 2. Phase 2: Build a new graph where communities become nodes
 * 3. Repeat until no improvement
 */

class Louvain {
public:
    using NodeID = Graph::NodeID;
    using EdgeID = Graph::EdgeID;
    using Weight = Graph::Weight;
    using Community = uint32_t;
    
    struct Result {
        std::vector<Community> communities;  // community[node] = community_id
        double modularity;
        uint32_t num_communities;
        uint32_t num_levels;
    };
    
private:
    const Graph* graph_;
    double resolution_;
    double min_modularity_gain_;
    uint32_t seed_;
    
    // Current partition
    std::vector<Community> node_to_community_;
    
    // Community statistics
    std::vector<Weight> community_weights_;  // Total weight of edges incident to community
    std::vector<Weight> community_internal_weights_;  // Weight of internal edges in community
    
    double total_weight_;
    std::mt19937 rng_;
    
public:
    Louvain(double resolution = 1.0, 
            double min_modularity_gain = 1e-7,
            uint32_t seed = 42)
        : graph_(nullptr)
        , resolution_(resolution)
        , min_modularity_gain_(min_modularity_gain)
        , seed_(seed)
        , total_weight_(0.0)
        , rng_(seed) {}
    
    Result detect_communities(const Graph& graph) {
        graph_ = &graph;
        total_weight_ = graph.total_weight();
        
        // Initialize dendrogram (hierarchy of partitions)
        std::vector<std::vector<Community>> dendrogram;
        
        // Start with each node in its own community
        initialize_partition();
        
        uint32_t level = 0;
        double modularity = compute_modularity();
        bool improvement = true;
        
        printf("Initial modularity: %.6f\n", modularity);
        
        while (improvement && level < 100) {
            printf("\n--- Level %u ---\n", level);
            
            // Phase 1: Optimize partition
            improvement = optimize_partition();
            
            if (!improvement) {
                printf("No improvement in phase 1, stopping\n");
                break;
            }
            
            double new_modularity = compute_modularity();
            printf("After phase 1: Modularity = %.6f, Communities = %u\n", 
                   new_modularity, count_communities());
            
            // Check if modularity improved enough
            if (new_modularity - modularity < min_modularity_gain_) {
                printf("Modularity improvement %.6f < threshold %.6f, stopping\n",
                       new_modularity - modularity, min_modularity_gain_);
                break;
            }
            
            modularity = new_modularity;
            
            // Save current partition
            dendrogram.push_back(node_to_community_);
            
            // Phase 2: Aggregate graph
            // For simplicity, we stop here (single level)
            // Full implementation would build new aggregated graph
            break;
            
            level++;
        }
        
        // Prepare result
        renumber_communities();
        
        Result result;
        result.communities = node_to_community_;
        result.modularity = compute_modularity();
        result.num_communities = count_communities();
        result.num_levels = level + 1;
        
        return result;
    }
    
private:
    void initialize_partition() {
        uint32_t num_nodes = graph_->num_nodes();
        node_to_community_.resize(num_nodes);
        community_weights_.resize(num_nodes, 0.0);
        community_internal_weights_.resize(num_nodes, 0.0);
        
        // Each node starts in its own community
        for (NodeID node = 0; node < num_nodes; ++node) {
            node_to_community_[node] = node;
            community_weights_[node] = graph_->degree(node);
            
            // Check for self-loops
            Weight self_loop_weight = graph_->get_edge_weight(node, node);
            community_internal_weights_[node] = self_loop_weight;
        }
    }
    
    bool optimize_partition() {
        bool improvement = false;
        bool node_moved = true;
        uint32_t iteration = 0;
        
        while (node_moved && iteration < 100) {
            node_moved = false;
            
            // Create random order of nodes
            std::vector<NodeID> nodes(graph_->num_nodes());
            for (NodeID i = 0; i < graph_->num_nodes(); ++i) {
                nodes[i] = i;
            }
            std::shuffle(nodes.begin(), nodes.end(), rng_);
            
            // Try to move each node
            for (NodeID node : nodes) {
                Community old_comm = node_to_community_[node];
                Community best_comm = find_best_community(node);
                
                if (best_comm != old_comm) {
                    move_node_to_community(node, old_comm, best_comm);
                    node_moved = true;
                    improvement = true;
                }
            }
            
            iteration++;
            if (node_moved) {
                printf("  Iteration %u: nodes moved\n", iteration);
            }
        }
        
        return improvement;
    }
    
    Community find_best_community(NodeID node) {
        Community current_comm = node_to_community_[node];
        
        // Compute weights to neighboring communities
        std::unordered_map<Community, Weight> neighbor_comm_weights;
        
        for (EdgeID edge_idx = graph_->neighbor_start(node);
             edge_idx < graph_->neighbor_end(node); ++edge_idx) {
            NodeID neighbor = graph_->neighbor(edge_idx);
            Weight weight = graph_->weight(edge_idx);
            Community neighbor_comm = node_to_community_[neighbor];
            
            neighbor_comm_weights[neighbor_comm] += weight;
        }
        
        // Remove node from current community (temporarily)
        Weight node_degree = graph_->degree(node);
        Weight ki_in_old = neighbor_comm_weights[current_comm];
        
        community_weights_[current_comm] -= node_degree;
        community_internal_weights_[current_comm] -= 2.0 * ki_in_old + graph_->get_edge_weight(node, node);
        
        // Find best community
        Community best_comm = current_comm;
        double best_delta_Q = 0.0;
        
        for (const auto& [comm, ki_in] : neighbor_comm_weights) {
            double delta_Q = compute_delta_modularity(node, comm, ki_in);
            
            if (delta_Q > best_delta_Q) {
                best_delta_Q = delta_Q;
                best_comm = comm;
            }
        }
        
        // Put node back in current community
        community_weights_[current_comm] += node_degree;
        community_internal_weights_[current_comm] += 2.0 * ki_in_old + graph_->get_edge_weight(node, node);
        
        return best_comm;
    }
    
    double compute_delta_modularity(NodeID node, Community target_comm, Weight ki_in) {
        Weight node_degree = graph_->degree(node);
        Weight sigma_tot = community_weights_[target_comm];
        
        // Standard Louvain modularity gain formula
        double delta_Q = ki_in - resolution_ * sigma_tot * node_degree / (2.0 * total_weight_);
        delta_Q /= total_weight_;
        
        return delta_Q;
    }
    
    void move_node_to_community(NodeID node, Community old_comm, Community new_comm) {
        if (old_comm == new_comm) {
            return;
        }
        
        Weight node_degree = graph_->degree(node);
        Weight self_loop = graph_->get_edge_weight(node, node);
        
        // Compute weight to old and new communities
        Weight ki_in_old = 0.0;
        Weight ki_in_new = 0.0;
        
        for (EdgeID edge_idx = graph_->neighbor_start(node);
             edge_idx < graph_->neighbor_end(node); ++edge_idx) {
            NodeID neighbor = graph_->neighbor(edge_idx);
            Weight weight = graph_->weight(edge_idx);
            Community neighbor_comm = node_to_community_[neighbor];
            
            if (neighbor_comm == old_comm) {
                ki_in_old += weight;
            }
            if (neighbor_comm == new_comm) {
                ki_in_new += weight;
            }
        }
        
        // Remove from old community
        community_weights_[old_comm] -= node_degree;
        community_internal_weights_[old_comm] -= 2.0 * ki_in_old + self_loop;
        
        // Add to new community
        community_weights_[new_comm] += node_degree;
        community_internal_weights_[new_comm] += 2.0 * ki_in_new + self_loop;
        
        // Update node's community
        node_to_community_[node] = new_comm;
    }
    
    double compute_modularity() const {
        double Q = 0.0;
        
        std::unordered_set<Community> communities;
        for (Community comm : node_to_community_) {
            communities.insert(comm);
        }
        
        for (Community comm : communities) {
            double l_c = community_internal_weights_[comm] / 2.0;  // Internal edges
            double d_c = community_weights_[comm];  // Total degree
            
            Q += l_c / total_weight_ - resolution_ * (d_c / (2.0 * total_weight_)) * (d_c / (2.0 * total_weight_));
        }
        
        return Q;
    }
    
    uint32_t count_communities() const {
        std::unordered_set<Community> unique_communities(
            node_to_community_.begin(),
            node_to_community_.end()
        );
        return unique_communities.size();
    }
    
    void renumber_communities() {
        std::unordered_set<Community> unique_comms(
            node_to_community_.begin(),
            node_to_community_.end()
        );
        
        std::unordered_map<Community, Community> comm_mapping;
        Community new_id = 0;
        for (Community old_id : unique_comms) {
            comm_mapping[old_id] = new_id++;
        }
        
        for (NodeID node = 0; node < graph_->num_nodes(); ++node) {
            node_to_community_[node] = comm_mapping[node_to_community_[node]];
        }
        
        // Update community statistics
        std::vector<Weight> new_weights(new_id, 0.0);
        std::vector<Weight> new_internal(new_id, 0.0);
        
        for (Community old_comm : unique_comms) {
            Community new_comm = comm_mapping[old_comm];
            new_weights[new_comm] = community_weights_[old_comm];
            new_internal[new_comm] = community_internal_weights_[old_comm];
        }
        
        community_weights_ = std::move(new_weights);
        community_internal_weights_ = std::move(new_internal);
    }
};


/* Example usage
#ifdef LOUVAIN_EXAMPLE
#include <iostream>

int main() {
    // Create Karate Club graph
    GraphBuilder builder(34, false);
    
    // Add Zachary's Karate Club edges
    std::vector<std::pair<uint32_t, uint32_t>> karate_edges = {
        {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8},
        {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 17}, {0, 19}, {0, 21}, {0, 31},
        {1, 2}, {1, 3}, {1, 7}, {1, 13}, {1, 17}, {1, 19}, {1, 21}, {1, 30},
        {2, 3}, {2, 7}, {2, 8}, {2, 9}, {2, 13}, {2, 27}, {2, 28}, {2, 32},
        {3, 7}, {3, 12}, {3, 13}, {4, 6}, {4, 10}, {5, 6}, {5, 10}, {5, 16},
        {6, 16}, {8, 30}, {8, 32}, {8, 33}, {9, 33}, {13, 33}, {14, 32}, {14, 33},
        {15, 32}, {15, 33}, {18, 32}, {18, 33}, {19, 33}, {20, 32}, {20, 33},
        {22, 32}, {22, 33}, {23, 25}, {23, 27}, {23, 29}, {23, 32}, {23, 33},
        {24, 25}, {24, 27}, {24, 31}, {25, 31}, {26, 29}, {26, 33}, {27, 33},
        {28, 31}, {28, 33}, {29, 32}, {29, 33}, {30, 32}, {30, 33}, {31, 32},
        {31, 33}, {32, 33}
    };
    
    for (const auto& [u, v] : karate_edges) {
        builder.add_edge(u, v, 1.0);
    }
    
    Graph graph = builder.build();
    graph.print_stats();
    
    printf("\n========================================\n");
    printf("Running Traditional Louvain Algorithm\n");
    printf("========================================\n\n");
    
    // Run Louvain with seed for reproducibility
    Louvain louvain(1.0, 1e-7, 42);
    auto result = louvain.detect_communities(graph);
    
    printf("\n========================================\n");
    printf("Final Results:\n");
    printf("  Communities: %u\n", result.num_communities);
    printf("  Modularity: %.6f\n", result.modularity);
    printf("  Levels: %u\n", result.num_levels);
    printf("========================================\n\n");
    
    // Group nodes by community
    std::unordered_map<uint32_t, std::vector<uint32_t>> comm_groups;
    for (uint32_t node = 0; node < result.communities.size(); ++node) {
        comm_groups[result.communities[node]].push_back(node);
    }
    
    printf("Community assignments:\n");
    for (auto& [comm_id, nodes] : comm_groups) {
        std::sort(nodes.begin(), nodes.end());
        printf("  Community %u: ", comm_id);
        for (size_t i = 0; i < nodes.size(); ++i) {
            printf("%u", nodes[i]);
            if (i < nodes.size() - 1) printf(", ");
        }
        printf("\n");
    }
    
    return 0;
}
#endif*/

#endif // LOUVAIN_HPP