#ifndef CUDA_IMPROVED_LOUVAIN_CUH
#define CUDA_IMPROVED_LOUVAIN_CUH

#include "graph.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * GPU-accelerated IMPROVED Fast Louvain Algorithm
 * Based on: Zhang et al. (2021) "An Improved Louvain Algorithm for Community Detection"
 * 
 * Key improvements implemented:
 * 1. Dynamic iteration - only processes active nodes
 * 2. Tree structure detection and splitting
 * 3. Active node tracking and propagation
 * 4. Early stopping based on modularity gain
 */

namespace cuda_improved_louvain {

using NodeID = uint32_t;
using EdgeID = uint64_t;
using Community = uint32_t;
using Weight = float;

// Device graph structure
struct DeviceGraph {
    EdgeID* d_row_ptr;
    NodeID* d_col_idx;
    Weight* d_weights;
    Weight* d_node_degrees;
    NodeID num_nodes;
    EdgeID num_edges;
    Weight total_weight;
};

// Kernel: BFS for tree detection (simplified version)
__global__ void bfs_step_kernel(
    const EdgeID* __restrict__ row_ptr,
    const NodeID* __restrict__ col_idx,
    NodeID num_nodes,
    const int* __restrict__ current_frontier,
    int* next_frontier,
    int* visited,
    NodeID* parent,
    int* component_id,
    int comp_id
) {
    NodeID node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes || !current_frontier[node]) return;
    
    EdgeID start = row_ptr[node];
    EdgeID end = row_ptr[node + 1];
    
    for (EdgeID e = start; e < end; ++e) {
        NodeID neighbor = col_idx[e];
        
        // Try to visit neighbor with properly aligned atomic operation
        if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
            next_frontier[neighbor] = 1;
            parent[neighbor] = node;
            component_id[neighbor] = comp_id;
        }
    }
}

// Kernel: Count edges in component for tree detection
__global__ void count_component_edges_kernel(
    const EdgeID* __restrict__ row_ptr,
    const NodeID* __restrict__ col_idx,
    const int* __restrict__ component_id,
    NodeID num_nodes,
    int comp_id,
    unsigned long long* edge_count
) {
    NodeID node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes || component_id[node] != comp_id) return;
    
    EdgeID start = row_ptr[node];
    EdgeID end = row_ptr[node + 1];
    
    for (EdgeID e = start; e < end; ++e) {
        NodeID neighbor = col_idx[e];
        // Count each edge once (only if neighbor > node and in same component)
        if (component_id[neighbor] == comp_id && neighbor > node) {
            atomicAdd(edge_count, 1ULL);
        }
    }
}

// Kernel: Compute best moves for ACTIVE nodes only
__global__ void compute_best_moves_active_kernel(
    const EdgeID* __restrict__ row_ptr,
    const NodeID* __restrict__ col_idx,
    const Weight* __restrict__ weights,
    const Weight* __restrict__ node_degrees,
    const Community* __restrict__ node_to_comm,
    const Weight* __restrict__ comm_degrees,
    const bool* __restrict__ is_active,
    NodeID num_nodes,
    Weight total_weight,
    Weight resolution,
    Weight min_gain,
    Community* best_comm,
    Weight* best_delta_Q,
    bool* has_improvement
) {
    NodeID node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes || !is_active[node]) return;
    
    Community current_comm = node_to_comm[node];
    Weight node_deg = node_degrees[node];
    
    // Count weights to each neighboring community
    EdgeID start = row_ptr[node];
    EdgeID end = row_ptr[node + 1];
    
    // First pass: compute k_i_in for current community
    Weight k_i_in_current = 0.0f;
    for (EdgeID e = start; e < end; ++e) {
        NodeID neighbor = col_idx[e];
        if (node_to_comm[neighbor] == current_comm) {
            k_i_in_current += weights[e];
        }
    }
    
    Weight sigma_tot_current = comm_degrees[current_comm];
    
    // Find best neighboring community
    Community best = current_comm;
    Weight best_delta = 0.0f;
    
    // Track visited communities to avoid recomputation
    const int MAX_NEIGHBOR_COMMS = 32;
    Community visited_comms[MAX_NEIGHBOR_COMMS];
    int num_visited = 0;
    
    for (EdgeID e = start; e < end; ++e) {
        NodeID neighbor = col_idx[e];
        Community neighbor_comm = node_to_comm[neighbor];
        
        if (neighbor_comm == current_comm) continue;
        
        // Check if already evaluated this community
        bool already_checked = false;
        for (int i = 0; i < num_visited; ++i) {
            if (visited_comms[i] == neighbor_comm) {
                already_checked = true;
                break;
            }
        }
        if (already_checked) continue;
        
        // Mark as visited
        if (num_visited < MAX_NEIGHBOR_COMMS) {
            visited_comms[num_visited++] = neighbor_comm;
        }
        
        // Compute k_i_in for this community
        Weight k_i_in_new = 0.0f;
        for (EdgeID e2 = start; e2 < end; ++e2) {
            if (node_to_comm[col_idx[e2]] == neighbor_comm) {
                k_i_in_new += weights[e2];
            }
        }
        
        Weight sigma_tot_new = comm_degrees[neighbor_comm];
        
        // Compute modularity delta (Improved Louvain formula)
        Weight delta_Q_in = k_i_in_new - resolution * sigma_tot_new * node_deg / total_weight;
        Weight delta_Q_out = k_i_in_current - resolution * (sigma_tot_current - node_deg) * node_deg / total_weight;
        Weight delta_Q = (delta_Q_in - delta_Q_out) / total_weight;
        
        if (delta_Q > best_delta) {
            best_delta = delta_Q;
            best = neighbor_comm;
        }
    }
    
    best_comm[node] = best;
    best_delta_Q[node] = best_delta;
    
    if (best != current_comm && best_delta > min_gain) {
        has_improvement[node] = true;
    } else {
        has_improvement[node] = false;
    }
}

// Kernel: Apply moves and mark nodes that moved
__global__ void apply_moves_kernel(
    const Community* __restrict__ best_comm,
    const bool* __restrict__ has_improvement,
    const bool* __restrict__ is_active,
    const Weight* __restrict__ node_degrees,
    NodeID num_nodes,
    Community* node_to_comm,
    Weight* comm_degrees,
    bool* moved
) {
    NodeID node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes || !is_active[node]) {
        if (node < num_nodes) moved[node] = false;
        return;
    }
    
    if (!has_improvement[node]) {
        moved[node] = false;
        return;
    }
    
    Community old_comm = node_to_comm[node];
    Community new_comm = best_comm[node];
    Weight node_deg = node_degrees[node];
    
    // Apply move
    node_to_comm[node] = new_comm;
    
    // Update community degrees atomically
    atomicAdd(&comm_degrees[new_comm], node_deg);
    atomicAdd(&comm_degrees[old_comm], -node_deg);
    
    moved[node] = true;
}

// Kernel: Mark nodes as active for next iteration (nodes that moved + their neighbors)
__global__ void update_active_nodes_kernel(
    const EdgeID* __restrict__ row_ptr,
    const NodeID* __restrict__ col_idx,
    const bool* __restrict__ moved,
    NodeID num_nodes,
    bool* next_active
) {
    NodeID node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;
    
    // If this node moved, it's active
    if (moved[node]) {
        next_active[node] = true;
        
        // Mark all neighbors as active
        EdgeID start = row_ptr[node];
        EdgeID end = row_ptr[node + 1];
        for (EdgeID e = start; e < end; ++e) {
            NodeID neighbor = col_idx[e];
            next_active[neighbor] = true;
        }
    }
}

// Kernel: Compute modularity
__global__ void compute_modularity_kernel(
    const EdgeID* __restrict__ row_ptr,
    const NodeID* __restrict__ col_idx,
    const Weight* __restrict__ weights,
    const Community* __restrict__ node_to_comm,
    const Weight* __restrict__ node_degrees,
    NodeID num_nodes,
    Weight total_weight,
    Weight resolution,
    Weight* internal_edges,
    Weight* total_degrees
) {
    NodeID node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;
    
    Community node_comm = node_to_comm[node];
    Weight node_deg = node_degrees[node];
    
    // Add to community total degree
    atomicAdd(&total_degrees[node_comm], node_deg);
    
    // Count internal edges
    EdgeID start = row_ptr[node];
    EdgeID end = row_ptr[node + 1];
    
    for (EdgeID e = start; e < end; ++e) {
        NodeID neighbor = col_idx[e];
        Community neighbor_comm = node_to_comm[neighbor];
        
        if (node_comm == neighbor_comm && node <= neighbor) {
            Weight w = weights[e];
            atomicAdd(&internal_edges[node_comm], w);
        }
    }
}

class CUDAImprovedLouvain {
public:
    DeviceGraph d_graph_;
    thrust::device_vector<Community> d_node_to_comm_;
    thrust::device_vector<Weight> d_comm_degrees_;
    thrust::device_vector<Community> d_best_comm_;
    thrust::device_vector<Weight> d_best_delta_Q_;
    thrust::device_vector<bool> d_has_improvement_;
    thrust::device_vector<bool> d_moved_;
    thrust::device_vector<bool> d_active_;
    thrust::device_vector<bool> d_is_tree_node_;
    
    double resolution_;
    double min_modularity_gain_;
    int block_size_;
    
public:
    struct Result {
        std::vector<Community> communities;
        double modularity;
        uint32_t num_communities;
        uint32_t num_iterations;
        uint32_t tree_nodes_detected;
    };
    
    CUDAImprovedLouvain(const Graph& graph, 
                        double resolution = 1.0,
                        double min_modularity_gain = 1e-7,
                        int block_size = 256)
        : resolution_(resolution)
        , min_modularity_gain_(min_modularity_gain)
        , block_size_(block_size) {
        
        upload_graph_to_device(graph);
        
        // Detect and split tree structures
        uint32_t tree_nodes = detect_and_split_trees();
        printf("Detected %u tree nodes\n", tree_nodes);
        
        initialize_communities(graph.num_nodes());
    }
    
    ~CUDAImprovedLouvain() {
        CUDA_CHECK(cudaFree(d_graph_.d_row_ptr));
        CUDA_CHECK(cudaFree(d_graph_.d_col_idx));
        CUDA_CHECK(cudaFree(d_graph_.d_weights));
        CUDA_CHECK(cudaFree(d_graph_.d_node_degrees));
    }
    
    Result detect_communities() {
        Result result;
        
        printf("Starting CUDA Improved Louvain algorithm...\n");
        
        uint32_t iteration = 0;
        double prev_modularity = -1.0;
        
        while (iteration < 100) {
            // Count active nodes
            uint32_t num_active = thrust::reduce(d_active_.begin(), d_active_.end(), 0u);
            printf("Iteration %u: Active nodes = %u\n", iteration, num_active);
            
            if (num_active == 0) {
                printf("No active nodes remaining, stopping\n");
                break;
            }
            
            // Phase 1: Dynamic iteration on active nodes only
            bool improvement = phase1_dynamic_iteration();
            
            if (!improvement) {
                printf("Iteration %u: No improvement, stopping\n", iteration);
                break;
            }
            
            double current_modularity = compute_modularity();
            uint32_t num_comms = count_communities();
            
            printf("Iteration %u: Modularity = %.6f, Communities = %u\n",
                   iteration, current_modularity, num_comms);
            
            if (iteration > 0 && current_modularity - prev_modularity < min_modularity_gain_) {
                printf("Iteration %u: Modularity gain < threshold, stopping\n", iteration);
                break;
            }
            
            prev_modularity = current_modularity;
            iteration++;
        }
        
        // Download results
        thrust::host_vector<Community> h_communities = d_node_to_comm_;
        thrust::host_vector<bool> h_tree_nodes = d_is_tree_node_;
        
        result.communities.assign(h_communities.begin(), h_communities.end());
        result.modularity = compute_modularity();
        result.num_communities = count_communities();
        result.num_iterations = iteration;
        result.tree_nodes_detected = thrust::reduce(h_tree_nodes.begin(), h_tree_nodes.end(), 0u);
        
        return result;
    }
    
public:
    void upload_graph_to_device(const Graph& graph) {
        d_graph_.num_nodes = graph.num_nodes();
        d_graph_.num_edges = graph.num_edges();
        d_graph_.total_weight = graph.total_weight();
        
        // Prepare host data
        std::vector<EdgeID> h_row_ptr(graph.num_nodes() + 1);
        std::vector<NodeID> h_col_idx;
        std::vector<Weight> h_weights;
        std::vector<Weight> h_degrees(graph.num_nodes());
        
        EdgeID offset = 0;
        for (NodeID i = 0; i < graph.num_nodes(); ++i) {
            h_row_ptr[i] = offset;
            h_degrees[i] = graph.degree(i);
            
            for (auto edge_idx = graph.neighbor_start(i);
                 edge_idx < graph.neighbor_end(i); ++edge_idx) {
                h_col_idx.push_back(graph.neighbor(edge_idx));
                h_weights.push_back(graph.weight(edge_idx));
                offset++;
            }
        }
        h_row_ptr[graph.num_nodes()] = offset;
        
        // Allocate and copy to device
        size_t row_ptr_size = h_row_ptr.size() * sizeof(EdgeID);
        size_t col_idx_size = h_col_idx.size() * sizeof(NodeID);
        size_t weights_size = h_weights.size() * sizeof(Weight);
        size_t degrees_size = h_degrees.size() * sizeof(Weight);
        
        CUDA_CHECK(cudaMalloc(&d_graph_.d_row_ptr, row_ptr_size));
        CUDA_CHECK(cudaMalloc(&d_graph_.d_col_idx, col_idx_size));
        CUDA_CHECK(cudaMalloc(&d_graph_.d_weights, weights_size));
        CUDA_CHECK(cudaMalloc(&d_graph_.d_node_degrees, degrees_size));
        
        CUDA_CHECK(cudaMemcpy(d_graph_.d_row_ptr, h_row_ptr.data(), 
                             row_ptr_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_graph_.d_col_idx, h_col_idx.data(),
                             col_idx_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_graph_.d_weights, h_weights.data(),
                             weights_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_graph_.d_node_degrees, h_degrees.data(),
                             degrees_size, cudaMemcpyHostToDevice));
    }
    
    uint32_t detect_and_split_trees() {
        NodeID num_nodes = d_graph_.num_nodes;
        int num_blocks = (num_nodes + block_size_ - 1) / block_size_;
        
        d_is_tree_node_.resize(num_nodes, false);
        
        thrust::device_vector<int> d_component_id(num_nodes, -1);
        thrust::device_vector<int> d_visited(num_nodes, 0);
        thrust::device_vector<NodeID> d_parent(num_nodes, num_nodes);
        thrust::device_vector<int> d_current_frontier(num_nodes, 0);
        thrust::device_vector<int> d_next_frontier(num_nodes, 0);
        
        int comp_id = 0;
        
        // Find connected components and check if they're trees
        for (NodeID start = 0; start < num_nodes; ++start) {
            int visited = 0;
            cudaMemcpy(&visited, thrust::raw_pointer_cast(d_visited.data()) + start,
                      sizeof(int), cudaMemcpyDeviceToHost);
            
            if (visited) continue;
            
            // Initialize BFS from this node
            thrust::fill(d_current_frontier.begin(), d_current_frontier.end(), 0);
            thrust::fill(d_next_frontier.begin(), d_next_frontier.end(), 0);
            d_current_frontier[start] = 1;
            d_visited[start] = 1;
            d_component_id[start] = comp_id;
            
            uint32_t component_size = 1;
            
            // BFS to find component
            bool has_frontier = true;
            while (has_frontier) {
                bfs_step_kernel<<<num_blocks, block_size_>>>(
                    d_graph_.d_row_ptr,
                    d_graph_.d_col_idx,
                    num_nodes,
                    thrust::raw_pointer_cast(d_current_frontier.data()),
                    thrust::raw_pointer_cast(d_next_frontier.data()),
                    thrust::raw_pointer_cast(d_visited.data()),
                    thrust::raw_pointer_cast(d_parent.data()),
                    thrust::raw_pointer_cast(d_component_id.data()),
                    comp_id
                );
                CUDA_CHECK(cudaDeviceSynchronize());
                
                uint32_t frontier_size = thrust::reduce(d_next_frontier.begin(),
                                                        d_next_frontier.end(), 0);
                component_size += frontier_size;
                
                if (frontier_size == 0) {
                    has_frontier = false;
                } else {
                    d_current_frontier = d_next_frontier;
                    thrust::fill(d_next_frontier.begin(), d_next_frontier.end(), 0);
                }
            }
            
            // Count edges in component
            thrust::device_vector<unsigned long long> d_edge_count(1, 0ULL);
            count_component_edges_kernel<<<num_blocks, block_size_>>>(
                d_graph_.d_row_ptr,
                d_graph_.d_col_idx,
                thrust::raw_pointer_cast(d_component_id.data()),
                num_nodes,
                comp_id,
                thrust::raw_pointer_cast(d_edge_count.data())
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            unsigned long long edge_count;
            cudaMemcpy(&edge_count, thrust::raw_pointer_cast(d_edge_count.data()),
                      sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            
            // Check if it's a tree: |E| = |V| - 1 and size > 2
            if (component_size > 2 && edge_count == component_size - 1) {
                // Mark all nodes in this component as tree nodes
                thrust::transform(
                    thrust::make_counting_iterator<NodeID>(0),
                    thrust::make_counting_iterator<NodeID>(num_nodes),
                    d_component_id.begin(),
                    d_is_tree_node_.begin(),
                    [comp_id] __device__ (NodeID node, int cid) {
                        return cid == comp_id;
                    }
                );
            }
            
            comp_id++;
        }
        
        return thrust::reduce(d_is_tree_node_.begin(), d_is_tree_node_.end(), 0u);
    }
    
    void initialize_communities(NodeID num_nodes) {
        d_node_to_comm_.resize(num_nodes);
        d_comm_degrees_.resize(num_nodes);
        d_best_comm_.resize(num_nodes);
        d_best_delta_Q_.resize(num_nodes);
        d_has_improvement_.resize(num_nodes);
        d_moved_.resize(num_nodes);
        d_active_.resize(num_nodes);
        
        // Initialize each node to its own community
        thrust::sequence(d_node_to_comm_.begin(), d_node_to_comm_.end());
        
        // Copy node degrees to community degrees
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_comm_degrees_.data()),
                             d_graph_.d_node_degrees,
                             num_nodes * sizeof(Weight),
                             cudaMemcpyDeviceToDevice));
        
        // Mark non-tree nodes as active initially
        thrust::transform(
            d_is_tree_node_.begin(),
            d_is_tree_node_.end(),
            d_active_.begin(),
            thrust::logical_not<bool>()
        );
    }
    
    bool phase1_dynamic_iteration() {
        NodeID num_nodes = d_graph_.num_nodes;
        int num_blocks = (num_nodes + block_size_ - 1) / block_size_;
        
        bool global_improvement = false;
        
        // Dynamic iteration: continue while there are active nodes
        while (true) {
            uint32_t num_active = thrust::reduce(d_active_.begin(), d_active_.end(), 0u);
            
            if (num_active == 0) {
                break;
            }
            
            // Reset flags
            thrust::fill(d_has_improvement_.begin(), d_has_improvement_.end(), false);
            thrust::fill(d_moved_.begin(), d_moved_.end(), false);
            
            // Compute best moves for active nodes only
            compute_best_moves_active_kernel<<<num_blocks, block_size_>>>(
                d_graph_.d_row_ptr,
                d_graph_.d_col_idx,
                d_graph_.d_weights,
                d_graph_.d_node_degrees,
                thrust::raw_pointer_cast(d_node_to_comm_.data()),
                thrust::raw_pointer_cast(d_comm_degrees_.data()),
                thrust::raw_pointer_cast(d_active_.data()),
                num_nodes,
                d_graph_.total_weight,
                resolution_,
                min_modularity_gain_,
                thrust::raw_pointer_cast(d_best_comm_.data()),
                thrust::raw_pointer_cast(d_best_delta_Q_.data()),
                thrust::raw_pointer_cast(d_has_improvement_.data())
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Check if any improvements found
            bool any_improvement = thrust::reduce(d_has_improvement_.begin(),
                                                 d_has_improvement_.end(),
                                                 false,
                                                 thrust::logical_or<bool>());
            
            if (!any_improvement) {
                break;
            }
            
            global_improvement = true;
            
            // Apply moves
            apply_moves_kernel<<<num_blocks, block_size_>>>(
                thrust::raw_pointer_cast(d_best_comm_.data()),
                thrust::raw_pointer_cast(d_has_improvement_.data()),
                thrust::raw_pointer_cast(d_active_.data()),
                d_graph_.d_node_degrees,
                num_nodes,
                thrust::raw_pointer_cast(d_node_to_comm_.data()),
                thrust::raw_pointer_cast(d_comm_degrees_.data()),
                thrust::raw_pointer_cast(d_moved_.data())
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Update active set: nodes that moved + their neighbors
            thrust::fill(d_active_.begin(), d_active_.end(), false);
            
            update_active_nodes_kernel<<<num_blocks, block_size_>>>(
                d_graph_.d_row_ptr,
                d_graph_.d_col_idx,
                thrust::raw_pointer_cast(d_moved_.data()),
                num_nodes,
                thrust::raw_pointer_cast(d_active_.data())
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Exclude tree nodes from active set
            thrust::transform(
                d_active_.begin(),
                d_active_.end(),
                d_is_tree_node_.begin(),
                d_active_.begin(),
                [] __device__ (bool active, bool is_tree) {
                    return active && !is_tree;
                }
            );
        }
        
        return global_improvement;
    }
    
    double compute_modularity() {
        NodeID num_nodes = d_graph_.num_nodes;
        int num_blocks = (num_nodes + block_size_ - 1) / block_size_;
        
        thrust::device_vector<Weight> d_internal_edges(num_nodes, 0.0f);
        thrust::device_vector<Weight> d_total_degrees(num_nodes, 0.0f);
        
        compute_modularity_kernel<<<num_blocks, block_size_>>>(
            d_graph_.d_row_ptr,
            d_graph_.d_col_idx,
            d_graph_.d_weights,
            thrust::raw_pointer_cast(d_node_to_comm_.data()),
            d_graph_.d_node_degrees,
            num_nodes,
            d_graph_.total_weight,
            resolution_,
            thrust::raw_pointer_cast(d_internal_edges.data()),
            thrust::raw_pointer_cast(d_total_degrees.data())
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Compute modularity on host
        thrust::host_vector<Weight> h_internal = d_internal_edges;
        thrust::host_vector<Weight> h_degrees = d_total_degrees;
        
        double Q = 0.0;
        for (size_t i = 0; i < h_internal.size(); ++i) {
            if (h_degrees[i] > 0) {
                double l_c = h_internal[i];
                double d_c = h_degrees[i];
                Q += l_c / d_graph_.total_weight - 
                     resolution_ * (d_c / (2.0 * d_graph_.total_weight)) * 
                                  (d_c / (2.0 * d_graph_.total_weight));
            }
        }
        
        return Q;
    }
    
    uint32_t count_communities() {
        thrust::device_vector<Community> sorted_comms = d_node_to_comm_;
        thrust::sort(sorted_comms.begin(), sorted_comms.end());
        auto new_end = thrust::unique(sorted_comms.begin(), sorted_comms.end());
        return thrust::distance(sorted_comms.begin(), new_end);
    }
};

} // namespace cuda_improved_louvain

#endif // CUDA_IMPROVED_LOUVAIN_CUH