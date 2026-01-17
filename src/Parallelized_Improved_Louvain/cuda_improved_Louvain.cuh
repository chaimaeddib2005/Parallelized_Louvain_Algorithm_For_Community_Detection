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
#include <thrust/execution_policy.h>

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

// Kernel: Initialize community weights (sum of edge weights within community)
__global__ void initialize_community_weights_kernel(
    const EdgeID* __restrict__ row_ptr,
    const NodeID* __restrict__ col_idx,
    const Weight* __restrict__ weights,
    const Community* __restrict__ node_to_comm,
    NodeID num_nodes,
    Weight* comm_internal_weight
) {
    NodeID u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;
    
    Community cu = node_to_comm[u];
    EdgeID start = row_ptr[u];
    EdgeID end = row_ptr[u + 1];
    
    for (EdgeID e = start; e < end; ++e) {
        NodeID v = col_idx[e];
        Community cv = node_to_comm[v];
        if (cu == cv) {
            Weight w = weights[e];
            atomicAdd(&comm_internal_weight[cu], w / 2.0f); // Divide by 2 to avoid double counting
        }
    }
}

// Kernel: Compute community degrees (total degree of all nodes in community)
__global__ void compute_community_degrees_kernel(
    const Weight* __restrict__ node_degrees,
    const Community* __restrict__ node_to_comm,
    NodeID num_nodes,
    Weight* comm_degrees
) {
    NodeID u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;
    
    Community cu = node_to_comm[u];
    Weight deg = node_degrees[u];
    atomicAdd(&comm_degrees[cu], deg);
}

// Kernel: Compute best community for each node
__global__ void compute_best_community_kernel(
    const EdgeID* __restrict__ row_ptr,
    const NodeID* __restrict__ col_idx,
    const Weight* __restrict__ weights,
    const Weight* __restrict__ node_degrees,
    const Community* __restrict__ node_to_comm,
    const Weight* __restrict__ comm_degrees,
    NodeID num_nodes,
    Weight total_weight,
    Weight resolution,
    Community* best_comm,
    Weight* best_delta_Q
) {
    NodeID u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;
    
    Community cu = node_to_comm[u];
    Weight ku = node_degrees[u];
    Weight m2 = total_weight;
    
    EdgeID start = row_ptr[u];
    EdgeID end = row_ptr[u + 1];
    
    // Compute weight to current community (excluding self-loops counted once)
    Weight k_u_in_cu = 0.0f;
    for (EdgeID e = start; e < end; ++e) {
        NodeID v = col_idx[e];
        if (node_to_comm[v] == cu) {
            k_u_in_cu += weights[e];
        }
    }
    
    // Try to find best neighboring community
    Community best = cu;
    Weight best_delta = 0.0f;
    
    // Use a simple array to track unique neighboring communities
    const int MAX_NEIGHBORS = 32;
    Community neighbor_comms[MAX_NEIGHBORS];
    Weight neighbor_weights[MAX_NEIGHBORS];
    int num_neighbors = 0;
    
    // Aggregate weights to each neighboring community
    for (EdgeID e = start; e < end; ++e) {
        NodeID v = col_idx[e];
        Community cv = node_to_comm[v];
        Weight w = weights[e];
        
        // Find or add this community
        bool found = false;
        for (int i = 0; i < num_neighbors; ++i) {
            if (neighbor_comms[i] == cv) {
                neighbor_weights[i] += w;
                found = true;
                break;
            }
        }
        if (!found && num_neighbors < MAX_NEIGHBORS) {
            neighbor_comms[num_neighbors] = cv;
            neighbor_weights[num_neighbors] = w;
            num_neighbors++;
        }
    }
    
    // Evaluate each neighboring community
    for (int i = 0; i < num_neighbors; ++i) {
        Community cv = neighbor_comms[i];
        Weight k_u_in_cv = neighbor_weights[i];
        
        if (cv == cu) continue; // Skip current community
        
        Weight sigma_tot_cv = comm_degrees[cv];
        Weight sigma_tot_cu = comm_degrees[cu];
        
        // Delta Q when moving from cu to cv
        // Based on standard Louvain formula
        Weight delta = (k_u_in_cv - k_u_in_cu) / m2 
                     - resolution * ku * (sigma_tot_cv - sigma_tot_cu + ku) / (m2 * m2);
        
        if (delta > best_delta) {
            best_delta = delta;
            best = cv;
        }
    }
    
    best_comm[u] = best;
    best_delta_Q[u] = best_delta;
}

// Kernel: Apply moves (single pass)
__global__ void apply_moves_kernel(
    const Community* __restrict__ best_comm,
    const Weight* __restrict__ best_delta_Q,
    NodeID num_nodes,
    Weight threshold,
    Community* node_to_comm,
    int* changed
) {
    NodeID u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;
    
    Community new_comm = best_comm[u];
    Community old_comm = node_to_comm[u];
    Weight delta = best_delta_Q[u];
    
    if (new_comm != old_comm && delta > threshold) {
        node_to_comm[u] = new_comm;
        atomicAdd(changed, 1);
    }
}

// Kernel: Compute modularity
__global__ void compute_modularity_kernel(
    const EdgeID* __restrict__ row_ptr,
    const NodeID* __restrict__ col_idx,
    const Weight* __restrict__ weights,
    const Community* __restrict__ node_to_comm,
    const Weight* __restrict__ comm_degrees,
    NodeID num_nodes,
    Weight total_weight,
    Weight resolution,
    Weight* partial_Q
) {
    NodeID u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;
    
    Community cu = node_to_comm[u];
    EdgeID start = row_ptr[u];
    EdgeID end = row_ptr[u + 1];
    
    Weight internal = 0.0f;
    for (EdgeID e = start; e < end; ++e) {
        NodeID v = col_idx[e];
        if (node_to_comm[v] == cu && u <= v) { // Count each edge once
            internal += weights[e];
        }
    }
    
    atomicAdd(partial_Q, internal);
}

class CUDAImprovedLouvain {
private:
    DeviceGraph d_graph_;
    thrust::device_vector<Community> d_node_to_comm_;
    thrust::device_vector<Weight> d_comm_degrees_;
    thrust::device_vector<Weight> d_comm_internal_;
    thrust::device_vector<Community> d_best_comm_;
    thrust::device_vector<Weight> d_best_delta_Q_;
    
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
        
        printf("Starting CUDA Louvain algorithm...\n");
        printf("Graph: %u nodes, %lu edges, total weight: %.2f\n",
               d_graph_.num_nodes, (unsigned long)d_graph_.num_edges, d_graph_.total_weight);
        
        uint32_t iteration = 0;
        double prev_modularity = -1.0;
        const uint32_t MAX_ITERATIONS = 100;
        
        while (iteration < MAX_ITERATIONS) {
            // Update community information
            update_community_info();
            
            // Compute best moves for all nodes
            int num_blocks = (d_graph_.num_nodes + block_size_ - 1) / block_size_;
            
            compute_best_community_kernel<<<num_blocks, block_size_>>>(
                d_graph_.d_row_ptr,
                d_graph_.d_col_idx,
                d_graph_.d_weights,
                d_graph_.d_node_degrees,
                thrust::raw_pointer_cast(d_node_to_comm_.data()),
                thrust::raw_pointer_cast(d_comm_degrees_.data()),
                d_graph_.num_nodes,
                d_graph_.total_weight,
                resolution_,
                thrust::raw_pointer_cast(d_best_comm_.data()),
                thrust::raw_pointer_cast(d_best_delta_Q_.data())
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Apply moves
            thrust::device_vector<int> d_changed(1, 0);
            apply_moves_kernel<<<num_blocks, block_size_>>>(
                thrust::raw_pointer_cast(d_best_comm_.data()),
                thrust::raw_pointer_cast(d_best_delta_Q_.data()),
                d_graph_.num_nodes,
                min_modularity_gain_,
                thrust::raw_pointer_cast(d_node_to_comm_.data()),
                thrust::raw_pointer_cast(d_changed.data())
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            int num_changed = d_changed[0];
            printf("Iteration %u: %d nodes changed community\n", iteration, num_changed);
            
            if (num_changed == 0) {
                printf("No changes, converged.\n");
                break;
            }
            
            // Compute current modularity
            double current_modularity = compute_modularity();
            uint32_t num_comms = count_communities();
            
            printf("Iteration %u: Modularity = %.6f, Communities = %u\n",
                   iteration, current_modularity, num_comms);
            
            if (iteration > 0) {
                double mod_gain = current_modularity - prev_modularity;
                printf("  Modularity gain: %.6e\n", mod_gain);
                
                if (mod_gain < min_modularity_gain_) {
                    printf("Modularity gain below threshold, stopping\n");
                    break;
                }
            }
            
            prev_modularity = current_modularity;
            iteration++;
        }
        
        printf("\nFinal results:\n");
        
        // Download results
        thrust::host_vector<Community> h_communities = d_node_to_comm_;
        
        result.communities.assign(h_communities.begin(), h_communities.end());
        result.modularity = compute_modularity();
        result.num_communities = count_communities();
        result.num_iterations = iteration;
        result.tree_nodes_detected = 0;
        
        printf("  Final modularity: %.6f\n", result.modularity);
        printf("  Final communities: %u\n", result.num_communities);
        printf("  Total iterations: %u\n", result.num_iterations);
        
        return result;
    }
    
private:
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
        CUDA_CHECK(cudaMalloc(&d_graph_.d_row_ptr, h_row_ptr.size() * sizeof(EdgeID)));
        CUDA_CHECK(cudaMalloc(&d_graph_.d_col_idx, h_col_idx.size() * sizeof(NodeID)));
        CUDA_CHECK(cudaMalloc(&d_graph_.d_weights, h_weights.size() * sizeof(Weight)));
        CUDA_CHECK(cudaMalloc(&d_graph_.d_node_degrees, h_degrees.size() * sizeof(Weight)));
        
        CUDA_CHECK(cudaMemcpy(d_graph_.d_row_ptr, h_row_ptr.data(), 
                             h_row_ptr.size() * sizeof(EdgeID), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_graph_.d_col_idx, h_col_idx.data(),
                             h_col_idx.size() * sizeof(NodeID), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_graph_.d_weights, h_weights.data(),
                             h_weights.size() * sizeof(Weight), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_graph_.d_node_degrees, h_degrees.data(),
                             h_degrees.size() * sizeof(Weight), cudaMemcpyHostToDevice));
    }
    
    void initialize_communities(NodeID num_nodes) {
        d_node_to_comm_.resize(num_nodes);
        d_comm_degrees_.resize(num_nodes);
        d_comm_internal_.resize(num_nodes);
        d_best_comm_.resize(num_nodes);
        d_best_delta_Q_.resize(num_nodes);
        
        // Initialize each node to its own community
        thrust::sequence(d_node_to_comm_.begin(), d_node_to_comm_.end());
    }
    
    void update_community_info() {
        NodeID num_nodes = d_graph_.num_nodes;
        int num_blocks = (num_nodes + block_size_ - 1) / block_size_;
        
        // Reset community info
        thrust::fill(d_comm_degrees_.begin(), d_comm_degrees_.end(), 0.0f);
        thrust::fill(d_comm_internal_.begin(), d_comm_internal_.end(), 0.0f);
        
        // Compute community degrees
        compute_community_degrees_kernel<<<num_blocks, block_size_>>>(
            d_graph_.d_node_degrees,
            thrust::raw_pointer_cast(d_node_to_comm_.data()),
            num_nodes,
            thrust::raw_pointer_cast(d_comm_degrees_.data())
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Compute community internal weights
        initialize_community_weights_kernel<<<num_blocks, block_size_>>>(
            d_graph_.d_row_ptr,
            d_graph_.d_col_idx,
            d_graph_.d_weights,
            thrust::raw_pointer_cast(d_node_to_comm_.data()),
            num_nodes,
            thrust::raw_pointer_cast(d_comm_internal_.data())
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    double compute_modularity() {
        NodeID num_nodes = d_graph_.num_nodes;
        int num_blocks = (num_nodes + block_size_ - 1) / block_size_;
        
        thrust::device_vector<Weight> d_partial_Q(1, 0.0f);
        
        compute_modularity_kernel<<<num_blocks, block_size_>>>(
            d_graph_.d_row_ptr,
            d_graph_.d_col_idx,
            d_graph_.d_weights,
            thrust::raw_pointer_cast(d_node_to_comm_.data()),
            thrust::raw_pointer_cast(d_comm_degrees_.data()),
            num_nodes,
            d_graph_.total_weight,
            resolution_,
            thrust::raw_pointer_cast(d_partial_Q.data())
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        Weight sum_in = d_partial_Q[0];
        
        // Compute sum of (degree_c / (2*m))^2 for all communities
        thrust::host_vector<Weight> h_comm_degrees = d_comm_degrees_;
        double sum_tot = 0.0;
        for (size_t c = 0; c < h_comm_degrees.size(); ++c) {
            if (h_comm_degrees[c] > 0) {
                double deg_c = h_comm_degrees[c];
                sum_tot += (deg_c / d_graph_.total_weight) * (deg_c / d_graph_.total_weight);
            }
        }
        
        double Q = sum_in / d_graph_.total_weight - resolution_ * sum_tot;
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