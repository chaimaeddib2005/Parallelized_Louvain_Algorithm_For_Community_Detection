#include <iostream>
#include<unordered_map>
#include "improved_louvain_cpu.hpp"
#include "classic_Louvain_cpu.hpp"
#include "cuda_Louvain.cuh"
#include "cuda_improved_Louvain.cuh"
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
    printf("Running Improved Fast Louvain Algorithm\n");
    printf("========================================\n\n");
    
    // Run Louvain
    MultiLevelLouvain louvain1(1.0, 1e-7, 42);
    auto result1 = louvain1.detect_communities(graph);
    
    printf("\n========================================\n");
    printf("Final Results:\n");
    printf("  Communities: %u\n", result1.num_communities);
    printf("  Modularity: %.6f\n", result1.modularity);
    printf("  Levels: %u\n", result1.num_levels);
    printf("========================================\n\n");
    
    // Group nodes by community
    std::unordered_map<ImprovedLouvain::Community, std::vector<ImprovedLouvain::NodeID>> comm_groups1;
    for (ImprovedLouvain::NodeID node = 0; node < result1.communities.size(); ++node) {
        comm_groups1[result1.communities[node]].push_back(node);
    }
    
    printf("Community assignments:\n");
    for (auto& [comm_id, nodes] : comm_groups1) {
        std::sort(nodes.begin(), nodes.end());
        printf("  Community %u: ", comm_id);
        for (size_t i = 0; i < nodes.size(); ++i) {
            printf("%u", nodes[i]);
            if (i < nodes.size() - 1) printf(", ");
        }
        printf("\n");
    }
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
    printf("\n========================================\n");
    printf("Running Parallelized Louvain Algorithm\n");
    printf("========================================\n\n");
    cuda_louvain::CUDALouvain cuda_louvain(graph);
    auto result2 = cuda_louvain.detect_communities();
    
    printf("Found %u communities with modularity %.6f\n",
           result2.num_communities, result2.modularity);
    return 0;
}
