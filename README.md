#### Louvain Community Detection - CPU & GPU Implementations
A comprehensive implementation of Louvain community detection algorithms with both sequential (CPU) and parallel (CUDA GPU) variants, including the improved Louvain algorithm with optimizations.
### Overview
This repository contains four implementations of the Louvain algorithm for community detection in graphs:

Classic Louvain (CPU) - Sequential implementation
Parallel Louvain (CUDA) - GPU-accelerated version
Improved Louvain (CPU) - Enhanced algorithm with optimizations
Parallel Improved Louvain (CUDA) - GPU-accelerated improved version

### Features
Classic Louvain

Standard modularity optimization
Two-phase approach: node reassignment and network aggregation
Best for: Small to medium graphs, baseline comparison

Parallel Louvain (CUDA)

GPU acceleration using CUDA
Parallel node evaluation and community assignment
10-100x speedup on large graphs
Best for: Large graphs (>10K nodes)

Improved Louvain
Based on Zhang et al. (2021) improvements:

Dynamic iteration: Only processes active nodes
Tree structure detection: Identifies and handles tree-like subgraphs
Early stopping: Modularity-based convergence criteria
Better quality and faster convergence

Parallel Improved Louvain (CUDA)
Combines GPU acceleration with algorithmic improvements:

Parallel tree detection using BFS
Active node tracking on GPU
Optimized memory access patterns
Best for: Very large graphs requiring both speed and quality