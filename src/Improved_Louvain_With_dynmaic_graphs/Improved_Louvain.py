"""
Improved Fast Louvain Algorithm for Community Detection
Based on: Zhang et al. (2021) "An Improved Louvain Algorithm for Community Detection"
DOI: 10.1155/2021/1485592

Key improvements over standard Louvain:
1. Dynamic iteration - only iterates over nodes that changed in previous iteration
2. Tree structure splitting - identifies and separates local tree structures
3. Early stopping based on modularity gain threshold
"""

import networkx as nx
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple


class ImprovedFastLouvain:
    def __init__(self, graph: nx.Graph, resolution: float = 1.0, 
                 min_modularity_gain: float = 1e-7):
        """
        Initialize the Improved Fast Louvain algorithm.
        
        Args:
            graph: NetworkX graph
            resolution: Resolution parameter for modularity
            min_modularity_gain: Minimum modularity gain to continue iteration
        """
        self.graph = graph.copy()
        self.resolution = resolution
        self.min_modularity_gain = min_modularity_gain
        self.m = graph.size(weight='weight')  # Total edge weight
        if self.m == 0:
            self.m = 1  # Avoid division by zero
            
        # Precompute node degrees
        self.node_degrees = {node: self.graph.degree(node, weight='weight') 
                            for node in self.graph.nodes()}
        
    def detect_tree_structures(self) -> List[Set[int]]:
        """
        Identify local tree structures in the network.
        A tree structure is a connected component with no cycles.
        """
        trees = []
        visited = set()
        
        for node in self.graph.nodes():
            if node in visited:
                continue
                
            # BFS to find connected component
            component = set()
            queue = deque([node])
            component.add(node)
            visited.add(node)
            
            while queue:
                current = queue.popleft()
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in component:
                        component.add(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Count actual edges in component
            edges_count = 0
            for n in component:
                for neighbor in self.graph.neighbors(n):
                    if neighbor in component and neighbor > n:  # Count each edge once
                        edges_count += 1
            
            # Check if it's a tree: |E| = |V| - 1
            if len(component) > 2 and edges_count == len(component) - 1:
                trees.append(component)
        
        return trees
    
    def split_tree_structures(self) -> Tuple[nx.Graph, List[Dict[int, int]]]:
        """
        Split tree structures from the main graph.
        Returns the reduced graph and community assignments for trees.
        """
        trees = self.detect_tree_structures()
        tree_communities = []
        nodes_to_remove = set()
        
        for tree in trees:
            # Assign all nodes in tree to same community
            tree_comm = {node: min(tree) for node in tree}
            tree_communities.append(tree_comm)
            nodes_to_remove.update(tree)
        
        # Create reduced graph without tree structures
        reduced_graph = self.graph.copy()
        reduced_graph.remove_nodes_from(nodes_to_remove)
        
        return reduced_graph, tree_communities
    
    def compute_modularity(self, communities: Dict[int, int]) -> float:
        """Compute modularity of the partition."""
        Q = 0.0
        community_set = set(communities.values())
        
        for comm in community_set:
            nodes_in_comm = [n for n, c in communities.items() if c == comm]
            
            # Internal edges (count each edge once)
            l_c = 0
            for u in nodes_in_comm:
                for v in self.graph.neighbors(u):
                    if v in nodes_in_comm and communities[v] == comm:
                        l_c += self.graph[u][v].get('weight', 1)
            l_c = l_c / 2  # Each edge counted twice
            
            # Sum of degrees
            d_c = sum(self.graph.degree(n, weight='weight') for n in nodes_in_comm)
            
            Q += l_c / self.m - self.resolution * (d_c / (2 * self.m)) ** 2
        
        return Q
    
    def compute_modularity_gain(self, node: int, target_comm: int, 
                                communities: Dict[int, int],
                                k_i_in: float, sigma_tot: float) -> float:
        """
        Compute modularity gain if node moves to target_comm.
        
        Standard Louvain formula:
        ΔQ = [Σin + k_i,in / 2m - (Σtot + k_i)²/4m²] - [Σin/2m - Σtot²/4m² - (k_i/2m)²]
        
        Simplified:
        ΔQ = [k_i,in - Σtot * k_i / m] / m
        """
        current_comm = communities[node]
        
        if current_comm == target_comm:
            return 0.0
        
        # Node's degree
        k_i = self.node_degrees[node]
        
        # Modularity gain from joining target community
        delta_Q = (k_i_in - self.resolution * sigma_tot * k_i / self.m) / self.m
        
        return delta_Q
    
    def remove_from_community(self, node: int, communities: Dict[int, int],
                             k_i_in_old: float, sigma_tot_old: float) -> float:
        """
        Compute modularity change when removing node from its current community.
        """
        k_i = self.node_degrees[node]
        
        # Loss from leaving current community
        delta_Q = -(k_i_in_old - self.resolution * (sigma_tot_old - k_i) * k_i / self.m) / self.m
        
        return delta_Q
    
    def phase1_dynamic(self, graph: nx.Graph) -> Tuple[Dict[int, int], bool]:
        """
        Phase 1: Dynamic node movement with improved iteration.
        Only processes nodes that changed or have neighbors that changed.
        """
        # Initialize: each node in its own community
        communities = {node: node for node in graph.nodes()}
        
        # Precompute community degrees (sigma_tot for each community)
        community_degrees = {node: self.node_degrees.get(node, graph.degree(node, weight='weight')) 
                            for node in graph.nodes()}
        
        # Track which nodes need to be checked
        active_nodes = set(graph.nodes())
        global_improvement = False
        iteration = 0
        
        while active_nodes:
            iteration += 1
            local_improvement = False
            next_active = set()
            
            # Shuffle for random order
            nodes_to_process = list(active_nodes)
            np.random.shuffle(nodes_to_process)
            
            for node in nodes_to_process:
                # Current community
                current_comm = communities[node]
                node_degree = self.node_degrees.get(node, graph.degree(node, weight='weight'))
                
                # Compute weights to neighboring communities
                neighbor_comm_weights = defaultdict(float)
                for neighbor in graph.neighbors(node):
                    comm = communities[neighbor]
                    weight = graph[node][neighbor].get('weight', 1)
                    neighbor_comm_weights[comm] += weight
                
                # Weight to current community
                k_i_in_old = neighbor_comm_weights.get(current_comm, 0)
                sigma_tot_old = community_degrees[current_comm]
                
                # Remove node from current community (conceptually)
                best_comm = current_comm
                best_delta_Q = 0.0
                
                # Try each neighboring community
                for comm, k_i_in_new in neighbor_comm_weights.items():
                    if comm == current_comm:
                        continue
                    
                    sigma_tot_new = community_degrees[comm]
                    
                    # Gain from adding to new community
                    delta_Q_add = self.compute_modularity_gain(
                        node, comm, communities, k_i_in_new, sigma_tot_new
                    )
                    
                    # Loss from removing from old community
                    delta_Q_remove = self.remove_from_community(
                        node, communities, k_i_in_old, sigma_tot_old
                    )
                    
                    # Total change
                    total_delta_Q = delta_Q_add + delta_Q_remove
                    
                    if total_delta_Q > best_delta_Q:
                        best_delta_Q = total_delta_Q
                        best_comm = comm
                
                # Move node if improvement found
                if best_comm != current_comm and best_delta_Q > self.min_modularity_gain:
                    # Update community degrees
                    community_degrees[current_comm] -= node_degree
                    community_degrees[best_comm] += node_degree
                    
                    # Move node
                    communities[node] = best_comm
                    local_improvement = True
                    global_improvement = True
                    
                    # Mark this node and neighbors for next iteration
                    next_active.add(node)
                    for neighbor in graph.neighbors(node):
                        next_active.add(neighbor)
            
            active_nodes = next_active
            
            # Stop if no improvement in this iteration
            if not local_improvement:
                break
        
        return communities, global_improvement
    
    def phase2_aggregate(self, graph: nx.Graph, 
                        communities: Dict[int, int]) -> nx.Graph:
        """
        Phase 2: Create new graph where nodes are communities.
        """
        # Build new graph
        new_graph = nx.Graph()
        
        # Group nodes by community
        comm_nodes = defaultdict(list)
        for node, comm in communities.items():
            comm_nodes[comm].append(node)
        
        # Add nodes (communities)
        for comm in comm_nodes.keys():
            new_graph.add_node(comm)
        
        # Add edges between communities
        comm_edges = defaultdict(float)
        for node in graph.nodes():
            node_comm = communities[node]
            for neighbor in graph.neighbors(node):
                neighbor_comm = communities[neighbor]
                weight = graph[node][neighbor].get('weight', 1)
                
                if node_comm <= neighbor_comm:  # Avoid double counting
                    edge = (node_comm, neighbor_comm)
                    comm_edges[edge] += weight
        
        # Add edges to new graph
        for (u, v), weight in comm_edges.items():
            if new_graph.has_edge(u, v):
                new_graph[u][v]['weight'] += weight
            else:
                new_graph.add_edge(u, v, weight=weight)
        
        return new_graph
    
    def detect_communities(self) -> Dict[int, int]:
        """
        Main algorithm: Improved Fast Louvain community detection.
        """
        # Step 1: Split tree structures (optional - can disable for debugging)
        # For now, let's skip tree splitting to focus on main algorithm
        working_graph = self.graph.copy()
        tree_communities = []
        
        # Step 2: Iterative refinement
        current_graph = working_graph
        dendrogram = []
        
        prev_modularity = -1
        max_iterations = 100
        
        for iteration in range(max_iterations):
            # Phase 1: Dynamic node assignment
            communities, changed = self.phase1_dynamic(current_graph)
            
            if not changed:
                print(f"Stopped at iteration {iteration}: no changes")
                break
            
            # Check modularity improvement
            current_modularity = self.compute_modularity(communities)
            print(f"Iteration {iteration}: Modularity = {current_modularity:.4f}, Communities = {len(set(communities.values()))}")
            
            if abs(current_modularity - prev_modularity) < self.min_modularity_gain:
                print(f"Stopped at iteration {iteration}: modularity change < threshold")
                break
            
            prev_modularity = current_modularity
            dendrogram.append(communities.copy())
            
            # Phase 2: Aggregate into new graph
            new_graph = self.phase2_aggregate(current_graph, communities)
            
            if new_graph.number_of_nodes() == current_graph.number_of_nodes():
                print(f"Stopped at iteration {iteration}: no aggregation")
                break
            
            # Update node degrees for new graph
            self.node_degrees = {node: new_graph.degree(node, weight='weight') 
                                for node in new_graph.nodes()}
            
            current_graph = new_graph
        
        # Step 3: Reconstruct full partition
        final_communities = {node: node for node in working_graph.nodes()}
        
        for level_communities in dendrogram:
            new_final = {}
            for node, comm in final_communities.items():
                new_final[node] = level_communities.get(comm, comm)
            final_communities = new_final
        
        # Renumber communities to be sequential
        unique_comms = list(set(final_communities.values()))
        comm_mapping = {old: new for new, old in enumerate(unique_comms)}
        final_communities = {node: comm_mapping[comm] 
                            for node, comm in final_communities.items()}
        
        return final_communities