import networkx as nx
import random
import numpy as np
from typing import List, Optional, Set, Tuple


def jaccard_similarity(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
    edges1, edges2 = set(g1.edges()), set(g2.edges())
    intersection, union = len(edges1.intersection(edges2)), len(edges1.union(edges2))
    return intersection / union if union > 0 else 1.0


class Generator:
    """
    Abstract base class for network generators.
    """
    def __init__(self, num_nodes: int, average_degree: float):
        if num_nodes <= 0:
            raise ValueError("Number of nodes must be positive.")
        if average_degree <= 0:
            raise ValueError("Average degree must be positive.")
        self.num_nodes = num_nodes
        self.average_degree = average_degree
    
    def generate_network(self) -> nx.DiGraph:
        """Generates a single network. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method!")

    def generate_similar_network(self, base_graph: nx.DiGraph, similarity: float) -> nx.DiGraph:
        """
        Creates a new network with a specified Jaccard similarity to the base graph,
        while preserving the model's structural characteristics.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method for model-specific similarity!")

    def generate_networks(self, n: int, similarities: Optional[List[float]] = None) -> List[nx.DiGraph]:
        """
        Generates a list of n networks, with optional inter-network similarity.
        """
        if n <= 0:
            return []
            
        if similarities is None:
            return [self.generate_network() for _ in range(n)]

        if len(similarities) != n - 1:
            raise ValueError(f"Length of similarities list must be n-1 ({n-1}), but got {len(similarities)}.")

        base_network = self.generate_network()
        networks = [base_network]

        for s in similarities:
            if s < 0:
                new_network = self.generate_network()
            else:
                # Use the polymorphic method, which will call the correct subclass implementation
                new_network = self.generate_similar_network(base_network, s)
            networks.append(new_network)
            
        return networks


class ERGenerator(Generator):
    """
    Generates Erdős-Rényi (ER) networks. The similar network generation
    preserves the probabilistic nature of the ER model.
    """
    def generate_network(self) -> nx.DiGraph:
        # Same as before
        graph = nx.DiGraph()
        graph.add_nodes_from(range(1, self.num_nodes + 1))
        if self.num_nodes <= 1: return graph
        p = min(1.0, self.average_degree / (2 * (self.num_nodes - 1)))
        for i in range(1, self.num_nodes + 1):
            for j in range(1, self.num_nodes + 1):
                if i != j and random.random() < p:
                    graph.add_edge(i, j)
        return graph

    def generate_similar_network(self, base_graph: nx.DiGraph, similarity: float) -> nx.DiGraph:
        base_edges = set(base_graph.edges())
        num_base_edges = len(base_edges)
        
        # Total possible edges in a directed graph without self-loops
        total_possible_edges = self.num_nodes * (self.num_nodes - 1)
        
        if total_possible_edges == 0 or num_base_edges == total_possible_edges:
             return base_graph.copy() # Cannot add or change edges

        # Probability of keeping an existing edge. Simplified to s.
        p_keep = min(max(similarity, 0.0), 1.0)
        
        # Probability of adding a new edge, calculated to maintain density.
        denominator = total_possible_edges - num_base_edges
        if denominator > 0:
            p_add = (num_base_edges * (1 - p_keep)) / denominator
        else: # This happens if base_graph is a complete graph
            p_add = 0
        p_add = min(max(p_add, 0.0), 1.0)
        
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(range(1, self.num_nodes + 1))
        
        for i in range(1, self.num_nodes + 1):
            for j in range(1, self.num_nodes + 1):
                if i == j: continue
                edge = (i, j)
                # Decide whether to add the edge based on its presence in the base graph
                if edge in base_edges:
                    if random.random() < p_keep:
                        new_graph.add_edge(i, j)
                else:
                    if random.random() < p_add:
                        new_graph.add_edge(i, j)
        return new_graph


class BAGenerator(Generator):
    """
    Generates a generalized Barabási-Albert (BA) network that can handle
    non-integer average degrees.
    """
    def generate_network(self) -> nx.DiGraph:
        return self._generate_ba_graph(base_edges=None, similarity_bias=0)

    def generate_similar_network(self, base_graph: nx.DiGraph, similarity: float) -> nx.DiGraph:
        base_edges = set(base_graph.edges())
        m_float = self.average_degree / 2
        
        similarity = min(max(similarity, 0.0), 0.9999)
        # The bias strength should scale with the average number of edges being added.
        bias_strength = m_float * (similarity / (1 - similarity))
        
        return self._generate_ba_graph(base_edges=base_edges, similarity_bias=bias_strength)

    def _generate_ba_graph(self, base_edges: Optional[Set] = None, similarity_bias: float = 0.0) -> nx.DiGraph:
        # m_float is the target average number of edges per new node
        m_float = self.average_degree / 2
        if m_float <= 0:
            g = nx.DiGraph()
            g.add_nodes_from(range(1, self.num_nodes + 1))
            return g
            
        # For the initial core, we use the rounded integer value of m
        m_core = max(1, round(m_float))

        if self.num_nodes < m_core:
            raise ValueError(f"Number of nodes ({self.num_nodes}) cannot be smaller than the initial core size ({m_core}).")

        # Initialize with a complete graph of size m_core
        nodes_initial = range(1, m_core + 1)
        graph = nx.complete_graph(nodes_initial, create_using=nx.DiGraph())

        # The integer and fractional parts of m for probabilistic edge addition
        m_int = int(m_float)
        m_frac = m_float - m_int

        for i in range(m_core + 1, self.num_nodes + 1):
            graph.add_node(i)
            
            # Determine the number of edges to add for this specific step
            num_to_pick = m_int
            if random.random() < m_frac:
                num_to_pick += 1

            if num_to_pick == 0:
                continue

            possible_targets = list(graph.nodes())
            possible_targets.remove(i)
            
            weights = []
            for target in possible_targets:
                weight = graph.in_degree(target) + 0.1
                if base_edges and (i, target) in base_edges:
                    weight += similarity_bias
                weights.append(weight)

            targets = set()
            num_to_pick = min(num_to_pick, len(possible_targets))
            
            if not possible_targets or sum(weights) == 0:
                 if possible_targets:
                    targets = set(random.sample(possible_targets, num_to_pick))
            else:
                 temp_possible_targets = list(possible_targets)
                 temp_weights = list(weights)
                 for _ in range(num_to_pick):
                    if not temp_possible_targets: break
                    chosen_node = random.choices(temp_possible_targets, weights=temp_weights, k=1)[0]
                    targets.add(chosen_node)
                    idx = temp_possible_targets.index(chosen_node)
                    temp_possible_targets.pop(idx)
                    temp_weights.pop(idx)

            for target in targets:
                graph.add_edge(i, target)
        return graph
    

class SFGenerator(Generator):
    """
    Generates a scale-free network using a static model.

    This model does not use growth or preferential attachment. Instead, it
    assigns a priori connection probabilities to nodes based on a power-law
    distribution tied to their index. This results in a network with a
    power-law degree distribution, but generated via a static mechanism.
    """
    def __init__(self, num_nodes: int, average_degree: float, gamma_in: float = 3.0, gamma_out: float = 3.0):
        super().__init__(num_nodes, average_degree)
        if gamma_in <= 1 or gamma_out <= 1:
            raise ValueError("Gamma values must be greater than 1.")
        self.gamma_in = gamma_in
        self.gamma_out = gamma_out
    
    def generate_network(self) -> nx.DiGraph:
        # Calculate target number of edges.
        num_edges = round(self.num_nodes * self.average_degree / 2)

        if num_edges == 0:
            g = nx.DiGraph()
            g.add_nodes_from(range(1, self.num_nodes + 1))
            return g
            
        alpha_in = 1.0 / (self.gamma_in - 1)
        alpha_out = 1.0 / (self.gamma_out - 1)
        
        # Create weights based on node index (1-based)
        node_indices = np.arange(1, self.num_nodes + 1)
        w_out = np.power(node_indices, -alpha_out)
        w_in = np.power(node_indices, -alpha_in)

        # Normalize to create a probability distribution
        w_out /= np.sum(w_out)
        w_in /= np.sum(w_in)

        graph = nx.DiGraph()
        graph.add_nodes_from(range(1, self.num_nodes + 1))
        
        # Generate edges by sampling from the static distributions
        added_edges = set()
        # Add a safeguard against infinite loops if num_edges is close to total possible edges
        max_attempts = num_edges * 10 
        attempts = 0
        while len(added_edges) < num_edges and attempts < max_attempts:
            # Choose source and target nodes based on their pre-defined weights
            source = np.random.choice(node_indices, p=w_out)
            target = np.random.choice(node_indices, p=w_in)
            
            if source != target and (source, target) not in added_edges:
                added_edges.add((source, target))
            attempts += 1
        
        graph.add_edges_from(added_edges)
        return graph
    
    def generate_similar_network(self, base_graph: nx.DiGraph, similarity: float) -> nx.DiGraph:
        """
        Generates a similar scale-free network by partitioning the edge selection
        process into 'keeping' and 'adding' phases, both guided by the static
        model's power-law probabilities to preserve structural properties.
        """
        similarity = min(max(similarity, 0.0), 1.0)
        
        if similarity == 1.0:
            return base_graph.copy()
        if similarity == 0.0:
            return self.generate_network()

        # 1. Calculate target edge counts
        base_edges = set(base_graph.edges())
        num_base_edges = len(base_edges)
        num_target_edges = round(self.num_nodes * self.average_degree / 2)

        if num_target_edges == 0:
            g = nx.DiGraph()
            g.add_nodes_from(range(1, self.num_nodes + 1))
            return g

        # From J = |I| / |U| => J = |I| / (|A|+|B|-|I|), solve for |I|
        # Let |I| = num_to_keep, |A| = num_base_edges, |B| = num_target_edges
        # num_to_keep = J * (|A| + |B|) / (1 + J)
        if (1 + similarity) > 0:
             num_to_keep = round(similarity * (num_base_edges + num_target_edges) / (1 + similarity))
        else: # Should not happen with similarity >= 0
             num_to_keep = 0
        
        num_to_keep = min(num_to_keep, num_base_edges, num_target_edges)
        num_to_add = num_target_edges - num_to_keep

        # 2. Calculate static model weights (same as in generate_network)
        alpha_in = 1.0 / (self.gamma_in - 1)
        alpha_out = 1.0 / (self.gamma_out - 1)
        node_indices = np.arange(1, self.num_nodes + 1)
        
        # Using a small epsilon to avoid zero weights for large indices if alpha is large
        w_out_raw = np.power(node_indices, -alpha_out, dtype=float) + 1e-12
        w_in_raw = np.power(node_indices, -alpha_in, dtype=float) + 1e-12
        w_out_prob = w_out_raw / np.sum(w_out_raw)
        w_in_prob = w_in_raw / np.sum(w_in_raw)

        kept_edges = set()

        # 3. Select edges to KEEP from the base graph, biased by SF probability
        if num_to_keep > 0 and num_base_edges > 0:
            candidate_edges_to_keep = list(base_edges)
            # Weight of an edge (u, v) is proportional to P(u)*P(v)
            # Node indices are 1-based, numpy arrays are 0-based
            keep_weights = np.array([
                w_out_raw[u - 1] * w_in_raw[v - 1] for u, v in candidate_edges_to_keep
            ])
            
            # Handle case where all weights are zero
            if np.sum(keep_weights) > 0:
                keep_prob = keep_weights / np.sum(keep_weights)
                # Sample indices of edges to keep
                chosen_indices = np.random.choice(
                    len(candidate_edges_to_keep),
                    size=num_to_keep,
                    replace=False,
                    p=keep_prob
                )
                kept_edges = {candidate_edges_to_keep[i] for i in chosen_indices}
            else: # Fallback to random sampling if weights are all zero
                kept_edges = set(random.sample(candidate_edges_to_keep, num_to_keep))


        # 4. ADD new edges using rejection sampling
        newly_added_edges = set()
        max_attempts = num_to_add * 20  # Safeguard
        attempts = 0
        while len(newly_added_edges) < num_to_add and attempts < max_attempts:
            source = np.random.choice(node_indices, p=w_out_prob)
            target = np.random.choice(node_indices, p=w_in_prob)
            edge = (source, target)
            
            # Add if it's not a self-loop, not in base graph, and not already added
            if source != target and edge not in base_edges and edge not in newly_added_edges:
                newly_added_edges.add(edge)
            attempts += 1
            
        # 5. Construct the final graph
        final_graph = nx.DiGraph()
        final_graph.add_nodes_from(range(1, self.num_nodes + 1))
        final_graph.add_edges_from(kept_edges)
        final_graph.add_edges_from(newly_added_edges)
        
        return final_graph