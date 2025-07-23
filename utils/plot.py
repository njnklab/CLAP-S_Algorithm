import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import datetime
import config
import os
from utils.utils import save_network


def get_node_positions(graph: nx.DiGraph, method: str):
    """
    Compute node positions for visualization based on the specified method.
    
    Parameters:
    - graph (nx.DiGraph): The network for which node positions are computed.
    - method (str): The method to use for computing node positions.
    
    Returns:
    - pos (dict): A dictionary of node positions.
    """
    if method == "bipartite":
        # Separate nodes into top (+) and bottom (-)
        out_nodes = [node for node in graph.nodes() if node.endswith("+")]
        in_nodes = [node for node in graph.nodes() if node.endswith("-")]
        
        # Calculate positions
        pos = {}
        for i, node in enumerate(out_nodes):
            pos[node] = (i, 1)  # x-coordinate varies, y-coordinate is 1 for top nodes
        for i, node in enumerate(in_nodes):
            pos[node] = (i, 0)  # x-coordinate varies, y-coordinate is 0 for bottom nodes
    else:
        pos = nx.shell_layout(graph)
        
    return pos


def transform_graph(graph: nx.DiGraph):
    """
    Transform a directed graph by splitting each node into an out-node and an in-node.
    For every directed edge (A, B) in the original graph, create a directed edge (A+, B-) in the transformed graph.
    """
    # Initialize the transformed graph
    bipartite_graph = nx.DiGraph()
    
    # Add nodes
    for node in graph.nodes():
        bipartite_graph.add_node(f"{node}+")
        bipartite_graph.add_node(f"{node}-")
        
    # Add edges
    for edge in graph.edges():
        source, target = edge
        bipartite_graph.add_edge(f"{source}+", f"{target}-")
        
    return bipartite_graph


def transform_graphs(graphs: list):
    """
    Transform a list of directed graphs by splitting each node in each graph into an out-node and an in-node.
    For every directed edge (A, B) in each original graph, create a directed edge (A+, B-) in the transformed graph.
    
    Parameters:
    - graphs (list): List of directed graphs to transform.
    
    Returns:
    - list: List of transformed graphs.
    """
    
    return [transform_graph(graph) for graph in graphs]


def save_graph(graph):
    os.makedirs(os.path.join(config.TEMP_PATH, "fig"), exist_ok=True)
    os.makedirs(os.path.join(config.TEMP_PATH, "net"), exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.join(config.TEMP_PATH, "fig", f"{current_time}.png"))
    if not isinstance(graph, list):
        save_network(graph, os.path.join(config.TEMP_PATH, "net", f"{current_time}.txt"))
    else:
        for i in range(len(graph)):
            save_network(graph[i], os.path.join(config.TEMP_PATH, "net", f"{current_time}_{i}.txt"))



def visualize_network(graph: nx.DiGraph, method: str=None):
    """
    Visualize a network using matplotlib.

    Parameters:
    - graph (nx.DiGraph): Network to visualize.
    - method (str): The method to use for computing node positions.
    """
    
    pos = get_node_positions(graph, method)
    
    # Draw the network
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", width=2)

    save_graph(graph)
    plt.show()


def visualize_network_with_matching(graph: nx.DiGraph, matched_edges: list, MDS: list, method: str=None):
    """
    Visualize a network with highlighted matched edges and Maximum Dominating Set (MDS) nodes using matplotlib.

    Parameters:
    - graph (nx.DiGraph): Network to visualize.
    - matched_edges (list): List of edges that are matched and should be highlighted.
    - MDS (list): List of nodes that form the Maximum Dominating Set and should be highlighted.
    - method (str): The method to use for computing node positions.
    """
    pos = get_node_positions(graph, method)

    # Draw the network
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", width=2)
    
    # Highlight MDS nodes in red
    nx.draw_networkx_nodes(graph, pos, nodelist=MDS, node_color='red')
    # Highlight matched edges in red
    valid_matched_edges = [edge for edge in matched_edges if edge in graph.edges()]
    nx.draw_networkx_edges(graph, pos, edgelist=valid_matched_edges, edge_color='red')

    save_graph(graph)
    plt.show()


def visualize_network_with_bipartite(graph: nx.DiGraph, bi_graph: nx.DiGraph):
    """
    Visualize a network using two horizontal subplots: one with the default layout and one with the bipartite layout.

    Parameters:
    - graph (nx.DiGraph): Network to visualize.
    - bi_graph (nx.DiGraph): Transformed network.
    """
    
    # Create a figure with two horizontal subplots
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    # Draw the graph with method=None on the left subplot
    pos = get_node_positions(graph, method=None)
    nx.draw(graph, pos, ax=axes[0], with_labels=True, node_color="skyblue", width=2)
    axes[0].set_title("Default Layout")
    
    # Draw the graph with method="bipartite" on the right subplot
    pos = get_node_positions(bi_graph, method="bipartite")
    nx.draw(bi_graph, pos, ax=axes[1], with_labels=True, node_color="skyblue", width=2)
    axes[1].set_title("Bipartite Layout")
    
    plt.tight_layout()
    save_graph(graph)
    plt.show()


def visualize_networks_with_bipartite(graphs: list, bi_graphs: list):
    """
    Visualize a list of networks using vertical subplots. Each row contains two subplots: 
    one for the original graph with the default layout and one for the bipartite layout.

    Parameters:
    - graphs (list): List of networks to visualize.
    - bi_graphs (list): List of transformed networks (bipartite layout).
    """
    
    # Ensure the lists have the same length
    assert len(graphs) == len(bi_graphs), "The number of original graphs and bipartite graphs must be the same."
    
    num_graphs = len(graphs)
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows=num_graphs, ncols=2)
    
    # If only one graph is provided, axes will be 1D, so we reshape it to be 2D for consistency
    if num_graphs == 1:
        axes = np.array(axes).reshape(1, -1)
    
    for idx, (graph, bi_graph) in enumerate(zip(graphs, bi_graphs)):
        # Draw the graph with method=None on the left subplot
        pos = get_node_positions(graph, method=None)
        nx.draw(graph, pos, ax=axes[idx, 0], with_labels=True, node_color="skyblue", width=2)
        axes[idx, 0].set_title("Default Layout")
        
        # Draw the graph with method="bipartite" on the right subplot
        pos = get_node_positions(bi_graph, method="bipartite")
        nx.draw(bi_graph, pos, ax=axes[idx, 1], with_labels=True, node_color="skyblue", width=2)
        axes[idx, 1].set_title("Bipartite Layout")
    
    plt.tight_layout()
    save_graph(graphs)
    plt.show()


def visualize_network_with_matching_and_bipartite(
        graph: nx.DiGraph, bi_graph: nx.DiGraph, 
        matched_edges: list, bi_matched_edges: list, 
        mds: list, bi_mds: list):
    """
    Visualize a network with highlighted matched edges and Maximum Dominating Set (MDS) nodes using two horizontal subplots: 
    one with the default layout and one with the bipartite layout.

    Parameters:
    - graph (nx.DiGraph): Network to visualize.
    - bi_graph (nx.DiGraph): Transformed network.
    - matched_edges (list): List of edges that are matched and should be highlighted.
    - bi_matched_edges (list): Transformed list of edges.
    - mds (list): List of nodes that form the Maximum Dominating Set and should be highlighted.
    - bi_mds (list): Transformed list of nodes.
    """
    
    # Create a figure with two horizontal subplots
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    # Draw the graph with method=None on the left subplot
    pos = get_node_positions(graph, method=None)
    nx.draw(graph, pos, ax=axes[0], with_labels=True, node_color="skyblue", width=2)
    nx.draw_networkx_nodes(graph, pos, ax=axes[0], nodelist=mds, node_color='red')
    valid_matched_edges = [edge for edge in matched_edges if edge in graph.edges()]
    nx.draw_networkx_edges(graph, pos, ax=axes[0], edgelist=valid_matched_edges, edge_color='red')
    axes[0].set_title("Default Layout")
    
    # Draw the graph with method="bipartite" on the right subplot
    pos = get_node_positions(bi_graph, method="bipartite")
    nx.draw(bi_graph, pos, ax=axes[1], with_labels=True, node_color="skyblue", width=2)
    nx.draw_networkx_nodes(bi_graph, pos, ax=axes[1], nodelist=bi_mds, node_color='red')
    valid_matched_edges = [edge for edge in bi_matched_edges if edge in bi_graph.edges()]
    nx.draw_networkx_edges(bi_graph, pos, ax=axes[1], edgelist=valid_matched_edges, edge_color='red')
    axes[1].set_title("Bipartite Layout")
    
    plt.tight_layout()
    save_graph(graph)
    plt.show()


def visualize_networks_with_matching_and_bipartite(
        graphs: list, bi_graphs: list, 
        matched_edges_list: list, bi_matched_edges_list: list, 
        mds_list: list, bi_mds_list: list):
    """
    Visualize a list of networks with highlighted matched edges and Maximum Dominating Set (MDS) nodes using vertical subplots. 
    Each row contains two subplots: one for the original graph with the default layout and one for the bipartite layout.

    Parameters:
    - graphs (list): List of networks to visualize.
    - bi_graphs (list): List of transformed networks.
    - matched_edges_list (list): List of lists of edges that are matched and should be highlighted for each graph.
    - bi_matched_edges_list (list): List of transformed lists of edges for each bipartite graph.
    - mds_list (list): List of lists of nodes that form the Maximum Dominating Set and should be highlighted for each graph.
    - bi_mds_list (list): List of transformed lists of nodes for each bipartite graph.
    """
    
    # Ensure the lists have the same length
    assert len(graphs) == len(bi_graphs) == len(matched_edges_list) == len(bi_matched_edges_list) == len(mds_list) == len(bi_mds_list), "All input lists must have the same length."
    
    num_graphs = len(graphs)
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows=num_graphs, ncols=2)
    
    # If only one graph pair is provided, axes will be 1D, so we reshape it to be 2D for consistency
    if num_graphs == 1:
        axes = np.array(axes).reshape(1, -1)
    
    for idx, (graph, bi_graph, matched_edges, bi_matched_edges, mds, bi_mds) in enumerate(zip(graphs, bi_graphs, matched_edges_list, bi_matched_edges_list, mds_list, bi_mds_list)):
        # Draw the graph with method=None on the left subplot
        pos = get_node_positions(graph, method=None)
        nx.draw(graph, pos, ax=axes[idx, 0], with_labels=True, node_color="skyblue", width=2)
        nx.draw_networkx_nodes(graph, pos, ax=axes[idx, 0], nodelist=mds, node_color='red')
        valid_matched_edges = [edge for edge in matched_edges if edge in graph.edges()]
        nx.draw_networkx_edges(graph, pos, ax=axes[idx, 0], edgelist=valid_matched_edges, edge_color='red')
        axes[idx, 0].set_title("Default Layout")
        
        # Draw the graph with method="bipartite" on the right subplot
        pos = get_node_positions(bi_graph, method="bipartite")
        nx.draw(bi_graph, pos, ax=axes[idx, 1], with_labels=True, node_color="skyblue", width=2)
        nx.draw_networkx_nodes(bi_graph, pos, ax=axes[idx, 1], nodelist=bi_mds, node_color='red')
        valid_matched_edges = [edge for edge in bi_matched_edges if edge in bi_graph.edges()]
        nx.draw_networkx_edges(bi_graph, pos, ax=axes[idx, 1], edgelist=valid_matched_edges, edge_color='red')
        axes[idx, 1].set_title("Bipartite Layout")
    
    plt.tight_layout()
    save_graph(graphs)
    plt.show()