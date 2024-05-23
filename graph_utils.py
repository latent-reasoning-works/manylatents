import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def get_graph_representation(K, remove_self_edges=False, weighted=True):
    
    # Remove self edges
    if remove_self_edges:
        K -= np.diag(np.diag(K))

    # Directed Graph
    # Use K directly to create weighted graph
    if weighted:
        dir_graph = nx.from_numpy_array(K, create_using=nx.DiGraph())
    else:
        # This is the adjacency matrix
        A = np.array(K>0, dtype=int)
        dir_graph = nx.from_numpy_array(A, create_using=nx.DiGraph())

    # graph represents the same matrix
    assert len(dir_graph.edges()) == (K>0).sum()
    return dir_graph

def get_groups_from_graph(K):
    # Directed Graph
    dir_graph = get_graph_representation(K)

    # Find weakly connected components
    weakly_connected = list(nx.weakly_connected_components(dir_graph))

    # Find strongly connected components
    strongly_connected = list(nx.strongly_connected_components(dir_graph))

    indices = np.zeros(K.shape[0])
    for i, group in enumerate(weakly_connected):
        indices[list(group)] = i

    return indices

def plot_graph(K, labels_dict, palettes, label_orders):
    # K = Kernel Matrix
    # labels_dict: dict of label_names:levels
    # palettes: dict of label_names:palette (palette maps levels to colors)

    # Directed Graph
    dir_graph = get_graph_representation(K, remove_self_edges=True)

    # Assign these labels to the nodes
    for i, node in enumerate(dir_graph.nodes()):
        for label_name, labels in labels_dict.items():
            dir_graph.nodes[node][label_name] = labels[i]

    # node colors is dict of label_names to colors per point
    node_colors = {label_name: [] for label_name in labels_dict.keys()}
    for label_name, palette in palettes.items():
        node_colors[label_name] = [palette[dir_graph.nodes[node][label_name]] \
                                   for node in dir_graph.nodes()]

    # Now plot the graph
    for label_name, labels in labels_dict.items(): 
        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(dir_graph, node_color=node_colors[label_name], ax=ax)

        # Create legend manually
        legend_elements = [Patch(facecolor=palettes[label_name][tgt_name], 
                                 edgecolor='k', 
                                 label=str(tgt_name)) \
                           for tgt_name in label_orders[label_name]]
        ax.legend(handles=legend_elements, 
                  bbox_to_anchor=(1.1, 1.05), 
                  loc='upper left')

        plt.show()