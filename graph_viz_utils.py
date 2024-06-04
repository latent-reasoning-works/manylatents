import os
import random
import math

import multiprocessing as mp
from multiprocessing import Pool
import functools
from functools import partial

import numpy as np
import matplotlib.pyplot as plt 
#import stlearn as st
import pandas as pd

import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout
#import graphviz
import scprep 


# Costly step!
def get_agg(subset, keys):
    out = []
    for key in keys:
        #  numeric
        if (subset[key].dtype == 'int') or (subset[key].dtype == 'float'):
            out.append(subset[key].mean())  # mean
        #  categorical
        else:
            out.append(subset[key].mode().values[0])  # most common category subsets[0][key]
    return out

def get_emb_table(emb, metadata, clusters, sizes, NxT, keys):
    subsets = [metadata[NxT == c] for c in np.unique(NxT)]
    n = len(subsets)
    if n > 7:
        # if its large use parallel processing
        with mp.Pool(processes=16) as pool:
            r = pool.map(partial(get_agg, keys=keys), subsets)

        out = np.array(r)
    else:
        # code breaks when small, so just run thru a loop
        out = []
        for s in subsets:
            out.append(get_agg(s, keys=keys))
        out = np.array(out)

    metadata_msphate = pd.DataFrame({'MSPHATE_1': emb[:,0],
                                     'MSPHATE_2': emb[:,1],
                                     'sizes': sizes,
                                     'msphate_clusters': clusters
                                    })
    for i, key in enumerate(keys):
        metadata_msphate[key] = out[:, i]

    return metadata_msphate

def make_table(algo_obj, metadata, keys):
    metadatas = []

    for level in algo_obj.levels:
        emb, clusters, sizes = algo_obj.transform(visualization_level=level, 
                                                  cluster_level=level)
        metadata_msphate_level = get_emb_table(emb, 
                                               metadata, 
                                               clusters, 
                                               sizes, 
                                               algo_obj.NxTs[level], 
                                               keys)
        metadata_msphate_level['level'] = level
        metadatas.append(metadata_msphate_level)
        print('done level {}'.format(level))

    metadata_msphate = pd.concat(metadatas)

    # The embeddings before are inconsistent across levels so we compute them again properly here
    tree = algo_obj.build_tree()
    to_keep = pd.DataFrame(tree[:,2]).isin(algo_obj.levels).values.flatten().tolist()
    embeddings = pd.DataFrame(tree[:,:2], 
                              index=tree[:,2]).loc[to_keep].values # emb at each selected levels

    metadata_msphate['MSPHATE_1'] = embeddings[:,0]
    metadata_msphate['MSPHATE_2'] = embeddings[:,1]
    metadata_msphate['NodeName'] = metadata_msphate.apply(lambda row: str(row.level) + '_' + str(row.msphate_clusters), axis=1).values
    
    return metadata_msphate


def make_directed_tree(msp_op):
    """
    Takes the MSPHATE operator and returns a simple tree representation of it.
    """
    DG = nx.DiGraph()

    levels_to_add = [len(msp_op.NxTs)-1] + msp_op.levels[::-1] # include highest level (root!)

    prev_level = levels_to_add[0]
    for counter, level in enumerate(levels_to_add):
        for indv in np.unique(msp_op.NxTs[level]):
            # add individual to the graph
            DG.add_node(str(level) + '_' + str(indv), subset=len(levels_to_add)-counter)
        # Now link to its child
        if counter > 0:
            for child in np.unique(msp_op.NxTs[level]):
                parents = np.unique(msp_op.NxTs[prev_level][msp_op.NxTs[level] == child]) 
                for parent in parents:
                    DG.add_edge(str(prev_level) + '_' + str(parent),
                                str(level) + '_' + str(child))
        prev_level = level # store previous level to be used in next iteration!
    return DG


def get_subgraph_up_to_level(G, root, max_level):
    subgraph_nodes = set([root])
    current_level_nodes = set([root])
    for level in range(max_level):
        next_level_nodes = set()
        for node in current_level_nodes:
            next_level_nodes.update(G.successors(node))
        subgraph_nodes.update(next_level_nodes)
        current_level_nodes = next_level_nodes
    return G.subgraph(subgraph_nodes)


def remove_unneeded_nodes(digraph):
    # Remove nodes that dont change the overall topology.
    # Nodes only remain if they have more than one child (or are a leaf)
    # Create a list to keep track of whether a node was removed
    
    # need to make a copy first
    digraph = digraph.copy()
    
    node_status = {node: True for node in digraph.nodes()}

    # Find nodes with exactly one child and one parent
    nodes_to_remove = [node for node in digraph if digraph.in_degree(node) == 1 and digraph.out_degree(node) == 1]

    for node in nodes_to_remove:
        parent = list(digraph.predecessors(node))[0]
        child = list(digraph.successors(node))[0]

        # Add edge from parent to child
        digraph.add_edge(parent, child)

        # Remove the node
        digraph.remove_node(node)
        # Mark node as removed
        node_status[node] = False

    # Convert the dictionary to a boolean list
    node_status_list = [node_status[node] for node in digraph.nodes()]
    
    return digraph, node_status_list


def get_node_color_size(tree, metadata, key, color_dict):
    node_colors = []
    node_sizes = []
    for node in tree.nodes():
        level, msphate_cluster = str(node).split('_')
        node_metadata = metadata[(metadata['msphate_clusters'] == int(msphate_cluster)) & (metadata['level'] == int(level))]
        if len(node_metadata) > 0:
            node_colors.append(color_dict[node_metadata[key].values[0]])
            node_sizes.append(100*np.sqrt(node_metadata['sizes'].values[0]))
        else:
            node_colors.append('black') # root doesn't have any metadata associated with it!
            node_sizes.append(100)
    return node_colors, node_sizes

def plot_msphate(msp_op, levels, level_ind, palette=None, labels=None):
    coarse_embedding, coarse_clusters, coarse_sizes = \
    msp_op.transform(visualization_level=levels[0],
                     cluster_level=levels[level_ind])
    
    if labels is None:
        point_labels = pd.Categorical(np.array(coarse_clusters, dtype=str))
        point_colors = point_labels # arbitrary colors
    else:
        point_labels = labels
        point_colors = [palette[indv] for indv in labels]

    fig, ax = plt.subplots(figsize=(10,8))
    scprep.plot.scatter2d(coarse_embedding, 
                          s=100*np.sqrt(coarse_sizes), 
                          c=point_colors,
                          fontsize=16, 
                          ticks=False,
                          label_prefix="Multiscale PHATE", 
                          ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for clustername in np.unique(point_labels):
        x_annot = coarse_embedding[point_labels == clustername][:,0].mean()
        y_annot = coarse_embedding[point_labels == clustername][:,1].mean()
        ax.text(x_annot, y_annot, clustername)

# Graph Layouts
def hierarchical_layout(DG, root, min_spacing=0.1):
    # Step 1: Assign initial positions from root to leaves
    pos = {}
    level = 0
    current_level_nodes = [root]
    pos[root] = (0, -level)
    level_width = {0: 1}

    while current_level_nodes:
        level += 1
        next_level_nodes = []

        for parent in current_level_nodes:
            children = list(DG.successors(parent))
            num_children = len(children)

            if num_children > 0:
                parent_x, parent_y = pos[parent]
                child_x_pos = np.linspace(parent_x - (num_children - 1) * min_spacing / 2.0,
                                          parent_x + (num_children - 1) * min_spacing / 2.0,
                                          num_children)

                for i, child in enumerate(children):
                    pos[child] = (child_x_pos[i], -level)
                    next_level_nodes.append(child)

                level_width[level] = max(level_width.get(level, 0), len(children))

        current_level_nodes = next_level_nodes

    # Step 2: Adjust parent positions to be centered above their children
    def adjust_parent_positions(node):
        children = list(DG.successors(node))
        if children:
            child_positions = [pos[child][0] for child in children]
            pos[node] = (np.mean(child_positions), pos[node][1])
            for child in children:
                adjust_parent_positions(child)

    adjust_parent_positions(root)

    # Step 3: Ensure minimal spacing at each level
    for lvl in range(level):
        nodes_at_level = [node for node, p in pos.items() if p[1] == -lvl]
        if len(nodes_at_level) > 1:
            node_positions = sorted([pos[node][0] for node in nodes_at_level])
            for i in range(1, len(node_positions)):
                if node_positions[i] - node_positions[i - 1] < min_spacing:
                    shift = min_spacing - (node_positions[i] - node_positions[i - 1])
                    for j in range(i, len(node_positions)):
                        pos[nodes_at_level[j]] = (pos[nodes_at_level[j]][0] + shift, pos[nodes_at_level[j]][1])
                    node_positions = [pos[node][0] for node in nodes_at_level]

    return pos

# taken from: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                    vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent=root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def radial_layout(G, root=None, vert_gap=0.2, vert_loc=0):
    pos = hierarchy_pos(G, root=root, width=2*math.pi, vert_gap=vert_gap, vert_loc=vert_loc, xcenter=0)
    new_pos = {u:(r*math.cos(theta),r*math.sin(theta)) for u, (theta, r) in pos.items()}
    return new_pos

def depricated_hierarchical_layout(DG, root):
    # Heirarchical Layout for networkx draw function
    pos = {}
    level = 0
    current_level_nodes = [root]
    next_level_nodes = []

    while current_level_nodes:
        x_pos = np.linspace(-1, 1, len(current_level_nodes))
        for i, node in enumerate(current_level_nodes):
            pos[node] = (x_pos[i], -level)
            children = list(DG.successors(node))
            next_level_nodes.extend(children)

        current_level_nodes = next_level_nodes
        next_level_nodes = []
        level += 1

    return pos