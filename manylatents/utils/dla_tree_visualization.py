"""
Graph visualization utilities for DLA Tree datasets.

This module contains visualization functions extracted from DLATreeFromGraph
to keep the core dataset class focused on data generation.
"""
import os
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional


class DLATreeGraphVisualizer:
    """Handles graph topology visualization for DLA tree datasets."""

    def __init__(self, graph_edges: List[Tuple], excluded_edges: set = None,
                 edge_renumbering: Dict = None, original_excluded_edges: set = None,
                 random_state: int = 42, save_dir: str = "outputs"):
        """
        Initialize the visualizer with graph topology information.

        Parameters:
        -----------
        graph_edges : list of tuples
            List of (from_node, to_node, edge_id, length) tuples
        excluded_edges : set, optional
            Set of edge_ids that are excluded from data generation
        edge_renumbering : dict, optional
            Mapping from original edge_ids to renumbered edge_ids
        original_excluded_edges : set, optional
            Set of original excluded edge_ids
        random_state : int
            Random seed for layout algorithms
        save_dir : str
            Directory to save visualizations
        """
        self.graph_edges = graph_edges
        self.excluded_edges = excluded_edges or set()
        self.edge_renumbering = edge_renumbering or {}
        self.original_excluded_edges = original_excluded_edges or set()
        self.random_state = random_state
        self.save_dir = save_dir

    def visualize_sample_graph(self, sample_graph, save_path="debug_outputs/sample_graph.png", max_nodes=500):
        """
        Visualize the sample-level graph where nodes are samples and edges connect them.

        Parameters:
        -----------
        sample_graph : networkx.Graph
            Graph where nodes are individual samples
        save_path : str
            Path to save the visualization
        max_nodes : int
            Maximum number of nodes to display (for performance)
        """
        if sample_graph is None:
            print("No sample graph available. Run generation first.")
            return

        G = sample_graph
        G_full = G

        # Filter to visible and gap nodes (exclude only junctions)
        display_nodes = [n for n in G.nodes()
                        if not G.nodes[n]['is_junction']]
        print(f"Showing {len(display_nodes)} nodes from {G.number_of_nodes()} total nodes (includes gap edges)")
        G = G.subgraph(display_nodes)

        # Create layout using first two dimensions of sample positions (fast and reliable)
        pos = {}
        for node in G.nodes():
            sample_pos = G.nodes[node]['pos']
            pos[node] = (sample_pos[0], sample_pos[1])  # Use first 2 dims for layout

        # Now we should have proper gap detection - let's verify
        gap_nodes = [n for n in G.nodes() if G.nodes[n]['is_gap']]
        visible_nodes = [n for n in G.nodes() if not G.nodes[n]['is_gap']]

        print(f"Gap nodes: {len(gap_nodes)}, Visible nodes: {len(visible_nodes)}")

        # Draw different edge types differently
        within_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'within_branch']
        junction_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'to_junction']

        plt.figure(figsize=(12, 10))

        # Draw within-branch edges (thin, colored by branch)
        from manylatents.utils.mappings import cmap_dla_tree
        for edge in within_edges:
            u, v = edge
            edge_data = G.edges[edge]
            branch_id = edge_data.get('branch_id', 1)
            color = cmap_dla_tree.get(branch_id, 'gray')
            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=color, width=0.5, alpha=0.7)

        # Draw junction edges (thick, black)
        nx.draw_networkx_edges(G, pos, edgelist=junction_edges, edge_color='black', width=2, alpha=0.8)

        # Draw nodes with different alphas for gap vs visible edges
        gap_nodes = [n for n in G.nodes() if G.nodes[n]['is_gap']]
        visible_nodes = [n for n in G.nodes() if not G.nodes[n]['is_gap']]

        # Draw gap nodes (light gray, transparent)
        if gap_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=gap_nodes, node_color='lightgray',
                                 node_size=15, alpha=0.3)

        # Draw visible nodes (colored by edge)
        if visible_nodes:
            visible_colors = []
            for node in visible_nodes:
                edge_id = G.nodes[node]['edge_id']
                try:
                    edge_id_int = int(edge_id)
                    if edge_id_int in cmap_dla_tree:
                        visible_colors.append(cmap_dla_tree[edge_id_int])
                    else:
                        visible_colors.append('gray')
                except (ValueError, TypeError):
                    visible_colors.append('gray')

            nx.draw_networkx_nodes(G, pos, nodelist=visible_nodes, node_color=visible_colors,
                                 node_size=15, alpha=0.8)

        # Create legend for edge colors (only visible edges)
        import matplotlib.patches as mpatches
        legend_elements = []
        edge_ids_with_colors = set()
        for node in G.nodes():
            if not G.nodes[node]['is_gap']:  # Only include non-gap edges in legend
                edge_id = G.nodes[node]['edge_id']
                try:
                    edge_id_int = int(edge_id)
                    if edge_id_int in cmap_dla_tree:
                        edge_ids_with_colors.add((edge_id_int, cmap_dla_tree[edge_id_int]))
                except (ValueError, TypeError):
                    pass

        for edge_id, color in sorted(edge_ids_with_colors):
            legend_elements.append(mpatches.Patch(color=color, label=f'Edge {edge_id}'))

        # Add gap edges entry
        legend_elements.append(mpatches.Patch(color='lightgray', alpha=0.3, label='Gap edges'))

        if legend_elements:
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.title(f'Sample Graph Visualization\n{G_full.number_of_nodes()} total nodes, {G_full.number_of_edges()} total edges\n(showing {G.number_of_nodes()} nodes)')
        plt.axis('equal')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logging.debug(f"Sample graph visualization saved to: {save_path}")

        # Print full graph statistics (not subsampled)
        logging.debug(f"Sample Graph Statistics (Full Graph): {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

        junction_nodes = [n for n in G_full.nodes() if G_full.nodes[n]['is_junction']]
        gap_nodes = [n for n in G_full.nodes() if G_full.nodes[n]['is_gap']]
        visible_nodes = [n for n in G_full.nodes() if G_full.nodes[n]['is_visible']]

        logging.debug(f"Node types: {len(junction_nodes)} junctions, {len(gap_nodes)} gaps, {len(visible_nodes)} visible")

        # Check connectivity of full graph
        if nx.is_connected(G_full):
            print(f"  Graph is connected ✓")
        else:
            components = list(nx.connected_components(G_full))
            print(f"  Graph has {len(components)} connected components ⚠️")
            for i, comp in enumerate(components):
                print(f"    Component {i+1}: {len(comp)} nodes")

    def visualize_and_save_graph(self):
        """Create and save two versions of graph topology visualization."""
        try:
            # Create networkx graph
            G = nx.DiGraph()

            # Add all edges
            for from_node, to_node, edge_id, length in self.graph_edges:
                G.add_edge(from_node, to_node, edge_id=edge_id, length=length)

            # Use custom semantic layout based on graph structure
            pos = self._create_semantic_layout(G)

            # Get colormap for edge coloring (same as embeddings)
            from manylatents.utils.mappings import cmap_dla_tree

            # All edges are data edges (simplified approach - no gaps)
            regular_edges = [(u, v, d) for u, v, d in G.edges(data=True)]

            # Separate edges into visible and gap edges for different drawing styles
            visible_edges = []
            gap_edges = []
            visible_edge_colors = []
            visible_edge_labels = {}
            gap_edge_labels = {}

            for u, v, d in regular_edges:
                original_edge_id = d['edge_id']

                if original_edge_id in self.original_excluded_edges:
                    # Gap edge: will be drawn as transparent outline
                    gap_edges.append((u, v))
                    gap_edge_labels[(u, v)] = f"E{original_edge_id}"  # E prefix for gap edges
                else:
                    # Visible edge: use renumbered ID for coloring and labeling
                    visible_edges.append((u, v))

                    if self.edge_renumbering:
                        display_edge_id = self.edge_renumbering.get(original_edge_id, original_edge_id)
                    else:
                        display_edge_id = original_edge_id

                    # Map display_edge_id to color index
                    if isinstance(display_edge_id, str):
                        color_idx = hash(display_edge_id) % len(cmap_dla_tree) + 1
                    else:
                        color_idx = display_edge_id

                    color = cmap_dla_tree[color_idx]
                    visible_edge_colors.append(color)
                    visible_edge_labels[(u, v)] = f"{display_edge_id}"  # No E prefix for display version

            os.makedirs(self.save_dir, exist_ok=True)

            # ===== Display Version (Clean, no node labels) =====
            plt.figure(figsize=(10, 8))
            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)

            # Draw visible edges (data branches) with colormap colors
            if visible_edges:
                nx.draw_networkx_edges(G, pos, edgelist=visible_edges,
                                     edge_color=visible_edge_colors, width=8, alpha=0.9, arrows=False,
                                     arrowstyle='-', arrowsize=20)

            # Draw gap edges as faint dashed lines
            if gap_edges:
                nx.draw_networkx_edges(G, pos, edgelist=gap_edges,
                                     edge_color='lightgray', width=3, alpha=0.4, arrows=False,
                                     style='dashed', arrowstyle='-')

            # Add clean edge labels for visible edges only
            if visible_edge_labels:
                nx.draw_networkx_edge_labels(G, pos, visible_edge_labels, font_size=12,
                                           font_weight='bold', font_family='sans-serif',
                                           bbox=dict(boxstyle='round,pad=0.3',
                                                    facecolor='white', alpha=0.9, edgecolor='none'))

            # Clean title for display
            plt.title("DLA Tree Graph Topology", fontsize=16, fontweight='bold', pad=20)
            plt.axis('off')

            # Save display version
            display_path = os.path.join(self.save_dir, "dla_tree_graph_topology.png")
            plt.savefig(display_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            print(f"Display graph visualization saved to: {display_path}")

            # ===== Debug Version (Detailed, with node and edge labels) =====
            plt.figure(figsize=(12, 10))
            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)

            # Draw nodes with labels
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, alpha=0.8)

            # Draw visible edges (data branches) with colormap colors
            if visible_edges:
                nx.draw_networkx_edges(G, pos, edgelist=visible_edges,
                                     edge_color=visible_edge_colors, width=6, alpha=0.8, arrows=True,
                                     arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1')

            # Draw gap edges as dashed outlines
            if gap_edges:
                nx.draw_networkx_edges(G, pos, edgelist=gap_edges,
                                     edge_color='gray', width=4, alpha=0.5, arrows=True,
                                     style='dashed', arrowstyle='->', arrowsize=15,
                                     connectionstyle='arc3,rad=0.1')

            # Add node labels (N1, N2, ...)
            node_labels = {node: f"N{node}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, node_labels, font_size=10, font_weight='bold',
                                  font_family='sans-serif')

            # Add edge labels for visible edges (E1, E2, ...)
            if visible_edge_labels:
                nx.draw_networkx_edge_labels(G, pos, visible_edge_labels, font_size=10,
                                           font_weight='bold', font_family='sans-serif',
                                           bbox=dict(boxstyle='round,pad=0.2',
                                                    facecolor='yellow', alpha=0.7, edgecolor='black'))

            # Add edge labels for gap edges (E9, E10, ... in parentheses)
            if gap_edge_labels:
                gap_labels_formatted = {edge: f"({label})" for edge, label in gap_edge_labels.items()}
                nx.draw_networkx_edge_labels(G, pos, gap_labels_formatted, font_size=9,
                                           font_color='darkgray', font_family='sans-serif',
                                           bbox=dict(boxstyle='round,pad=0.2',
                                                    facecolor='lightgray', alpha=0.6, edgecolor='gray'))

            # Detailed title with gap information
            if self.original_excluded_edges:
                gap_info = f"Gaps (Excluded): {sorted(self.original_excluded_edges)}"
                plt.title(f"DLA Tree Graph Topology - DEBUG\n{gap_info}\nSolid edges = Data populations, Dashed edges = Excluded (no data)",
                         fontsize=14, pad=20)
            else:
                plt.title("DLA Tree Graph Topology - DEBUG\nAll Edges Visible",
                         fontsize=14, pad=20)
            plt.axis('off')

            # Save debug version
            debug_path = os.path.join(self.save_dir, "dla_tree_graph_topology_debug.png")
            plt.savefig(debug_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

            print(f"Debug graph visualization saved to: {debug_path}")

        except ImportError:
            print("NetworkX or matplotlib not available for graph visualization")

    def _create_semantic_layout(self, G):
        """
        Create a layout based purely on graph structure without automatic layout algorithms.
        Positions nodes based on their semantic relationships in the graph.
        """
        # Find the root node (node that appears as source but not as target)
        all_nodes = set(G.nodes())
        target_nodes = {v for u, v in G.edges()}
        root_candidates = all_nodes - target_nodes
        root = min(root_candidates) if root_candidates else min(G.nodes())

        pos = {}

        # Build tree levels using BFS traversal
        levels = {0: [root]}
        visited = {root}
        queue = [(root, 0)]

        while queue:
            node, level = queue.pop(0)
            # Add children to next level
            children = [neighbor for neighbor in G.neighbors(node) if neighbor not in visited]
            if children:
                next_level = level + 1
                if next_level not in levels:
                    levels[next_level] = []
                for child in children:
                    levels[next_level].append(child)
                    visited.add(child)
                    queue.append((child, next_level))

        # Position nodes semantically
        # Root at origin, each level gets positioned based on graph structure
        pos[root] = (0, 0)

        for level_num, nodes in levels.items():
            if level_num == 0:  # Root level already positioned
                continue

            y_pos = -level_num * 2  # Move down for each level

            # For each node in this level, position based on its parent
            for i, node in enumerate(nodes):
                # Find parent (node that connects to this one from previous level)
                parent = None
                for prev_level_node in levels.get(level_num - 1, []):
                    if G.has_edge(prev_level_node, node):
                        parent = prev_level_node
                        break

                if parent and parent in pos:
                    # Position relative to parent
                    parent_x = pos[parent][0]
                    # Spread children horizontally around parent
                    parent_children = [n for n in nodes if any(G.has_edge(p, n) for p in levels.get(level_num - 1, []) if p == parent)]
                    child_index = parent_children.index(node)
                    n_children = len(parent_children)

                    if n_children == 1:
                        x_pos = parent_x
                    else:
                        # Spread children around parent
                        spacing = 3.0  # Horizontal spacing between siblings
                        x_offset = (child_index - (n_children - 1) / 2) * spacing
                        x_pos = parent_x + x_offset
                else:
                    # Fallback: spread nodes horizontally at this level
                    spacing = 4.0
                    x_pos = (i - (len(nodes) - 1) / 2) * spacing

                pos[node] = (x_pos, y_pos)

        # Handle any unpositioned nodes (shouldn't happen with well-formed trees)
        for node in G.nodes():
            if node not in pos:
                pos[node] = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))

        return pos