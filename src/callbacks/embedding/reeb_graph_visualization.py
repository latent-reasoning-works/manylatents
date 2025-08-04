import matplotlib.pyplot as plt
import networkx as nx
import wandb
from src.callbacks.embedding.base import EmbeddingCallback
from src.metrics.reeb_graph import ReebGraphNodesEdges  # to compute the graph if needed

class ReebGraphVisualizationCallback(EmbeddingCallback):
    """
    Embedding callback to visualize and log Reeb graphs after DR computation.
    """

    def __init__(self, figsize=(8, 6)):
        super().__init__()  # Important: ensures proper EmbeddingCallback initialization
        self.figsize = figsize

    def on_dr_end(self, dataset, embeddings):
        """
        Called when dimensionality reduction completes.
        Automatically computes and visualizes the Reeb graph.
        """
        reeb_data = ReebGraphNodesEdges(embeddings["embeddings"])
        self._plot_and_log(reeb_data["graph"])
        # Optionally register output so it's stored in callback_outputs
        self.register_output("reeb_graph", reeb_data)
        return self.callback_outputs

    def _plot_and_log(self, G):
        """Plot the Reeb graph and log it to WandB."""
        plt.figure(figsize=self.figsize)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos,
            node_size=50,
            node_color='skyblue',
            edge_color='gray',
            with_labels=False
        )
        plt.title("Reeb Graph Visualization")
        if wandb.run is not None:
            wandb.log({"Reeb Graph": wandb.Image(plt)})
        plt.close()
