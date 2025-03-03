import logging
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor
from typing_extensions import NotRequired, Required, TypedDict

from src.utils.utils import save_embeddings
from src.metrics.handler import MetricsHandler

logger = logging.getLogger(__name__)

class DimensionalityReductionOutputs(TypedDict, total=False):
    """
    Defines the minimal output contract for DR models.
    All DR LightningModules should return at least an 'embeddings' tensor,
    along with optional 'loss' and 'label' entries.
    """
    embeddings: Required[Tensor]
    """Reduced embeddings from the model, e.g. PHATE, UMAP, PCA."""
    
    loss: NotRequired[Union[torch.Tensor, float]]
    """Optional loss value from the training/validation step."""
    
    label: NotRequired[Tensor]
    """Optional labels for the embeddings, if available."""
    
class DimensionalityReductionCallback(ABC):
    @abstractmethod
    def on_dr_end(self, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        """
        Called when the Dimensionality Reduction process is complete.
        
        Args:
            embeddings (np.ndarray): The computed embeddings.
            labels (np.ndarray, optional): Optional labels associated with the embeddings.
        """
        pass

class SaveEmbeddings(DimensionalityReductionCallback):
    def __init__(self, save_dir: str = "outputs", save_format: str = "npy") -> None:
        self.save_dir = save_dir
        self.save_format = save_format
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"SaveEmbeddings initialized with directory: {self.save_dir} and format: {self.save_format}")

    def save_embeddings(self, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        logger.debug(f"save_embeddings() called with embeddings shape: {embeddings.shape}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{timestamp}.{self.save_format}"
        save_path = os.path.join(self.save_dir, filename)
        logger.info(f"Computed save path: {save_path}")
        
        metadata = {"labels": labels} if labels is not None else None

        try:
            save_embeddings(embeddings, save_path, format=self.save_format, metadata=metadata)
            logger.info(f"Saved embeddings successfully to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            warnings.warn(f"Failed to save embeddings: {e}")

    def plot_embeddings(self, embeddings: np.ndarray, labels: np.ndarray = None, save_path: str = None) -> None:
            """
            Generates and saves a scatter plot of the embeddings.
            """
            if embeddings.shape[1] != 2:
                logger.info("Skipping plot: Embeddings must be 2D for visualization.")
                return

            logger.info("Generating embedding plot...")
            plt.figure(figsize=(8, 6))

            if labels is not None:
                sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels, palette="viridis", alpha=0.7)
            else:
                plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7)

            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.title("Dimensionality Reduction Visualization")
            plt.legend()

            if save_path:
                plot_path = save_path.replace(f".{self.save_format}", ".png")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(self.save_dir, f"embedding_plot_{timestamp}.png")

            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved plot to {plot_path}")

    def on_dr_end(self, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        logger.debug("on_dr_end() called; delegating to save_embeddings()")
        self.save_embeddings(embeddings, labels)

class AdditionalMetrics(DimensionalityReductionCallback):
    def __init__(self, metrics_config, metadata: np.ndarray = None):
        """
        Initializes the callback with a metrics configuration and optionally the original data.

        Args:
            metrics_config: A Hydra config (or dict) specifying the metrics to compute.
            metadata (np.ndarray, optional): Additional experimental data for metrics computation.
        """
        self.metrics_handler = MetricsHandler(metrics_config)
        self.metadata = metadata

    def on_dr_end(self, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        """
        Computes additional metrics using the MetricsHandler and logs the results.
        """
        if self.metadata is None:
            logger.warning("No metadata provided for metrics computation. Skipping additional metrics.")
            return

        try:
            results = self.metrics_handler.compute_all(
                original=self.metadata, 
                embedded=embeddings,
                labels=labels
            )
            logger.info(f"Computed DR metrics: {results}")
        except Exception as e:
            logger.error(f"Error while computing DR metrics: {e}")
