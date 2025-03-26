import logging
import os
from datetime import datetime

from src.callbacks.dimensionality_reduction.base import DimensionalityReductionCallback
from src.utils.utils import save_embeddings

logger = logging.getLogger(__name__)

class SaveEmbeddings(DimensionalityReductionCallback):
    def __init__(self, save_dir: str = "outputs", 
                 save_format: str = "npy", 
                 experiment_name: str = "experiment") -> None:
        
        self.save_dir = save_dir
        self.save_format = save_format
        self.experiment_name = experiment_name
        
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"SaveEmbeddings initialized with directory: {self.save_dir} and format: {self.save_format}")

    def save_embeddings(self, dr_outputs: dict) -> None:
        # Directly extract the embeddings (and labels, if available) from the unified dr_outputs.
        embeddings = dr_outputs["embeddings"]
        metadata = dr_outputs.get("metadata")
        if metadata is None:
            metadata = {}
        if "labels" not in metadata and "label" in dr_outputs:
            metadata["labels"] = dr_outputs["label"]

        logger.debug(f"save_embeddings() called with embeddings shape: {embeddings.shape}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{self.experiment_name}_{timestamp}.{self.save_format}"
        self.save_path = os.path.join(self.save_dir, filename)
        logger.info(f"Computed save path: {self.save_path}")

        save_embeddings(embeddings, self.save_path, format=self.save_format, metadata=metadata)
        logger.info(f"Saved embeddings successfully to {self.save_path}")

    def on_dr_end(self, dataset: any, dr_outputs: dict) -> str:
        logger.debug("on_dr_end() called; delegating to save_embeddings()")
        # If labels weren't provided in dr_outputs, extract them from the dataset.
        # may be a redundant check, given how they're accessed in main
        if "label" not in dr_outputs and hasattr(dataset, "get_labels"):
            dr_outputs["label"] = dataset.get_labels()
        self.save_embeddings(dr_outputs)
        return self.save_path

