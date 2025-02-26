import logging
import os
import warnings
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from typing_extensions import NotRequired, Required, TypedDict, override

logger = logging.getLogger(__name__)

class DimensionalityReductionOutputs(TypedDict, total=False):
    """
    Minimally returned output from the training/val Dimensionality Reduction
    LightningModules so that metrics can be automatically casted into the 
    DimensionalityReductionMetricsCallback.
    """
    embeddings: Required[Tensor]
    """Reduced embeddings from the model, e.g. PHATE, UMAP, PCA."""
    
    loss: NotRequired[Union[torch.Tensor, float]]      
    """Loss value from the training/val step."""
    
    label: NotRequired[Tensor]
    """Labels for the embeddings, if available."""

class DimensionalityReductionMetricsCallback(Callback):
    def __init__(self, save_dir: str = "outputs", save_format: str = "npy") -> None:
        """
        Callback to store dimensionality reduction embeddings and optionally compute Rand Score.
        
        Args:
            save_dir (str): Directory where embeddings will be saved.
            save_format (str): Format for saving ('npy', 'csv', 'pt', 'h5').
        """
        super().__init__()
        self.embeddings_buffer = []
        self.labels_buffer = []
        self.save_dir = save_dir
        self.save_format = save_format

        os.makedirs(self.save_dir, exist_ok=True)

    @override
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: DimensionalityReductionOutputs,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Collect embeddings and labels at the end of each validation batch.
        """
        if "embeddings" in outputs:
            self.embeddings_buffer.append(outputs["embeddings"].detach().cpu())
        if "label" in outputs:
            self.labels_buffer.append(outputs["label"].detach().cpu())

    @override
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        At the end of validation, concatenate stored embeddings and save them.
        """
        if not self.embeddings_buffer:
            return
        
        all_embeddings = torch.cat(self.embeddings_buffer, dim=0).numpy()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"embeddings_{timestamp}.{self.save_format}")

        # Save embeddings in the chosen format
        if self.save_format == 'npy':
            np.save(save_path, all_embeddings)
        elif self.save_format == 'csv':
            df = pd.DataFrame(all_embeddings, columns=[f"dim_{i}" for i in range(all_embeddings.shape[1])])
            df.to_csv(save_path, index=False)
        elif self.save_format == 'pt':
            torch.save(torch.tensor(all_embeddings), save_path)
        elif self.save_format == 'h5':
            import h5py
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('embeddings', data=all_embeddings)
        else:
            warnings.warn(f"Unsupported save format: {self.save_format}")

        logger.info(f"Saved embeddings to {save_path}")

        """
        Currently throws an import error, verify torchmetrics
        # Compute Rand Score (if labels are available)
        ## This should be generalized for n many metrics
        ## and proper flow control should be implemented
        if self.labels_buffer:
            all_labels = torch.cat(self.labels_buffer, dim=0).numpy()
            try:
                rand_score = RandScore()(torch.tensor(all_embeddings), torch.tensor(all_labels))
                pl_module.log("val/RandScore", rand_score, prog_bar=True)
            except Exception as e:
                warnings.warn(f"Could not compute RandScore: {e}")
        """

        self.embeddings_buffer.clear()
        self.labels_buffer.clear()
