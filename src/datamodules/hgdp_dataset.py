import os
import hashlib
import tqdm
from typing import Optional, Any, Tuple
import numpy as np
import pandas as pd
from pyplink import PyPlink
from torch.utils.data import Dataset

from plink_dataset import PlinkDataset

class HGDPDataset(PlinkDataset):
    """
    PyTorch  Dataset for the Thousand Genomes Project + Human Genome Diversity Project (HGDP) dataset.
    """

    def extract_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        filters = ["filter_pca_outlier", "hard_filtered", "filter_contaminated"]
        _filtered_indices = self.metadata[self.metadata[filters].any(axis=1)].index
        filtered_indices = ~self.metadata.index.isin(_filtered_indices)
        related_indices = ~self.metadata['filter_king_related'].values

        to_fit_on = related_indices & filtered_indices
        to_transform_on = (~related_indices) & filtered_indices

        return to_fit_on, to_transform_on

    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        metadata = pd.read_csv(metadata_path)

        # because HGDP metadata is missing first row, we manually add dummy first row
        null_row = pd.DataFrame([{col: np.nan for col in metadata.columns}])
        for _filter in  ["filter_king_related", "filter_pca_outlier", "hard_filtered", "filter_contaminated"]:
            null_row[_filter] = False
        metadata = pd.concat([null_row, metadata], ignore_index=True)
        return metadata.set_index('project_meta.sample_id')
