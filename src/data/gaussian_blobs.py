
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

logger = logging.getLogger(__name__)


class GaussianBlobDataset:
    """
    Gaussian K blocs 
    """

    def __init__(self,
                 n_samples: Union[int, List[int]] = 100,
                 n_features: int = 2,
                 centers: Optional[Union[int, np.ndarray, List[List[float]]]] = None,
                 cluster_std: Union[float, List[float]] = 1.0,
                 center_box: Tuple[float, float] = (-10.0, 10.0),
                 shuffle: bool = True,
                 random_state: Optional[int] = 42,
                 return_centers: bool = False):
        
        logger.info("generation")

        # From  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
        result = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            shuffle=shuffle,
            random_state=random_state,
            return_centers=return_centers
        )

        if return_centers:
            self.X, self.y, self.centers = result
        else:
            self.X, self.y = result
            self.centers = None

        # Dataframe
        self.FinalData = pd.DataFrame({
            'sample_id': [f"sample_{i}" for i in range(len(self.X))],
            'label': self.y
        }).set_index("sample_id")

    def get_data(self) -> np.ndarray:
        return self.X

    def get_labels(self) -> np.ndarray:
        return self.y

    def get_FinalData(self) -> pd.DataFrame:
        return self.FinalData

    def get_centers(self) -> Optional[np.ndarray]:
        return self.centers

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return {
            "data": self.X[idx],
            "metadata": self.y[idx]
        }