import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional
# DEPRECATED: PrecomputedMixin removed. Use PrecomputedDataModule for loading saved embeddings.
import scanpy as sc


class SinglecellDataset(Dataset):
    """
    PyTorch Dataset for single-cell AnnData objects.
    """
   
    def __init__(self,
                 adata_path: str = None,
                 label_key: str = None,
                ): 
        """
        Initializes the single-cell dataset.

        Parameters
        ----------
        adata : AnnData
            AnnData object containing gene expression data and metadata.

        label_key : str or None
            Key in adata.obs used as cell labels (e.g., cell type or condition).
            If None, assigns all-zero dummy labels.
        """
        adata = sc.read_h5ad(adata_path)
        self.data = torch.tensor(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=torch.float32)

        if self.data is None:
            raise ValueError("No data source found: either AnnData or precomputed embeddings are missing.")  
        
        if label_key is not None and label_key in adata.obs:
            self.metadata = adata.obs[label_key].astype(str).values
        else:
            self.metadata = np.array([0] * adata.shape[0])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.metadata[idx] if self.metadata is not None else -1
        return {"data": x, "metadata": y}

    def get_labels(self):
        return self.metadata

    def get_data(self):
        return self.data


class EmbryoidBody(SinglecellDataset):
    def __init__(
        self,
        adata_path: str,
        label_key: str = "cell_type",
        precomputed_path: Optional[str] = None,
        mmap_mode: Optional[str] = None,
    ):
        """
        PyTorch Dataset for Embryoid body single-cell dataset loaded from an .h5ad file.

        Parameters
        ----------
        adata_path : str
            Path to the raw .h5ad file containing the full dataset.

        label_key : str, default="cell_type"
            Key in adata.obs used to retrieve cell type or condition labels.

        precomputed_path : str or None, optional
            Optional path to a precomputed .h5ad file. If provided and exists, this file is used instead of adata_path.

        mmap_mode: str or None, optional
            Memory-mapping mode for large datasets.
        """
        super().__init__(
            adata_path=adata_path,
            label_key=label_key,
            precomputed_path=precomputed_path,
            mmap_mode=mmap_mode,
        )



if __name__ == "__main__":
    dataset = EmbryoidBody("./data/scRNAseq/EBT_counts.h5ad", label_key="sample_labels")
    print("Data shape:", dataset.data.shape)
    print("Labels shape:", dataset.metadata.shape)
    print("First sample:", dataset[0])


    
