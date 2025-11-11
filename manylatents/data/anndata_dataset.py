import torch
from torch.utils.data import Dataset
import numpy as np
import scanpy as sc


class AnnDataset(Dataset):
    """
    PyTorch Dataset for AnnData objects.
    """
   
    def __init__(self,
                 adata_path: str = None,
                 label_key: str = None,
                ): 
        """
        Initializes the  AnnDataset.

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


if __name__ == "__main__":
    dataset = AnnDataset("./data/scRNAseq/EBT_counts.h5ad", label_key="sample_labels")
    print("Data shape:", dataset.data.shape)
    print("Labels shape:", dataset.metadata.shape)
    print("First sample:", dataset[0])


    
