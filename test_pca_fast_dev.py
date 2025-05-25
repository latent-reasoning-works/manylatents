import torch
from src.algorithms.pca import PCAModule
import numpy as np

def test_pca_fast_dev():
    # Create a large synthetic dataset
    n_samples = 1000
    n_features = 50
    X = torch.randn(n_samples, n_features)
    
    # Initialize PCA with fast_dev_run_dr enabled
    pca = PCAModule(
        n_components=2,
        random_state=42,
        fast_dev_run_dr=True,
        n_samples_fast_dev=100
    )
    
    # Fit PCA
    pca.fit(X)
    
    # Transform the data
    X_transformed = pca.transform(X)
    
    # Print shapes to verify
    print(f"Original data shape: {X.shape}")
    print(f"Transformed data shape: {X_transformed.shape}")
    
    # Verify that the model was fitted on a subset of data
    print(f"Number of components: {pca.n_components}")
    print(f"Was fast_dev_run_dr used? {pca.fast_dev_run_dr}")
    print(f"Number of samples used for fitting: {pca.n_samples_fast_dev}")

if __name__ == "__main__":
    test_pca_fast_dev() 