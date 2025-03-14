import numpy as np
from sklearn.manifold import trustworthiness as sk_trustworthiness


def Trustworthiness(dataset, embeddings: np.ndarray,
                    n_neighbors, metric) -> float:
    return sk_trustworthiness(X=dataset.full_data, 
                              X_embedded=embeddings, 
                              n_neighbors=n_neighbors, 
                              metric=metric)