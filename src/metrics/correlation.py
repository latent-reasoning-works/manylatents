import numpy as np
from scipy.spatial.distance import pdist


def PearsonCorrelation(original_x, embeddings) -> float:
    """
    Compute the Pearson correlation between the pairwise distances
    of the original data and the embeddings. Accepts either torch.Tensors
    or numpy arrays.
    """
    # Convert to numpy arrays if needed.
    if hasattr(original_x, "detach"):
        orig_np = original_x.detach().cpu().numpy()
    else:
        orig_np = original_x

    if hasattr(embeddings, "detach"):
        emb_np = embeddings.detach().cpu().numpy()
    else:
        emb_np = embeddings

    orig_dists = pdist(orig_np)
    emb_dists = pdist(emb_np)
    corr = np.corrcoef(orig_dists, emb_dists)[0, 1]
    return corr
