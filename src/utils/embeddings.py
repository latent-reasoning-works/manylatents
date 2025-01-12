import os

import numpy as np
import phate
from sklearn.manifold import TSNE

from .utils import load_pickle, save_pickle


# Compute PHATE embeddings or load if available
def compute_or_load_phate(pca_input, fit_idx, transform_idx, ts, phate_dir, **phate_params):
    os.makedirs(phate_dir, exist_ok=True)
    param_str = "_".join([f"{k}_{v}" for k, v in phate_params.items()])
    file_path = os.path.join(phate_dir, f"phate_{param_str}.pkl")

    if os.path.exists(file_path):
        print(f"Loading PHATE operator from {file_path}")
        phate_operator = load_pickle(file_path)
    else:
        phate_operator = phate.PHATE(random_state=42, **phate_params)
        phate_operator.fit(pca_input[fit_idx])
        save_pickle(phate_operator, file_path)
        print(f"Saved PHATE operator to {file_path}")

    output_embs = []
    for t in ts:
        phate_operator.set_params(t=t)
        phate_emb = np.zeros((len(pca_input), 2))
        phate_emb[fit_idx] = phate_operator.transform(pca_input[fit_idx])
        phate_emb[transform_idx] = phate_operator.transform(pca_input[transform_idx])
        output_embs.append(phate_emb)

    return output_embs, phate_operator

def compute_tsne(pca_input, fit_idx, transform_idx, **tsne_params):
    tsne_obj = TSNE(n_components=2, **tsne_params)
    tsne_emb = np.zeros(shape=(len(pca_input), 2))
    tsne_out = tsne_obj.fit_transform(pca_input[fit_idx | transform_idx])
    tsne_emb[fit_idx | transform_idx] = tsne_out
    return tsne_emb, tsne_obj
