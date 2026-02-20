#!/usr/bin/env python
"""One-time data prep: normalize + log1p + PCA(50) for embryoid body h5ad.

Saves PCA features and labels as .npy for use with precomputed data config.
Run once before submitting artifact_grid_sweep experiments.

Usage:
    .venv/bin/python scripts/prep_embryoid_body_pca.py
"""
import numpy as np
import scanpy as sc

DATA = "data/scRNAseq/EBT_2k_hvg.h5ad"
OUT_PCA = "data/scRNAseq/EBT_2k_hvg_pca50.npy"
OUT_LABELS = "data/scRNAseq/EBT_2k_hvg_labels.npy"

print(f"Loading {DATA} ...")
adata = sc.read_h5ad(DATA)
print(f"  Shape: {adata.shape}")

print("Preprocessing: normalize_total + log1p + PCA(50) ...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, n_comps=50, svd_solver="arpack")

np.save(OUT_PCA, adata.obsm["X_pca"].astype(np.float32))
np.save(OUT_LABELS, adata.obs["sample_labels"].values)
print(f"  Saved: {OUT_PCA} ({adata.obsm['X_pca'].shape})")
print(f"  Saved: {OUT_LABELS} ({adata.obs['sample_labels'].nunique()} unique labels)")
