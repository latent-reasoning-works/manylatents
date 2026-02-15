<div align="center">

<pre>
        ·  ·  ·
      · · · · · ·                 · · ·
     · · · · · · · ·  ────────► · · · · ·
      · · · · · ·                 · · ·
        ·  ·  ·

           m a n y l a t e n t s

    one geometry, learned through many latents
</pre>

Dimensionality reduction and neural network analysis.
Built on PyTorch Lightning + Hydra.

[![license](https://img.shields.io/badge/license-MIT-a0a0a0.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10+-a0a0a0.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/pkg-uv-a0a0a0.svg)](https://docs.astral.sh/uv/)

</div>

---

## quickstart

```bash
git clone https://github.com/latent-reasoning-works/manylatents.git
cd manylatents && uv sync

# embed a swiss roll with UMAP
uv run python -m manylatents.main algorithms/latent=umap data=swissroll

# add metrics, log to wandb
uv run python -m manylatents.main \
  algorithms/latent=umap data=swissroll \
  metrics/embedding=trustworthiness logger=wandb

# sweep algorithms and neighborhood sizes
uv run python -m manylatents.main --multirun \
  algorithms/latent=umap,phate,tsne \
  neighborhood_size=5,10,15,30 \
  data=swissroll metrics/embedding=trustworthiness
```

```python
from manylatents.api import run

result = run(
    data="swissroll",
    algorithms={"latent": "pca"},
    metrics={"embedding": {"trustworthiness": {
        "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
        "_partial_": True, "n_neighbors": 5
    }}}
)

embeddings = result["embeddings"]   # (n, d) ndarray
scores     = result["scores"]       # {"embedding.trustworthiness": 0.95}

# chain: PCA 50d -> PHATE 2d
result2 = run(input_data=result["embeddings"], algorithms={"latent": "phate"})
```

---

## architecture

```
┌────────────┐      ┌───────────────────┐      ┌────────────────┐
│   Config   │─────►│    Algorithm      │─────►│ EmbeddingOutputs│
│            │      │                   │      │                │
│ algorithms │      │  LatentModule     │      │ dict[str, Any] │
│ data       │      │    fit(x)         │      │ "embeddings"   │
│ metrics    │      │    transform(x)   │      └───────┬────────┘
│ callbacks  │      │                   │              │
│ logger     │      │  LightningModule  │       ┌──────▼────────┐
└────────────┘      │    trainer.fit()  │       │   Evaluate    │
                    │    encode(x)      │       │               │
                    └───────────────────┘       │ prewarm_cache │
                                               │ compute_knn   │
                                               │ metric_fn(    │
                                               │  ...,         │
                                               │  cache=cache) │
                                               └───────────────┘
```

Two base classes, one decision rule:

| if the algorithm... | use | interface |
|---|---|---|
| has no training loop | `LatentModule` | `fit(x)` / `transform(x)` |
| trains with backprop | `LightningModule` | `trainer.fit()` / `encode(x)` |

Both produce `EmbeddingOutputs` — a dict keyed by `"embeddings"`. All metrics receive a shared `cache` dict for deduplicated kNN and eigenvalue computation.

---

## algorithms

> 8 latent modules, 4 lightning modules

| algorithm | type | config | neighborhood param |
|---|---|---|---|
| PCA | latent | `algorithms/latent=pca` | -- |
| t-SNE | latent | `algorithms/latent=tsne` | `perplexity` |
| UMAP | latent | `algorithms/latent=umap` | `n_neighbors` |
| PHATE | latent | `algorithms/latent=phate` | `knn` |
| DiffusionMap | latent | `algorithms/latent=diffusionmap` | `knn` |
| MDS | latent | `algorithms/latent=mds` | -- |
| Archetypes | latent | `algorithms/latent=aa` | -- |
| MultiscalePHATE | latent | `algorithms/latent=multiscale_phate` | `knn` |
| Autoencoder | lightning | `algorithms/lightning=ae_reconstruction` | -- |
| AANet | lightning | `algorithms/lightning=aanet_reconstruction` | -- |
| LatentODE | lightning | `algorithms/lightning=latent_ode` | -- |
| HF Trainer | lightning | `algorithms/lightning=hf_trainer` | -- |

`neighborhood_size=k` sweeps kNN uniformly across algorithms. Maps to each algorithm's native parameter (UMAP `n_neighbors`, PHATE `knn`, t-SNE `perplexity * 3`).

---

## metrics

> 20+ metrics across three evaluation contexts

### embedding metrics

Compare high-dimensional input to low-dimensional output.

| metric | config | measures |
|---|---|---|
| Trustworthiness | `trustworthiness` | local neighborhood preservation |
| Continuity | `continuity` | reverse neighborhood preservation |
| kNN Preservation | `knn_preservation` | kNN graph overlap |
| Local Intrinsic Dim | `local_intrinsic_dimensionality` | per-point dimensionality |
| Participation Ratio | `participation_ratio` | effective dimensionality |
| Fractal Dimension | `fractal_dimension` | box-counting dimension |
| Tangent Space | `tangent_space` | local manifold alignment |
| Persistent Homology | `persistent_homology` | topological features (via ripser) |
| Diffusion Spectral Entropy | `diffusion_spectral_entropy` | spectral complexity |
| Diffusion Curvature | `diffusion_curvature` | local curvature |
| Anisotropy | `anisotropy` | embedding uniformity |
| Pearson Correlation | `pearson_correlation` | distance correlation |

Config pattern: `metrics/embedding=<name>`

### module metrics

Evaluate algorithm internals. Require a fitted module exposing `affinity_matrix()` or `kernel_matrix()`.

| metric | config | measures |
|---|---|---|
| AffinitySpectrum | `affinity_spectrum` | top-k eigenvalues |
| SpectralGapRatio | `spectral_gap_ratio` | lambda_1 / lambda_2 |
| SpectralDecayRate | `spectral_decay_rate` | log-eigenvalue slope |
| Connected Components | `connected_components` | graph connectivity |
| Kernel Sparsity | `kernel_matrix_sparsity` | kernel matrix density |
| DiffusionMap Correlation | `diffusion_map_correlation` | diffusion distance fidelity |

Config pattern: `metrics/module=<name>`

### dataset metrics

Evaluate original data properties, independent of embedding.

| metric | config | measures |
|---|---|---|
| Stratification | `stratification` | population structure |
| Admixture Laplacian | `admixture_laplacian` | admixture gradients |

Config pattern: `metrics/dataset=<name>`

### sampling

Large datasets are subsampled before metric evaluation.

| strategy | config | method |
|---|---|---|
| Random | `sampling/random` | uniform without replacement |
| Stratified | `sampling/stratified` | preserves label distribution |
| Farthest Point | `sampling/farthest_point` | maximum coverage of embedding space |

---

## cache protocol

All metrics share a single `cache` dict. The config sleuther discovers k-values from metric configs and pre-warms kNN and eigenvalues before any metric runs.

```python
# this happens automatically inside evaluate_embeddings()
cache = {}
compute_knn(high_dim_data, k=25, cache=cache)    # computed once
compute_knn(embeddings,     k=25, cache=cache)    # computed once
compute_eigenvalues(module, cache=cache)           # computed once

# every metric reuses the same cache — zero redundant work
trustworthiness(emb, dataset=ds, cache=cache)
continuity(emb, dataset=ds, cache=cache)
```

`compute_knn` selects the fastest available backend: FAISS-GPU > FAISS-CPU > sklearn.

---

## data

| dataset | config | geometry |
|---|---|---|
| Swiss Roll | `data=swissroll` | 3D spiral manifold |
| Torus | `data=torus` | toroidal surface |
| Saddle Surface | `data=saddle_surface` | hyperbolic paraboloid |
| Gaussian Blobs | `data=gaussian_blobs` | isotropic clusters |
| DLA Tree | `data=dla_tree` | diffusion-limited aggregation |
| Precomputed | `data=precomputed` | load from .npy / .npz |

Domain-specific datasets (genomics, single-cell) available via [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics).

---

## extensions

Namespace packages that add algorithms, metrics, and data modules to the `manylatents` namespace. Core never imports from extensions; extensions import from core.

| extension | adds | install |
|---|---|---|
| [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) | foundation encoders, population genetics metrics, single-cell data | `uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git` |

See [docs/extensions.md](docs/extensions.md) for creating your own.

---

<div align="center">

MIT License -- Cesar Miguel Valdez Cordova, Matthew Scicluna, Shuang Ni, and contributors -- [Mila](https://mila.quebec)

</div>
