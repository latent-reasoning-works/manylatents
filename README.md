<div align="center">

<pre>
     .  . .
  .  .. . . ..       .  . .
 . . .  . .. . . -->  . .. .  -->  λ(·)
  . ..  . .  .       .  . .
    .  . .

        m a n y l a t e n t s

  one geometry, learned through many latents
</pre>

[![license](https://img.shields.io/badge/license-MIT-2DD4BF.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11+-2DD4BF.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/pkg-uv-2DD4BF.svg)](https://docs.astral.sh/uv/)
[![PyPI](https://img.shields.io/badge/PyPI-manylatents-2DD4BF.svg)](https://pypi.org/project/manylatents/)

</div>

---

## install

```bash
uv add manylatents
```

Optional extras:

```bash
uv add manylatents[hf]         # HuggingFace trainer
uv add manylatents[torchdr]    # GPU-accelerated DR via TorchDR
uv add manylatents[jax]        # JAX backend (MIOFlow JAX, diffrax)
uv add manylatents[all]        # everything
```

<details>
<summary>pip also works</summary>

```bash
pip install manylatents
```

</details>

## quickstart

```bash
# embed a swiss roll with UMAP
manylatents algorithms/latent=umap data=swissroll

# add metrics
manylatents algorithms/latent=umap data=swissroll \
  metrics/embedding=trustworthiness

# sweep algorithms
manylatents --multirun \
  algorithms/latent=umap,phate,tsne \
  data=swissroll metrics/embedding=trustworthiness
```

<details>
<summary>development install</summary>

```bash
git clone https://github.com/latent-reasoning-works/manylatents.git
cd manylatents && uv sync
```

</details>

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
│   Config   │─────►│    Algorithm      │─────►│  LatentOutputs  │
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

Both produce `LatentOutputs` — a dict keyed by `"embeddings"`. All metrics receive a shared `cache` dict for deduplicated kNN and eigenvalue computation.

---

## [algorithms](https://latent-reasoning-works.github.io/manylatents/algorithms/)

> 12 algorithms -- 8 latent modules, 4 lightning modules

PCA, t-SNE, UMAP, PHATE, DiffusionMap, MDS, Archetypes, MultiscalePHATE,
Autoencoder, AANet, LatentODE, HF Trainer.

`neighborhood_size=k` sweeps kNN uniformly across algorithms.

---

## [metrics](https://latent-reasoning-works.github.io/manylatents/metrics/)

> 20+ metrics across three evaluation contexts

Embedding fidelity (trustworthiness, continuity, kNN preservation), spectral
analysis (affinity spectrum, spectral decay rate), topological features
(persistent homology), and dataset properties (stratification).

All metrics share a `cache` dict for deduplicated kNN computation.
List-valued parameters expand via `flatten_and_unroll_metrics()` --
one kNN computation covers the entire sweep.

Config pattern: `metrics/embedding=<name>`, `metrics/module=<name>`, `metrics/dataset=<name>`

---

## [data](https://latent-reasoning-works.github.io/manylatents/data/)

> 6 synthetic manifolds + precomputed loader

Swiss roll, torus, saddle surface, gaussian blobs, DLA tree, and custom `.npy`/`.npz` files.
Domain-specific datasets (genomics, single-cell) available via extensions.

---

## citing

If manylatents was useful in your research, a citation goes a long way:

```bibtex
@software{manylatents2026,
  title     = {manyLatents: Unified Dimensionality Reduction and Neural Network Analysis},
  author    = {Valdez C{\'o}rdova, C{\'e}sar Miguel and Scicluna, Matthew and Ni, Shuang},
  year      = {2026},
  url       = {https://github.com/latent-reasoning-works/manylatents},
  license   = {MIT}
}
```

---

<br><br>

<p align="center">
<sub>MIT License &middot; <a href="https://github.com/latent-reasoning-works">Latent Reasoning Works</a></sub>
</p>
