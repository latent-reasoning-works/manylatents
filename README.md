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

[![license](https://img.shields.io/badge/license-Apache--2.0-2DD4BF.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11+-2DD4BF.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/pkg-uv-2DD4BF.svg)](https://docs.astral.sh/uv/)
[![PyPI](https://img.shields.io/badge/PyPI-manylatents-2DD4BF.svg)](https://pypi.org/project/manylatents/)

<sub>Nine files ported from KrishnaswamyLab code (MIOFlow, cflows, AAnet) are Yale non-commercial, not Apache-2.0; see <a href="THIRD_PARTY_NOTICES.md">THIRD_PARTY_NOTICES.md</a>.</sub>

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
uv add manylatents[jax]        # JAX backend (diffrax)
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
  metrics=trustworthiness

# sweep algorithms
manylatents --multirun \
  algorithms/latent=umap,phate,tsne \
  data=swissroll metrics=trustworthiness
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

result = run(data="swissroll", algorithm="pca", metrics=["trustworthiness"])

embeddings = result["embeddings"]   # (n, d) ndarray
scores     = result["scores"]       # {"trustworthiness": 0.95}

# chain: PCA 50d -> PHATE 2d
result2 = run(input_data=result["embeddings"], algorithm="phate")
```

---

## architecture

```
  CLI                         API
  manylatents ...             run(data=, algorithm=, ...)
       │                           │
       ▼                           ▼
  ┌─────────┐               ┌───────────┐
  │ main.py │               │  api.py   │
  │ Hydra   │               │ registries│
  └────┬────┘               └─────┬─────┘
       │    instantiate objects    │
       └──────────┬───────────────┘
                  ▼
        ┌──────────────────┐      ┌────────────────┐
        │  experiment.py   │─────►│  evaluate.py   │
        │  run_experiment()│      │  evaluate()    │
        │                  │      │  prewarm_cache │
        │  LatentModule    │      │  compute_knn   │
        │    fit/transform │      │  metric_fn(    │
        │  LightningModule │      │   cache=cache) │
        │    trainer.fit   │      └────────────────┘
        └──────────────────┘
```

Two base classes, one decision rule:

| if the algorithm... | use | interface |
|---|---|---|
| has no training loop | `LatentModule` | `fit(x)` / `transform(x)` |
| trains with backprop | `LightningModule` | `trainer.fit()` / `encode(x)` |

Both produce `LatentOutputs` — a dict keyed by `"embeddings"`. All metrics receive a shared `cache` dict for deduplicated kNN and eigenvalue computation.

---

## [algorithms](https://latent-reasoning-works.github.io/manylatents/algorithms/)

> 17 algorithms -- 12 latent modules, 5 lightning modules

PCA, t-SNE, UMAP, PHATE, DiffusionMap, MDS, MultiscalePHATE, Merging,
Classifier, Leiden, ReebGraph, SelectiveCorrection,
Autoencoder, AANet, LatentODE, HF Trainer, Reconstruction AE.

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

Config pattern: `metrics=<name>` (each metric config has an `at:` field for evaluation context)

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
  license   = {Apache-2.0}
}
```

---

<br><br>

<p align="center">
<sub>Apache-2.0 License &middot; <a href="https://github.com/latent-reasoning-works">Latent Reasoning Works</a></sub>
</p>
