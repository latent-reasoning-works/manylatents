# ManyLatents

**Part of the [Latent Reasoning Works](https://github.com/latent-reasoning-works) ecosystem**

ManyLatents is a unified framework for dimensionality reduction and neural network analysis. Built on **PyTorch Lightning** and **Hydra**, it bridges traditional DR techniques with modern neural architectures.

---

## Quick Start

```bash
git clone https://github.com/latent-reasoning-works/manylatents.git
cd manylatents
uv sync
source .venv/bin/activate

python -m manylatents.main data=swissroll algorithms/latent=pca
```

---

## Documentation

- **[Algorithms](algorithms.md)** — LatentModule (fit/transform) and LightningModule (trainable) algorithms, networks, and losses
- **[Metrics](metrics.md)** — Three-level evaluation system: embedding, dataset, and module metrics
- **[Extensions](extensions.md)** — Install, use, and develop domain extensions
- **[API](api_usage.md)** — Programmatic API for agent-driven multi-step workflows
- **[Probing](probing.md)** — Representation probing for auditing algorithm internals during training
- **[Testing](testing.md)** — CI pipeline, namespace integration testing, and mock package patterns

---

## LRW Ecosystem

| Project | Description |
|---------|-------------|
| [manylatents](https://github.com/latent-reasoning-works/manylatents) | Core DR framework |
| [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) | Genomics, population genetics, single-cell extensions |
| [shop](https://github.com/latent-reasoning-works/shop) | Multi-cluster SLURM management |
