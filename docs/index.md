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

### Core

- **[Extensions](extensions.md)** — Install, use, and develop domain extensions (Usage | Architecture | Development)
- **[API Usage](api_usage.md)** — Programmatic API for chaining algorithms
- **[Metrics Architecture](metrics_architecture.md)** — Three-level metrics system design

### Guides

- **[Representation Probing](guides/representation-probing.md)** — Probe neural network representations during training
- **[Null Metrics](null_metrics.md)** — Running experiments without metrics
- **[Testing](testing.md)** — Testing infrastructure and CI pipeline
- **[Integration Testing](integration_testing_guide.md)** — Namespace integration testing
- **[Local Namespace Testing](local_namespace_testing.md)** — Mock package testing for CI

### Architecture

- **[Probing Architecture](designs/probing-architecture.md)** — ADR for representation probing design
- **[HF Representation Audit](designs/hf-representation-audit-architecture.md)** — Architecture for HuggingFace representation auditing

---

## LRW Ecosystem

| Project | Description |
|---------|-------------|
| [manylatents](https://github.com/latent-reasoning-works/manylatents) | Core DR framework |
| [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) | Genomics, population genetics, single-cell extensions |
| [shop](https://github.com/latent-reasoning-works/shop) | Multi-cluster SLURM management |
