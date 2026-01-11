# Welcome to ManyLatents

**Part of the [Latent Reasoning Works](https://github.com/latent-reasoning-works) ecosystem**

ManyLatents is a unified framework for dimensionality reduction and neural network analysis on diverse datasets. Built on **PyTorch Lightning** and **Hydra**, it bridges traditional DR techniques with modern neural architectures.

---

## What You'll Find Here

- **[Extensions Guide](extensions.md)**: Install domain-specific extensions (genomics, etc.)
- **[API Usage Guide](api_usage.md)**: Programmatic API for chaining algorithms
- **[Metrics Architecture](metrics_architecture.md)**: Three-level metrics system design
- **[Testing Guide](testing.md)**: Testing infrastructure and best practices
- **[Null Metrics](null_metrics.md)**: Running experiments without metrics

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/latent-reasoning-works/manylatents.git
cd manylatents
uv sync
source .venv/bin/activate

# Run an experiment
python -m manylatents.main data=swissroll algorithms/latent=pca
```

---

## Extensions

ManyLatents supports domain-specific extensions through namespace packages:

- **[manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics)**: Genomics and population genetics
  - PLINK/VCF data loaders
  - Geographic preservation metrics
  - Ancestry-specific algorithms

Install extensions:
```bash
uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
```

[Learn more about extensions →](extensions.md)

---

## LRW Ecosystem

ManyLatents is part of the **Latent Reasoning Works** suite:

| Project | Description |
|---------|-------------|
| [manylatents](https://github.com/latent-reasoning-works/manylatents) | Core DR framework |
| [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) | Genomics extensions |
| [shop](https://github.com/latent-reasoning-works/shop) | Multi-cluster SLURM management |
