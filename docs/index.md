# manylatents

Dimensionality reduction and neural network analysis. Built on **PyTorch Lightning** and **Hydra**.

---

## Quick Start

```bash
git clone https://github.com/latent-reasoning-works/manylatents.git
cd manylatents && uv sync

uv run python -m manylatents.main data=swissroll algorithms/latent=pca
```

---

## Documentation

- **[Algorithms](algorithms.md)** — LatentModule (fit/transform) and LightningModule (trainable) algorithms, networks, and losses
- **[Data](data.md)** — Synthetic manifolds, precomputed data, and sampling strategies
- **[Metrics](metrics.md)** — Three-level evaluation system: embedding, dataset, and module metrics
- **[Evaluation](evaluation.md)** — Algorithm dispatch, sampling strategies, and shared caching
- **[Cache Protocol](cache.md)** — Shared kNN cache, config sleuther, metric expansion
- **[Callbacks](callbacks.md)** — Embedding callbacks (save, plot, wandb) and trainer callbacks (probing)
- **[Extensions](extensions.md)** — Install, use, and develop domain extensions
- **[API](api_usage.md)** — Programmatic API for agent-driven multi-step workflows
- **[Probing](probing.md)** — Representation probing for auditing algorithm internals during training
- **[Testing](testing.md)** — CI pipeline, namespace integration testing, and mock package patterns
