# Data

manyLatents provides synthetic manifold datasets for benchmarking and a precomputed loader for custom data.

{{ data_table() }}

Domain-specific datasets (genomics, single-cell) are available via the [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) extension.

## Precomputed Data

Load your own data from `.npy` or `.npz` files:

```bash
python -m manylatents.main data=precomputed data.path=/path/to/data.npy algorithms/latent=umap
```

## Sampling

Large datasets are subsampled before metric evaluation. Configure under `metrics.sampling`:

{{ sampling_table() }}
