# Cache Protocol

All metrics share a single `cache` dict. The config sleuther discovers k-values from metric configs and pre-warms kNN and eigenvalues before any metric runs.

```python
# this happens automatically inside evaluate_embeddings()
cache = {}
compute_knn(high_dim_data, k=25, cache=cache)    # computed once
compute_knn(embeddings,     k=25, cache=cache)    # computed once
compute_eigenvalues(module, cache=cache)           # computed once

# every metric reuses the same cache
trustworthiness(emb, dataset=ds, cache=cache)
continuity(emb, dataset=ds, cache=cache)
```

`compute_knn` selects the fastest available backend: FAISS-GPU > FAISS-CPU > sklearn.

## How It Works

`evaluate_embeddings()` uses `extract_k_requirements()` to discover all `k`/`n_neighbors` values from metric configs, then calls `prewarm_cache()` to compute kNN once with `max(k)`:

1. **Sleuther** extracts requirements from metric configs
2. **Pre-warm** computes kNN and eigenvalues at optimal k values
3. **Metrics** receive the shared cache — `compute_knn()` slices cached results for smaller k values

## Metric Expansion

List-valued parameters expand via Cartesian product through `flatten_and_unroll_metrics()`:

```yaml
trustworthiness:
  _target_: manylatents.metrics.trustworthiness.Trustworthiness
  _partial_: true
  n_neighbors: [5, 10, 20]

# expands to three evaluations:
# embedding.trustworthiness__n_neighbors_5
# embedding.trustworthiness__n_neighbors_10
# embedding.trustworthiness__n_neighbors_20
```

All expanded k-values contribute to the shared cache — one kNN computation covers the entire sweep.

## Extension Metrics

Extension metrics that don't accept `cache=` are handled gracefully via a `TypeError` fallback. A warning is logged suggesting the extension add `cache=None` to its signature.
