# Evaluation

How manyLatents dispatches, evaluates, and samples embeddings. The core engine lives in `experiment.py`; evaluation helpers live in `evaluate.py`.

=== "Dispatch"

    ## Algorithm Dispatch

    manyLatents uses a two-level dispatch system to handle both LatentModule (fit/transform) and LightningModule (training loop) algorithms through a unified interface.

    ### Algorithm Resolution

    `run_engine()` determines which algorithm type to instantiate from the algorithm dict:

    ```python
    if "latent" in algorithms and algorithms["latent"] is not None:
        algorithm = instantiate_algorithm(algorithms["latent"], datamodule)
    elif "lightning" in algorithms and algorithms["lightning"] is not None:
        algorithm = instantiate_algorithm(algorithms["lightning"], datamodule)
    else:
        raise ValueError("No algorithm specified in configuration")
    ```

    Only one of `algorithms["latent"]` or `algorithms["lightning"]` should be set per run. The key determines which path is taken.

    ### Execution

    `run_engine()` routes via `isinstance()` checks (the former `execute_step()` logic is now inlined in `run_engine()`):

    ```python
    if isinstance(algorithm, LatentModule):
        algorithm.fit(train_tensor, train_labels)
        latents = algorithm.transform(test_tensor)

    elif isinstance(algorithm, LightningModule):
        trainer.fit(algorithm, datamodule=datamodule)
        latents = algorithm.encode(test_tensor)
    ```

    **LatentModule path:** Direct `fit()` on training data, then `transform()` on test data. Labels are passed for supervised modules (e.g., `ClassifierModule`) and ignored by unsupervised ones.

    **LightningModule path:** Full Lightning training loop via `trainer.fit()`, optional pretrained checkpoint loading, model evaluation via `evaluate()`, then embedding extraction via `encode()`.

    ### Evaluation: `evaluate()` in `evaluate.py`

    The unified `evaluate()` function (in `evaluate.py`) handles both metric formats:

    ```python
    def evaluate(
        embeddings,
        *,
        dataset=None,
        module=None,
        metrics=None,       # list[str] OR dict[str, DictConfig]
        sampling=None,       # dict of instantiated samplers
        cache_dir=None,
        cache=None,
    ) -> dict[str, Any]:
    ```

    | Metric format | Path | When used |
    |---------------|------|-----------|
    | `list[str]` (registry names) | `_evaluate_registry()` | Python API (`run(metrics=["trustworthiness"])`) |
    | `dict[str, DictConfig]` (Hydra configs) | `_evaluate_hydra()` | CLI path (configs with `_target_` and `at` fields) |

    For LightningModule runs, `_evaluate_lightningmodule()` in `experiment.py` handles model-level metrics (`trainer.test()`), then `evaluate()` runs on the extracted embeddings.

=== "Sampling"

    ## Sampling Strategies

    Large datasets can make metric computation expensive. manyLatents provides pluggable sampling strategies that subsample embeddings and datasets before evaluation.

    ### Protocol

    All strategies implement the `SamplingStrategy` protocol:

    ```python
    class SamplingStrategy(Protocol):
        def sample(
            self,
            embeddings: np.ndarray,
            dataset: object,
            n_samples: Optional[int] = None,
            fraction: Optional[float] = None,
            seed: int = 42,
        ) -> Tuple[np.ndarray, object, np.ndarray]:
            # Returns (subsampled_embeddings, subsampled_dataset, indices)
            ...
    ```

    The returned dataset is a deep copy with subsampled `data`, `latitude`, `longitude`, and `population_label` attributes (when present).

    ### Available Strategies

    | Strategy | Config | Use Case |
    |----------|--------|----------|
    | `RandomSampling` | `sampling/random` | Default. Uniform random without replacement |
    | `StratifiedSampling` | `sampling/stratified` | Preserves label distribution across strata |
    | `FarthestPointSampling` | `sampling/farthest_point` | Maximum coverage of embedding space |
    | `FixedIndexSampling` | (programmatic) | Reproducible cross-setting comparisons |

    ### Configuration

    Sampling is configured under top-level `sampling` in Hydra, keyed by output name:

    ```yaml
    # Pre-fit: subsample dataset before algorithm fitting
    sampling:
      dataset:
        _target_: manylatents.utils.sampling.RandomSampling
        seed: 42
        fraction: 0.5

    # Post-fit: subsample embeddings before metric evaluation
    sampling:
      embedding:
        _target_: manylatents.utils.sampling.RandomSampling
        seed: 42
        fraction: 0.1
    ```

    ### Deterministic Indices

    `RandomSampling.get_indices()` precomputes indices for reproducible comparisons:

    ```python
    sampler = RandomSampling(seed=42)
    indices = sampler.get_indices(data, fraction=0.1)
    np.save('shared_indices.npy', indices)

    # Reuse across runs
    fixed = FixedIndexSampling(indices=np.load('shared_indices.npy'))
    emb_sub, ds_sub, _ = fixed.sample(embeddings, dataset)
    ```

    ### How Sampling Integrates

    In `evaluate()`, post-fit sampling runs before any metrics:

    ```python
    # sampling is a dict of pre-instantiated sampler objects
    if sampling is not None:
        for output_name, sampler in sampling.items():
            if output_name == "dataset":
                continue  # pre-fit sampling handled in run_engine()
            indices = sampler.get_indices(outputs[output_name])
            outputs[output_name] = outputs[output_name][indices]
    ```

    Pre-fit sampling (`sampling["dataset"]`) runs in `run_engine()` before `fit()`, reducing the data the algorithm sees. Post-fit sampling (e.g., `sampling["embedding"]`) runs in `evaluate()` before metrics. If no sampling is configured, metrics run on the full dataset.

=== "Caching"

    ## Shared Cache

    Several metrics need k-nearest neighbors, SVD decompositions, or eigenvalue computations. Computing these per-metric would be redundant. manyLatents pre-warms a shared `cache` dict and passes it to all metrics.

    ### How It Works

    `evaluate()` uses the config sleuther (`extract_k_requirements`, in `evaluate.py`) to discover all `k`/`n_neighbors` values from metric configs, then calls `prewarm_cache()` (also in `evaluate.py`) to compute kNN and eigenvalues once with `max(k)`:

    ```python
    # 1. Sleuther extracts requirements from metric configs
    reqs = extract_k_requirements(metric_cfgs)
    # reqs = {"emb_k": {5, 10, 25}, "data_k": {10, 25}, "spectral": True}

    # 2. Pre-warm cache with optimal k values
    cache = prewarm_cache(metric_cfgs, embeddings, dataset, module)
    # cache is keyed by id(data) for kNN, "eigenvalues" for spectral

    # 3. All metrics receive the same cache dict
    result = metric_fn(embeddings=emb, dataset=ds, module=module, cache=cache)
    ```

    ### compute_knn with cache

    `compute_knn()` uses the `cache` dict to avoid redundant computation. If a cached result exists with `k >= requested k`, it slices and returns immediately:

    ```python
    from manylatents.utils.metrics import compute_knn

    cache = {}
    # First call: computes kNN with k=25
    dists, idxs = compute_knn(data, k=25, cache=cache)

    # Second call: reuses cached result, slices to k=10
    dists, idxs = compute_knn(data, k=10, cache=cache)  # instant
    ```

    `compute_knn()` automatically selects the fastest backend: FAISS-GPU > FAISS-CPU > sklearn.

    ### SVD Cache

    `compute_svd_cache()` batches local SVD computation with GPU acceleration (torch) when CUDA is available, falling back to CPU numpy. Results are stored in the same `cache` dict.

    ### Metric Protocol

    All metrics receive `cache=` as a keyword argument. Metrics that need kNN call `compute_knn(..., cache=cache)` internally — the cache ensures no redundant computation. Extension metrics that don't accept `cache=` are handled gracefully via a `TypeError` fallback.

    ### Metric Expansion

    `flatten_and_unroll_metrics()` handles list-valued parameters via Cartesian product:

    ```yaml
    # This config:
    trustworthiness:
      _target_: manylatents.metrics.trustworthiness.Trustworthiness
      _partial_: true
      n_neighbors: [5, 10, 20]

    # Expands to three separate evaluations:
    # trustworthiness__n_neighbors_5
    # trustworthiness__n_neighbors_10
    # trustworthiness__n_neighbors_20
    ```

    This expansion happens before kNN extraction, so all k values from expanded metrics contribute to the shared cache.
