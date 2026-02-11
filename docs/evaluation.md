# Evaluation

How manyLatents dispatches, evaluates, and samples embeddings. The core engine lives in `experiment.py`.

=== "Dispatch"

    ## Algorithm Dispatch

    manyLatents uses a two-level dispatch system to handle both LatentModule (fit/transform) and LightningModule (training loop) algorithms through a unified interface.

    ### Algorithm Resolution

    `run_algorithm()` determines which algorithm type to instantiate from the Hydra config:

    ```python
    if hasattr(cfg.algorithms, 'latent') and cfg.algorithms.latent is not None:
        algorithm = instantiate_algorithm(cfg.algorithms.latent, datamodule)
    elif hasattr(cfg.algorithms, 'lightning') and cfg.algorithms.lightning is not None:
        algorithm = instantiate_algorithm(cfg.algorithms.lightning, datamodule)
    else:
        raise ValueError("No algorithm specified in configuration")
    ```

    Only one of `algorithms/latent` or `algorithms/lightning` should be set per run. The config group determines which path is taken.

    ### Execution: `execute_step()`

    `execute_step()` routes via `isinstance()` checks:

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

    ### Evaluation: `@functools.singledispatch`

    The `evaluate()` function uses Python's `@functools.singledispatch` to dispatch on the first argument's type:

    ```python
    @functools.singledispatch
    def evaluate(algorithm: Any, /, **kwargs):
        raise NotImplementedError(...)

    @evaluate.register(dict)
    def evaluate_embeddings(EmbeddingOutputs: dict, *, cfg, datamodule, **kwargs):
        # Handles embedding-level metrics (trustworthiness, continuity, etc.)
        ...

    @evaluate.register(LightningModule)
    def evaluate_lightningmodule(algorithm: LightningModule, *, cfg, trainer, datamodule, **kwargs):
        # Handles trainer.test() and model-specific metrics
        ...
    ```

    | Dispatch Type | Handler | Evaluates |
    |---------------|---------|-----------|
    | `dict` (EmbeddingOutputs) | `evaluate_embeddings()` | Embedding metrics (trustworthiness, continuity, kNN preservation, etc.) |
    | `LightningModule` | `evaluate_lightningmodule()` | `trainer.test()` results + custom model metrics |

    Both paths are called during a LightningModule run: first `evaluate_lightningmodule` during `execute_step()`, then `evaluate_embeddings` on the extracted embeddings.

    ### Pipeline Mode

    `run_pipeline()` chains multiple steps sequentially, where step N's output embeddings become step N+1's input. The dispatch logic is reused per step via `execute_step()`.

    ```bash
    # PCA (1000→50) → PHATE (50→2)
    python -m manylatents.main experiment=my_pipeline
    ```

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

    Sampling is configured under `metrics.sampling` in Hydra:

    ```yaml
    # Random (default)
    metrics:
      sampling:
        _target_: manylatents.utils.sampling.RandomSampling
        seed: 42
        fraction: 0.1

    # Stratified by population label
    metrics:
      sampling:
        _target_: manylatents.utils.sampling.StratifiedSampling
        stratify_by: population_label
        seed: 42
        fraction: 0.1

    # Farthest point (O(n*k) — slower but better coverage)
    metrics:
      sampling:
        _target_: manylatents.utils.sampling.FarthestPointSampling
        seed: 42
        fraction: 0.1
    ```

    ### Deterministic Indices

    `RandomSampling.get_indices()` precomputes indices without requiring data, enabling reproducible comparisons:

    ```python
    sampler = RandomSampling(seed=42)
    indices = sampler.get_indices(n_total=1000, fraction=0.1)
    np.save('shared_indices.npy', indices)

    # Reuse across runs
    fixed = FixedIndexSampling(indices=np.load('shared_indices.npy'))
    emb_sub, ds_sub, _ = fixed.sample(embeddings, dataset)
    ```

    ### How Sampling Integrates

    In `evaluate_embeddings()`, sampling runs before any metrics:

    ```python
    sampling_cfg = cfg.metrics.get("sampling", None)
    if sampling_cfg is not None:
        sampler = hydra.utils.instantiate(sampling_cfg)
        emb_sub, ds_sub, _ = sampler.sample(embeddings, ds)
    ```

    All metrics then operate on the subsampled data. If no sampling config is provided, metrics run on the full dataset.

=== "Caching"

    ## Shared Caches

    Several metrics need k-nearest neighbors or SVD decompositions. Computing these per-metric would be redundant. manyLatents computes them once and shares the results.

    ### kNN Cache

    `evaluate_embeddings()` extracts all `k` and `n_neighbors` values from metric configs, computes kNN once with `max(k)`, and passes the cache to metrics that accept `_knn_cache`:

    ```python
    # 1. Extract all k values from metric configs
    k_values = _extract_k_values(metric_cfgs)  # e.g., {5, 10, 20}

    # 2. Compute kNN once with max(k)
    knn_cache = compute_knn(embeddings, k=max(k_values), include_self=True)

    # 3. Pass to metrics that accept it
    if "_knn_cache" in sig.parameters:
        call_kwargs["_knn_cache"] = knn_cache
    ```

    `compute_knn()` automatically selects the fastest backend: FAISS-GPU > FAISS-CPU > sklearn.

    ### SVD Cache

    For metrics like participation ratio and tangent space alignment that need local SVD:

    ```python
    # Discover metrics accepting _svd_cache
    for metric_cfg in metric_cfgs.values():
        sig = inspect.signature(hydra.utils.instantiate(metric_cfg))
        if "_svd_cache" in sig.parameters:
            svd_k_values.add(...)

    # Compute once
    svd_cache = compute_svd_cache(embeddings, knn_indices, svd_k_values)
    ```

    `compute_svd_cache()` batches SVD computation with GPU acceleration (torch) when CUDA is available, falling back to numpy.

    ### Cache Injection

    Caches are injected only into metrics whose signatures accept them — metrics that don't need caches are unaffected:

    ```python
    sig = inspect.signature(metric_fn)
    call_kwargs = dict(embeddings=emb_sub, dataset=ds_sub, module=module)

    if "_knn_cache" in sig.parameters and knn_cache is not None:
        call_kwargs["_knn_cache"] = knn_cache
    if "_svd_cache" in sig.parameters and svd_cache is not None:
        call_kwargs["_svd_cache"] = svd_cache

    result = metric_fn(**call_kwargs)
    ```

    ### Metric Expansion

    `flatten_and_unroll_metrics()` handles list-valued parameters via Cartesian product:

    ```yaml
    # This config:
    trustworthiness:
      _target_: manylatents.metrics.trustworthiness.Trustworthiness
      _partial_: true
      n_neighbors: [5, 10, 20]

    # Expands to three separate evaluations:
    # embedding.trustworthiness__n_neighbors_5
    # embedding.trustworthiness__n_neighbors_10
    # embedding.trustworthiness__n_neighbors_20
    ```

    This expansion happens before kNN extraction, so all k values from expanded metrics contribute to the shared cache.
