# Experiment Engine Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `experiment.py` into a Hydra-free engine, move evaluation into unified `evaluate.py`, rewrite `main.py` as a thin Hydra translation layer, and rewrite `api.py` to call the engine directly without Hydra.

**Architecture:** Two entry paths (CLI via Hydra in `main.py`, Python API in `api.py`) both call a single Hydra-free `run_engine()` in `experiment.py`. All metric evaluation converges in a unified `evaluate()` in `evaluate.py`. The circular import between evaluate.py and experiment.py is eliminated.

**Tech Stack:** Python, PyTorch Lightning, Hydra (CLI only), OmegaConf (CLI only), numpy, wandb

---

## File Map

| File | Responsibility | Changes |
|---|---|---|
| `manylatents/evaluate.py` | Unified `evaluate()` + `extract_k_requirements()` + `prewarm_cache()` + `_flatten_metric_result()` | Major rewrite — absorbs functions from experiment.py |
| `manylatents/experiment.py` | `run_engine()` — Hydra-free engine | Major rewrite — delete old functions, create `run_engine()` |
| `manylatents/main.py` | CLI entry: `@hydra.main` → instantiate → `run_engine()` | Major rewrite — absorbs `instantiate_*` helpers |
| `manylatents/api.py` | Python API: resolve strings → `run_engine()` directly | Major rewrite — remove all Hydra dependency |
| `manylatents/__init__.py` | Lazy imports | Update `evaluate_metrics` → `evaluate` |
| `tests/test_sleuther.py` | Tests for `extract_k_requirements`, `prewarm_cache` | Update imports from `experiment` → `evaluate` |
| `tests/test_evaluate_metrics.py` | Tests for `evaluate_metrics` | Update to test unified `evaluate()` |
| `tests/test_full_cache_integration.py` | Integration tests for cache pipeline | Update imports |
| `tests/test_compute_knn_cache.py` | Cache tests using `prewarm_cache` | Update imports |
| `tests/test_evaluate_extension_metrics.py` | Legacy metric tests | Update imports |
| `tests/test_dict_result_flattening.py` | `_flatten_metric_result` tests | Update imports from `experiment` → `evaluate` |
| `tests/test_full_api_integration.py` | API integration test | Update `evaluate_metrics` → `evaluate` |
| `tests/test_metric_routing_integration.py` | Metric routing integration | Update `prewarm_cache` import |

---

### Task 1: Move evaluation functions to evaluate.py

Move `extract_k_requirements()`, `prewarm_cache()` from `experiment.py` to `evaluate.py`. Merge `evaluate_outputs()` (from experiment.py) and `evaluate_metrics()` into a single `evaluate()` function. This eliminates the circular import.

**Files:**
- Modify: `manylatents/evaluate.py` (full rewrite)
- Modify: `manylatents/experiment.py` (remove moved functions + the `from manylatents.evaluate import _flatten_metric_result` line)
- Test: Run existing tests to verify nothing breaks yet (they'll fail on imports — that's expected until Task 5)

- [ ] **Step 1: Write the new evaluate.py**

Replace the entire contents of `manylatents/evaluate.py` with the unified module. This file absorbs `extract_k_requirements`, `prewarm_cache`, `_flatten_metric_result`, and merges `evaluate_outputs` + `evaluate_metrics` into one `evaluate()` function.

```python
"""Unified metric evaluation.

Single entry point ``evaluate()`` handles both registry-based metrics
(``list[str]`` from the Python API) and Hydra config-based metrics
(``dict[str, DictConfig]`` from the CLI). Eliminates the old circular
import between evaluate.py and experiment.py.
"""
import copy
import inspect
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from manylatents.callbacks.embedding.base import ColormapInfo
from manylatents.metrics.registry import get_metric
from manylatents.utils.metrics import (
    _content_key,
    compute_eigenvalues,
    compute_knn,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_metric_result(metric_name: str, raw_result: Any) -> dict[str, Any]:
    """Flatten a metric result into a dict of name->value pairs."""
    if isinstance(raw_result, dict):
        return {f"{metric_name}.{k}": v for k, v in raw_result.items()}
    return {metric_name: raw_result}


# ---------------------------------------------------------------------------
# Cache sleuther
# ---------------------------------------------------------------------------

def extract_k_requirements(metrics) -> dict:
    """Scan metrics for kNN and spectral requirements, grouped by ``at`` value.

    Args:
        metrics: Either a Dict[str, DictConfig] (Hydra configs from
                 flatten_and_unroll_metrics) or a list[str] (metric registry
                 names from the programmatic API).

    Returns:
        Dict with keys:
            knn: dict mapping at_value -> set of k values needed
            spectral: whether eigendecomposition is needed
    """
    knn: dict[str, set[int]] = {}
    spectral = False

    if isinstance(metrics, list):
        # Registry path -- no ``at`` field, assume embedding
        for name in metrics:
            spec = get_metric(name)
            sig = inspect.signature(spec.func)
            k_val = None
            for param_name in ("k", "n_neighbors"):
                if param_name in spec.params:
                    val = spec.params[param_name]
                    if isinstance(val, (int, float)) and val > 0:
                        k_val = int(val)
                        break
                elif param_name in sig.parameters:
                    default = sig.parameters[param_name].default
                    if default is not inspect.Parameter.empty and isinstance(default, (int, float)) and default > 0:
                        k_val = int(default)
                        break

            if k_val is not None:
                knn.setdefault("embedding", set()).add(k_val)

    else:
        # DictConfig path -- use ``at`` field for routing
        for metric_name, metric_cfg in metrics.items():
            at_value = getattr(metric_cfg, "at", "embedding")

            # Module metrics need eigendecomposition
            if at_value == "module":
                spectral = True

            # Extract k value
            k_val = None
            for param in ("k", "n_neighbors"):
                if hasattr(metric_cfg, param):
                    val = getattr(metric_cfg, param)
                    if isinstance(val, (int, float)) and val > 0:
                        k_val = int(val)
                        break

            if k_val is not None:
                knn.setdefault(at_value, set()).add(k_val)

    return {"knn": knn, "spectral": spectral}


def prewarm_cache(
    metrics,  # Dict[str, DictConfig] or list[str]
    embeddings: np.ndarray,
    dataset,
    module=None,
    knn_cache_dir=None,
    cache: Optional[dict] = None,
    outputs: Optional[dict] = None,
) -> dict:
    """Pre-compute kNN and eigenvalues based on metric requirements.

    Uses the ``at`` field from each metric config to determine which data
    source needs kNN pre-computation.

    Args:
        metrics: Flattened metric configs (Dict[str, DictConfig]) or
            list of metric registry names (list[str]).
        embeddings: Embedding array.
        dataset: Dataset object with .data attribute.
        module: Optional fitted LatentModule.
        knn_cache_dir: Optional directory for disk-persisted dataset kNN.
        cache: Optional pre-existing cache dict. Created if None.
        outputs: Optional dict mapping at_value -> data arrays.
            Allows prewarming for any ``at`` value.

    Returns:
        Populated cache dict.
    """
    reqs = extract_k_requirements(metrics)
    if cache is None:
        cache = {}

    # kNN for each at_value that has requirements
    for at_value, k_set in reqs["knn"].items():
        data = None
        if outputs and at_value in outputs:
            data = outputs[at_value]
        elif at_value == "embedding":
            data = embeddings
        elif at_value == "dataset" and dataset is not None and hasattr(dataset, "data"):
            data = dataset.data

        if data is not None and k_set:
            max_k = max(k_set)

            # Disk cache for dataset kNN
            if at_value == "dataset" and knn_cache_dir is not None:
                content_hash = _content_key(data)
                npz_path = Path(knn_cache_dir) / "knn" / f"{content_hash}_k{max_k}.npz"
                if npz_path.exists():
                    saved = np.load(npz_path)
                    cache[content_hash] = (max_k, saved["distances"], saved["indices"])
                    logger.info(f"Loaded dataset kNN from disk cache: {npz_path}")
                    continue

            logger.info(f"Pre-warming cache: {at_value} kNN with max_k={max_k}")
            compute_knn(data, k=max_k, cache=cache)

            # Save dataset kNN to disk cache
            if at_value == "dataset" and knn_cache_dir is not None:
                content_hash = _content_key(data)
                knn_dir = Path(knn_cache_dir) / "knn"
                knn_dir.mkdir(parents=True, exist_ok=True)
                npz_path = knn_dir / f"{content_hash}_k{max_k}.npz"
                _, dists, idxs = cache[content_hash]
                np.savez(npz_path, distances=dists, indices=idxs)
                logger.info(f"Saved dataset kNN to disk cache: {npz_path}")

    # Eigenvalues if spectral metrics present
    if reqs["spectral"] and module is not None:
        logger.info("Pre-warming cache: eigenvalue decomposition")
        compute_eigenvalues(module, cache=cache)

    return cache


# ---------------------------------------------------------------------------
# Unified evaluate
# ---------------------------------------------------------------------------

def evaluate(
    embeddings: np.ndarray,
    *,
    dataset=None,
    module=None,
    metrics=None,       # list[str] OR dict[str, DictConfig]
    sampling=None,       # dict of instantiated samplers for post-fit
    cache_dir=None,
    cache=None,
) -> dict[str, Any]:
    """Unified metric evaluation — handles both API and CLI metric formats.

    Args:
        embeddings: (n_samples, n_features) array of embeddings.
        dataset: Dataset object with ``.data`` attribute (for cross-space metrics).
        module: Optional fitted LatentModule (for module-context metrics).
        metrics: Either a ``list[str]`` of metric registry names (API path)
            or a ``dict[str, DictConfig]`` of Hydra metric configs (CLI path).
            If None, returns empty dict.
        sampling: Optional dict of instantiated samplers keyed by output name.
            Post-fit sampling applied before metric evaluation.
        cache_dir: Optional directory for disk-persisted dataset kNN.
        cache: Optional shared cache dict. Created internally if None.

    Returns:
        Flat dict of metric name -> value.
    """
    if metrics is None:
        return {}

    if cache is None:
        cache = {}

    # Build outputs dict for routing and sampling
    outputs: dict[str, Any] = {
        "embedding": embeddings,
    }
    if dataset is not None and hasattr(dataset, "data"):
        outputs["dataset"] = dataset.data
    if module is not None:
        outputs["module"] = module
        for key, val in module.extra_outputs().items():
            outputs[key] = val

    # --- Post-fit sampling ---
    embedding_sample_indices = None
    if sampling is not None:
        for output_name, sampler in sampling.items():
            if output_name == "dataset":
                continue  # pre-fit sampling handled in run_engine()
            if output_name not in outputs or not isinstance(outputs[output_name], np.ndarray):
                continue
            indices = sampler.get_indices(outputs[output_name])
            outputs[output_name] = outputs[output_name][indices]
            logger.info(f"Post-fit sampling on '{output_name}': {len(indices)} samples using {type(sampler).__name__}")
            if output_name == "embedding":
                embedding_sample_indices = indices

        # If embedding was sampled, slice dataset to matching indices
        if embedding_sample_indices is not None and dataset is not None and hasattr(dataset, 'data'):
            from manylatents.utils.sampling import _subsample_dataset_metadata
            dataset = _subsample_dataset_metadata(dataset, embedding_sample_indices)
            outputs["dataset"] = dataset.data

        # Sync local embeddings ref with sampled outputs
        embeddings = outputs["embedding"]

    # Log dataset capabilities
    if dataset is not None:
        from manylatents.data.capabilities import log_capabilities
        log_capabilities(dataset)

    # Pre-warm cache
    prewarm_cache(
        metrics, embeddings, dataset, module,
        knn_cache_dir=cache_dir, cache=cache, outputs=outputs,
    )

    # --- Dispatch based on metric format ---
    if isinstance(metrics, list):
        return _evaluate_registry(embeddings, dataset, module, metrics, cache)
    else:
        return _evaluate_hydra(embeddings, dataset, module, metrics, outputs, cache)


def _evaluate_registry(
    embeddings: np.ndarray,
    dataset,
    module,
    metrics: list[str],
    cache: dict,
) -> dict[str, Any]:
    """Evaluate metrics by registry name (API path)."""
    results: dict[str, Any] = {}
    for name in metrics:
        spec = get_metric(name)
        raw_result = spec(
            embeddings=embeddings,
            dataset=dataset,
            module=module,
            cache=cache,
        )
        results.update(_flatten_metric_result(name, raw_result))
    return results


def _evaluate_hydra(
    embeddings: np.ndarray,
    dataset,
    module,
    metrics: dict,  # dict[str, DictConfig]
    outputs: dict[str, Any],
    cache: dict,
) -> dict[str, Any]:
    """Evaluate metrics from Hydra DictConfig (CLI path)."""
    import hydra

    results: dict[str, Any] = {}
    for metric_name, metric_cfg in metrics.items():
        cfg_copy = copy.deepcopy(metric_cfg)
        at_value = getattr(cfg_copy, "at", "embedding")
        if hasattr(cfg_copy, "at"):
            try:
                delattr(cfg_copy, "at")
            except Exception:
                pass  # read-only struct flags

        # Resolve primary data from outputs dict
        if at_value == "module":
            primary_data = None
        else:
            primary_data = outputs.get(at_value)
            if primary_data is None:
                algo_name = type(module).__name__ if module else "unknown"
                logger.warning(
                    f"Skipping metric '{metric_name}': output '{at_value}' "
                    f"not available from {algo_name}. "
                    f"Check that the algorithm produces this output."
                )
                continue

        metric_fn = hydra.utils.instantiate(cfg_copy)

        raw_result = metric_fn(
            embeddings=primary_data if primary_data is not None else embeddings,
            dataset=dataset,
            module=module,
            cache=cache,
        )

        # Unpack (value, ColormapInfo) tuples
        if (
            isinstance(raw_result, tuple)
            and len(raw_result) == 2
            and isinstance(raw_result[1], ColormapInfo)
        ):
            value, viz_info = raw_result
            results[metric_name] = value
            results[f"{metric_name}__viz"] = viz_info
        else:
            results.update(_flatten_metric_result(metric_name, raw_result))

    return results


# ---------------------------------------------------------------------------
# Backward-compat alias (deprecated)
# ---------------------------------------------------------------------------

def evaluate_metrics(
    embeddings: np.ndarray,
    *,
    metrics: list[str],
    module=None,
    dataset=None,
    cache=None,
    **kwargs,
) -> dict[str, Any]:
    """Deprecated — use ``evaluate()`` instead.

    Kept for backward compatibility with existing callers.
    """
    return evaluate(
        embeddings,
        dataset=dataset,
        module=module,
        metrics=metrics,
        cache=cache,
    )
```

- [ ] **Step 2: Remove moved functions from experiment.py**

Delete these from `experiment.py`:
- The `from manylatents.evaluate import _flatten_metric_result` import (line 264)
- `extract_k_requirements()` function (lines 122-183)
- `prewarm_cache()` function (lines 186-261)
- `evaluate_outputs()` function (the `@evaluate.register(dict)` block, lines 267-394)
- The `_flatten_metric_result` re-export (it's now only in evaluate.py)

Keep in experiment.py for now (they'll be refactored in Task 2):
- `instantiate_*` helpers
- `evaluate` singledispatch + `evaluate_lightningmodule`
- `execute_step`
- `run_algorithm`

After this step, experiment.py no longer imports from evaluate.py, breaking the circular dependency.

- [ ] **Step 3: Run tests to confirm evaluate.py loads**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -c "from manylatents.evaluate import evaluate, extract_k_requirements, prewarm_cache, _flatten_metric_result; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add manylatents/evaluate.py manylatents/experiment.py
git commit -m "refactor: unify evaluation in evaluate.py — move extract_k_requirements, prewarm_cache, create unified evaluate()"
```

---

### Task 2: Create run_engine() in experiment.py

Rewrite `experiment.py` to contain only `run_engine()` and `evaluate_lightningmodule()`. Inline `execute_step()` logic. Delete `run_algorithm()`, the `evaluate` singledispatch, and all `instantiate_*` helpers.

**Files:**
- Modify: `manylatents/experiment.py` (full rewrite)

- [ ] **Step 1: Write the new experiment.py**

Replace the entire contents of `manylatents/experiment.py`:

```python
"""Hydra-free experiment engine.

``run_engine()`` is the single execution entry point. It accepts Python
objects (no DictConfig, no Hydra imports) and orchestrates: seed, data
extraction, pre-fit sampling, algorithm fit/transform, evaluation, callbacks,
and wandb logging.

Two callers:
- ``main.py`` (CLI): instantiates objects from Hydra config, then calls run_engine
- ``api.py`` (Python API): constructs objects directly, then calls run_engine
"""
import logging
import time
from typing import Any, Optional

import lightning
import numpy as np
import torch

try:
    import wandb

    wandb.init  # verify real package, not wandb/ output dir
except (ImportError, AttributeError):
    wandb = None
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from torch.utils.data import DataLoader

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.evaluate import evaluate
from manylatents.utils.data import determine_data_source
from manylatents.utils.utils import load_precomputed_embeddings

logger = logging.getLogger(__name__)


def evaluate_lightningmodule(
    algorithm: LightningModule,
    *,
    trainer: Trainer,
    datamodule,
    metrics_cfg=None,
) -> tuple[dict, Optional[float]]:
    """Evaluate LightningModule on test set and compute custom metrics.

    Returns:
        (combined_metrics, error_value). If no test_step or empty results,
        returns ({}, None).
    """
    if not hasattr(algorithm, "test_step"):
        logger.info("Model does not define a test_step(); skipping evaluation.")
        return {}, None

    results = trainer.test(model=algorithm, datamodule=datamodule)
    if not results:
        return {}, None

    base_metrics = results[0]
    custom_metrics = {}

    if metrics_cfg:
        import hydra
        model_metrics_cfg = metrics_cfg.get("model", {}) if hasattr(metrics_cfg, "get") else {}
        for metric_key, metric_params in model_metrics_cfg.items():
            if metric_params.get("enabled", True):
                metric_fn = hydra.utils.instantiate(metric_params)
                name, value = metric_fn(algorithm, test_results=base_metrics)
                custom_metrics[name] = value

    combined_metrics = {**base_metrics, **custom_metrics}
    error_value = next(iter(combined_metrics.values())) if combined_metrics else None
    return combined_metrics, error_value


def run_engine(
    datamodule: LightningDataModule,
    algorithm,
    trainer: Trainer,
    *,
    embedding_callbacks: Optional[list[EmbeddingCallback]] = None,
    metrics=None,           # list[str] or dict[str, DictConfig]
    metrics_cfg=None,       # raw cfg.metrics for LightningModule model metrics
    sampling=None,          # dict keyed by output name, values are instantiated samplers
    seed: int = 42,
    eval_only: bool = False,
    pretrained_ckpt: Optional[str] = None,
    cache_dir: Optional[str] = None,
    wandb_run=None,
) -> dict[str, Any]:
    """Hydra-free experiment engine.

    Args:
        datamodule: Configured and setup-ready LightningDataModule.
        algorithm: LatentModule or LightningModule instance.
        trainer: Lightning Trainer instance.
        embedding_callbacks: Post-embedding callbacks (on_latent_end).
        metrics: list[str] (registry names) or dict[str, DictConfig] (Hydra configs).
        metrics_cfg: Raw metrics config for LightningModule model-level metrics.
        sampling: Dict of instantiated samplers keyed by output name.
        seed: Random seed.
        eval_only: If True, load precomputed embeddings instead of running.
        pretrained_ckpt: Path to pretrained checkpoint (skip training).
        cache_dir: Directory for kNN disk cache.
        wandb_run: Pre-initialized wandb run, or None.

    Returns:
        Dict with keys: embeddings, label, metadata, scores.
    """
    lightning.seed_everything(seed, workers=True)

    # --- Data extraction ---
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    field_index, data_source = determine_data_source(train_loader)

    # --- Algorithm Execution ---
    embeddings: dict[str, Any] = {}

    if eval_only:
        logger.info("Evaluation-only mode: Loading precomputed latent outputs.")
        # eval_only requires a cfg-like object with data.precomputed_path.
        # For now, return empty — callers should pass precomputed embeddings via input_data.
        embeddings = load_precomputed_embeddings_from_datamodule(datamodule)
    else:
        # Unroll dataloaders to tensors
        train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
        test_tensor = torch.cat([b[field_index].cpu() for b in test_loader], dim=0)

        # Extract labels for supervised LatentModules
        train_labels = None
        train_dataset = getattr(datamodule, "train_dataset", None)
        if train_dataset is not None and hasattr(train_dataset, "get_labels"):
            labels = train_dataset.get_labels()
            if labels is not None:
                train_labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                logger.info(f"Extracted {len(train_labels)} training labels for supervised learning")

        # --- Pre-fit sampling ---
        if sampling is not None and "dataset" in sampling:
            dataset_sampler = sampling["dataset"]
            pre_fit_indices = dataset_sampler.get_indices(
                train_tensor.numpy() if torch.is_tensor(train_tensor) else train_tensor
            )
            train_tensor = train_tensor[pre_fit_indices]
            test_tensor = test_tensor[pre_fit_indices]
            if train_labels is not None:
                train_labels = train_labels[pre_fit_indices]
            logger.info(f"Pre-fit sampling: {len(pre_fit_indices)} samples using {type(dataset_sampler).__name__}")

        logger.info(
            f"Running algorithm on {data_source}:\n"
            f"Train tensor shape: {train_tensor.shape}\n"
            f"Test tensor shape: {test_tensor.shape}\n"
            f"Algorithm: {type(algorithm).__name__}"
        )

        # --- Fit / Transform (inlined execute_step) ---
        t_total_start = time.perf_counter()
        t_step_start = time.perf_counter()
        latents = None

        if isinstance(algorithm, LatentModule):
            algorithm.fit(train_tensor, train_labels)
            try:
                latents = algorithm.transform(test_tensor)
            except NotImplementedError:
                logger.warning(
                    f"{type(algorithm).__name__} does not support transform(). "
                    "Falling back to fit_transform() on test data (transductive mode)."
                )
                latents = algorithm.fit_transform(test_tensor)
            logger.info(f"LatentModule embedding shape: {latents.shape}")

        elif isinstance(algorithm, LightningModule):
            if pretrained_ckpt:
                logger.info(f"Loading pretrained model from {pretrained_ckpt}")
                algorithm = LightningModule.load_from_checkpoint(pretrained_ckpt)
            else:
                logger.info("Running training...")
                trainer.fit(algorithm, datamodule=datamodule)

            # Model-level evaluation (test_step metrics)
            model_metrics, model_error = evaluate_lightningmodule(
                algorithm,
                trainer=trainer,
                datamodule=datamodule,
                metrics_cfg=metrics_cfg,
            )
            logger.info(f"Model evaluation completed. Error: {model_error}, Metrics: {model_metrics}")

            # Extract embeddings
            if hasattr(algorithm, "encode"):
                logger.info("Extracting embeddings using network encoder...")
                latents = algorithm.encode(test_tensor)
                latents = latents.detach().cpu().numpy() if isinstance(latents, torch.Tensor) else latents
                logger.info(f"LightningModule embedding shape: {latents.shape}")
            else:
                logger.warning(f"LightningModule {type(algorithm).__name__} has no 'encode' method - skipping")

        step_time = time.perf_counter() - t_step_start

        # --- Package results ---
        if latents is not None:
            embeddings = {
                "embeddings": latents,
                "label": getattr(getattr(datamodule, "test_dataset", None), "get_labels", lambda: None)(),
                "metadata": {
                    "source": "single_algorithm",
                    "algorithm_type": type(algorithm).__name__,
                    "data_shape": test_tensor.shape,
                    "step_time": step_time,
                },
            }

            # Attach extra outputs
            if isinstance(algorithm, LatentModule):
                extras = algorithm.extra_outputs()
                for key, val in extras.items():
                    embeddings[key] = val
                    shape_info = f" shape={val.shape}" if hasattr(val, 'shape') else ""
                    logger.info(f"Extra output attached: {key}{shape_info}")

            # --- Evaluate ---
            # Determine dataset for evaluation
            mode = getattr(datamodule, 'mode', None) or getattr(datamodule.hparams, 'mode', 'full')
            ds = datamodule.test_dataset if mode == "split" else datamodule.train_dataset
            logger.info(f"Reference data shape: {ds.data.shape}")

            # Ensure embeddings are numpy
            eval_embeddings = latents
            if torch.is_tensor(eval_embeddings):
                eval_embeddings = eval_embeddings.cpu().numpy()

            # Post-fit samplers (minus dataset which is pre-fit)
            post_fit_sampling = None
            if sampling is not None:
                post_fit_sampling = {k: v for k, v in sampling.items() if k != "dataset"}
                if not post_fit_sampling:
                    post_fit_sampling = None

            logger.info(f"Evaluating embeddings from {type(algorithm).__name__}...")
            t_eval_start = time.perf_counter()
            embeddings["scores"] = evaluate(
                eval_embeddings,
                dataset=ds,
                module=algorithm if isinstance(algorithm, LatentModule) else None,
                metrics=metrics,
                sampling=post_fit_sampling,
                cache_dir=cache_dir,
            )
            eval_time = time.perf_counter() - t_eval_start
            total_time = time.perf_counter() - t_total_start

            embeddings["metadata"]["eval_time"] = eval_time
            embeddings["metadata"]["total_time"] = total_time

    # --- Callbacks ---
    callback_outputs = {}
    if embeddings and embedding_callbacks:
        for cb in embedding_callbacks:
            cb_result = cb.on_latent_end(dataset=datamodule.test_dataset, embeddings=embeddings)
            if isinstance(cb_result, dict):
                callback_outputs.update(cb_result)
                logger.info(f"Callback {cb.__class__.__name__} returned: {list(cb_result.keys())}")

    if callback_outputs:
        embeddings['callback_outputs'] = callback_outputs
        logger.info(f"Added callback outputs to embeddings: {list(callback_outputs.keys())}")

    logger.info("Experiment complete.")

    # --- WandB logging ---
    if wandb_run is not None and wandb is not None and embeddings.get("scores"):
        scores = embeddings["scores"]
        scalar_metrics = {}
        for name, val in scores.items():
            if isinstance(val, tuple) and len(val) == 2:
                scalar_metrics[f"metrics/{name}"] = float(val[0])
            elif np.ndim(val) == 0:
                scalar_metrics[f"metrics/{name}"] = float(val)
        if scalar_metrics:
            wandb.log(scalar_metrics)
            logger.info(f"Auto-logged {len(scalar_metrics)} metrics to wandb: {list(scalar_metrics.keys())}")

    return embeddings


def load_precomputed_embeddings_from_datamodule(datamodule) -> dict:
    """Load precomputed embeddings from a datamodule's configured path.

    This is the eval_only path — the datamodule was configured with a
    precomputed_path, and we just need to load the stored embeddings.
    """
    path = getattr(datamodule.hparams, 'precomputed_path', None) or getattr(datamodule.hparams, 'path', None)
    if path is None:
        logger.warning("eval_only=True but no precomputed path found in datamodule.")
        return {}

    from manylatents.utils.utils import _load_embeddings_from_path
    return _load_embeddings_from_path(path)
```

- [ ] **Step 2: Verify experiment.py loads**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -c "from manylatents.experiment import run_engine; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Verify no circular import**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -c "from manylatents.evaluate import evaluate; from manylatents.experiment import run_engine; print('No circular import')"`

Expected: `No circular import`

- [ ] **Step 4: Commit**

```bash
git add manylatents/experiment.py
git commit -m "refactor: replace run_algorithm with Hydra-free run_engine, inline execute_step"
```

---

### Task 3: Rewrite main.py as Hydra translation layer

Move `instantiate_*` helpers into `main.py` and wire `@hydra.main` → instantiate → `run_engine()`.

**Files:**
- Modify: `manylatents/main.py` (full rewrite)

- [ ] **Step 1: Write the new main.py**

```python
"""CLI entry point for manylatents experiments.

This is the only file that imports ``hydra.utils.instantiate``. It translates
Hydra DictConfig into Python objects, then calls the Hydra-free engine.

Usage:
    python -m manylatents.main data=swissroll algorithms/latent=pca
    python -m manylatents.main -m experiment=spectral_discrimination cluster=mila resources=cpu_light
"""
import functools
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
from lightning import Callback, LightningDataModule, Trainer
from omegaconf import DictConfig, OmegaConf

try:
    import wandb

    wandb.init  # verify real package
except (ImportError, AttributeError):
    wandb = None

import manylatents.configs  # noqa: F401 — registers SearchPathPlugin
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.experiment import run_engine
from manylatents.extensions import discover_extensions
from manylatents.utils.metrics import flatten_and_unroll_metrics
from manylatents.utils.utils import check_or_make_dirs, setup_logging, should_disable_wandb

discover_extensions()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instantiation helpers (Hydra → Python objects)
# ---------------------------------------------------------------------------

def _instantiate_datamodule(cfg: DictConfig, input_data_holder: Optional[Dict] = None) -> LightningDataModule:
    check_or_make_dirs(cfg.cache_dir)
    datamodule_cfg = {k: v for k, v in cfg.data.items() if k != "debug"}
    if input_data_holder is not None and 'data' in input_data_holder:
        datamodule_cfg['data'] = input_data_holder['data']
    return hydra.utils.instantiate(datamodule_cfg)


def _instantiate_algorithm(algorithm_config: DictConfig, datamodule=None):
    algo_or_partial = hydra.utils.instantiate(algorithm_config, datamodule=datamodule)
    if isinstance(algo_or_partial, functools.partial):
        return algo_or_partial(datamodule=datamodule) if datamodule else algo_or_partial()
    return algo_or_partial


def _instantiate_callbacks(
    trainer_cb_cfg: Dict[str, Any] = None,
    embedding_cb_cfg: Dict[str, Any] = None,
) -> Tuple[List[Callback], List[EmbeddingCallback]]:
    trainer_cb_cfg = trainer_cb_cfg or {}
    embedding_cb_cfg = embedding_cb_cfg or {}
    lightning_cbs, embedding_cbs = [], []

    for name, one_cfg in trainer_cb_cfg.items():
        cb = hydra.utils.instantiate(one_cfg)
        if isinstance(cb, Callback):
            lightning_cbs.append(cb)
        else:
            logger.warning(f"Ignoring non-Lightning callback '{name}'")

    for name, one_cfg in embedding_cb_cfg.items():
        cb = hydra.utils.instantiate(one_cfg)
        if isinstance(cb, EmbeddingCallback):
            embedding_cbs.append(cb)
        else:
            logger.warning(f"Ignoring non-Embedding callback '{name}'")

    return lightning_cbs, embedding_cbs


def _instantiate_trainer(
    cfg: DictConfig,
    lightning_callbacks: Optional[List] = None,
    loggers: Optional[List] = None,
) -> Trainer:
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.pop("_target_", None)
    trainer_kwargs.pop("callbacks", None)
    trainer_kwargs.pop("logger", None)
    if lightning_callbacks:
        trainer_kwargs["callbacks"] = lightning_callbacks
    if loggers:
        trainer_kwargs["logger"] = loggers
    return Trainer(**trainer_kwargs)


def _instantiate_sampling(cfg: DictConfig) -> Optional[dict]:
    """Instantiate sampling strategies from config."""
    sampling_cfg = (
        OmegaConf.to_container(cfg.sampling, resolve=True)
        if hasattr(cfg, 'sampling') and cfg.sampling is not None
        else None
    )
    if sampling_cfg is None:
        return None
    samplers = {}
    for output_name, sampler_cfg in sampling_cfg.items():
        samplers[output_name] = hydra.utils.instantiate(sampler_cfg)
    return samplers


def _init_wandb(cfg: DictConfig):
    """Initialize wandb run from config. Returns the run or None."""
    wandb_disabled = should_disable_wandb(cfg) or wandb is None
    if wandb_disabled or cfg.logger is None:
        logger.info("WandB logging disabled - skipping wandb initialization")
        return None

    logger.info(f"Initializing wandb logger: {OmegaConf.to_yaml(cfg.logger)}")
    return wandb.init(
        project=cfg.logger.get("project", cfg.project),
        name=cfg.logger.get("name", cfg.name),
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logger.get("mode", "online"),
        dir=os.environ.get("WANDB_DIR", "logs"),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: DictConfig) -> Dict[str, Any]:
    setup_logging(debug=cfg.debug, log_level=getattr(cfg, "log_level", "warning"))
    logger.info("Final Config:\n" + OmegaConf.to_yaml(cfg))

    wandb_run = _init_wandb(cfg)

    # --- Instantiate all objects from config ---
    datamodule = _instantiate_datamodule(cfg)

    if hasattr(cfg.algorithms, 'latent') and cfg.algorithms.latent is not None:
        algorithm = _instantiate_algorithm(cfg.algorithms.latent, datamodule)
    elif hasattr(cfg.algorithms, 'lightning') and cfg.algorithms.lightning is not None:
        algorithm = _instantiate_algorithm(cfg.algorithms.lightning, datamodule)
    else:
        raise ValueError("No algorithm specified in configuration")

    # Callbacks
    trainer_cb_cfg = dict(cfg.trainer.get("callbacks") or {})
    if hasattr(cfg, "callbacks") and cfg.callbacks is not None:
        extra_trainer_cbs = cfg.callbacks.get("trainer") or {}
        trainer_cb_cfg.update(extra_trainer_cbs)
    embedding_cb_cfg = cfg.callbacks.get("embedding") if hasattr(cfg, "callbacks") else None

    lightning_cbs, embedding_cbs = _instantiate_callbacks(trainer_cb_cfg, embedding_cb_cfg)
    if lightning_cbs:
        logger.info(f"Instantiated {len(lightning_cbs)} Lightning callbacks: {[type(cb).__name__ for cb in lightning_cbs]}")
    if not embedding_cbs:
        logger.info("No embedding callbacks configured; skip embedding-level hooks.")

    # Loggers
    wandb_disabled = should_disable_wandb(cfg) or wandb is None
    loggers = []
    if not wandb_disabled and cfg.logger is not None:
        for lg_conf in cfg.trainer.get("logger", {}).values():
            loggers.append(hydra.utils.instantiate(lg_conf))
        logger.info(f"Trainer loggers enabled: {len(loggers)} logger(s)")
    else:
        logger.info("Trainer loggers disabled (WandB disabled)")

    trainer = _instantiate_trainer(cfg, lightning_callbacks=lightning_cbs, loggers=loggers)

    # Metrics: flatten and unroll for the engine
    metrics = flatten_and_unroll_metrics(cfg.metrics) if cfg.metrics is not None else None

    # Sampling: instantiate sampler objects
    sampling = _instantiate_sampling(cfg)

    # --- Run engine ---
    result = run_engine(
        datamodule=datamodule,
        algorithm=algorithm,
        trainer=trainer,
        embedding_callbacks=embedding_cbs,
        metrics=metrics,
        metrics_cfg=cfg.metrics,
        sampling=sampling,
        seed=cfg.seed,
        eval_only=getattr(cfg, "eval_only", False),
        pretrained_ckpt=getattr(cfg, "pretrained_ckpt", None),
        cache_dir=getattr(cfg, "cache_dir", None),
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb.finish()

    return result


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify main.py loads**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -c "from manylatents.main import main; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add manylatents/main.py
git commit -m "refactor: rewrite main.py as Hydra translation layer calling run_engine"
```

---

### Task 4: Rewrite api.py as Hydra-free Python interface

Remove all Hydra imports. Resolve string names to objects directly. Call `run_engine()`.

**Files:**
- Modify: `manylatents/api.py` (full rewrite)

- [ ] **Step 1: Write the new api.py**

```python
"""Programmatic API for manylatents.

Hydra-free interface — resolves string names to Python objects and calls
``run_engine()`` directly. No DictConfig building, no compose(), no GlobalHydra.

Example:
    from manylatents.api import run

    # With string names
    result = run(data="swissroll", algorithm="pca", metrics=["trustworthiness"])

    # With pre-built objects
    from manylatents.algorithms.latent.pca import PCAModule
    result = run(input_data=my_array, algorithm=PCAModule(n_components=2))

    # Chaining (multi-step via sequential API calls)
    result1 = run(data="swissroll", algorithm="pca")
    result2 = run(input_data=result1["embeddings"], algorithm="phate")
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from lightning import Trainer

from manylatents.experiment import run_engine

logger = logging.getLogger(__name__)


def _resolve_datamodule(
    input_data: Optional[np.ndarray] = None,
    data: Optional[str] = None,
    **kwargs,
):
    """Resolve data argument to a LightningDataModule.

    Accepts either an in-memory array (wrapped in PrecomputedDataModule)
    or a string name (instantiated via Hydra config lookup).
    """
    if input_data is not None:
        from manylatents.data.precomputed_datamodule import PrecomputedDataModule
        return PrecomputedDataModule(
            data=input_data,
            batch_size=kwargs.get("batch_size", 128),
            num_workers=kwargs.get("num_workers", 0),
            seed=kwargs.get("seed", 42),
        )

    if data is None:
        raise ValueError("Either 'input_data' or 'data' must be provided.")

    # String name → Hydra config lookup (minimal Hydra usage for config resolution)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import hydra as _hydra
    from omegaconf import OmegaConf

    config_dir = str((Path(__file__).parent / "configs").resolve())

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=[f"data={data}"])

    datamodule_cfg = {k: v for k, v in cfg.data.items() if k != "debug"}
    dm = _hydra.utils.instantiate(datamodule_cfg)
    return dm


def _resolve_algorithm(
    algorithm=None,
    algorithms: Optional[dict] = None,
    **kwargs,
):
    """Resolve algorithm argument to a LatentModule or LightningModule.

    Accepts a string name, a pre-built instance, or a dict config.
    """
    import functools

    # Pre-built instance — pass through
    from manylatents.algorithms.latent.latent_module_base import LatentModule
    from lightning import LightningModule

    if isinstance(algorithm, (LatentModule, LightningModule)):
        return algorithm

    # Dict config (from old API) — instantiate via Hydra
    if algorithms is not None:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        import hydra as _hydra
        from omegaconf import OmegaConf, DictConfig

        config_dir = str((Path(__file__).parent / "configs").resolve())

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Build override list from the algorithms dict
        override_list = []
        algo_dict_overrides = {}

        for algo_type, algo_cfg in algorithms.items():
            if isinstance(algo_cfg, str):
                override_list.append(f"algorithms/{algo_type}={algo_cfg}")
            elif isinstance(algo_cfg, dict):
                algo_dict_overrides[algo_type] = algo_cfg

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=override_list)

        OmegaConf.set_struct(cfg, False)

        for algo_type, algo_cfg in algo_dict_overrides.items():
            OmegaConf.update(cfg, f"algorithms.{algo_type}", algo_cfg, merge=True)

        if hasattr(cfg.algorithms, 'latent') and cfg.algorithms.latent is not None:
            algo_cfg = cfg.algorithms.latent
        elif hasattr(cfg.algorithms, 'lightning') and cfg.algorithms.lightning is not None:
            algo_cfg = cfg.algorithms.lightning
        else:
            raise ValueError("No algorithm resolved from config")

        algo_or_partial = _hydra.utils.instantiate(algo_cfg)
        if isinstance(algo_or_partial, functools.partial):
            return algo_or_partial()
        return algo_or_partial

    # String name → simple lookup
    if isinstance(algorithm, str):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        import hydra as _hydra

        config_dir = str((Path(__file__).parent / "configs").resolve())
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=[f"algorithms/latent={algorithm}"])

        algo_or_partial = _hydra.utils.instantiate(cfg.algorithms.latent)
        if isinstance(algo_or_partial, functools.partial):
            return algo_or_partial()
        return algo_or_partial

    raise ValueError(f"Cannot resolve algorithm from: {algorithm}")


def _resolve_metrics(metrics, cfg_metrics=None):
    """Resolve metrics argument.

    Returns (engine_metrics, metrics_cfg) tuple where engine_metrics is
    list[str] or dict[str, DictConfig] for the engine, and metrics_cfg
    is the raw config for LightningModule model-level metrics.
    """
    if metrics is None and cfg_metrics is None:
        return None, None

    # list[str] — registry names, pass through
    if isinstance(metrics, list):
        return metrics, None

    # dict — could be Hydra-style metric configs
    if isinstance(metrics, dict) or cfg_metrics is not None:
        metric_input = cfg_metrics if cfg_metrics is not None else metrics
        from omegaconf import OmegaConf, DictConfig
        from manylatents.utils.metrics import flatten_and_unroll_metrics

        if isinstance(metric_input, str):
            # String like "standard" — resolve from config
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra

            config_dir = str((Path(__file__).parent / "configs").resolve())
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="config", overrides=[f"metrics={metric_input}"])
            metric_input = cfg.metrics

        if not isinstance(metric_input, DictConfig):
            metric_input = OmegaConf.create(metric_input)

        flattened = flatten_and_unroll_metrics(metric_input)
        return flattened, metric_input

    # String name — resolve from config
    if isinstance(metrics, str):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import DictConfig
        from manylatents.utils.metrics import flatten_and_unroll_metrics

        config_dir = str((Path(__file__).parent / "configs").resolve())
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=[f"metrics={metrics}"])

        flattened = flatten_and_unroll_metrics(cfg.metrics)
        return flattened, cfg.metrics

    return None, None


def _resolve_sampling(sampling) -> Optional[dict]:
    """Resolve sampling config to instantiated sampler objects."""
    if sampling is None:
        return None

    # Already instantiated sampler objects — pass through if they have get_indices
    result = {}
    for key, val in sampling.items():
        if hasattr(val, 'get_indices'):
            result[key] = val
        elif isinstance(val, dict):
            import hydra as _hydra
            result[key] = _hydra.utils.instantiate(val)
        else:
            result[key] = val
    return result


def run(
    input_data: Optional[np.ndarray] = None,
    data: Optional[str] = None,
    algorithm=None,
    algorithms: Optional[dict] = None,
    metrics=None,
    sampling=None,
    seed: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    """Run a manylatents experiment.

    Args:
        input_data: In-memory array to use as input data.
        data: Dataset name (e.g., "swissroll").
        algorithm: Algorithm name (str), instance, or None.
        algorithms: Dict config for algorithms (e.g., {"latent": "pca"}).
            Used when algorithm is None.
        metrics: Metric specification — list[str] of registry names,
            dict of Hydra configs, or string bundle name (e.g., "standard").
        sampling: Dict of sampling configs keyed by output name.
        seed: Random seed (default 42).
        **kwargs: Additional options (batch_size, num_workers, etc.).

    Returns:
        Dict with keys: embeddings, label, metadata, scores.
    """
    from manylatents.utils.utils import setup_logging
    setup_logging(debug=kwargs.get("debug", False), log_level=kwargs.get("log_level", "warning"))

    datamodule = _resolve_datamodule(input_data, data, seed=seed, **kwargs)
    algo = _resolve_algorithm(algorithm, algorithms, **kwargs)
    engine_metrics, metrics_cfg = _resolve_metrics(metrics)
    resolved_sampling = _resolve_sampling(sampling)

    trainer = Trainer(enable_progress_bar=False, logger=False)

    return run_engine(
        datamodule=datamodule,
        algorithm=algo,
        trainer=trainer,
        metrics=engine_metrics,
        metrics_cfg=metrics_cfg,
        sampling=resolved_sampling,
        seed=seed,
        cache_dir=kwargs.get("cache_dir"),
    )
```

- [ ] **Step 2: Verify api.py loads**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -c "from manylatents.api import run; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add manylatents/api.py
git commit -m "refactor: rewrite api.py as Hydra-free interface calling run_engine directly"
```

---

### Task 5: Update imports and tests

Fix all test files that import old function names/locations.

**Files:**
- Modify: `manylatents/__init__.py`
- Modify: `tests/test_sleuther.py`
- Modify: `tests/test_evaluate_metrics.py`
- Modify: `tests/test_full_cache_integration.py`
- Modify: `tests/test_compute_knn_cache.py`
- Modify: `tests/test_evaluate_extension_metrics.py`
- Modify: `tests/test_dict_result_flattening.py`
- Modify: `tests/test_full_api_integration.py`
- Modify: `tests/test_metric_routing_integration.py`

- [ ] **Step 1: Update `__init__.py` lazy imports**

In `manylatents/__init__.py`, the lazy import already points to `manylatents.evaluate` for `evaluate_metrics`, which still exists as a backward-compat alias. No change needed. But also add `evaluate` to the lazy imports:

```python
_LAZY_IMPORTS = {
    "evaluate_metrics": "manylatents.evaluate",
    "evaluate": "manylatents.evaluate",
}
```

- [ ] **Step 2: Update test_sleuther.py imports**

Change all `from manylatents.experiment import extract_k_requirements` to `from manylatents.evaluate import extract_k_requirements`.
Change all `from manylatents.experiment import prewarm_cache` to `from manylatents.evaluate import prewarm_cache`.

- [ ] **Step 3: Update test_dict_result_flattening.py imports**

Change all `from manylatents.experiment import _flatten_metric_result` to `from manylatents.evaluate import _flatten_metric_result`.

- [ ] **Step 4: Update test_full_cache_integration.py imports**

Change all `from manylatents.experiment import extract_k_requirements, prewarm_cache` to `from manylatents.evaluate import extract_k_requirements, prewarm_cache`.

- [ ] **Step 5: Update test_compute_knn_cache.py imports**

Change `from manylatents.experiment import prewarm_cache` to `from manylatents.evaluate import prewarm_cache`.

- [ ] **Step 6: Update test_evaluate_extension_metrics.py imports**

Change `from manylatents.experiment import prewarm_cache` to `from manylatents.evaluate import prewarm_cache`.

- [ ] **Step 7: Update test_metric_routing_integration.py imports**

Change `from manylatents.experiment import prewarm_cache` to `from manylatents.evaluate import prewarm_cache`.

- [ ] **Step 8: Run all tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/ -x -q`

Expected: All tests pass.

- [ ] **Step 9: Run callback tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest manylatents/callbacks/tests/ -x -q`

Expected: All tests pass.

- [ ] **Step 10: Commit**

```bash
git add manylatents/__init__.py tests/
git commit -m "refactor: update all imports for experiment engine split"
```

---

### Task 6: Fix eval_only path and edge cases

The old `run_algorithm` used `load_precomputed_embeddings(cfg)` which takes a DictConfig. The new engine is Hydra-free. Fix the `eval_only` path to work without cfg.

**Files:**
- Modify: `manylatents/experiment.py` (if `load_precomputed_embeddings_from_datamodule` needs adjustment)
- Modify: `manylatents/utils/utils.py` (extract the file-loading logic if needed)

- [ ] **Step 1: Check if load_precomputed_embeddings can be adapted**

Read `manylatents/utils/utils.py` function `load_precomputed_embeddings` to see what it needs from cfg. If it only needs `cfg.data.precomputed_path`, the datamodule hparams approach works. If not, adapt.

The `eval_only` path is rarely used (it loads embeddings from disk rather than computing them). The `load_precomputed_embeddings_from_datamodule` helper in the new experiment.py handles this by reading from `datamodule.hparams.path`.

- [ ] **Step 2: Verify the helper exists and test it manually if needed**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -c "from manylatents.experiment import load_precomputed_embeddings_from_datamodule; print('OK')"`

- [ ] **Step 3: Commit if changes were needed**

```bash
git add manylatents/experiment.py manylatents/utils/utils.py
git commit -m "fix: adapt eval_only path for Hydra-free engine"
```

---

### Task 7: Update documentation

Update all docs that reference old function names.

**Files:**
- Modify: `docs/evaluation.md`
- Modify: `docs/metrics.md`
- Modify: `docs/cache.md`
- Modify: `docs/callbacks.md`
- Modify: `docs/api_usage.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update docs/evaluation.md**

Replace references:
- `run_algorithm()` → `run_engine()`
- `execute_step()` → inlined in `run_engine()`
- `evaluate_outputs()` → `evaluate()` in evaluate.py
- `evaluate_lightningmodule` → stays, but now in experiment.py called from `run_engine()`

- [ ] **Step 2: Update docs/metrics.md**

Replace references:
- `run_algorithm()` → `run_engine()`
- `evaluate_outputs()` → `evaluate()`

- [ ] **Step 3: Update docs/cache.md**

Replace references:
- `evaluate_outputs()` → `evaluate()`
- `extract_k_requirements()` now lives in `evaluate.py`

- [ ] **Step 4: Update docs/callbacks.md**

Replace `run_algorithm()` → `run_engine()`

- [ ] **Step 5: Update docs/api_usage.md**

The `evaluate_metrics` import still works (backward-compat alias). Optionally update examples to use `evaluate` directly.

- [ ] **Step 6: Update CLAUDE.md Key Files table**

Update the table to reflect the new responsibilities:
- `experiment.py` → "Hydra-free engine: `run_engine()`"
- `evaluate.py` → "Unified `evaluate()`, `extract_k_requirements()`, `prewarm_cache()`"
- `main.py` → "CLI entry point + Hydra instantiation layer"
- `api.py` → "Python API: `run()` — Hydra-free"

- [ ] **Step 7: Update CLAUDE.md Entry Points**

Update the Python API example to show the new signature:
```python
from manylatents.api import run
result = run(data="swissroll", algorithm="pca", metrics=["trustworthiness"])
```

- [ ] **Step 8: Commit**

```bash
git add docs/ CLAUDE.md
git commit -m "docs: update references for experiment engine split"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/ -x -q`

- [ ] **Step 2: Run callback tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest manylatents/callbacks/tests/ -x -q`

- [ ] **Step 3: Verify CLI still works**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -m manylatents.main algorithms/latent=pca data=swissroll metrics=trustworthiness 2>&1 | tail -5`

- [ ] **Step 4: Verify API still works**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -c "
from manylatents.api import run
result = run(data='swissroll', algorithms={'latent': 'pca'}, metrics={'trustworthiness': {'_target_': 'manylatents.metrics.trustworthiness.Trustworthiness', '_partial_': True, 'n_neighbors': 5, 'at': 'embedding'}})
print('Scores:', result['scores'])
print('Shape:', result['embeddings'].shape)
"`

- [ ] **Step 5: Verify no circular imports**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -c "import manylatents.evaluate; import manylatents.experiment; import manylatents.api; import manylatents.main; print('All modules import cleanly')"`
