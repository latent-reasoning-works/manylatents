from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import torch
from lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.outputs import collect_outputs
from manylatents.utils.data import determine_data_source

logger = logging.getLogger(__name__)


from manylatents.evaluate import (  # noqa: F401  -- backward compat re-exports
    extract_k_requirements,
    prewarm_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_precomputed_from_datamodule(datamodule: LightningDataModule) -> dict[str, Any] | None:
    """Load precomputed embeddings from a datamodule's configured path.

    Looks for ``path`` or ``precomputed_path`` in ``datamodule.hparams`` and
    delegates to :func:`~manylatents.utils.utils.load_precomputed_embeddings`
    style loading (npy / csv / pt).

    Returns ``None`` when no path is configured.
    """
    hparams = getattr(datamodule, "hparams", {})
    precomputed_path = getattr(hparams, "precomputed_path", None) or getattr(hparams, "path", None)

    if not precomputed_path:
        return None

    ext = os.path.splitext(precomputed_path)[-1].lower()
    embeddings = None

    if ext == ".npy":
        embeddings = np.load(precomputed_path)
    elif ext == ".csv":
        delimiter = ","
        with open(precomputed_path, "r") as f:
            first_line = f.readline().strip()
        first_line_fields = first_line.split(delimiter)
        if any(not field.replace(".", "").replace("-", "").replace("e", "").isdigit() for field in first_line_fields):
            embeddings = np.genfromtxt(precomputed_path, delimiter=delimiter, skip_header=1)
        else:
            embeddings = np.loadtxt(precomputed_path, delimiter=delimiter)
    elif ext in [".pt", ".pth"]:
        loaded = torch.load(precomputed_path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            embeddings = loaded.numpy()
        elif isinstance(loaded, dict):
            if "embeddings" in loaded:
                emb = loaded["embeddings"]
                embeddings = emb.numpy() if hasattr(emb, "numpy") else np.array(emb)
            else:
                raise ValueError("Checkpoint dictionary does not contain 'embeddings' key.")
        else:
            raise ValueError(f"Unsupported type loaded from {precomputed_path}: {type(loaded)}")
    else:
        raise ValueError(f"Unsupported precomputed embedding file extension: {ext}")

    return {
        "embeddings": embeddings,
        "label": None,
        "metadata": None,
        "scores": None,
    }


# ---------------------------------------------------------------------------
# Hydra-free engine
# ---------------------------------------------------------------------------


def _index_ambient(value: Any, idx: list[int]) -> Any:
    """Restrict an ambient array (or a dict of them, e.g. ``metadata``) to ``idx``."""
    if isinstance(value, dict):
        return {k: _index_ambient(v, idx) for k, v in value.items()}
    try:
        return value[idx]                      # numpy / torch fancy indexing
    except Exception:                          # noqa: BLE001 - lists, or anything unindexable
        try:
            return [value[i] for i in idx]
        except Exception:                      # noqa: BLE001
            return value


class _SubsetView:
    """Attribute view over a torch ``Subset`` that keeps ambient arrays row-aligned.

    Metrics receive the evaluation dataset and read ``.data`` (the ambient matrix) or
    ``.metadata`` (population labels) to compare against the embeddings. ``random_split``
    hands back a ``torch.utils.data.Subset``, which exposes neither, so `trustworthiness`,
    `continuity`, `knn_preservation` and `kmeans_stratification` all raised
    ``AttributeError: 'Subset' object has no attribute 'data'`` for every datamodule that
    splits (torus, saddle_surface) while working fine for those running ``mode="full"``.

    Slicing by ``subset.indices`` is the point: forwarding the UNDERLYING dataset instead
    would hand N embeddings alongside M > N ambient rows, which `correlation` catches as a
    shape mismatch but the neighbourhood metrics would silently score against the wrong rows.
    """

    def __init__(self, subset: Any) -> None:
        self._subset = subset
        self._base = subset.dataset
        self._idx = list(subset.indices)

    #: Row-indexed attributes we know how to realign, by kind.
    _SLICED_ATTRS = ("data", "metadata")            # (N, …) arrays → slice rows
    _SLICED_CALLABLES = ("get_labels", "step_trace_ids")   # callables returning (N, …)
    _SQUARE_CALLABLES = ("get_gt_dists",)           # callables returning (N, N) → slice BOTH

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):               # never recurse into our own slots
            raise AttributeError(name)
        value = getattr(self._base, name)

        if name in self._SLICED_ATTRS:
            return _index_ambient(value, self._idx)

        if name in self._SLICED_CALLABLES or name in self._SQUARE_CALLABLES:
            square = name in self._SQUARE_CALLABLES

            def _realigned(*args: Any, **kwargs: Any) -> Any:
                out = value(*args, **kwargs)
                if out is None:
                    return None
                out = _index_ambient(out, self._idx)
                if square and getattr(out, "ndim", 0) == 2:
                    out = out[:, self._idx]        # a distance matrix needs both axes
                return out

            return _realigned

        # Anything else is NOT realigned, and forwarding it whole is how a protective guard
        # becomes a silent wrong answer: metrics gate on `hasattr(dataset, 'get_gt_dists')`
        # (geodesic_distance_correlation.py:43, trajectory_geometry.py:86) and
        # `assert hasattr(dataset, 'get_gt_dists')` (preservation.py:175). If those guards see
        # a full-dataset attribute behind a subset view, they pass and then index N_full rows
        # against N_subset embeddings — either an IndexError or, worse, a plausible number
        # computed against the wrong ground truth. Raising keeps the guards protective.
        raise AttributeError(
            f"{name!r} is not row-realigned for a dataset Subset. Add it to "
            "_SubsetView._SLICED_* if it is row-indexed, or read it from the base dataset "
            "explicitly if it is not."
        )

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, i: Any) -> Any:
        return self._subset[i]


def _unwrap_for_metrics(ds: Any) -> Any:
    """A ``Subset`` becomes a row-aligned view; anything else passes through untouched."""
    from torch.utils.data import Subset

    return _SubsetView(ds) if isinstance(ds, Subset) else ds


def run_experiment(
    datamodule: LightningDataModule,
    algorithm,
    trainer: Trainer,
    *,
    embedding_callbacks: list[EmbeddingCallback] | None = None,
    metrics=None,
    metrics_cfg=None,
    sampling=None,
    seed: int = 42,
    eval_only: bool = False,
    pretrained_ckpt: str | None = None,
    cache_dir: str | None = None,
    wandb_run=None,
) -> dict[str, Any]:
    """Hydra-free experiment engine.

    Runs the full experiment pipeline — seed, data extraction, optional
    pre-fit sampling, algorithm fit/transform, evaluation, callbacks, and
    wandb logging — without depending on any Hydra / OmegaConf objects.

    Args:
        datamodule: An already-instantiated LightningDataModule.
        algorithm: A :class:`LatentModule` or :class:`LightningModule` instance.
        trainer: An already-instantiated Lightning :class:`Trainer`.
        embedding_callbacks: Optional list of :class:`EmbeddingCallback` objects
            to run after embedding computation.
        metrics: ``list[str]`` of registry metric names **or**
            ``dict[str, DictConfig]`` of Hydra metric configs (from
            ``flatten_and_unroll_metrics``).  Passed to
            :func:`manylatents.evaluate.evaluate`.
        metrics_cfg: Raw ``cfg.metrics`` DictConfig for LightningModule model
            metrics (used by ``evaluate_lightningmodule``).  May be ``None``
            for LatentModule runs.
        sampling: Dict keyed by output name whose values are already-instantiated
            sampler objects.  A ``"dataset"`` key triggers pre-fit subsampling;
            other keys are forwarded to ``evaluate()`` for post-fit sampling.
        seed: Random seed (default 42).
        eval_only: If ``True``, skip fit/transform and load precomputed
            embeddings from the datamodule.
        pretrained_ckpt: Optional path to a pretrained checkpoint
            (LightningModule only).
        cache_dir: Optional directory for disk-persisted kNN caches.
        wandb_run: An already-initialized wandb run object. If provided,
            scalar metrics are logged and the run is finished on exit.

    Returns:
        LatentOutputs dict with keys ``"embeddings"``, ``"label"``,
        ``"metadata"``, ``"scores"``, and optionally ``"callback_outputs"``.
    """
    from manylatents.evaluate import evaluate as _evaluate

    # ---- 1. Seed ----
    seed_everything(seed, workers=True)

    # ---- 2. Data setup ----
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    field_index, data_source = determine_data_source(train_loader)

    results: dict[str, Any] = {}

    # ---- 3. Eval-only path ----
    if eval_only:
        logger.info("Evaluation-only mode: loading precomputed embeddings from datamodule.")
        results = _load_precomputed_from_datamodule(datamodule) or {}
    else:
        # ---- 4a. Unroll dataloaders to tensors ----
        #
        # The FIT tensor is unrolled in dataset order, not loader order. `train_dataloader()`
        # shuffles by default (every synthetic datamodule sets `shuffle_traindata=True` in
        # Python, though every YAML says false — which is why the Hydra CLI never hit this and
        # the programmatic API always did), while `test_dataloader()` does not. In the default
        # `mode='full'` the two datasets are THE SAME OBJECT, so a LatentModule was being
        # fitted on a permutation of the exact array it was then asked to transform. Any module
        # whose output is defined on its fit rows returned the right shape with the wrong
        # pairing, silently: measured on gaussian_blob, `leiden` AMI against ground truth
        # -0.0090 versus +1.0000, `reeb_graph` -0.0098 versus +0.6877.
        #
        # Shuffling exists for SGD in the LightningModule branch, which still uses
        # `train_loader` below and is unaffected. A fit/transform estimator gains nothing from
        # it and, here, was actively broken by it.
        train_dataset = getattr(datamodule, "train_dataset", None)
        if isinstance(algorithm, LatentModule) and train_dataset is not None:
            from torch.utils.data import DataLoader

            ordered = DataLoader(train_dataset, batch_size=getattr(datamodule, "batch_size", 128),
                                 shuffle=False)
            train_tensor = torch.cat([b[field_index].cpu() for b in ordered], dim=0)
        else:
            train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
        test_tensor = torch.cat([b[field_index].cpu() for b in test_loader], dim=0)

        # ---- 4b. Extract train labels if available ----
        train_labels = None
        train_dataset = getattr(datamodule, "train_dataset", None)
        if train_dataset is not None and hasattr(train_dataset, "get_labels"):
            labels = train_dataset.get_labels()
            if labels is not None:
                train_labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                logger.info(f"Extracted {len(train_labels)} training labels for supervised learning")

        # Labels to return with embeddings (used by plotting callbacks)
        output_labels = None
        test_dataset = getattr(datamodule, "test_dataset", None)
        if test_dataset is not None and hasattr(test_dataset, "get_labels"):
            output_labels = test_dataset.get_labels()

        # ---- 4c. Pre-fit sampling ----
        pre_fit_indices = None
        if sampling is not None and "dataset" in sampling:
            dataset_sampler = sampling["dataset"]
            sampler_input = train_tensor.numpy() if torch.is_tensor(train_tensor) else train_tensor
            sampler_kwargs: dict[str, Any] = {}
            if train_dataset is not None:
                sampler_kwargs["dataset"] = train_dataset
            if train_labels is not None:
                sampler_kwargs["labels"] = (
                    train_labels.cpu().numpy()
                    if isinstance(train_labels, torch.Tensor)
                    else np.asarray(train_labels)
                )
            try:
                pre_fit_indices = dataset_sampler.get_indices(
                    sampler_input, **sampler_kwargs
                )
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                pre_fit_indices = dataset_sampler.get_indices(sampler_input)
            train_tensor = train_tensor[pre_fit_indices]
            test_tensor = test_tensor[pre_fit_indices]
            if train_labels is not None:
                train_labels = train_labels[pre_fit_indices]
            if output_labels is not None:
                if isinstance(output_labels, torch.Tensor):
                    output_labels = output_labels[torch.as_tensor(pre_fit_indices, dtype=torch.long)]
                else:
                    output_labels = np.asarray(output_labels)[pre_fit_indices]
            logger.info(f"Pre-fit sampling: {len(pre_fit_indices)} samples using {type(dataset_sampler).__name__}")

        logger.info(
            f"Running algorithm on {data_source}:\n"
            f"Train tensor shape: {train_tensor.shape}\n"
            f"Test tensor shape: {test_tensor.shape}\n"
            f"Algorithm: {type(algorithm).__name__}"
        )

        # ---- 4d/4e. Execute algorithm ----
        t_total_start = time.perf_counter()
        t_step_start = time.perf_counter()
        latents = None
        model_metrics: dict[str, Any] = {}

        if isinstance(algorithm, LatentModule):
            # ---- LatentModule path ----
            #
            # When the fit rows and the eval rows are the SAME SET, fit once and use that
            # embedding. `mode='full'` datamodules set `test_dataset = train_dataset`, so the
            # old `fit(train); transform(test)` fitted on a SHUFFLED view of exactly the array
            # it then transformed — `train_dataloader()` shuffles by default,
            # `test_dataloader()` does not. Any module whose output is defined on its fit rows
            # (a clusterer, an MDS embedding, a Reeb membership matrix) then returned the right
            # shape with the wrong row pairing, silently. Measured on gaussian_blob: `leiden`
            # AMI against ground truth -0.0090 versus +1.0000, `reeb_graph` -0.0098 versus
            # +0.6877 — repaired by this alone, with their transform() bodies untouched.
            #
            # Note the YAMLs all say `shuffle_traindata: false` while the Python defaults say
            # True, so the Hydra CLI never hit this and the programmatic API always did.
            # `fit_fraction < 1` is deliberately excluded from the shortcut. PHATE and TSNE
            # override `fit_transform` to embed only the fitted subset, so it returns `n_fit`
            # rows — correct for that method, but it would then trip the row-cardinality
            # postcondition below and turn a shipped, documented parameter into a hard raise.
            # Those modules keep the fit-then-transform path, where `transform` extends the
            # embedding back over all rows.
            fits_all_rows = float(getattr(algorithm, "fit_fraction", 1.0)) >= 1.0
            same_rows = (fits_all_rows
                         and train_tensor.shape == test_tensor.shape
                         and torch.equal(train_tensor, test_tensor))
            if same_rows:
                latents = algorithm.fit_transform(train_tensor, train_labels)
            else:
                algorithm.fit(train_tensor, train_labels)
                try:
                    latents = algorithm.transform(test_tensor)
                except NotImplementedError:
                    logger.warning(
                        f"{type(algorithm).__name__} does not support transform(). "
                        "Falling back to fit_transform() on test data (transductive mode)."
                    )
                    # `train_labels` belong to the fit rows, not these — pass the eval labels
                    # if we have them. Previously nothing was passed, so a supervised
                    # transductive module silently retrained unsupervised.
                    fallback_y = output_labels if output_labels is not None else None
                    if fallback_y is not None and not isinstance(fallback_y, torch.Tensor):
                        fallback_y = torch.as_tensor(np.asarray(fallback_y))
                    latents = algorithm.fit_transform(test_tensor, fallback_y)

            # Postcondition, not a type: whatever a module returns, it must return one row per
            # input row. This is the axis on which `reeb_graph`, `merging`, `multiscale_phate`
            # and `diffusion_map(mode='cluster')` all fail — they return their FIT row count
            # regardless of the array handed in. Checkable here, once, without any per-module
            # cooperation or a declaration on the ABC.
            n_in = (train_tensor if same_rows else test_tensor).shape[0]
            if latents is not None and latents.shape[0] != n_in:
                raise ValueError(
                    f"{type(algorithm).__name__}.transform returned {latents.shape[0]} rows "
                    f"for {n_in} input rows. A latent module must emit one row per input row; "
                    "returning stored fit-time output silently mispairs rows with results."
                )
            logger.info(f"LatentModule embedding shape: {latents.shape}")

        elif isinstance(algorithm, LightningModule):
            # ---- LightningModule path ----
            if pretrained_ckpt:
                logger.info(f"Loading pretrained model from {pretrained_ckpt}")
                algorithm = LightningModule.load_from_checkpoint(pretrained_ckpt)
            else:
                logger.info("Running training...")
                trainer.fit(algorithm, datamodule=datamodule)

            # Model evaluation (uses metrics_cfg, not the full Hydra cfg)
            logger.info("Running model evaluation.")
            model_metrics, model_error = _evaluate_lightningmodule(
                algorithm,
                trainer=trainer,
                datamodule=datamodule,
                metrics_cfg=metrics_cfg,
            )
            logger.info(f"Model evaluation completed. Error: {model_error}, Metrics: {model_metrics}")

            # Extract embeddings from encoder
            if hasattr(algorithm, "encode"):
                logger.info("Extracting embeddings using network encoder...")
                latents = algorithm.encode(test_tensor)
                latents = latents.detach().cpu().numpy() if isinstance(latents, torch.Tensor) else latents
                logger.info(f"LightningModule embedding shape: {latents.shape}")
            else:
                logger.warning(f"LightningModule {type(algorithm).__name__} has no 'encode' method — skipping")

        step_time = time.perf_counter() - t_step_start

        # ---- 4f. Package results ----
        if latents is not None:
            results = {
                "embeddings": latents,
                "label": output_labels,
                "metadata": {
                    "source": "single_algorithm",
                    "algorithm_type": type(algorithm).__name__,
                    "data_shape": test_tensor.shape,
                    "step_time": step_time,
                },
            }

            # Merge model metrics (LightningModule path)
            if model_metrics:
                results.setdefault("scores", {}).update(model_metrics)

            # ---- 4g. Attach extra outputs from any algorithm that exposes them ----
            # Two halves, one call. The GENERIC outputs (trajectories/affinity/
            # adjacency/kernel) come from the registry in `manylatents.outputs`, so they
            # no longer depend on the algorithm inheriting LatentModule — that inheritance
            # is why MIOFlow's trajectories were unreachable (#295). ALGORITHM-SPECIFIC
            # ones still come from the algorithm's own extra_outputs() (Cflows' GRN head).
            # The `hasattr` gate is gone because collect_outputs() handles absence.
            for key, val in collect_outputs(algorithm).items():
                results[key] = val
                shape_info = f" shape={val.shape}" if hasattr(val, "shape") else ""
                logger.info(f"Extra output attached: {key}{shape_info}")

            # ---- 4h. Evaluate embedding metrics ----
            if metrics is not None:
                logger.info(f"Evaluating embeddings from {type(algorithm).__name__}...")
                # Build post-fit sampling dict (exclude pre-fit "dataset" key)
                post_fit_sampling = None
                if sampling is not None:
                    post_fit_sampling = {k: v for k, v in sampling.items() if k != "dataset"}
                    if not post_fit_sampling:
                        post_fit_sampling = None

                # Determine dataset for evaluation
                mode = getattr(datamodule, "mode", None) or getattr(getattr(datamodule, "hparams", None), "mode", "full")
                if mode == "split":
                    ds = datamodule.test_dataset
                else:
                    ds = datamodule.train_dataset
                ds = _unwrap_for_metrics(ds)   # random_split yields a Subset; metrics need .data

                t_eval_start = time.perf_counter()
                embedding_scores = _evaluate(
                    results["embeddings"],
                    dataset=ds,
                    module=algorithm if isinstance(algorithm, LatentModule) else None,
                    metrics=metrics,
                    sampling=post_fit_sampling,
                    cache_dir=cache_dir,
                )
                eval_time = time.perf_counter() - t_eval_start
                results.setdefault("scores", {}).update(embedding_scores)
                total_time = time.perf_counter() - t_total_start

                results["metadata"]["eval_time"] = eval_time
                results["metadata"]["total_time"] = total_time

    # ---- 5. Run embedding callbacks ----
    callback_outputs: dict[str, Any] = {}
    if results and embedding_callbacks:
        for cb in embedding_callbacks:
            cb_result = cb.on_latent_end(dataset=datamodule.test_dataset, embeddings=results)
            if isinstance(cb_result, dict):
                callback_outputs.update(cb_result)
                logger.info(f"Callback {cb.__class__.__name__} returned: {list(cb_result.keys())}")

    if callback_outputs:
        results["callback_outputs"] = callback_outputs
        logger.info(f"Added callback outputs to results: {list(callback_outputs.keys())}")

    # ---- 6. Log to wandb ----
    if wandb_run is not None and results.get("scores"):
        scores = results["scores"]
        scalar_metrics = {}
        for name, val in scores.items():
            if isinstance(val, tuple) and len(val) == 2:
                scalar_metrics[f"metrics/{name}"] = float(val[0])
            elif np.ndim(val) == 0:
                scalar_metrics[f"metrics/{name}"] = float(val)
        if scalar_metrics:
            wandb_run.log(scalar_metrics)
            logger.info(f"Auto-logged {len(scalar_metrics)} metrics to wandb: {list(scalar_metrics.keys())}")

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Engine run complete.")

    # ---- 7. Return results ----
    return results


def _evaluate_lightningmodule(
    algorithm: LightningModule,
    *,
    trainer: Trainer,
    datamodule,
    metrics_cfg=None,
) -> tuple[dict[str, Any], float | None]:
    """Evaluate a LightningModule without requiring a full Hydra cfg.

    This is the Hydra-free counterpart of the old ``evaluate_lightningmodule``.
    It runs ``trainer.test()`` and then any model-level metrics specified via
    *metrics_cfg*.

    Args:
        algorithm: The LightningModule to evaluate.
        trainer: Lightning Trainer.
        datamodule: DataModule or DataLoader for testing.
        metrics_cfg: Optional metric configs (DictConfig or dict).  If provided
            and contains a ``"model"`` key, those model metrics are instantiated
            via ``hydra.utils.instantiate`` and applied.

    Returns:
        (combined_metrics, error_value) tuple.
    """
    if not hasattr(algorithm, "test_step"):
        logger.info("Model does not define a test_step() method; skipping evaluation.")
        return {}, None

    results = trainer.test(model=algorithm, datamodule=datamodule)
    if not results:
        return {}, None

    base_metrics = results[0]
    custom_metrics: dict[str, Any] = {}

    # Model-level metrics from config (if any)
    model_metrics_cfg: dict[str, Any] = {}
    if metrics_cfg is not None:
        if hasattr(metrics_cfg, "get"):
            model_metrics_cfg = metrics_cfg.get("model", {}) or {}
        elif isinstance(metrics_cfg, dict):
            model_metrics_cfg = metrics_cfg.get("model", {})

    for metric_key, metric_params in model_metrics_cfg.items():
        if isinstance(metric_params, dict) and not metric_params.get("enabled", True):
            continue
        if hasattr(metric_params, "get") and not metric_params.get("enabled", True):
            continue
        import hydra as _hydra
        metric_fn = _hydra.utils.instantiate(metric_params)
        name, value = metric_fn(algorithm, test_results=base_metrics)
        custom_metrics[name] = value

    combined_metrics = {**base_metrics, **custom_metrics}
    error_value = next(iter(combined_metrics.values())) if combined_metrics else None
    return combined_metrics, error_value
