# Callbacks

manyLatents has two callback systems: **embedding callbacks** for post-embedding processing, and **trainer callbacks** (Lightning) for training-time hooks.

=== "Architecture"

    ## Callback Hierarchy

    ```
    BaseCallback (ABC)
    ├── on_experiment_start(cfg)
    ├── on_experiment_end()
    ├── on_latent_end(dataset, embeddings)
    ├── on_training_start()
    └── on_training_end()

    EmbeddingCallback(BaseCallback, ABC)
    ├── on_latent_end(dataset, embeddings)  ← abstract, must implement
    ├── register_output(key, output)        ← store results for downstream
    └── callback_outputs: dict              ← accumulated outputs

    lightning.Callback                      ← PyTorch Lightning's callback
    ├── on_fit_start()
    ├── on_train_batch_end()
    ├── on_train_epoch_end()
    ├── on_validation_end()
    └── ...
    ```

    `EmbeddingCallback` runs **after** embeddings are computed (for both LatentModule and LightningModule). Lightning `Callback` runs **during** training (LightningModule only).

    ## Instantiation

    Callbacks are instantiated from two config groups and routed by type:

    ```python
    def instantiate_callbacks(trainer_cb_cfg, embedding_cb_cfg):
        lightning_cbs, embedding_cbs = [], []

        for name, cfg in trainer_cb_cfg.items():
            cb = hydra.utils.instantiate(cfg)
            if isinstance(cb, Callback):
                lightning_cbs.append(cb)

        for name, cfg in embedding_cb_cfg.items():
            cb = hydra.utils.instantiate(cfg)
            if isinstance(cb, EmbeddingCallback):
                embedding_cbs.append(cb)

        return lightning_cbs, embedding_cbs
    ```

    ## Config Structure

    ```yaml
    # configs/callbacks/default.yaml
    defaults:
      - trainer: null         # Lightning callbacks (probing, etc.)
      - embedding: null       # Embedding callbacks (save, plot, wandb)
      - _self_
    ```

    ```yaml
    # configs/callbacks/embedding/default.yaml
    defaults:
      - save_embeddings
      - plot_embeddings
      - wandb_log_scores
      - _self_
    ```

    ## Execution Flow

    In `run_algorithm()`:

    1. Callbacks instantiated from config
    2. Lightning callbacks passed to `Trainer(callbacks=[...])`
    3. Algorithm executes (fit/transform or trainer.fit)
    4. Embeddings wrapped as `EmbeddingOutputs` dict
    5. Metrics evaluated
    6. Each embedding callback's `on_latent_end()` called with dataset + embeddings
    7. Callback outputs merged into the embeddings dict

    ```python
    for cb in embedding_cbs:
        cb_result = cb.on_latent_end(dataset=datamodule.test_dataset, embeddings=embeddings)
        if isinstance(cb_result, dict):
            callback_outputs.update(cb_result)
    ```

    ## EmbeddingOutputs

    The standard interchange format passed to all embedding callbacks:

    ```python
    EmbeddingOutputs = dict[str, Any]
    # Required: "embeddings" (np.ndarray)
    # Optional: "label", "metadata", "scores", "callback_outputs"
    ```

    `validate_embedding_outputs()` checks the required key exists.

=== "Embedding Callbacks"

    ## SaveEmbeddings

    Saves embeddings to disk in CSV or NPY format. Optionally saves metric tables (scalar summary and per-sample).

    ```yaml
    # configs/callbacks/embedding/save_embeddings.yaml
    save_embeddings:
      _target_: manylatents.callbacks.embedding.save_embeddings.SaveEmbeddings
      save_dir: ${hydra:runtime.output_dir}
      save_format: "csv"
      experiment_name: ${name}
      save_metric_tables: false
    ```

    | Parameter | Default | Description |
    |-----------|---------|-------------|
    | `save_dir` | Hydra output dir | Base directory for saved files |
    | `save_format` | `"csv"` | Format: `"csv"` or `"npy"` |
    | `save_metric_tables` | `false` | Save separate scalar + per-sample metric CSVs |
    | `save_additional_outputs` | `false` | Save non-embedding keys as separate files |

    When running under Geomancer orchestration, also writes to the shared metrics directory via `atomic_writer`.

    ## PlotEmbeddings

    Creates 2D scatter plots of embeddings with customizable colormaps and optional WandB upload.

    ```yaml
    # configs/callbacks/embedding/plot_embeddings.yaml
    plot_embeddings:
      _target_: manylatents.callbacks.embedding.plot_embeddings.PlotEmbeddings
      save_dir: ${hydra:runtime.output_dir}
      experiment_name: "${name}.png"
      figsize: [8, 6]
      label_col: Population
      legend: false
      color_by_score: null
    ```

    ### Colormap Resolution

    PlotEmbeddings resolves colormaps from multiple sources (highest priority first):

    1. **User overrides** — `cmap_override`, `is_categorical_override` in config
    2. **Metric-declared** — `scores["<metric>__viz"]` containing a `ColormapInfo`
    3. **Dataset-provided** — via the `ColormapProvider` protocol
    4. **Defaults** — `"viridis"`

    Datasets can implement `ColormapProvider` to declare their preferred visualization:

    ```python
    class MyDataset(ColormapProvider):
        def get_colormap_info(self) -> ColormapInfo:
            return ColormapInfo(
                cmap={"A": "#ff0000", "B": "#00ff00"},
                label_names={0: "Class A", 1: "Class B"},
                is_categorical=True,
            )
    ```

    ### Coloring by Score

    Color points by a metric value instead of labels:

    ```yaml
    plot_embeddings:
      color_by_score: "embedding.local_intrinsic_dimensionality"
      legend: false  # Uses colorbar instead
    ```

    ## WandbLogScores

    Logs metric scores to WandB in three formats:

    ```yaml
    # configs/callbacks/embedding/wandb_log_scores.yaml
    wandb_log_scores:
      _target_: manylatents.callbacks.embedding.wandb_log_scores.WandbLogScores
    ```

    | Log Type | WandB Key | Content |
    |----------|-----------|---------|
    | Summary scalars | `{tag}/metric_name` | 0-D metrics as `wandb.log()` |
    | Per-sample table | `{tag}/per_sample_metrics` | 1-D arrays as `wandb.Table` |
    | k-curve tables | `{tag}/metric__k_curve_table` | Swept `n_neighbors` values grouped into tables |

    k-curve tables automatically detect metrics swept over `n_neighbors` (e.g., `trustworthiness__n_neighbors_5`, `_10`, `_20`) and group them into a single curve.

    ## LoadingsAnalysisCallback

    Analyzes shared vs modality-specific components in multi-modal loadings (e.g., DNA + RNA + Protein fusion).

    ```yaml
    callbacks:
      embedding:
        loadings:
          _target_: manylatents.callbacks.embedding.loadings_analysis.LoadingsAnalysisCallback
          modality_dims: [1920, 256, 1536]
          modality_names: [dna, rna, protein]
          threshold: 0.1
    ```

    Requires the algorithm module to have a `get_loadings()` method (e.g., `MergingModule` with `concat_pca`).

=== "Trainer Callbacks"

    ## Lightning Callbacks

    Trainer callbacks extend `lightning.Callback` and run during the training loop. They are passed to `Trainer(callbacks=[...])`.

    ### RepresentationProbeCallback

    The primary trainer callback. Extracts activations from model layers at configurable triggers and computes diffusion operators to track representation geometry.

    ```yaml
    # configs/callbacks/trainer/probe.yaml
    probe:
      _target_: manylatents.lightning.callbacks.probing.RepresentationProbeCallback
      layer_specs:
        - _target_: manylatents.lightning.hooks.LayerSpec
          path: "transformer.h[-1]"
          extraction_point: "output"
          reduce: "mean"
      trigger:
        _target_: manylatents.lightning.callbacks.probing.ProbeTrigger
        every_n_steps: 500
        on_checkpoint: true
        on_validation_end: true
      gauge:
        _target_: manylatents.callbacks.probing.DiffusionGauge
        knn: 15
        alpha: 1.0
        symmetric: false
      log_to_wandb: true
    ```

    See **[Probing](probing.md)** for full documentation of layer specs, triggers, gauge configuration, and programmatic access.

    ### Adding a Trainer Callback

    1. Create a class extending `lightning.Callback`
    2. Implement the relevant hooks (`on_train_batch_end`, `on_train_epoch_end`, etc.)
    3. Create a config in `configs/callbacks/trainer/your_callback.yaml`
    4. Add to your experiment config:

    ```yaml
    callbacks:
      trainer:
        your_callback:
          _target_: manylatents.your_module.YourCallback
          param: value
    ```

    ### Adding an Embedding Callback

    1. Create a class extending `EmbeddingCallback`
    2. Implement `on_latent_end(self, dataset, embeddings) -> Any`
    3. Use `self.register_output(key, value)` to store results
    4. Create a config in `configs/callbacks/embedding/your_callback.yaml`
    5. Add to the embedding defaults or your experiment config
