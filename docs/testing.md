# Testing

Testing strategy and CI pipeline for manyLatents.

=== "CI Pipeline"

    ## GitHub Actions Pipeline

    ### Two-Phase Testing

    1. **Build & Unit Tests** (20 min): Unit tests + CLI smoke tests for both LatentModule and LightningModule paths
    2. **Discovery Smoke Tests** (conditional): Automatic testing of all algorithm/metric configs when their source changes

    ## Discovery Smoke Tests

    **Triggers on changes to**:

    - `manylatents/algorithms/latent/**` → runs `.github/workflows/scripts/test_latent_algorithms.sh`
    - `manylatents/algorithms/lightning/**` → runs `.github/workflows/scripts/test_lightning_algorithms.sh`
    - `manylatents/metrics/**` → runs `.github/workflows/scripts/test_metrics.sh`

    Each script dynamically discovers all configs in its directory and runs quick instantiation tests using minimal test data.

    **Validates**:

    - Algorithm/metric config is valid YAML
    - Target class can be instantiated via Hydra
    - Full pipeline completes without errors

    **Does NOT test**: mathematical correctness (unit tests cover that).

    ## CLI Smoke Tests

    Every CI run includes two fixed smoke tests:

    | Test | Config | Purpose |
    |------|--------|---------|
    | LatentModule CLI | `experiment=single_algorithm metrics=noop` | Validates non-neural path |
    | LightningModule CLI | `algorithms/lightning=ae_reconstruction trainer.fast_dev_run=true` | Validates neural path |

    ## Local Testing

    ```bash
    # Quick smoke test (LatentModule)
    uv run python -m manylatents.main \
      algorithms/latent=noop data=test_data metrics=noop logger=none

    # PCA + SwissRoll with metrics
    uv run python -m manylatents.main \
      algorithms/latent=pca data=swissroll \
      metrics/embedding=trustworthiness logger=none

    # Autoencoder + SwissRoll (fast dev run)
    uv run python -m manylatents.main \
      algorithms/lightning=ae_reconstruction data=swissroll \
      trainer=default trainer.max_epochs=2 trainer.fast_dev_run=true \
      logger=none

    # Full pytest suite
    uv run pytest tests/ -x -q

    # Callback tests
    uv run pytest manylatents/callbacks/tests/ -x -q

    # Docs coverage check
    uv run python scripts/check_docs_coverage.py
    ```

    ## Adding New Algorithm Tests

    Discovery scripts auto-detect new configs. Just add a YAML file to the right directory:

    - `manylatents/configs/algorithms/latent/your_algo.yaml` — auto-discovered by `test_latent_algorithms.sh`
    - `manylatents/configs/algorithms/lightning/your_algo.yaml` — auto-discovered by `test_lightning_algorithms.sh`
    - `manylatents/configs/metrics/embedding/your_metric.yaml` — auto-discovered by `test_metrics.sh`

    ## Failure Investigation

    1. Check test-specific artifacts in GitHub Actions
    2. Run the same configuration locally
    3. Enable `HYDRA_FULL_ERROR=1` for detailed stack traces

=== "Namespace Extensions"

    ## Testing manylatents + Extension Packages

    manyLatents uses `pkgutil.extend_path()` for namespace extensions. The `manylatents-omics` package adds:

    - `manylatents.dogma` — foundation model encoders (Evo2, ESM3, Orthrus, AlphaGenome)
    - `manylatents.popgen` — population genetics datasets and metrics
    - `manylatents.singlecell` — AnnData integration

    ## Verification

    ### Package Discovery

    ```python
    import manylatents
    print(manylatents.__path__)  # Should show multiple paths if extension installed
    ```

    ### Module Loading

    ```python
    # Only available with manylatents-omics installed
    from manylatents.dogma.encoders import Evo2Encoder
    ```

    ## Key Principle

    Core `manylatents` never imports from extensions. Extensions register configs via Hydra's `SearchPathPlugin` and are discovered at runtime.

    ### Namespace Declaration

    Both packages must have in `manylatents/__init__.py`:

    ```python
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)
    ```

    ## Troubleshooting

    ### "No module named 'manylatents.dogma'"

    Install the extension package:

    ```bash
    pip install manylatents-omics
    # or for development
    uv add -e /path/to/manylatents-omics --no-deps
    ```

    ### CLI Breaks With Extension

    The `manylatents.dogma` namespace is imported conditionally. If the extension has unmet dependencies, the CLI may fail. Install with all extras: `pip install manylatents-omics[all]`.
