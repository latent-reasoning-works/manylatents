# Testing

Testing strategy, CI pipeline, and namespace integration testing for manyLatents.

=== "CI Pipeline"

    ## GitHub Actions Pipeline

    ### Three-Phase Testing

    1. **Build & Unit Tests** (20 min): Fast validation of code quality and unit tests
    2. **LatentModule Smoke Tests** (conditional): Automatic testing of all DR algorithms when configs change
    3. **Integration Tests** (25 min): Matrix testing of algorithm combinations with real data

    ## LatentModule Smoke Tests

    **Triggers on changes to**:

    - `manylatents/algorithms/latent/**`
    - `manylatents/configs/algorithms/latent/**`

    **Script**: `.github/workflows/scripts/test_latent_algorithms.sh`

    Dynamically discovers all algorithm configs and runs quick instantiation tests using minimal test data (swissroll: 10 distributions x 20 points). Currently tests: `aa`, `diffusionmap`, `mds`, `noop`, `pca`, `phate`, `tsne`, `umap`.

    **Validates**:

    - Algorithm config is valid YAML
    - Algorithm class can be instantiated via Hydra
    - Algorithm can fit and transform data
    - Full pipeline completes without errors

    **Does NOT test**: mathematical correctness (unit tests), Lightning modules, or comprehensive data/metric combinations (integration tests).

    ## Integration Test Matrix

    | Test | Algorithm | Data | Timeout |
    |------|-----------|------|---------|
    | smoke-test | `latent/noop` | `test_data` | 2 min |
    | pca-swissroll | `latent/pca` | `swissroll` | 8 min |
    | umap-swissroll | `latent/umap` | `swissroll` | 10 min |
    | autoencoder-swissroll | `lightning/ae_reconstruction` | `swissroll` | 15 min |

    ## Local Testing

    ```bash
    # Smoke test
    uv run python -m manylatents.main \
      algorithms/latent=noop data=test_data metrics=test_metric debug=true

    # PCA + SwissRoll
    uv run python -m manylatents.main \
      algorithms/latent=pca data=swissroll metrics=synthetic_data_metrics \
      debug=true trainer.max_epochs=1

    # Autoencoder + SwissRoll
    uv run python -m manylatents.main \
      algorithms/lightning=ae_reconstruction data=swissroll \
      debug=true trainer.max_epochs=1

    # Full pytest suite
    pytest --cov=manylatents --cov-report=xml
    ```

    ## Adding New Algorithm Tests

    Edit `.github/workflows/build.yml`:

    ```yaml
    - name: "new-algorithm-test"
      algorithm: "latent/your_algorithm"
      data: "test_dataset"
      metrics: "appropriate_metrics"
      timeout: 5
    ```

    ## Timeout Guidelines

    | Type | Timeout |
    |------|---------|
    | Smoke tests | 2-3 min |
    | Traditional DR | 5-10 min |
    | Neural networks | 10-20 min |
    | Large datasets | 20-30 min |

    ## Failure Investigation

    1. Check test-specific artifacts in GitHub Actions
    2. Run the same configuration locally
    3. Enable `HYDRA_FULL_ERROR=1` for detailed stack traces
    4. Use `debug=true` for verbose logging

=== "Namespace Integration"

    ## Testing manylatents + manylatents-omics Integration

    This guide validates that the namespace integration works end-to-end.

    ## Setup

    ```bash
    # Install core package
    uv add -e /path/to/manylatents --no-deps

    # Install mock omics package for testing
    uv add -e /path/to/manylatents/tests/mock_omics_package --no-deps

    # Verify imports
    python3 -c "
    from manylatents.omics.data import PlinkDataset
    from manylatents.omics.metrics import GeographicPreservation
    print('Namespace integration successful')
    "
    ```

    ## Integration Test Config

    ```yaml
    # configs/experiment/integration_test.yaml
    data:
      _target_: manylatents.data.synthetic_dataset.SwissRoll
      n_samples: 100
      noise: 0.01
      seed: 42

    algorithms:
      latent:
        _target_: manylatents.algorithms.latent.pca.PCAModule
        n_components: 2

    metrics:
      embedding:
        - continuity
        - trustworthiness

    logger: none
    trainer:
      max_epochs: 1
    ```

    ## Verification Levels

    ### Level 1: Package Discovery

    ```python
    import manylatents
    print(manylatents.__path__)  # Should show multiple paths
    ```

    ### Level 2: Module Loading

    ```python
    from manylatents.omics import data, metrics
    ```

    ### Level 3: Class Instantiation

    ```python
    from manylatents.omics.data import PlinkDataset
    from manylatents.omics.metrics import GeographicPreservation
    dataset = PlinkDataset()
    metric = GeographicPreservation()
    ```

    ## Troubleshooting

    ### "No module named 'manylatents.omics'"

    Install the extension:
    ```bash
    uv add -e ./tests/mock_omics_package --no-deps
    ```

    ### Namespace not extending

    Ensure both packages have the declaration in `manylatents/__init__.py`:
    ```python
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)
    ```

    ## Checklist

    - [ ] Core manylatents imports work
    - [ ] Omics namespace is importable
    - [ ] Mock classes can be instantiated
    - [ ] Both packages coexist in the same environment

=== "Mock Package Pattern"

    ## Local Namespace Testing Without Private Repos

    The organization is on a free GitHub plan, so CI cannot access private repos like `manylatents-omics`. The solution: a mock package that simulates the extension structure.

    ## Directory Structure

    ```
    tests/
    ├── mock_omics_package/
    │   ├── setup.py
    │   └── manylatents/
    │       ├── __init__.py           # Namespace declaration
    │       └── omics/
    │           ├── __init__.py
    │           ├── data/
    │           │   └── __init__.py   # Mock PlinkDataset
    │           └── metrics/
    │               └── __init__.py   # Mock GeographicPreservation
    └── test_namespace_integration.py
    ```

    ## How It Works

    1. Declares `manylatents` as a namespace package using `pkgutil.extend_path()`
    2. Provides mock implementations of key classes
    3. Allows testing namespace integration without the private repo

    ## Running

    ```bash
    python3 tests/test_namespace_integration.py
    ```

    The script creates a temporary venv, installs both packages, verifies namespace merging, and cleans up.

    ## Why This Approach

    - No CI secrets needed
    - Validates the namespace architecture
    - Fast (no large repo cloning)
    - Part of the repo and under version control

    ## When to Update

    Update the mock package if:

    - `manylatents-omics` adds new top-level modules
    - Namespace package organization changes
    - New integration scenarios need testing

    Mock classes don't need implementation details — they just need to exist so Python can import them.
