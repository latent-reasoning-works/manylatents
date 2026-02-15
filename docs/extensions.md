# Extensions

manyLatents uses a modular extension system that allows domain-specific functionality to be installed as separate packages. Extensions integrate seamlessly through Python's namespace package system and Hydra's config composition.

## Available Extensions

### manylatents-omics

The genomics, population genetics, and single-cell extension for manyLatents.

**Repository**: [github.com/latent-reasoning-works/manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics)

**Adds three submodules:**

- `manylatents.dogma` — Foundation model encoders (Evo2, ESM3, Orthrus, AlphaGenome) and fusion algorithms
- `manylatents.popgen` — Population genetics data modules and metrics (GeographicPreservation, AdmixturePreservation)
- `manylatents.singlecell` — Single-cell AnnData data modules

---

=== "Usage"

    ## Installing

    ### Quick Install (Recommended)

    ```bash
    uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
    ```

    ### Interactive Setup

    ```bash
    bash setup_extensions.sh
    ```

    ### Using Git Submodules

    For contributors working on both core and extensions:

    ```bash
    git submodule add https://github.com/latent-reasoning-works/manylatents-omics.git extensions/manylatents-omics
    uv add git+file://extensions/manylatents-omics
    ```

    ## Development Workflow

    **Working FROM the manylatents-omics repo (recommended for omics development):**

    ```bash
    cd manylatents-omics
    uv sync  # Pulls manylatents from git automatically

    # IMPORTANT: Use omics entry point for omics configs
    uv run python -m manylatents.omics.main experiment=single_algorithm
    ```

    **Working FROM the manylatents repo (core development only):**

    ```bash
    cd manylatents
    uv sync
    uv run python -m manylatents.main experiment=single_algorithm
    ```

    For omics experiments, always work from the omics repo and use `manylatents.omics.main`. The omics entry point registers configs before Hydra initializes.

    ## Using Extensions in Code

    Once installed, extension features are available through the `manylatents` namespace:

    ```python
    # Core imports (always available)
    from manylatents.data import SwissRoll
    from manylatents.algorithms.latent import PCAModule

    # Extension imports (available when manylatents-omics is installed)
    from manylatents.popgen.data import HGDPDataset
    from manylatents.popgen.metrics import GeographicPreservation
    from manylatents.dogma.encoders import Evo2Encoder
    from manylatents.singlecell.data import AnnDataModule
    ```

    ## Using Extensions with Hydra

    Extensions integrate directly with Hydra configs:

    ```bash
    python -m manylatents.main \
      data=hgdp_1kgp \
      algorithms/latent=pca \
      metrics=genetic_metrics \
      logger=wandb
    ```

    ## Checking What's Installed

    ```python
    import pkgutil
    import manylatents

    for importer, modname, ispkg in pkgutil.iter_modules(manylatents.__path__):
        print(f"- {modname}")
    ```

    ## Troubleshooting

    ### Hydra Config Discovery Error

    **Problem**: `ConfigAttributeError: Key 'experiment' is not in struct`

    **Cause**: Hydra SearchPathPlugin not being discovered.

    **Solutions**:

    1. Ensure the extension is installed (not just cloned):
        ```bash
        uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
        ```
    2. If developing both packages, work from the omics repo
    3. Verify plugin discovery:
        ```python
        from hydra.core.plugins import Plugins
        from hydra.plugins.search_path_plugin import SearchPathPlugin
        plugins = list(Plugins.instance().discover(SearchPathPlugin))
        print([p.__name__ for p in plugins])
        ```

    ### Extension Not Found

    **Problem**: `ModuleNotFoundError: No module named 'manylatents.omics'`

    **Solution**: Install the extension:
    ```bash
    uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
    ```

    ### Import Conflicts

    **Problem**: Namespace package not merging correctly.

    **Solution**: Ensure both packages have the namespace declaration in `manylatents/__init__.py`:
    ```python
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)
    ```

=== "Architecture"

    ## Design Philosophy

    manyLatents is built around a simple idea: **every interface between stages is a file with a known schema**. This matters because the agents and scripts that compose manyLatents into larger workflows are stateless — they don't remember what happened in the last call. If the output of one step doesn't fully describe itself, the next step can't use it.

    This constraint shapes everything:

    - **EmbeddingOutputs** is a `dict[str, Any]`, not a dataclass. When a new metric injects a custom field, every downstream consumer still works without schema migration.
    - **Metrics** are registered via Hydra configs with `_target_` and `_partial_: True`. Parameters are bound at config time, not at call time, so the evaluation engine doesn't need to know what parameters each metric takes.
    - **Algorithms** are either `LatentModule` (fit/transform) or `LightningModule` subclasses (training loops). The execution engine dispatches on type, not on name.

    The result is a system where you can add a new algorithm, metric, dataset, or entire domain extension without touching core code.

    ## Two Execution Modes

    **CLI** (`python -m manylatents.main`) executes a single step: one algorithm + metrics on one dataset. This is the primary user-facing interface and what SLURM jobs invoke.

    **Python API** (`manylatents.api.run()`) is the programmatic interface designed for agent-driven workflows. It accepts `input_data` to chain the output of one call into the next, and supports `pipeline` configs for sequential steps within a single process.

    ```python
    from manylatents.api import run

    # Single step
    result = run(
        data='swissroll',
        algorithms={'latent': {
            '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
            'n_components': 50
        }}
    )

    # Chaining: feed output of one step into the next
    result2 = run(
        input_data=result['embeddings'],
        algorithms={'latent': {
            '_target_': 'manylatents.algorithms.latent.phate.PHATEModule',
            'n_components': 2
        }}
    )
    ```

    ## Namespace Extension via pkgutil

    The core package's `__init__.py` contains one line:

    ```python
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)
    ```

    This tells Python: "if another installed package also defines a `manylatents` directory, merge its contents into mine." The rule is simple: **core never imports from extensions; extensions import from core.**

    Extensions also register a Hydra `SearchPathPlugin` so their configs are discovered automatically:

    ```python
    class OmicsSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(self, search_path):
            search_path.append(
                provider="manylatents-omics",
                path="pkg://manylatents.dogma.configs",
            )
    ```

    ## Four Extension Axes

    ### 1. Algorithms

    Two base classes, binary decision rule:

    - **`LatentModule`** — fit/transform for non-neural algorithms (PCA, UMAP, PHATE, etc.). The **FoundationEncoder pattern** is a LatentModule where `fit()` is a no-op and `transform()` wraps a pretrained model.
    - **`LightningModule` subclasses** — trainable neural networks with Lightning training loops (autoencoders, Latent ODEs).

    Optional methods `kernel_matrix()` and `affinity_matrix()` enable module-level metrics like `KernelMatrixSparsity` and `AffinitySpectrum`.

    ### 2. Metrics

    Metrics follow the `Metric` protocol with three evaluation contexts:

    | Context | `embeddings` | `dataset` | `module` | Use case |
    |---------|-------------|-----------|----------|----------|
    | `embedding` | Low-dim output | Source dataset | - | Trustworthiness, continuity |
    | `dataset` | - | Source dataset | - | Stratification, admixture |
    | `module` | - | Source dataset | Fitted LatentModule | Affinity spectrum, kernel sparsity |

    List-valued parameters in configs expand via Cartesian product through `flatten_and_unroll_metrics()`. Metrics sharing kNN graphs use a shared `cache`.

    ### 3. Data Modules

    Data modules provide `get_data()` and are auto-discovered at import time. Synthetic datasets generate on-the-fly; file-based datasets load from disk. For LightningModule algorithms, they also implement `LightningDataModule`.

    ### 4. Domain Extensions

    A domain extension is a separate installable package that adds algorithms, metrics, and data modules to the `manylatents` namespace. See the **Development** tab for how to create one.

    ## Hydra Configuration

    Every extensible component has a corresponding Hydra config group:

    ```
    configs/
      algorithms/
        latent/         # LatentModule configs
        lightning/      # LightningModule configs
          loss/         # Loss function configs
          network/      # Network architecture configs
          optimizer/    # Optimizer configs
      data/             # Dataset configs
      metrics/
        embedding/      # Embedding-level metric configs
        dataset/        # Dataset-level metric configs
        module/         # Module-level metric configs
        sampling/       # Metric sampling strategies
      callbacks/embedding/
      experiment/       # Experiment preset configs
      trainer/          # Lightning trainer configs
      logger/           # Logger configs (none, wandb)
      cluster/          # SLURM cluster configs (via Shop)
      launcher/         # Job launcher configs (via Shop)
    ```

    ## Scope Boundaries

    manyLatents owns single-step execution and the Python API for composable workflows. It does NOT own:

    - **Multi-step orchestration** — manyAgents calls `manylatents.api.run()` to compose steps
    - **RL / reward-driven training** — Geomancer
    - **Cluster job submission** — Shop provides Hydra launcher plugins

=== "Development"

    ## Creating an Extension

    This guide documents how to create extension packages for manyLatents, following the patterns established by `manylatents-omics`.

    ### Architecture Overview

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    Your Application                          │
    ├─────────────────────────────────────────────────────────────┤
    │  manylatents-yourextension   │   manylatents-omics          │
    │  (your namespace package)    │   (popgen, dogma, singlecell)│
    ├─────────────────────────────────────────────────────────────┤
    │                        shop (optional)                       │
    │              (shared SLURM launchers, logging utils)         │
    ├─────────────────────────────────────────────────────────────┤
    │                      manylatents (core)                      │
    │         (LatentModule, metrics, data, experiment runner)     │
    └─────────────────────────────────────────────────────────────┘
    ```

    ## Package Structure

    ```
    manylatents-yourextension/
    ├── pyproject.toml
    ├── README.md
    ├── CLAUDE.md                    # AI assistant instructions
    ├── manylatents/
    │   ├── __init__.py              # Namespace package declaration (CRITICAL)
    │   ├── yourext_plugin.py        # Hydra SearchPathPlugin
    │   └── yourext/
    │       ├── __init__.py
    │       ├── algorithms/
    │       │   ├── __init__.py
    │       │   └── your_algorithm.py
    │       ├── data/
    │       │   ├── __init__.py
    │       │   └── your_dataset.py
    │       ├── metrics/
    │       │   ├── __init__.py
    │       │   └── your_metric.py
    │       └── configs/
    │           ├── __init__.py      # Empty, required for pkg://
    │           ├── data/
    │           │   └── your_data.yaml
    │           ├── algorithms/
    │           │   └── latent/
    │           │       └── your_algo.yaml
    │           ├── metrics/
    │           │   └── dataset/
    │           │       └── your_metric.yaml
    │           └── experiment/
    │               └── your_experiment.yaml
    └── tests/
        ├── __init__.py
        ├── test_imports.py
        └── test_config_e2e.py
    ```

    ### Critical File: `manylatents/__init__.py`

    This file MUST contain the namespace package declaration:

    ```python
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)
    ```

    Without this, Python won't merge your package with core manyLatents.

    ## Hydra Config Integration

    ### SearchPathPlugin

    Create `manylatents/yourext_plugin.py`:

    ```python
    from hydra.core.config_search_path import ConfigSearchPath
    from hydra.plugins.search_path_plugin import SearchPathPlugin


    class YourExtSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
            search_path.append(
                provider="manylatents",
                path="pkg://manylatents.configs",
            )
            # Higher priority for YOUR configs
            search_path.prepend(
                provider="manylatents-yourext",
                path="pkg://manylatents.yourext.configs",
            )
    ```

    Use `prepend()` if your configs should override core configs with the same name, `append()` if core should take precedence.

    ### Entry Point Registration

    In `pyproject.toml`:

    ```toml
    [project.entry-points."hydra.searchpath"]
    manylatents-yourext = "manylatents.yourext_plugin:YourExtSearchPathPlugin"
    ```

    Hydra 1.3 doesn't reliably discover entry-point plugins. You should also auto-register in your package's `__init__.py` or provide an alternative entry point.

    ### Alternative Entry Point (Recommended)

    Create `manylatents/yourext/main.py`:

    ```python
    # Register SearchPathPlugin BEFORE importing manylatents.main
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin
    from manylatents.yourext_plugin import YourExtSearchPathPlugin

    plugins = Plugins.instance()
    existing = list(plugins.discover(SearchPathPlugin))
    if YourExtSearchPathPlugin not in existing:
        plugins.register(YourExtSearchPathPlugin)

    from manylatents.main import main

    if __name__ == "__main__":
        main()
    ```

    ### pyproject.toml

    ```toml
    [project]
    name = "manylatents-yourext"
    version = "0.1.0"
    requires-python = ">=3.10, <3.13"

    dependencies = [
        "manylatents",
    ]

    [project.entry-points."hydra.searchpath"]
    manylatents-yourext = "manylatents.yourext_plugin:YourExtSearchPathPlugin"

    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"

    [tool.hatch.build.targets.wheel]
    packages = ["manylatents"]  # CRITICAL: Package the manylatents/ directory

    [tool.uv]
    managed = true

    [tool.uv.sources]
    manylatents = { git = "https://github.com/latent-reasoning-works/manylatents.git" }
    ```

    ## Component Types

    ### Custom LatentModule (Algorithm)

    ```python
    from torch import Tensor
    from manylatents.algorithms.latent.latent_module_base import LatentModule


    class YourAlgorithm(LatentModule):
        def __init__(self, n_components=2, your_param=1.0, **kwargs):
            super().__init__(n_components=n_components, **kwargs)
            self.your_param = your_param

        def fit(self, x: Tensor) -> None:
            x_np = x.detach().cpu().numpy()
            # Your fitting logic
            self._is_fitted = True

        def transform(self, x: Tensor) -> Tensor:
            if not self._is_fitted:
                raise RuntimeError("Model not fitted. Call fit() first.")
            x_np = x.detach().cpu().numpy()
            embedding = ...  # Your embedding computation
            return torch.tensor(embedding, device=x.device, dtype=x.dtype)
    ```

    Config: `manylatents/yourext/configs/algorithms/latent/your_algo.yaml`

    ```yaml
    _target_: manylatents.yourext.algorithms.YourAlgorithm
    n_components: 2
    your_param: 1.0
    ```

    ### Custom Dataset

    ```python
    import numpy as np


    class YourDataset:
        def __init__(self, data_path: str, n_samples=None):
            self.data_path = data_path
            self._data = np.load(data_path)
            if n_samples:
                self._data = self._data[:n_samples]

        def get_data(self) -> np.ndarray:
            return self._data

        @property
        def data(self) -> np.ndarray:
            return self._data
    ```

    Config: `manylatents/yourext/configs/data/your_data.yaml`

    ```yaml
    _target_: manylatents.yourext.data.YourDataset
    data_path: ${paths.data_dir}/your_data.npy
    n_samples: null
    ```

    ### Custom Metric

    ```python
    import numpy as np
    from typing import Optional
    from manylatents.algorithms.latent.latent_module_base import LatentModule


    def YourMetric(
        embeddings: np.ndarray,
        dataset: object,
        module: Optional[LatentModule] = None,
        threshold: float = 0.5,
        return_per_sample: bool = False,
    ) -> float:
        scores = ...  # Your metric computation
        if return_per_sample:
            return float(np.mean(scores)), scores
        return float(np.mean(scores))
    ```

    Config: `manylatents/yourext/configs/metrics/dataset/your_metric.yaml`

    ```yaml
    _target_: manylatents.yourext.metrics.YourMetric
    _partial_: true  # CRITICAL: deferred parameter binding
    threshold: 0.5
    return_per_sample: false
    ```

    ### Experiment Config

    `manylatents/yourext/configs/experiment/your_experiment.yaml`:

    ```yaml
    # @package _global_
    name: your_experiment
    project: your_project

    defaults:
      - override /algorithms/latent: your_algo
      - override /data: your_data
      - override /callbacks/embedding: default
      - override /metrics: default

    seed: 42
    ```

    ## CI Requirements

    ### Import Tests

    ```python
    def test_namespace_package():
        import manylatents.yourext
        from manylatents.yourext.algorithms import YourAlgorithm
        from manylatents.data import SwissRoll  # Core still works
        assert YourAlgorithm is not None

    def test_algorithm_interface():
        from manylatents.yourext.algorithms import YourAlgorithm
        import torch

        algo = YourAlgorithm(n_components=2)
        X = torch.randn(100, 50)
        embedding = algo.fit_transform(X)
        assert embedding.shape == (100, 2)
    ```

    ### Config Resolution Tests

    ```python
    from omegaconf import OmegaConf
    from pathlib import Path

    CONFIGS_DIR = Path(__file__).parent.parent / "manylatents" / "yourext" / "configs"

    def test_all_targets_importable():
        for config_file in CONFIGS_DIR.rglob("*.yaml"):
            cfg = OmegaConf.load(config_file)
            if hasattr(cfg, "_target_"):
                assert cfg._target_.startswith("manylatents")
    ```

    ### GitHub Actions Workflow

    ```yaml
    name: CI
    on:
      push:
        branches: [main]
      pull_request:
        branches: [main]

    jobs:
      test:
        runs-on: ubuntu-latest
        strategy:
          matrix:
            python-version: ["3.10", "3.11", "3.12"]
        steps:
        - uses: actions/checkout@v4
        - uses: astral-sh/setup-uv@v5
        - run: uv sync
        - run: uv run pytest tests/ -v
        - run: |
            uv run python -c "
            import manylatents
            assert len(manylatents.__path__) >= 2
            print('Namespace package OK')
            "
    ```

    ## Testing Checklist

    ### Namespace Package
    - [ ] `manylatents/__init__.py` has `extend_path` line
    - [ ] `import manylatents.yourext` works
    - [ ] Core manylatents still importable

    ### Hydra Config
    - [ ] SearchPathPlugin registered (entry-point + manual)
    - [ ] All configs have valid `_target_` paths
    - [ ] Metrics use `_partial_: true`
    - [ ] Experiment configs use `# @package _global_`

    ### Interface Compliance
    - [ ] LatentModule subclasses implement `fit()` and `transform()`
    - [ ] Datasets have `get_data()` method
    - [ ] Metrics accept `(embeddings, dataset, module, **kwargs)`

    ### CI
    - [ ] Import tests pass on Python 3.10-3.12
    - [ ] Config resolution tests pass
    - [ ] Core tests still pass with extension installed

    ## Quick Reference

    ```bash
    # Using extension entry point (recommended)
    python -m manylatents.yourext.main experiment=your_experiment

    # Using environment variable (requires shop)
    HYDRA_SEARCH_PACKAGES="manylatents.configs:manylatents.yourext.configs" \
        python -m manylatents.main experiment=your_experiment
    ```

    ### Debugging Config Discovery

    ```python
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    plugins = list(Plugins.instance().discover(SearchPathPlugin))
    print([p.__name__ for p in plugins])
    ```
