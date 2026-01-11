<div align="center">

# ManyLatents

**"One geometry, learned through many latents"**

**Part of the [Latent Reasoning Works](https://github.com/latent-reasoning-works) ecosystem**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-blue)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/Framework-PyTorch%20Lightning-792ee5)](https://lightning.ai/)
[![uv](https://img.shields.io/badge/Package-uv-DE5FE9)](https://docs.astral.sh/uv/)

*A unified framework for dimensionality reduction and neural network analysis on diverse datasets*

</div>

---

## 🌟 Overview

**ManyLatents** is a comprehensive framework that bridges traditional dimensionality reduction techniques with modern neural networks. Built on **PyTorch Lightning** and **Hydra**, it provides a unified interface for:

- **Traditional DR methods**: PCA, t-SNE, PHATE, UMAP
- **Neural architectures**: Autoencoders, VAEs, and custom networks
- **Diverse datasets**: Single-cell data, synthetic manifolds, genetics data (with extensions)
- **🧬 Extensions**: Domain-specific packages like [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) for genomics

### ✨ Key Features

- 🔧 **Modular Architecture**: Unified `LatentModule` interface for all algorithms
- ⚡ **Lightning Integration**: Seamless neural network training with PyTorch Lightning
- 🎛️ **Hydra Configuration**: Flexible, composable experiment configurations
- 📊 **Rich Evaluation**: 23+ metrics for embedding quality assessment
- 🔌 **Python API**: Programmatic access for orchestration and workflow integration
- 🖥️ **SLURM Support**: Multi-cluster job submission via [shop](https://github.com/latent-reasoning-works/shop)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/latent-reasoning-works/manylatents.git
cd manylatents

# Install with uv
uv sync
source .venv/bin/activate

# Optional: Install extensions for domain-specific functionality
uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git  # Genomics support
```

### SLURM Cluster Support (Optional)

For multi-cluster job submission via [shop](https://github.com/latent-reasoning-works/shop):

```bash
# Install with SLURM support
uv sync --extra slurm

# Submit to cluster
python -m manylatents.main experiment=single_algorithm hydra/launcher=slurm_cluster
```

### Extensions

manylatents supports domain-specific extensions that add specialized data loaders, metrics, and algorithms:

- **🧬 [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics)**: Genetics and population genetics support
  ```bash
  uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
  ```

See [docs/extensions.md](docs/extensions.md) for full documentation on installing and using extensions.

### Single Algorithm Usage

```bash
# Run pre-configured experiment (PCA on Swiss roll)
python -m manylatents.main experiment=single_algorithm

# Run custom algorithm combinations
python -m manylatents.main \
  algorithms/latent=pca \
  data=swissroll \
  algorithms.latent.n_components=3

# Train an autoencoder
python -m manylatents.main \
  algorithms/lightning=ae_reconstruction \
  data=swissroll \
  trainer.max_epochs=50
```

### Python API Usage

For programmatic access and workflow orchestration:

```python
from manylatents.api import run

# Run with experiment config
result = run(experiment="single_algorithm", algorithms={"latent": {"n_components": 10}})

# Run with direct config
result = run(
    algorithms={"latent": {"_target_": "manylatents.algorithms.latent.pca.PCAModule", "n_components": 2}},
    data={"_target_": "manylatents.data.swissroll.SwissRollDataModule"},
    metrics="test_metric"
)

# Access results
embeddings = result["embeddings"]  # numpy array
scores = result["scores"]          # dict of metrics
metadata = result["metadata"]      # dict of metadata

# Chain algorithms by passing embeddings
result2 = run(input_data=result["embeddings"], algorithms={"latent": "phate"})
```

**Key Features**:
- 🔗 **In-memory data passing**: Pass numpy arrays between calls
- 🚀 **No subprocess overhead**: Direct Python function calls
- 📊 **Flexible metrics**: Returns float, tuple, or dict metric values

See [API Documentation](docs/api_usage.md) for complete reference.

---

## 🏗️ Architecture

### Core Components

```
manylatents/
├── algorithms/           # Algorithm implementations
│   ├── latent_module_base.py    # Base LatentModule class
│   ├── pca.py, umap.py, ...     # Traditional DR methods
│   └── networks/                # Neural network modules
├── configs/              # Hydra configurations
│   ├── experiment/              # Pre-defined experiments
│   ├── algorithm/               # Algorithm configs
│   └── data/                   # Dataset configurations
├── data/                # Dataset loaders and processors
├── metrics/             # Evaluation metrics
└── main.py             # Unified experiment pipeline
```

### Algorithm Types

- **`LatentModule`**: Traditional DR with `fit()`/`transform()` interface
- **`LightningModule`**: Neural networks with full training loops

---

## 🧪 Supported Algorithms & Datasets

### Dimensionality Reduction
- ✅ **PCA** - Principal Component Analysis
- ✅ **t-SNE** - t-distributed Stochastic Neighbor Embedding  
- ✅ **PHATE** - Potential of Heat-diffusion Affinity-based Transition Embedding
- ✅ **UMAP** - Uniform Manifold Approximation and Projection

### Neural Networks
- ✅ **Autoencoder** - Reconstruction-based dimensionality reduction
- ✅ **VAE** - Variational Autoencoder (coming soon)
- ✅ **Custom architectures** - Extensible neural network support

### Datasets
- 🔬 **Single-cell**: Anndata, scRNA-seq data in h5ad format
- 📐 **Synthetic**: Swiss roll, saddle surface, custom manifolds
- 🧪 **Test data**: Built-in synthetic datasets for validation
- 🧬 **Genomics**: Available via [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) extension (HGDP, AOU, UKBB)

---

## 🔧 Adding New Components

### New Algorithm (Traditional DR)

1. **Create algorithm class**:
```python
# manylatents/algorithms/myalgo.py
from manylatents.algorithms.latent_module_base import LatentModule

class MyAlgorithm(LatentModule):
    def __init__(self, n_components=2):
        super().__init__()
        self.n_components = n_components
    
    def fit(self, x):
        # Implementation here
        return self
    
    def transform(self, x):
        # Transform implementation
        return transformed_x
```

2. **Add Hydra config**:
```yaml
# manylatents/configs/algorithm/latent/myalgo.yaml
_target_: manylatents.algorithms.myalgo.MyAlgorithm
n_components: 2
```

3. **Run compliance test**:
```bash
pytest manylatents/tests/algorithms/dr_compliance_test.py
```

### New Neural Network

1. **Create Lightning module**:
```python
# manylatents/algorithms/networks/mynet.py  
from lightning import LightningModule

class MyNetwork(LightningModule):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Network definition
    
    def training_step(self, batch, batch_idx):
        # Training logic
        return loss
```

2. **Add to CI matrix** (`.github/workflows/build.yml`):
```yaml
- name: "mynet-test"
  algorithm: "lightning/mynet"
  data: "swissroll"
  metrics: "synthetic_data_metrics"
  timeout: 10
```

---

## 🧪 Testing & CI

### Local Testing
```bash
# Full test suite
pytest

# Test single algorithm
python -m manylatents.main experiment=single_algorithm

# Quick smoke test with minimal data
python -m manylatents.main \
  experiment=single_algorithm \
  metrics=test_metric \
  callbacks/embedding=minimal

# Test specific combination
python -m manylatents.main \
  algorithms/latent=pca \
  data=swissroll \
  trainer.max_epochs=1
```

### GitHub Actions Matrix
Our CI runs comprehensive testing across algorithm-dataset combinations:
- **Smoke tests**: Basic functionality validation
- **Traditional DR**: PCA, UMAP on synthetic data
- **Neural networks**: Autoencoder training validation
- **Integration tests**: End-to-end testing

See [Testing Documentation](docs/testing.md) for detailed information.

---

## 📊 Evaluation & Metrics

### Embedding Quality Metrics (23+ available)

**Neighborhood Preservation:**
- **Trustworthiness**: Local neighborhood preservation in embedding
- **Continuity**: Reverse trustworthiness (embedding → original space)
- **k-NN Preservation**: k-nearest neighbor graph preservation

**Dimensionality Analysis:**
- **Local Intrinsic Dimensionality (LID)**: Per-point dimensionality estimation
- **Participation Ratio**: Effective dimensionality via eigenvalue analysis
- **Fractal Dimension**: Box-counting dimension estimation
- **Tangent Space Approximation**: Local manifold dimension

**Topological & Geometric:**
- **Persistent Homology**: Topological feature counting via ripser
- **Diffusion Spectral Entropy**: Spectral complexity measure
- **Diffusion Curvature**: Local curvature estimation
- **Reeb Graph**: Topological skeleton analysis
- **Anisotropy**: Embedding uniformity measure

### Sampling Strategies

For large datasets, pluggable sampling strategies reduce computation:

```python
# In config
metrics:
  sampling:
    _target_: manylatents.utils.sampling.RandomSampling
    fraction: 0.1
    seed: 42
```

Available strategies: `RandomSampling`, `StratifiedSampling`, `FarthestPointSampling`, `FixedIndexSampling`

### Dataset-Specific Metrics
- **Single-cell**: Cell type separation, trajectory preservation
- **Synthetic**: Ground-truth manifold recovery
- **Genomic data**: Available via [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) (geographic preservation, admixture analysis)

### Usage
```bash
python -m manylatents.main \
  experiment=single_algorithm \
  metrics=synthetic_data_metrics
```

---

## 🛠️ Development

### Code Quality
```bash
# Run test suite
pytest

# Linting and formatting (if pre-commit is configured)
pre-commit run --all-files
```

### Project Structure
```bash
# View project structure
tree -I '__pycache__|*.pyc|.git|outputs|.venv'
```

---

## 📚 Documentation

- 📖 **[Full Documentation](docs/)**: Comprehensive guides and API reference
- 🧪 **[Testing Strategy](docs/testing.md)**: CI/CD and local testing practices  
- 🔧 **[Configuration Guide](manylatents/configs/)**: Hydra configuration examples
- 🎯 **[Examples](experiments/)**: Pre-configured experiment templates

---

## 🤝 Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write** tests for your changes
4. **Ensure** CI passes (`pytest` and GitHub Actions)
5. **Submit** a pull request

### Development Setup
```bash
git clone https://github.com/latent-reasoning-works/manylatents.git
cd manylatents
uv sync
source .venv/bin/activate
pre-commit install
```

---

## 🌐 LRW Ecosystem

ManyLatents is part of the **Latent Reasoning Works** suite of tools:

| Project | Description |
|---------|-------------|
| [manylatents](https://github.com/latent-reasoning-works/manylatents) | Core DR framework (this repo) |
| [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) | Genomics & population genetics extensions |
| [shop](https://github.com/latent-reasoning-works/shop) | Multi-cluster SLURM job management |

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Authors**: César Miguel Valdez Córdova, Shuang Ni, Matthew Scicluna
- **Affiliation**: Mila - Quebec AI Institute
- **Built with**: PyTorch Lightning, Hydra, uv dependency manager

---

<div align="center">

**🚀 Ready to explore the latent space? Get started with ManyLatents!**

[📖 Documentation](docs/) • [🐛 Issues](https://github.com/latent-reasoning-works/manylatents/issues) • [💬 Discussions](https://github.com/latent-reasoning-works/manylatents/discussions)

</div>
