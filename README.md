<div align="center">

# ManyLatents

**"One geometry, learned through many latents"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Build Status](https://github.com/username/manylatents/workflows/Build%20and%20Integration%20Tests/badge.svg)](https://github.com/username/manylatents/actions)
[![Hydra](https://img.shields.io/badge/Config-Hydra-blue)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/Framework-PyTorch%20Lightning-792ee5)](https://lightning.ai/)
[![Docs](https://img.shields.io/badge/docs-MkDocs-blue)](https://squidfunk.github.io/mkdocs-material/)

*A unified framework for dimensionality reduction and neural network analysis on diverse datasets*

</div>

---

## 🌟 Overview

**ManyLatents** is a comprehensive framework that bridges traditional dimensionality reduction techniques with modern neural networks. Built on **PyTorch Lightning** and **Hydra**, it provides a unified interface for:

- **Traditional DR methods**: PCA, t-SNE, PHATE, UMAP
- **Neural architectures**: Autoencoders, VAEs, and custom networks  
- **Sequential workflows**: Chain multiple algorithms (e.g., PCA → neural network → final embedding)
- **Diverse datasets**: Genomic data (HGDP, UKBB), single-cell data, synthetic manifolds

### ✨ Key Features

- 🔧 **Modular Architecture**: Unified `LatentModule` interface for all algorithms
- ⚡ **Lightning Integration**: Seamless neural network training with PyTorch Lightning
- 🎛️ **Hydra Configuration**: Flexible, composable experiment configurations
- 🔄 **Sequential Processing**: Chain algorithms for complex pipelines
- 📊 **Rich Evaluation**: Comprehensive metrics for embedding quality
- 🧪 **Extensive Testing**: Matrix testing across algorithm-dataset combinations

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/manylatents.git
cd manylatents

# Install with uv (recommended)
uv sync
source .venv/bin/activate

# Or with pip
pip install -e .
```

### Basic Usage

```bash
# Run PCA on synthetic data
python -m manylatents.main experiment=hgdp_pca

# Override hyperparameters
python -m manylatents.main \
  algorithm=latent/pca \
  data=swissroll \
  algorithm.n_components=3

# Train an autoencoder
python -m manylatents.main \
  algorithm=lightning/ae_reconstruction \
  data=swissroll \
  trainer.max_epochs=50
```

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
- **Sequential workflows**: Chain multiple algorithms automatically

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
- 🧬 **Genomic**: HGDP, All of Us (AOU), UK Biobank (UKBB)
- 🔬 **Single-cell**: Embryoid body, custom scRNA-seq data
- 📐 **Synthetic**: Swiss roll, saddle surface, custom manifolds
- 🧪 **Test data**: Built-in synthetic datasets for validation

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

# Quick smoke test  
python -m manylatents.main \
  algorithm=latent/noop \
  data=test_data \
  debug=true

# Test specific combination
python -m manylatents.main \
  algorithm=latent/pca \
  data=swissroll \
  trainer.max_epochs=1
```

### GitHub Actions Matrix
Our CI runs comprehensive testing across algorithm-dataset combinations:
- **Smoke tests**: Basic functionality validation
- **Traditional DR**: PCA, UMAP on synthetic data
- **Neural networks**: Autoencoder training validation
- **Integration tests**: Full pipeline testing

See [Testing Documentation](docs/testing.md) for detailed information.

---

## 📊 Evaluation & Metrics

### Embedding Quality Metrics
- **Trustworthiness**: Preservation of local neighborhoods
- **Continuity**: Smooth embedding properties  
- **Participation Ratio**: Effective dimensionality
- **Fractal Dimension**: Intrinsic dimensionality estimation

### Dataset-Specific Metrics
- **Genomic data**: Geographic preservation, admixture analysis
- **Single-cell**: Cell type separation, trajectory preservation
- **Synthetic**: Ground-truth manifold recovery

### Usage
```bash
python -m manylatents.main \
  experiment=hgdp_pca \
  metrics=genomic_metrics
```

---

## 🛠️ Development

### Code Quality
```bash
# Linting and formatting
pre-commit run --all-files

# Type checking (if available)
mypy manylatents/

# Coverage report
pytest --cov=manylatents --cov-report=html
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
git clone https://github.com/your-username/manylatents.git
cd manylatents
uv sync
source .venv/bin/activate
pre-commit install
```

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

[📖 Documentation](docs/) • [🐛 Issues](https://github.com/username/manylatents/issues) • [💬 Discussions](https://github.com/username/manylatents/discussions)

</div>
