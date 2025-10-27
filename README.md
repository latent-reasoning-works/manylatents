<div align="center">

# ManyLatents

**"One geometry, learned through many latents"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
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
- **Diverse datasets**: Single-cell data, synthetic manifolds
- **Extensions**: [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) for genomics datasets

### ✨ Key Features

- 🔧 **Modular Architecture**: Unified `LatentModule` interface for all algorithms
- ⚡ **Lightning Integration**: Seamless neural network training with PyTorch Lightning
- 🎛️ **Hydra Configuration**: Flexible, composable experiment configurations
- 🔄 **Sequential Pipelines**: Chain multiple algorithms (e.g., PCA → Autoencoder → final embedding)
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

### Sequential Pipeline Usage

For multi-step sequential workflows, use pipeline configurations from the `pipeline/` folder:

```bash
# Run a sequential pipeline (PCA → Autoencoder)
python -m manylatents.main experiment=multiple_algorithms

# The pipeline will:
# 1. Run PCA on input data
# 2. Pass PCA output to autoencoder
# 3. Generate final embedding
# 4. Track progress with W&B logging
```

**Note**: Pipeline configurations are located in `manylatents/configs/experiment/pipeline/` for chainable multi-algorithm workflows.

Pipeline configurations support:
- **Sequential processing**: Each algorithm receives output from previous step
- **Mixed algorithms**: Combine traditional DR with neural networks
- **Per-step overrides**: Custom hyperparameters for each algorithm
- **Unified interface**: Same main.py entry point as single algorithms

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
```

**Key Features**:
- 🔗 **In-memory data passing**: Pass numpy arrays between pipeline steps
- 🚀 **No subprocess overhead**: Direct Python function calls
- 🎯 **Orchestration-ready**: Designed for integration with manyAgents
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
- **Sequential workflows**: Chain multiple algorithms automatically

---

## 🔄 Pipeline Workflows

### Sequential Pipeline Architecture

The unified pipeline system enables complex multi-step workflows through the main interface:

```bash
# Basic pipeline syntax
python -m manylatents.main experiment=<pipeline_name>

# Example: PCA followed by autoencoder
python -m manylatents.main experiment=multiple_algorithms
```

### Pipeline Configuration

Define pipelines in YAML with sequential algorithm execution:

```yaml
# manylatents/configs/experiment/pipeline/my_pipeline.yaml
name: my_custom_pipeline

defaults:
  - _self_
  - override /data: swissroll
  - override /callbacks/embedding: default
  - override /trainer: default

pipeline:
  - experiment: pipeline_step_latent    # Step 1: Traditional DR method
    overrides:
      - algorithms/latent=pca
      - algorithms.latent.n_components=50
  - experiment: pipeline_step_lightning # Step 2: Neural network
    overrides:
      - algorithms/lightning=ae_reconstruction
      - algorithms.lightning.network.latent_dim=2
      - algorithms.lightning.network.input_dim=50
```

### Pipeline Features

- **🔗 Automatic chaining**: Each algorithm receives output from previous step
- **📊 W&B integration**: Unified experiment tracking across all steps  
- **🎛️ Per-step configuration**: Custom hyperparameters for each algorithm
- **📁 Output management**: Organized intermediate results tracking
- **🔄 Mixed workflows**: Combine traditional DR methods with neural networks
- **⚡ Unified interface**: Same main.py entry point as single algorithms

### Pipeline vs Single Algorithm

| Feature | Single Algorithm | Pipeline |
|---------|------------------|----------|
| **Entry point** | `python -m manylatents.main` | `python -m manylatents.main` |
| **Configuration** | `algorithms/latent=pca` | `experiment=multiple_algorithms` |
| **Data flow** | Single input/output | Sequential algorithm chaining |
| **Use case** | Single DR method | Multi-step workflows |

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

### Sequential Pipelines
- 🔗 **Algorithm Chaining** - Connect traditional DR methods with neural networks
- ⚙️ **Per-Step Configuration** - Custom hyperparameters for each pipeline stage
- 📊 **Unified Tracking** - Single W&B experiment across all pipeline steps

### Datasets
- 🔬 **Single-cell**: Embryoid body, custom scRNA-seq data
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

# Test pipeline workflow
python -m manylatents.main experiment=multiple_algorithms

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
