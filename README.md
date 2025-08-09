# Many Latents
"One geometry, learned through many latents"

A framework for applying dimensionality reduction techniques (PCA, PHATE, t-SNE, etc.) on diverse datasets, built with **PyTorch Lightning**, **Hydra**, and **uv** for reproducibility and scalability.  
TODO: Fill description with the neural net coupling functionality + other relevant stuff...

---

## 🛠️ Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies. To set up the project locally:

1. **Install dependencies**:

    ```bash
    uv sync  # Creates a virtual environment and installs dependencies
    ```

2. **Activate the virtual environment**:

    ```bash
    source .venv/bin/activate
    ```

3. **(Optional) Run Tests**:

    ```bash
    pytest
    ```

---

## 📁 Project Structure

To be finalized..

## 🚀 Running Experiments

1. **Run a dimensionality reduction experiment** (e.g., PCA on HGDP dataset):

    ```bash
    python -m manylatents.main experiment=hgdp_pca
    ```

2. **Override hyperparameters from the command line**:

    ```bash
    python -m manylatents.main experiment=hgdp_pca algorithm.dimensionality_reduction.n_components=10
    ```

3. **Logs and Outputs** will be saved under:

    ```
    outputs/<YYYY-MM-DD>/<time>/main.log
    ```

---

## 📊 Supported Dimensionality Reduction Techniques

- ✅ **PCA** (Principal Component Analysis)
- ✅ **PHATE** (Potential of Heat-diffusion for Affinity-based Transition Embedding)
- ✅ **t-SNE** (t-distributed Stochastic Neighbor Embedding)

---

## 🟡 **Adding a New Dimensionality Reduction (DR) Module**

Want to add a new DR method (e.g., UMAP, Isomap)? Follow these steps:

### 1️⃣ **Create a New DR Class**

1. In `src/algorithms/`, create a new file (e.g., `yourmodel.py`).

2. Inherit from `DimensionalityReduction` and implement the required methods:

```python
from manylatents.algorithms.dimensionality_reduction import DimensionalityReduction
import torch

class YourModelModule(DimensionalityReduction):
    def __init__(self, n_components=2):
        super().__init__(n_components)
        self.model = None  # Placeholder for your model

    def fit(self, x: torch.Tensor):
        # Implement model fitting logic here
        self._is_fitted = True

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before calling transform()")
        # Transform data here
        return x  # Placeholder
```

### 2️⃣ Add a Hydra Config

Add a new config file under `src/configs/algorithm/dimensionality_reduction/yourmodel.yaml`:

```yaml
_target_: manylatents.algorithms.yourmodel.YourModelModule
n_components: 2
```

## 3️⃣ Write Unit & Compliance Tests

### ✅ Unit Test

Create a unit test in src/algorithms/yourmodel_test.py:

```python
import torch
from manylatents.algorithms.yourmodel import YourModelModule

def test_yourmodel_instantiation():
    model = YourModelModule(n_components=3)
    assert model.n_components == 3

def test_yourmodel_fit_transform():
    model = YourModelModule(n_components=2)
    data = torch.randn(100, 10)
    model.fit(data)
    transformed = model.transform(data)
    assert transformed.shape == (100, 2)
```

### ✅ Compliance Test (WIP)
Ensure it passes the compliance test (dr_compliance_test.py):

```bash
pytest src/tests/algorithms/dr_compliance_test.py
```

## 4️⃣ Run the Experiment
Run the experiment using your new model:

```bash
python -m manylatents.main experiment=data_yourmodel
```

## ✅ Contributing
1. Fork the repo.

2. Raise an issue with your proposed functionality to avoid redundancy.

3. Create your feature branch:

```bash
git checkout -b feature/new-dr-method
```
4. Write code and tests.

5. Run tests

6. Submit a pull request 🚀.

## 💡 Tips & Best Practices

📁 Organize configs under src/configs for Hydra CLI compatibility.

🔍 Run tests frequently using:

``` bash
pytest
```
🧹 Use pre-commit hooks to ensure code formatting and linting:

```bash
pre-commit run --all-files
```

## 📚 Resources
- ⚡ [PyTorch Lightning Docs](https://lightning.ai/docs/)
- 💧 [Hydra Docs](https://hydra.cc/docs/)
- 📦 [uv Dependency Manager](https://docs.astral.sh/uv/)



# 🚀 Happy Experimenting!
