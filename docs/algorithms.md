# Algorithms

manyLatents provides two algorithm base classes. The decision rule is binary: if the algorithm trains with backprop, use LightningModule. If not, use LatentModule.

=== "LatentModule"

    ## fit/transform Algorithms

    `LatentModule` is the base class for non-neural algorithms. Subclass it, implement `fit()` and `transform()`, and you're done.

    ```python
    from manylatents.algorithms.latent.latent_module_base import LatentModule

    class MyAlgorithm(LatentModule):
        def __init__(self, n_components=2, my_param=1.0, **kwargs):
            super().__init__(n_components=n_components, **kwargs)
            self.my_param = my_param

        def fit(self, x, y=None):
            self._is_fitted = True

        def transform(self, x):
            return x[:, :self.n_components]
    ```

    ### Available Algorithms

    {{ algorithm_table("latent") }}

    ### FoundationEncoder Pattern

    Frozen pretrained models also use LatentModule — `fit()` is a no-op and `transform()` wraps the model's forward pass. This is a usage convention, not a separate class. Implementations live in `manylatents-omics/manylatents/dogma/encoders/` (Evo2, ESM3, Orthrus, AlphaGenome).

    ### Optional Methods

    If your algorithm uses a kernel-based approach, implement these to enable module-level metrics:

    ```python
    def kernel_matrix(self, ignore_diagonal=False) -> np.ndarray:
        """Raw similarity matrix (N x N)."""
        ...

    def affinity_matrix(self, ignore_diagonal=False, use_symmetric=False) -> np.ndarray:
        """Normalized transition matrix."""
        ...
    ```

    This enables metrics like `KernelMatrixSparsity`, `AffinitySpectrum`, and `ConnectedComponents`.

    ### Running

    ```bash
    python -m manylatents.main algorithms/latent=pca data=swissroll
    ```

    ### Adding a New LatentModule

    1. Create `manylatents/algorithms/latent/your_algo.py` inheriting from `LatentModule`
    2. Implement `fit(x, y=None)` and `transform(x)`
    3. Create `manylatents/configs/algorithms/latent/your_algo.yaml` with `_target_`
    4. Import in `manylatents/algorithms/latent/__init__.py`

=== "LightningModule"

    ## Trainable Algorithms

    Neural network algorithms use PyTorch Lightning's `LightningModule` with training loops. Implement `setup()`, `training_step()`, `encode()`, and `configure_optimizers()`.

    ### Available Algorithms

    {{ algorithm_table("lightning") }}

    ### Pattern

    All LightningModule algorithms follow the same pattern:

    ```python
    class MyTrainableAlgorithm(LightningModule):
        def __init__(self, network, optimizer, loss, datamodule=None, init_seed=42):
            super().__init__()
            self.save_hyperparameters(ignore=["datamodule", "network", "loss"])
            self.network_config = network
            self.optimizer_config = optimizer
            self.loss_fn = loss

        def setup(self, stage=None):
            # Deferred network construction — input_dim from datamodule
            input_dim = self.trainer.datamodule.data_dim
            self.network = hydra.utils.instantiate(self.network_config, input_dim=input_dim)

        def training_step(self, batch, batch_idx):
            outputs = self.network(batch)
            return self.loss_fn(outputs, batch)

        def encode(self, x):
            return self.network.encode(x)

        def configure_optimizers(self):
            return hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
    ```

    Key conventions:

    - `save_hyperparameters(ignore=["datamodule", "network", "loss"])` — Lightning can't serialize nn.Modules
    - `setup()` defers network construction until `input_dim` is known from the datamodule
    - `encode()` extracts embeddings for evaluation after training
    - Use the project's `MSELoss` from `manylatents.algorithms.lightning.losses.mse`, not `torch.nn.MSELoss`

    ### Latent ODE

    The `LatentODE` algorithm integrates neural ODEs for learning continuous-time dynamics in latent space:

    ```bash
    python -m manylatents.main \
      algorithms/lightning=latent_ode \
      data=swissroll \
      trainer.max_epochs=10
    ```

    Configuration supports custom integration times and ODE solver options via `torchdiffeq`.

    ### Running

    ```bash
    python -m manylatents.main \
      algorithms/lightning=ae_reconstruction \
      data=swissroll \
      trainer.max_epochs=10

    # Fast dev run for testing
    python -m manylatents.main \
      algorithms/lightning=ae_reconstruction \
      data=swissroll \
      trainer.fast_dev_run=true
    ```

    ### Adding a New LightningModule

    1. Create `manylatents/algorithms/lightning/your_algo.py` inheriting from `LightningModule`
    2. Implement `setup()`, `training_step()`, `encode()`, `configure_optimizers()`
    3. Use `self.save_hyperparameters(ignore=["datamodule", "network", "loss"])`
    4. Create config in `manylatents/configs/algorithms/lightning/your_algo.yaml`
    5. Test with `trainer.fast_dev_run=true`

=== "Networks & Losses"

    ## Network Architectures

    Networks are `nn.Module` classes used by LightningModule algorithms. They define the architecture; the LightningModule wraps the training logic.

    ### Available Networks

    | Network | Class | Config | Description |
    |---------|-------|--------|-------------|
    | Autoencoder | `Autoencoder` | `algorithms/lightning/network=autoencoder` | Symmetric encoder-decoder with configurable layers |
    | AANet | `AANet` | `algorithms/lightning/network=aanet` | Archetypal analysis network |
    | LatentODENetwork | `LatentODENetwork` | (configured via `latent_ode.yaml`) | ODE function for continuous dynamics |

    ### Autoencoder Config

    ```yaml
    # configs/algorithms/lightning/network/autoencoder.yaml
    _target_: manylatents.algorithms.lightning.networks.autoencoder.Autoencoder
    input_dim: ???  # Set by setup() from datamodule
    hidden_dims: [128, 64]
    latent_dim: 2
    activation: relu
    ```

    ## Loss Functions

    Use the project's loss functions, not PyTorch's directly.

    | Loss | Class | Config | Description |
    |------|-------|--------|-------------|
    | MSELoss | `MSELoss` | `algorithms/lightning/loss=default` | Reconstruction loss |
    | GeometricLoss | `GeometricLoss` | `algorithms/lightning/loss=ae_dim` | Dimensionality-aware loss |
    | GeometricLoss | `GeometricLoss` | `algorithms/lightning/loss=ae_neighbors` | Neighborhood-preserving loss |
    | GeometricLoss | `GeometricLoss` | `algorithms/lightning/loss=ae_shape` | Shape-preserving loss |

    The project's `MSELoss` (from `manylatents.algorithms.lightning.losses.mse`) accepts `(outputs, targets, **kwargs)`, unlike `torch.nn.MSELoss`. Always use the project's version.

    ## Optimizer Config

    ```yaml
    # configs/algorithms/lightning/optimizer/adam.yaml
    _target_: torch.optim.Adam
    _partial_: true
    lr: 0.001
    ```

    The `_partial_: true` flag creates a partial that receives `params=` from `configure_optimizers()`.

    ## Composing a Full Config

    ```yaml
    # configs/algorithms/lightning/ae_reconstruction.yaml
    _target_: manylatents.algorithms.lightning.reconstruction.Reconstruction
    init_seed: 42

    defaults:
      - network: autoencoder
      - optimizer: adam
      - loss: default
    ```
