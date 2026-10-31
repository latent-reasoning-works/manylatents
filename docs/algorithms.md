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

    ### Reeb graph: cell cap

    `ReebGraphModule` builds an (N, N) distance matrix and a Vietoris-Rips complex, so its
    memory grows as N². `max_N` (default `5000`; `None` disables it) caps the number of rows
    the graph is built on. When N > `max_N`, `fit` draws `max_N` rows with
    `random.Random(random_state).sample(range(N), k=max_N)`. This is the procedure that
    `PersistentHomology` uses, and the global `random` state is not changed. The seed is the
    existing `random_state`, which the Hydra configs set to `${seed}`.

    The lens, the distance matrix, the Reeb graph, `node_coordinates` and
    `structural_summary` come from the sampled rows only. The membership output still has
    one row per input row: each row that was not sampled copies the membership row of its
    nearest sampled row (Euclidean distance in input space), and each sampled row keeps its
    own. `extra_outputs()["subsample_indices"]` holds the sampled row indices (1-D `int64`,
    in draw order; `arange(N)` when no sampling happened). For N ≤ `max_N` the result is the
    same as without a cap.

    The generic `graph` output turns the Reeb `adjacency()` and `node_coordinates` into
    `edges`, `edge_weights`, `node_ids` and `node_coordinates`. `SaveOutputs` with
    `save_format: csv` writes it as nodes and edges CSV files (see Callbacks).

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
    uv run python -m manylatents.main algorithms/latent=pca data=swissroll
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
    uv run python -m manylatents.main \
      algorithms/lightning=latent_ode \
      data=swissroll \
      trainer.max_epochs=10
    ```

    Configuration supports custom integration times and ODE solver options via `torchdiffeq`.

    ### Running

    ```bash
    uv run python -m manylatents.main \
      algorithms/lightning=ae_reconstruction \
      data=swissroll \
      trainer.max_epochs=10

    # Fast dev run for testing
    uv run python -m manylatents.main \
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
    _recursive_: false
    datamodule: ${data}
    network:
      _target_: manylatents.algorithms.lightning.networks.autoencoder.Autoencoder
      input_dim: null  # inferred from data in setup()
      hidden_dims: [512, 256, 128]
      latent_dim: 50
      activation: relu
      batchnorm: true
      dropout: 0.1
    optimizer:
      _target_: torch.optim.Adam
      _partial_: true
      lr: 0.001
    loss:
      _target_: manylatents.algorithms.lightning.losses.mse.MSELoss
    ```

    `_recursive_: false` prevents Hydra from eagerly instantiating nested configs — the `Reconstruction` module handles deferred instantiation in `setup()` once `input_dim` is known from the datamodule.


## Parameter declarations

A parameter declaration says which values each parameter of a method accepts, in plain data
that any program can read. A caller that offers a method's parameters reads the declaration
instead of copying each library's checks. The same format describes a tool from another
producer: a declaration is JSON or YAML, and no Python type is needed to read it.

The definition of the format is this section, the function
`manylatents.params.check_declaration` and the fixtures in
`tests/fixtures/param_declarations/`. There is no schema file.

### The declaration

A declaration, version 1, is a mapping with exactly these keys:

| Key | Value |
|---|---|
| `schema_version` | `1` |
| `method` | the method name, for example `phate` or `cflows` |
| `group` | `latent`, `lightning` or `tool` |
| `origin` | the producer; `manylatents` for the declarations of this package |
| `params` | a mapping of parameter name to parameter entry |

A parameter entry has exactly these keys:

| Key | Value |
|---|---|
| `legal` | `{"type", "min", "min_open", "max", "max_open", "choices", "also"}`. `type` is `int`, `float`, `bool`, `choice` or `str`. Bounds are numbers or `null`; `min_open: true` excludes the bound. `choices` is the list for `choice` (and for a `str` with named options), else `[]`. `also` lists other legal values, such as `["auto"]` or `[null]`. This is what is checked. |
| `default` | the constructor default, the config value, or the manifest value |
| `sensible` | `null`, or `{"lo", "hi", "scale", "source"}`: the range where the output changes, on `scale` `linear` or `log`, with `source` `{"kind": "docs" or "paper" or "measured", "ref": <text>}`. It is `null` when no source says where the output changes. |
| `ladder` | `null`, or a list of preset values, for example `[5, 15, 30]` |
| `limits` | a list of `{"op": "le" or "lt", "fact": "n_samples" or "n_features", "offset": <int>}`. Each means `value <op> fact + offset`. |
| `role` | `knob`, a value a caller may set; `fixed`, a value that is part of the result and is recorded, such as a seed; or `infra`, a value that changes how a fit runs but not what it computes, such as `n_jobs`, `verbose`, `device` or `backend`. Only a `knob` is offered. |
| `precomputed` | `true` when the value was recorded from a precomputation; then `role` is `fixed` |
| `cost` | `null`, or `{"time", "memory", "note"}`: how a larger value moves compute. `time` and `memory` are each `up`, `down` or `none`; `note` is text or `null`. |
| `meaning` | what the parameter controls, 1 to 120 characters |
| `more`, `less` | what a larger and a smaller value does, 1 to 120 characters, or `null` |

`check_declaration(doc)` returns one message per problem, and `[]` for a valid declaration.
Each message starts with the dotted path of its field, for example
`params.knn.default: 0 is refused by legal (below min 1)`. It refuses an unknown or a missing
key, an unknown `legal.type`, a `default`, `ladder` value or `sensible` bound that `legal`
refuses, a `sensible` range with `lo > hi`, an unknown `scale`, a `log` scale with `lo <= 0`,
an unknown `source.kind`, an unknown `limits` fact or op, an unknown `role`, `precomputed: true`
with a role other than `fixed`, a `cost` with other keys or other moves, and a `meaning`,
`more` or `less` of the wrong length.

`ladder` and `sensible` describe; they never restrict. Any legal value may be submitted.

### Where the declarations of manylatents come from

`legal.type` and `default` come from the constructor signature of a latent module, or from the
composed config of a Lightning config. They are never written twice. The other keys come from a
source:

- A latent module has a class attribute `PARAMS`, with one entry per constructor parameter
  except `self` and `**kwargs`. A comment beside an entry names the library check that its
  bounds mirror. Shared fragments are in `manylatents.params.sources`.
- A Lightning config `configs/algorithms/lightning/<name>.yaml` has the source file
  `configs/algorithms/lightning/domains/<name>.yaml`. Its keys are dotted override paths: under
  `algorithms.lightning` (for example `network.hidden_dim`, `optimizer.lr`, `loss.which`), and
  `trainer.*` for the trainer group `trainer=default`. Every scalar of the composed config is
  declared, except `_target_`-style keys, `datamodule`, `input_dim` (set from the data), lists,
  interpolations and the trainer's `logger` and `callbacks`.

A source entry holds only `type` (when the signature or the value does not give it), the
bounds, `choices`, `also`, `sensible`, `ladder`, `limits`, `role`, `precomputed`, `cost` and the
texts. It never holds `default`. A value that is not simple, such as a dict, a list of channels
or another module, is declared `type: str` with `role: infra`.

### Reading and checking

```python
from manylatents import params

params.names("latent")                     # every latent method with a declaration
params.domain("phate")["params"]["t"]      # {"legal": {"type": "int", "min": 1, "also": ["auto"], ...}, ...}
params.domain("cflows", group="lightning")
params.check(params.domain("phate"), {"knn": 100}, n_samples=50)
# ['knn=100: must be <= n_samples - 2 (n_samples=50)']
```

| Function | Returns |
|---|---|
| `domain(name, group="latent")` | the declaration of one method. A latent name resolves through the registry, then through a config `algorithms/latent/<name>.yaml` on the Hydra search path, extensions included. `KeyError` for an unknown name; `ValueError` for a class without `PARAMS`. |
| `names(group="latent")` | every method name of the group that has a declaration, sorted |
| `check_declaration(doc)` | the problems of a declaration from any producer |
| `check(doc, values, *, n_samples=None, n_features=None)` | one message per value that the declaration refuses: an unknown parameter, a role other than `knob` (the message names the role), a value that `legal` refuses, or a value that a limit refuses when its fact is given. It does not read `sensible` or `ladder`. |

Reading a Lightning declaration composes the config with Hydra and imports no Lightning module.
`check` is for callers; each library keeps its own checks.

### Command line

| Command | Output | Exit |
|---|---|---|
| `python -m manylatents.params show NAME [--group G]` | the declaration as JSON | 0; 2 for an unknown name |
| `python -m manylatents.params check FILE` | `{"ok": bool, "problems": [{"path", "message"}]}` for a JSON or YAML declaration from any producer | 0 valid; 1 problems; 2 unreadable file |
| `python -m manylatents.params dump DIR` | one `<group>.<name>.json` per method | 0; 2 when `DIR` cannot be written |

A new latent module must declare `PARAMS`, and a new Lightning config must have a domain file.
`tests/test_param_declarations.py` fails otherwise.


## Licensing of Yale-derived modules

manylatents is Apache-2.0, but nine of its files are Yale non-commercial. They are ported
from KrishnaswamyLab repositories (MIOFlow, cflows and AAnet), whose code Yale University
publishes under a non-commercial license. The Apache-2.0 license does not apply to them.
Each file starts with `# SPDX-License-Identifier: LicenseRef-Yale-Non-Commercial`.

| Yale-derived file | Upstream |
|---|---|
| `manylatents/algorithms/lightning/mioflow.py` | KrishnaswamyLab/MIOFlow |
| `manylatents/algorithms/lightning/networks/mioflow_net.py` | KrishnaswamyLab/MIOFlow |
| `manylatents/algorithms/lightning/networks/gaga_net.py` | KrishnaswamyLab/MIOFlow |
| `manylatents/configs/algorithms/lightning/mioflow.yaml` | KrishnaswamyLab/MIOFlow |
| `manylatents/configs/algorithms/lightning/mioflow_gaga.yaml` | KrishnaswamyLab/MIOFlow |
| `manylatents/algorithms/cflows_granger.py` | KrishnaswamyLab/cflows |
| `manylatents/algorithms/lightning/cflows.py` | KrishnaswamyLab/cflows |
| `manylatents/algorithms/lightning/losses/cflows.py` | KrishnaswamyLab/cflows |
| `manylatents/algorithms/lightning/networks/aanet.py` | KrishnaswamyLab/AAnet |

The core imports (`import manylatents`, `manylatents.algorithms.latent`, `manylatents.api`)
do not load any of these modules. The upstream license texts are in
[THIRD_PARTY_NOTICES.md](https://github.com/latent-reasoning-works/manylatents/blob/main/THIRD_PARTY_NOTICES.md).

## MIOFlow encoders and checkpoints

MIOFlow accepts an optional `encoder` Hydra config, like Reconstruction's
`network` config. `configure_model()` instantiates the encoder, preprocessing
buffers, and flow once. Repeated calls retain trained weights. The encoder must
be an `nn.Module` satisfying `HasEncode` (`networks/network.py`): `encode(x)` maps
`(batch, ambient_dim)` to `(batch, latent_dim)`, and `latent_dim` is a positive
integer. Configs use `input_dim` for the ambient dimension; it can be inferred
from data, or supplied with `ambient_dim`.

Decoding is optional because the flow integrates fixed latent coordinates under
`no_grad()`. The flow optimizer includes only flow parameters. Encoder dropout
and BatchNorm remain in evaluation mode. `encode()` returns latent endpoints;
generated trajectories are decoded to ambient space if the encoder offers
`decode(z)`, otherwise they remain latent.

The `mioflow_gaga` catalog config selects `GAGANetwork` and
`encoder_pretraining: gaga`. Its PHATE statistics and two-phase fitting run in
`on_fit_start()`, after Lightning restores checkpoint state. The legacy
`use_gaga=True` shortcut remains available. For a fixed plain autoencoder, use:

```yaml
encoder:
  _target_: manylatents.algorithms.lightning.networks.autoencoder.Autoencoder
  input_dim: 15
  hidden_dims: [32, 16]
  latent_dim: 3
encoder_pretraining: none
```

`none` uses the supplied encoder weights with identity preprocessing; it does
not pretrain an autoencoder. An already-trained encoder instance may also be
passed. GAGA's pretraining routine specifically requires `GAGANetwork`; the
minimal encoder contract does not require its reconstruction training structure.

Config-based checkpoints persist resolved architecture configs, ambient/latent
dimensions, normalization statistics, training time bounds, and encoder-stage
completion. `MIOFlow.load_from_checkpoint(path, strict=True)` restores without a
datamodule or fitting. A completed stage is also skipped when resuming with
`Trainer.fit(ckpt_path=...)`; an unfinished stage is fitted when training resumes.
For live network/encoder objects, construction metadata cannot be inferred in
general: supply the corresponding `network=` or `encoder=` object again to
`load_from_checkpoint`. Old PR checkpoints lack the required construction
metadata and are not automatically migrated.

Explicit batch `time` takes precedence over `labels`, then `label`; populations
are ordered by time. `MIOFlowODEFunc` retains positional `init_seed` and makes
`momentum_beta` keyword-only. Momentum starts from zero after reset, so its first
velocity is `(1 - beta) * raw_velocity`, matching the reference.
