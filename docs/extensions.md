# Working with Extensions

`manylatents` is designed with a modular extension system that allows domain-specific functionality to be installed as separate packages. Extensions integrate seamlessly through Python's namespace package system.

## Available Extensions

### manylatents-omics

The genetics and population genetics extension for `manylatents`.

**Repository**: [https://github.com/latent-reasoning-works/manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics)

**Features**:
- Genetics data loaders (PLINK, VCF, etc.)
- Population genetics metrics (Geographic Preservation, FST, etc.)
- Ancestry-specific algorithms
- Integration with genetic databases

## Installing Extensions

### Quick Install (Recommended)

Install directly from GitHub:

```bash
uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
```

### Interactive Setup

Use the provided setup script:

```bash
bash setup_extensions.sh
```

This will:
1. Detect available extensions
2. Prompt you to choose which to install
3. Handle authentication if needed
4. Install the extensions automatically

### Development Workflow

**Working FROM the manylatents-omics repo (Recommended for omics development):**

```bash
cd manylatents-omics
uv sync  # Pulls manylatents from git automatically

# IMPORTANT: Use omics entry point for omics configs
uv run python -m manylatents.omics.main experiment=single_algorithm
uv run python -m manylatents.omics.main experiment=central_dogma_fusion
```

**Working FROM the manylatents repo (core development only):**

```bash
cd manylatents
uv sync
uv run python -m manylatents.main experiment=single_algorithm
```

**Note**: For omics experiments, always work from the omics repo and use `manylatents.omics.main`.
The omics entry point registers configs before Hydra initializes.

### Using Git Submodules

For contributors working on both core and extensions:

```bash
# Add as submodule
git submodule add https://github.com/latent-reasoning-works/manylatents-omics.git extensions/manylatents-omics

# Install from submodule (use git install, not editable)
uv add git+file://extensions/manylatents-omics
```

## Using Extensions

Once installed, extension features are available through the `manylatents` namespace:

```python
# Core imports (always available)
from manylatents.data import SwissRoll
from manylatents.algorithms.latent import PCAModule

# Extension imports (available when manylatents-omics is installed)
from manylatents.popgen.data import HGDPDataset      # Population genetics
from manylatents.popgen.metrics import GeographicPreservation
from manylatents.dogma.encoders import Evo2Encoder   # Foundation models
from manylatents.singlecell.data import AnnDataModule # Single-cell
```

## Using Extensions in Workflows

Extensions integrate directly with Hydra configs:

```yaml
# Using genetics data from manylatents-omics
data:
  _target_: manylatents.omics.data.PlinkDataset
  file_path: /path/to/data.bed
  
algorithms:
  latent:
    _target_: manylatents.algorithms.latent.pca.PCAModule
    n_components: 10

# Using genetics-specific metrics
metrics:
  embedding:
    - trustworthiness
    - continuity
  dataset:
    - geographic_preservation  # From manylatents-omics
```

Run as usual:

```bash
python -m manylatents.main \
  data=hgdp_1kgp \
  algorithms/latent=pca \
  metrics=genetic_metrics \
  logger=wandb
```

## Namespace Package Architecture

Extensions use Python's namespace package system, which means:

1. **Seamless integration**: Extension code lives under `manylatents.omics.*` (or similar)
2. **No conflicts**: Core and extensions can be installed/uninstalled independently
3. **Clean imports**: Use `from manylatents.omics import ...` naturally
4. **Optional dependencies**: Extensions only need to be installed if you use them

### How It Works

```
manylatents (core package)
├── data/
├── algorithms/
├── metrics/
└── ...

manylatents-omics (extension package)
└── manylatents/
    └── omics/         # Extends the manylatents namespace
        ├── data/
        ├── metrics/
        └── algorithms/
```

Both packages declare `manylatents` as a namespace package, so Python automatically merges them:

```python
import manylatents
print(manylatents.__path__)
# Output: ['/path/to/manylatents', '/path/to/manylatents-omics/manylatents']
```

## Checking What's Installed

Verify which extensions are available:

```python
import importlib

# Check if omics extension is installed
try:
    importlib.import_module("manylatents.omics")
    print("✅ manylatents-omics is installed")
except ImportError:
    print("❌ manylatents-omics is not installed")
```

Or programmatically list available features:

```python
import pkgutil
import manylatents

# List all top-level packages under manylatents
for importer, modname, ispkg in pkgutil.iter_modules(manylatents.__path__):
    print(f"- {modname}")
# Output might include: data, algorithms, metrics, omics, ...
```

## Developing Your Own Extension

Want to create a new extension for `manylatents`? Here's how:

### 1. Package Structure

```
manylatents-yourextension/
├── setup.py
└── manylatents/
    ├── __init__.py          # Namespace package declaration
    └── yourextension/
        ├── __init__.py
        ├── data/
        │   └── __init__.py
        ├── metrics/
        │   └── __init__.py
        └── algorithms/
            └── __init__.py
```

### 2. Namespace Package Declaration

In `manylatents/__init__.py`:

```python
# This makes manylatents a namespace package
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

### 3. Setup Configuration

In `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="manylatents-yourextension",
    version="0.1.0",
    packages=find_packages(),
    namespace_packages=["manylatents"],
    install_requires=[
        "manylatents>=0.1.0",
        # Your extension-specific dependencies
    ],
)
```

### 4. Register with Core

Submit a PR to add your extension to:
- `EXTENSIONS.md` - User-facing documentation
- `setup_extensions.sh` - Auto-detection in setup script
- `docs/extensions.md` - This documentation

## Troubleshooting

### Hydra Config Discovery Error

**Problem**: `ConfigAttributeError: Key 'experiment' is not in struct`

**Cause**: Hydra SearchPathPlugin not being discovered - configs not on search path.

**Solutions**:
1. Ensure manylatents-omics is installed (not just cloned):
   ```bash
   uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
   ```
2. If developing both packages, work from the omics repo (not manylatents repo)
3. Or use explicit config path:
   ```bash
   python -m manylatents.main --config-path=path/to/manylatents/configs
   ```

**Verification**: Check that the plugin is discovered:
```python
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
plugins = list(Plugins.instance().discover(SearchPathPlugin))
print([p.__name__ for p in plugins])
# Should show both ManylatentsSearchPathPlugin and OmicsSearchPathPlugin
```

### Extension Not Found

**Problem**: `ModuleNotFoundError: No module named 'manylatents.omics'`

**Solution**: Install the extension:
```bash
uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
```

### Import Conflicts

**Problem**: Namespace package not merging correctly

**Solution**: Ensure both packages have proper namespace declarations:
```python
# In manylatents/__init__.py (both core and extension)
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

### Authentication Issues

**Problem**: Can't clone private extension repository

**Solution**: Set up GitHub authentication:
```bash
# Using SSH
git clone git@github.com:latent-reasoning-works/manylatents-omics.git

# Using HTTPS with token
git clone https://TOKEN@github.com/latent-reasoning-works/manylatents-omics.git
```

## Best Practices

1. **Install only what you need**: Extensions are optional - only install those you'll use
2. **Keep extensions updated**: Run `uv add --upgrade git+https://...` periodically
3. **Check compatibility**: Some extensions may require specific core versions
4. **Use virtual environments**: Keep extension sets isolated per project
5. **Document dependencies**: If your workflow uses extensions, note them in your README

## Learn More

- See `EXTENSIONS.md` for complete installation guide
- Check extension repositories for specific documentation
- Review `examples/` for workflows using extensions
- Join discussions on extension development in GitHub Issues
