# manylatents Extensions

The `manylatents` framework supports extensions that add domain-specific functionality while maintaining a clean separation of concerns.

## Available Extensions

### manylatents-omics (Genetics & Population Genetics)

Adds support for genetics data and population genetics metrics.

**Repository**: https://github.com/latent-reasoning-works/manylatents-omics

**Features**:
- PLINK/BED format dataset loaders
- Population genetics metrics (geographic preservation, admixture analysis, etc.)
- Genetics-specific algorithms
- Integration with genomics databases

#### Installation Options

##### Option 1: Install as Separate Package (Recommended)

```bash
# Install manylatents core
uv sync  # or: pip install -e .

# Install manylatents-omics extension
uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
```

##### Option 2: Clone and Install Locally

```bash
# Clone both repositories
git clone https://github.com/latent-reasoning-works/manylatents.git
git clone https://github.com/latent-reasoning-works/manylatents-omics.git

# Install core
cd manylatents
pip install -e .

# Install omics extension
cd ../manylatents-omics
pip install -e .
```

##### Option 3: Using Git Submodule (For Developers)

```bash
# In your manylatents clone
git submodule add https://github.com/latent-reasoning-works/manylatents-omics.git extensions/manylatents-omics
git submodule update --init --recursive

# Install both packages
pip install -e .
pip install -e extensions/manylatents-omics
```

#### Verify Installation

```python
# Test that both packages work together
from manylatents.data import SwissRoll
from manylatents.omics.data import PlinkDataset
from manylatents.omics.metrics import GeographicPreservation

print("✅ manylatents-omics extension is working!")
```

#### Usage

Once installed, omics functionality is available through the `manylatents.omics` namespace:

```python
# Use in your code
from manylatents.omics.data import PlinkDataset
from manylatents.omics.metrics import GeographicPreservation

# Use in Hydra configs
# configs/data/genetics.yaml
data:
  _target_: manylatents.omics.data.PlinkDataset
  bed_file: path/to/data.bed
  ...

# configs/metrics/population.yaml
metrics:
  dataset:
    - geographic_preservation
    - admixture_coherence
```

## Architecture: Namespace Packages

Both `manylatents` and `manylatents-omics` use Python's namespace package feature, which allows multiple packages to share the `manylatents` namespace:

```
manylatents (core)               manylatents-omics (extension)
├── data/                        ├── manylatents/
├── algorithms/                  │   └── omics/
├── metrics/                     │       ├── data/
└── ...                          │       ├── metrics/
                                 │       └── algorithms/
```

When both are installed:
```python
import manylatents          # Core package
import manylatents.omics    # Extension package
# Both coexist seamlessly!
```

## Creating Your Own Extension

Want to create your own domain-specific extension? Follow the namespace package pattern:

1. **Create package structure**:
```
your-extension/
├── setup.py
└── manylatents/
    ├── __init__.py  # Must declare namespace
    └── yourext/
        ├── __init__.py
        ├── data/
        ├── metrics/
        └── algorithms/
```

2. **Declare namespace** in `manylatents/__init__.py`:
```python
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

3. **Install alongside core**:
```bash
pip install -e /path/to/manylatents
pip install -e /path/to/your-extension
```

4. **Use in workflows**:
```python
from manylatents.yourext.data import YourDataset
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'manylatents.omics'"

The extension isn't installed. Install it with:
```bash
pip install git+https://github.com/latent-reasoning-works/manylatents-omics.git
```

### Both packages installed but imports fail

Ensure both use namespace package declarations. Check that `manylatents/__init__.py` in each package contains:
```python
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

### Extension conflicts with core

Extensions should only add new submodules (like `manylatents.omics`), never override core modules. If you see conflicts, check the package structure.

## Available Extensions

| Extension | Description | Repository |
|-----------|-------------|------------|
| **manylatents-omics** | Genetics & population genetics | https://github.com/latent-reasoning-works/manylatents-omics |
| *(more coming)* | - | - |

## Need Help?

- **Core package issues**: Open issue at https://github.com/latent-reasoning-works/manylatents/issues
- **Extension issues**: Open issue at the extension's repository
- **General questions**: Check documentation or open a discussion
