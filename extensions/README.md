# Extensions Directory

This directory is for locally cloned manylatents extensions.

## Quick Start

To install extensions, run:
```bash
bash setup_extensions.sh
```

Or manually:
```bash
# Clone extension here
git clone https://github.com/latent-reasoning-works/manylatents-omics.git

# Install in editable mode
pip install -e manylatents-omics
```

## Using Git Submodules

For development workflows, you can use git submodules:
```bash
# Add extension as submodule
git submodule add https://github.com/latent-reasoning-works/manylatents-omics.git extensions/manylatents-omics

# Initialize and update
git submodule update --init --recursive

# Install
pip install -e extensions/manylatents-omics
```

## Available Extensions

- **manylatents-omics**: Genetics and population genetics support
  - Repository: https://github.com/latent-reasoning-works/manylatents-omics

See [../EXTENSIONS.md](../EXTENSIONS.md) for complete documentation.
