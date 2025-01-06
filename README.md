## Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies. Once uv is installed locally, all dependencies can be installed with:

```bash
uv sync  # Creates a virtual environment and installs dependencies in it.
```
You can then activate the virtual environment with:

```bash
source .venv/bin/activate # Activates the phate-genetics environment
```

## Project layout

```
pyproject.toml   # Project metadata and dependencies
requirements.txt # same as above, deprecated? 
src/
    mainfile.py  # to fill... currently doesn't exist
notebooks/       # description
data/            # data
```