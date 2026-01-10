"""
manyLatents data module.

Provides datasets and data modules for dimensionality reduction experiments.

Dataset Registry (Auto-discovered)
----------------------------------
Use `get_dataset(name)` for simple data loading by string name:

    >>> from manylatents.data import get_dataset
    >>> data = get_dataset('swissroll').get_data()  # Returns (5000, 3) array
    >>> print(list_datasets())  # See all available datasets

Datasets are auto-discovered from this module. Any class with a `get_data()`
method is registered automatically.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# Base imports that are always available
from .swissroll import SwissRollDataModule
from .synthetic_dataset import SwissRoll

__all__ = ["get_dataset", "list_datasets", "SwissRollDataModule", "SwissRoll"]

# Registry populated by auto-discovery
_DATASET_REGISTRY: Dict[str, Type] = {}
_DATAMODULE_REGISTRY: Dict[str, Type] = {}


def _has_get_data_method(cls: Type) -> bool:
    """Check if class has get_data method (duck typing for datasets)."""
    return hasattr(cls, 'get_data') and callable(getattr(cls, 'get_data', None))


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _discover_datasets() -> None:
    """
    Auto-discover dataset classes from all .py files in this directory.

    Registers any class that:
    1. Has a get_data() method (datasets)
    2. Inherits from LightningDataModule (datamodules)

    Classes are registered under multiple name variants:
    - Original class name (e.g., 'SwissRoll')
    - snake_case version (e.g., 'swiss_roll')
    - lowercase version (e.g., 'swissroll')
    """
    global _DATASET_REGISTRY, _DATAMODULE_REGISTRY

    data_dir = Path(__file__).parent
    module_files = [f.stem for f in data_dir.glob("*.py") if not f.stem.startswith('_')]

    for module_name in module_files:
        try:
            module = importlib.import_module(f".{module_name}", package="manylatents.data")
        except ImportError as e:
            logger.debug(f"Could not import {module_name}: {e}")
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip imports from other packages
            if not obj.__module__.startswith("manylatents.data"):
                continue

            # Skip base classes and test classes
            if name.startswith('_') or 'Test' in name or name == 'SyntheticDataset':
                continue

            # Register datasets (classes with get_data method)
            if _has_get_data_method(obj):
                # Register under multiple name variants for flexibility
                variants = [
                    name,  # SwissRoll
                    _to_snake_case(name),  # swiss_roll
                    name.lower(),  # swissroll
                ]
                for variant in set(variants):
                    if variant not in _DATASET_REGISTRY:
                        _DATASET_REGISTRY[variant] = obj

            # Register DataModules
            if 'DataModule' in name:
                base_name = name.replace('DataModule', '')
                variants = [
                    name,  # SwissRollDataModule
                    base_name,  # SwissRoll (as datamodule)
                    _to_snake_case(base_name) + '_datamodule',  # swiss_roll_datamodule
                ]
                for variant in set(variants):
                    if variant not in _DATAMODULE_REGISTRY:
                        _DATAMODULE_REGISTRY[variant] = obj


def get_dataset(name: str, **kwargs: Any) -> Any:
    """
    Get a dataset instance by name.

    This is the recommended way for external tools (manyAgents, Geomancy)
    to load datasets without importing specific classes.

    Args:
        name: Dataset name (case-insensitive, supports snake_case and CamelCase).
        **kwargs: Additional arguments passed to dataset constructor.

    Returns:
        Dataset instance with .get_data() method.

    Raises:
        ValueError: If dataset name is not found in registry.

    Examples:
        >>> from manylatents.data import get_dataset
        >>> sr = get_dataset('swissroll')
        >>> data = sr.get_data()  # (5000, 3) numpy array

        >>> # All these are equivalent:
        >>> get_dataset('SwissRoll')
        >>> get_dataset('swiss_roll')
        >>> get_dataset('swissroll')
    """
    # Ensure registry is populated
    if not _DATASET_REGISTRY:
        _discover_datasets()

    # Normalize name
    name_lower = name.lower().replace('-', '_')

    # Try exact match first
    if name in _DATASET_REGISTRY:
        return _DATASET_REGISTRY[name](**kwargs)

    # Try normalized name
    if name_lower in _DATASET_REGISTRY:
        return _DATASET_REGISTRY[name_lower](**kwargs)

    # Not found
    available = sorted(set(k.lower() for k in _DATASET_REGISTRY.keys()))
    raise ValueError(
        f"Unknown dataset: '{name}'. "
        f"Available: {available}"
    )


def get_datamodule(name: str, **kwargs: Any) -> Any:
    """Get a LightningDataModule instance by name."""
    if not _DATAMODULE_REGISTRY:
        _discover_datasets()

    name_lower = name.lower().replace('-', '_')

    if name in _DATAMODULE_REGISTRY:
        return _DATAMODULE_REGISTRY[name](**kwargs)
    if name_lower in _DATAMODULE_REGISTRY:
        return _DATAMODULE_REGISTRY[name_lower](**kwargs)

    available = sorted(set(k.lower() for k in _DATAMODULE_REGISTRY.keys()))
    raise ValueError(f"Unknown datamodule: '{name}'. Available: {available}")


def list_datasets() -> List[str]:
    """Return list of available dataset names (deduplicated, lowercase)."""
    if not _DATASET_REGISTRY:
        _discover_datasets()
    return sorted(set(k.lower() for k in _DATASET_REGISTRY.keys()))


def list_datamodules() -> List[str]:
    """Return list of available datamodule names."""
    if not _DATAMODULE_REGISTRY:
        _discover_datasets()
    return sorted(set(k.lower() for k in _DATAMODULE_REGISTRY.keys()))


# Run discovery on import
_discover_datasets()
