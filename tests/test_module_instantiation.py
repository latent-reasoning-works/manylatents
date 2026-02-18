"""Sweep test: every algorithm config can be instantiated via Hydra.

Dynamically discovers all algorithm configs (latent + lightning) and verifies
that Hydra can compose and instantiate each one without errors.
"""

import functools
import importlib
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import InstantiationException
import hydra.utils

# Trigger ConfigStore registration of base_config
import manylatents.configs  # noqa: F401

CONFIGS = Path(__file__).parent.parent / "manylatents" / "configs"
LATENT_DIR = CONFIGS / "algorithms" / "latent"
LIGHTNING_DIR = CONFIGS / "algorithms" / "lightning"


def _collect_configs(config_dir: Path) -> list[str]:
    """Collect config names from a directory, skipping non-algorithm files."""
    if not config_dir.is_dir():
        return []
    skip = {"__init__", "default"}
    return sorted(
        p.stem
        for p in config_dir.glob("*.yaml")
        if p.stem not in skip and not p.stem.startswith("_")
    )


def _target_importable(config_path: Path) -> bool:
    """Check whether the _target_ class in a config can be imported."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    target = cfg.get("_target_")
    if not target:
        return False
    module_path, class_name = target.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
        getattr(mod, class_name)
        return True
    except (ImportError, AttributeError):
        return False


def _compose(overrides: list[str]):
    """Compose a Hydra config with proper cleanup."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(CONFIGS.resolve()), version_base="1.3"
    ):
        return compose(config_name="config", overrides=overrides)


def _instantiate_or_skip(cfg, algo_name: str):
    """Instantiate a Hydra config, skipping on missing optional dependencies."""
    try:
        obj = hydra.utils.instantiate(cfg)
    except InstantiationException as e:
        if isinstance(e.__cause__, ImportError):
            pytest.skip(f"optional dependency missing for {algo_name}: {e.__cause__}")
        raise
    if isinstance(obj, functools.partial):
        obj = obj()
    return obj


# --- Latent modules ---

LATENT_CONFIGS = _collect_configs(LATENT_DIR)


@pytest.mark.parametrize("algo_name", LATENT_CONFIGS)
def test_latent_module_instantiation(algo_name):
    """Each latent algorithm config can be composed and instantiated."""
    config_path = LATENT_DIR / f"{algo_name}.yaml"
    if not _target_importable(config_path):
        pytest.skip(f"optional dependency missing for {algo_name}")

    cfg = _compose([f"algorithms/latent={algo_name}", "data=test_data"])
    algo = _instantiate_or_skip(cfg.algorithms.latent, algo_name)

    from manylatents.algorithms.latent.latent_module_base import LatentModule

    assert isinstance(algo, LatentModule), (
        f"{algo_name}: expected LatentModule, got {type(algo).__name__}"
    )


# --- Lightning modules ---

LIGHTNING_CONFIGS = _collect_configs(LIGHTNING_DIR)


@pytest.mark.parametrize("algo_name", LIGHTNING_CONFIGS)
def test_lightning_module_instantiation(algo_name):
    """Each lightning algorithm config can be composed and instantiated."""
    config_path = LIGHTNING_DIR / f"{algo_name}.yaml"
    if not _target_importable(config_path):
        pytest.skip(f"optional dependency missing for {algo_name}")

    cfg = _compose([
        f"algorithms/lightning={algo_name}",
        "data=swissroll",
        "trainer=default",
    ])

    # Lightning configs use _recursive_: false, so instantiate gives us the
    # module with DictConfig children (network, optimizer, loss) — not yet
    # fully built. That's correct; setup() finishes construction.
    algo = _instantiate_or_skip(cfg.algorithms.lightning, algo_name)

    from lightning.pytorch import LightningModule

    assert isinstance(algo, LightningModule), (
        f"{algo_name}: expected LightningModule, got {type(algo).__name__}"
    )
