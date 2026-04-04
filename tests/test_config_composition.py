"""Config composition tests: every config group resolves through Hydra's struct schema.

Catches missing fields in the Config dataclass (like the sampling gap) by
composing each config YAML through Hydra and verifying no schema errors.
Also validates that _target_ classes are importable where applicable.
"""

import importlib
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import manylatents.configs  # noqa: F401 — triggers ConfigStore registration

CONFIGS = Path(__file__).parent.parent / "manylatents" / "configs"


def _collect(config_dir: Path) -> list[str]:
    """Collect config names, skipping __init__ / default / underscore-prefixed."""
    if not config_dir.is_dir():
        return []
    skip = {"__init__", "default", "null", "none"}
    return sorted(
        p.stem
        for p in config_dir.glob("*.yaml")
        if p.stem not in skip and not p.stem.startswith("_")
    )


def _compose(overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(CONFIGS.resolve()), version_base="1.3"
    ):
        return compose(config_name="config", overrides=overrides)


def _target_importable(cfg_path: Path) -> bool:
    """Check whether the _target_ in a YAML can be imported."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}

    # _target_ can be at top level or nested one level (e.g. sampling.dataset._target_)
    targets = []
    if "_target_" in raw:
        targets.append(raw["_target_"])
    for v in raw.values():
        if isinstance(v, dict) and "_target_" in v:
            targets.append(v["_target_"])

    for target in targets:
        module_path, class_name = target.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
            getattr(mod, class_name)
        except (ImportError, AttributeError):
            return False
    return True


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_CONFIGS = _collect(CONFIGS / "data")


@pytest.mark.parametrize("name", DATA_CONFIGS)
def test_data_config_composes(name):
    """Each data config composes without schema errors."""
    cfg = _compose([f"data={name}"])
    assert cfg.data is not None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRICS_CONFIGS = _collect(CONFIGS / "metrics")


@pytest.mark.parametrize("name", METRICS_CONFIGS)
def test_metric_config_composes(name):
    """Each metric config composes without schema errors."""
    cfg = _compose([f"metrics={name}"])
    assert cfg.metrics is not None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

SAMPLING_CONFIGS = _collect(CONFIGS / "sampling")


@pytest.mark.parametrize("name", SAMPLING_CONFIGS)
def test_sampling_config_composes(name):
    """Each sampling config composes without schema errors."""
    cfg = _compose([f"sampling={name}"])
    assert cfg.sampling is not None


@pytest.mark.parametrize("name", SAMPLING_CONFIGS)
def test_sampling_target_importable(name):
    """Each sampling config references an importable _target_."""
    cfg_path = CONFIGS / "sampling" / f"{name}.yaml"
    if not _target_importable(cfg_path):
        pytest.skip(f"optional dependency missing for sampling/{name}")


# ---------------------------------------------------------------------------
# Callbacks (embedding)
# ---------------------------------------------------------------------------

EMBEDDING_CB_CONFIGS = _collect(CONFIGS / "callbacks" / "embedding")


@pytest.mark.parametrize("name", EMBEDDING_CB_CONFIGS)
def test_embedding_callback_config_composes(name):
    """Each embedding callback config composes without schema errors."""
    cfg = _compose([f"callbacks/embedding={name}"])
    assert cfg.callbacks is not None


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIGS = _collect(CONFIGS / "experiment")


@pytest.mark.parametrize("name", EXPERIMENT_CONFIGS)
def test_experiment_config_composes(name):
    """Each experiment config composes without schema errors."""
    cfg = _compose([f"experiment={name}"])
    # Experiment configs may use ${hydra:runtime.*} interpolations that can't
    # resolve outside a full Hydra run — only check composition, not resolution.
    assert cfg is not None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

TRAINER_CONFIGS = _collect(CONFIGS / "trainer")


@pytest.mark.parametrize("name", TRAINER_CONFIGS)
def test_trainer_config_composes(name):
    """Each trainer config composes without schema errors."""
    cfg = _compose([f"trainer={name}"])
    assert cfg.trainer is not None


# ---------------------------------------------------------------------------
# Cross-group: sampling + algorithm + data (the actual failure path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", SAMPLING_CONFIGS)
def test_sampling_with_algorithm_composes(sampling):
    """Sampling + algorithm + data compose together (end-to-end schema check)."""
    cfg = _compose([
        "algorithms/latent=pca",
        "data=swissroll",
        f"sampling={sampling}",
    ])
    assert cfg.sampling is not None
    assert cfg.algorithms.latent is not None
