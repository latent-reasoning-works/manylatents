from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

@dataclass
class AlgorithmSpec:
    id: str                              # "latent/pca" or "lightning/ae_reconstruction"
    kind: Literal["latent", "lightning"] # derived from id
    target: str                          # _target_ from the algo file
    params: DictConfig                   # effective params after merge

def _cfg_root() -> Path:
    # lazy to avoid hydra init ordering issues
    return Path(hydra.utils.get_original_cwd()) / "manylatents" / "configs"

def _is_full_id(s: Any) -> bool:
    return isinstance(s, str) and "/" in s

def _parse_item(item: Any) -> tuple[str, DictConfig]:
    """
    Accepts "latent/pca" or {"latent/pca": {...}}.
    Returns (id, overrides).
    """
    if isinstance(item, str) and _is_full_id(item):
        return item, OmegaConf.create({})
    if isinstance(item, dict) and len(item) == 1:
        k, v = next(iter(item.items()))
        if not _is_full_id(k):
            raise ValueError(f"Pipeline key must be 'kind/name', got '{k}'")
        return k, OmegaConf.create(v or {})
    raise ValueError(f"Invalid pipeline item: {item!r}")

def _normalize_pipeline(val: Any) -> list[Any]:
    # YAML list preferred; CLI "pipeline=latent/pca,lightning/ae_reconstruction" also ok
    if isinstance(val, list):
        return list(val)
    if isinstance(val, str):
        toks = [t.strip() for t in val.split(",") if t.strip()]
        if not all(_is_full_id(t) for t in toks):
            raise ValueError("CLI pipeline tokens must be full ids like 'latent/pca'")
        return toks
    raise ValueError("pipeline must be a list or a comma-separated string")

def _load_algo_file(full_id: str) -> DictConfig:
    path = _cfg_root() / "algorithm" / f"{full_id}.yaml"
    if not path.exists():
        raise ValueError(f"No algorithm config for '{full_id}' (expected {path})")
    base = OmegaConf.load(str(path))
    if "_target_" not in base:
        raise ValueError(f"Algorithm '{full_id}' missing _target_")
    base.setdefault("params", {})
    return base

def _selected_single_mode_ids(cfg: DictConfig) -> list[str]:
    # preserve your existing UX: override /algorithm/<kind>: <name>
    try:
        choices = HydraConfig.get().runtime.choices
    except Exception:
        choices = {}
    ids: list[str] = []
    lat = choices.get("algorithm/latent")
    lig = choices.get("algorithm/lightning")
    if lat not in (None, "null"): ids.append(f"latent/{lat}")
    if lig not in (None, "null"): ids.append(f"lightning/{lig}")
    # optional: respect algorithm.order: [latent, lightning]
    if "algorithm" in cfg and hasattr(cfg.algorithm, "order"):
        order = list(cfg.algorithm.order)
        ids.sort(key=lambda s: order.index(s.split("/",1)[0]) if s.split("/",1)[0] in order else 99)
    return ids

def resolve_algorithm_specs(cfg: DictConfig) -> List[AlgorithmSpec]:
    """
    Config -> specs:
      - If cfg.pipeline: parse it strictly (full ids only).
      - Else: derive ids from /algorithm group choices (single mode).
      - Merge once: file params <- experiment overrides (single mode) <- CLI.
    """
    pairs: list[tuple[str, DictConfig]]
    if getattr(cfg, "pipeline", None):
        items = _normalize_pipeline(cfg.pipeline)
        pairs = [_parse_item(x) for x in items]
    else:
        ids = _selected_single_mode_ids(cfg)
        pairs = [(i, OmegaConf.create({})) for i in ids]

    specs: list[AlgorithmSpec] = []
    for full_id, overrides in pairs:
        base = _load_algo_file(full_id)
        # In single mode, allow flat overrides under algorithm.<kind> (back-compat)
        if not getattr(cfg, "pipeline", None) and "algorithm" in cfg:
            kind = full_id.split("/", 1)[0]
            if kind in cfg.algorithm and isinstance(cfg.algorithm[kind], DictConfig):
                overrides = OmegaConf.merge(overrides, cfg.algorithm[kind])
        params = OmegaConf.merge(base.params, overrides)  # experiment/CLI always win
        kind = "lightning" if full_id.startswith("lightning/") else "latent"
        specs.append(AlgorithmSpec(id=full_id, kind=kind, target=base._target_, params=params))
    return specs

def requires_gpu(specs: List[AlgorithmSpec]) -> bool:
    # simple & explicit: lightning => GPU; latent => CPU
    return any(s.kind == "lightning" for s in specs)
