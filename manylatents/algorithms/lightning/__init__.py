# LightningModule algorithms (training-based pattern)
#
# Unlike `algorithms/latent`, the useful unit here is a CONFIG, not a class. `Reconstruction`
# is archetypal analysis when its `network` is AAnet and a plain autoencoder when it is not,
# so class-name lookup would be ambiguous. The packaged configs under
# `configs/algorithms/lightning/*.yaml` name the composition, and they are what
# `manylatents.api.run(algorithms={"lightning": "<name>"})` resolves.
#
# This module previously held only the comment on line 1 — no imports, no registry, no listing
# function. `algorithms.latent.list_algorithms()` therefore enumerated the *latent* group while
# reading like a catalogue of the whole engine, and a consumer building its catalog from that
# one call concluded that archetypal analysis did not exist here at all.
from typing import List

__all__ = ["list_algorithms"]


def list_algorithms() -> List[str]:
    """Every lightning algorithm reachable by name, from the packaged configs.

    Sorted, and excluding composition stubs that carry no ``_target_`` (``default.yaml``).
    Returns ``[]`` rather than raising when the configs are absent, so a partial install
    degrades instead of breaking enumeration.
    """
    from importlib import resources

    try:
        d = resources.files("manylatents") / "configs" / "algorithms" / "lightning"
        if not d.is_dir():
            return []
    except (ModuleNotFoundError, FileNotFoundError):  # pragma: no cover - packaging edge
        return []

    from omegaconf import OmegaConf

    names = []
    for p in d.iterdir():
        if p.name.endswith(".yaml"):
            try:
                if "_target_" in OmegaConf.load(str(p)):
                    names.append(p.name[: -len(".yaml")])
            except Exception:  # noqa: BLE001 - an unreadable file is not an algorithm
                continue
    return sorted(names)
