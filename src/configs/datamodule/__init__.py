from dataclasses import dataclass
from typing import Optional

from config import BaseDataModuleConfig
from hydra.core.config_store import ConfigStore


@dataclass
class HGDPDataModuleConfig(BaseDataModuleConfig):
    """
    Specialized config that *extends* the base data module config,
    adding the `_target_` and HGDP-specific fields.
    """
    _target_: str = "src.datamodules.hgdp.HGDPDataModule"
    plink_prefix: str = None
    metadata_path: str = None
    mode: str = None
    mmap_mode: Optional[str] = None

cs = ConfigStore.instance()
cs.store(
    group="datamodule",
    name="hgdp",
    node=HGDPDataModuleConfig(),
)
