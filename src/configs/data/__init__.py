from dataclasses import dataclass
from typing import Dict, Optional

from config import BaseDataModuleConfig
from hydra_zen import store


@dataclass
class HGDPDataModuleConfig(BaseDataModuleConfig):
    """
    Specialized config that *extends* the base data module config,
    adding the `_target_` and HGDP-specific fields.
    """
    _target_: str = "src.datamodules.hgdp.HGDPDataModule"
    filenames: Dict[str, str] = None 
    plink_path: str = "${paths.genotype_dir}/${datamodule.filenames.plink}"
    metadata_path: str = "${paths.data_dir}/${datamodule.filenames.metadata}" 
    mode: Optional[str] = None
    mmap_mode: Optional[str] = None
    precomputed_path: Optional[str] = None
    debug: bool = False

store(
    HGDPDataModuleConfig,
    group="data",
    name="hgdp"
)
