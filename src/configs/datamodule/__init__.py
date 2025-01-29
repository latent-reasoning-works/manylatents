from dataclasses import dataclass
from typing import Optional

from config import BaseDataModuleConfig

#from hydra.core.config_store import ConfigStore
from hydra_zen import store

datamodule_store = store(group="datamodule")

@dataclass
class HGDPDataModuleConfig(BaseDataModuleConfig):
    """
    Specialized config that *extends* the base data module config,
    adding the `_target_` and HGDP-specific fields.
    """
    _target_: str = "src.datamodules.hgdp.HGDPDataModule"
    filenames: dict = None 
    plink_path: str = "${paths.genotype_dir}/${datamodule.filenames.plink}"
    metadata_path: str = "${paths.data_dir}/${datamodule.filenames.metadata}" 
    mode: str = None
    mmap_mode: Optional[str] = None

#cs = ConfigStore.instance()
#cs.store(
#    group="datamodule",
#    name="hgdp",
#    node=HGDPDataModuleConfig(),
#)
