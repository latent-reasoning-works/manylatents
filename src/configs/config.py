from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore

from .configs import Config


@dataclass
class DataModuleConfig:
    batch_size: int = 32
    num_workers: int = 4
    cache_dir: str = "${paths.cache_dir}"  

@dataclass
class AlgorithmConfig:
    _target_: str = "src.algorithms.BaseAlgorithm"
    name: str = "default_algorithm"

@dataclass
class HydraConfig:
    run_dir: str = "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"

@dataclass
class PathsConfig:
    data_dir: str = "data/"
    genotype_dir: str = "data/genotypes/"
    admixture_dir: str = "data/admixture/"
    output_dir: str = "outputs/"
    ckpt_dir: str = "outputs/ckpt/"
    cache_dir: str = "outputs/cache/"
    plot_dir: str = "outputs/plots/"

@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    seed: int = 42
    
@dataclass
class Config:
    """
    The top-level config for your experiment.
    """
    algorithm: Optional[AlgorithmConfig] = None
    datamodule: DataModuleConfig = DataModuleConfig()
    hydra: HydraConfig = HydraConfig()
    paths: PathsConfig = PathsConfig()
    experiment: Optional[ExperimentConfig] = None

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)