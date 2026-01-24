# LatentModule algorithms (fit/transform pattern)
from .pca import PCAModule
from .tsne import TSNEModule
from .umap import UMAPModule
from .phate import PHATEModule
from .multiscale_phate import MultiscalePHATEModule
from .diffusion_map import DiffusionMapModule
from .multi_dimensional_scaling import MDSModule
from .merging import MergingModule, ChannelLoadings

__all__ = [
    "PCAModule",
    "TSNEModule",
    "UMAPModule",
    "PHATEModule",
    "MultiscalePHATEModule",
    "DiffusionMapModule",
    "MDSModule",
    "MergingModule",
    "ChannelLoadings",
]