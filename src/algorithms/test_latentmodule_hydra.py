import pytest
import hydra
from hydra.utils import instantiate
from src.algorithms import get_all_latent_modules

# Minimal configs for each module (add more as needed)
MINIMAL_CONFIGS = {
    'PCAModule': {'_target_': 'src.algorithms.pca.PCAModule', 'n_components': 2},
    'UMAPModule': {'_target_': 'src.algorithms.umap.UMAPModule', 'n_components': 2},
    'PHATEModule': {'_target_': 'src.algorithms.phate.PHATEModule', 'n_components': 2},
    'TSNEModule': {'_target_': 'src.algorithms.tsne.TSNEModule', 'n_components': 2},
    'DiffusionMapModule': {'_target_': 'src.algorithms.diffusionmap.DiffusionMapModule', 'n_components': 2},
    'MDSModule': {'_target_': 'src.algorithms.mds.MDSModule', 'ndim': 2},
    'ArchetypalAnalysisModule': {'_target_': 'src.algorithms.aa.ArchetypalAnalysisModule', 'n_components': 2},
    'NoOpModule': {'_target_': 'src.algorithms.dr_noop.NoOpModule'},
}

def get_minimal_config(cls):
    return MINIMAL_CONFIGS.get(cls.__name__, None)

@pytest.mark.parametrize('module_cls', get_all_latent_modules())
def test_latentmodule_hydra_instantiation(module_cls):
    cfg = get_minimal_config(module_cls)
    assert cfg is not None, f"No minimal config for {module_cls.__name__}"
    instance = instantiate(cfg)
    assert isinstance(instance, module_cls) 