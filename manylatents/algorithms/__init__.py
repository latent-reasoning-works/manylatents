
from hydra_zen import builds, store
import inspect

from .latent.pca import PCAModule
from .latent.phate import PHATEModule
from .latent.tsne import TSNEModule
from .latent.latent_module_base import LatentModule

algorithm_store = store(group="algorithm")

pca_config = builds(
    PCAModule,
    n_components="${cfg.algorithm.n_components:50}",
    populate_full_signature=True  # Ensures all parameters are exposed for overrides
)
algorithm_store(pca_config, name="pca")


phate_config = builds(
    PHATEModule,
    n_components="${cfg.algorithm.n_components:50}",
    knn="${cfg.algorithm.knn:5}",
    gamma="${cfg.algorithm.gamma:0.5}",
    populate_full_signature=True
)
algorithm_store(phate_config, name="phate")

tsne_config = builds(
    TSNEModule,
    n_components="${cfg.algorithm.n_components:2}",
    random_state="${cfg.algorithm.random_state:42}",
    perplexity="${cfg.algorithm.perplexity:30.0}",
    learning_rate="${cfg.algorithm.learning_rate:200.0}",
    n_iter_early="${cfg.algorithm.n_iter_early:250}",
    n_iter_late="${cfg.algorithm.n_iter_late:750}",
    metric="${cfg.algorithm.metric:euclidean}",
    populate_full_signature=True
)
algorithm_store(tsne_config, name="tsne")


def get_all_latent_modules():
    """Return all LatentModule subclasses defined in this package (excluding the base)."""
    modules = []
    for name, obj in globals().items():
        if inspect.isclass(obj) and issubclass(obj, LatentModule) and obj is not LatentModule:
            modules.append(obj)
    return modules