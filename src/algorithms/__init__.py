
from hydra_zen import builds, store

from .pca import PCAModule
from .phate import PHATE
from .tsne import tSNE

algorithm_store = store(group="algorithm")

pca_config = builds(
    PCAModule,
    n_components="${cfg.algorithm.n_components:50}",
    populate_full_signature=True  # Ensures all parameters are exposed for overrides
)
algorithm_store(pca_config, name="pca")

phate_config = builds(
    PHATE,
    n_components="${cfg.algorithm.n_components:50}",
    knn="${cfg.algorithm.knn:30}",
    gamma="${cfg.algorithm.gamma:0.5}",
    populate_full_signature=True
)
algorithm_store(phate_config, name="phate")

tsne_config = builds(
    tSNE,
    n_components="${cfg.algorithm.n_components:50}",
    perplexity="${cfg.algorithm.perplexity:30.0}",
    learning_rate="${cfg.algorithm.learning_rate:200.0}",
    n_iter="${cfg.algorithm.n_iter:1000}",
    populate_full_signature=True
)
algorithm_store(tsne_config, name="tsne")
