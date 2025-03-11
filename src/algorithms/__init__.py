
from hydra_zen import builds, store

from .pca import PCAModule
from .phate import PHATEModule
from .tsne import TSNEModule
from .umap import UMAPModule

algorithm_store = store(group="algorithm")

pca_config = builds(
    PCAModule,
    n_components="${cfg.algorithm.n_components:50}",
    populate_full_signature=True  # Ensures all parameters are exposed for overrides
)
algorithm_store(pca_config, name="pca")


## review PHATE and TSNE after they're implemented, currently placeholders
phate_config = builds(
    PHATEModule,
    n_components="${cfg.algorithm.n_components:2}",
    random_state="${cfg.algorithm.random_state:42}",
    t="${cfg.algorithm.t:5}",
    knn="${cfg.algorithm.knn:30}",
    gamma="${cfg.algorithm.gamma:0.5}",
    decay="${cfg.algorithm.decay:40}",
    n_landmark="${cfg.algorithm.n_landmark:2000}",
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

umap_config = builds(
    UMAPModule,
    n_components="${cfg.algorithm.n_components:2}",
    random_state="${cfg.algorithm.random_state:42}",
    n_neighbors="${cfg.algorithm.n_neighbors:15}",
    n_epochs="${cfg.algorithm.n_epochs:500}",
    learning_rate="${cfg.algorithm.learning_rate:1.0}",
    populate_full_signature=True
)
algorithm_store(umap_config, name="umap")