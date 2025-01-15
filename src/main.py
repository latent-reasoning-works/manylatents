import logging
from pathlib import Path

import hydra
import pandas as pd
import tqdm
from omegaconf import DictConfig, OmegaConf

from utils.data import preprocess_data
from utils.embeddings import compute_or_load_phate, compute_tsne
from utils.metrics import compute_and_append_metrics
from utils.plotting import plot_phate_results
from utils.utils import prepare_directories

logger = logging.getLogger(__name__)

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base="1.2",
)

def main(cfg: DictConfig):
    """
    Main entry point for the PHATE hyperparameter search script.
    """
    logger.setLevel(cfg.project.log_level)
    logger.info("Starting the PHATE hyperparameter search...")
    logger.info(f"Loaded configuration:\n{OmegaConf.to_yaml(cfg)}")

    prepare_directories(cfg)

    # Load data
    pca_emb, metadata, to_fit_on, to_transform_on, admixture_ratios_list, _ = preprocess_data(
        admixtures_k=cfg.data.admixtures_k,
        data_dir=cfg.paths.data_dir,
        admixture_dir=cfg.paths.admixture_dir,
        genotype_dir=cfg.paths.genotype_dir,
        pca_file=cfg.data.pca_file,
        metadata_file=cfg.data.metadata_file,
        relatedness_file="HGDP+1KGP_MattEstimated_related_samples.tsv",
        filters=cfg.data.filters
    )


    results = []

    # Compute PCA and t-SNE metrics
    logger.info("Computing PCA and t-SNE metrics...")
    compute_and_append_metrics(
        "pca (50D)", pca_input, pca_input, metadata, cfg.data.admixtures_k, admixture_ratios_list,
        {"gamma": "NA", "decay": "NA", "knn": "NA", "t": "NA"}, None, results
    )
    compute_and_append_metrics(
        "pca (2D)", pca_input[:, :2], pca_input, metadata, cfg.data.admixtures_k, admixture_ratios_list,
        {"gamma": "NA", "decay": "NA", "knn": "NA", "t": "NA"}, None, results
    )
    tsne_emb, tsne_obj = compute_tsne(pca_input, fit_idx, transform_idx, init="pca")
    compute_and_append_metrics(
        "t-SNE", tsne_emb, pca_input, metadata, cfg.data.admixtures_k, admixture_ratios_list,
        {"gamma": "NA", "decay": "NA", "knn": "NA", "t": "NA"}, None, results
    )

    # Hyperparameter search
    logger.info("Starting hyperparameter search...")
    for gamma in cfg.hyperparameters.gammas:
        for decay in cfg.hyperparameters.decays:
            embeddings_list_k = []
            for knn in tqdm.tqdm(cfg.hyperparameters.knns, desc=f"gamma={gamma}, decay={decay}"):
                embeddings_list, phate_operator = compute_or_load_phate(
                    pca_input, fit_idx, transform_idx, cfg.hyperparameters.ts,
                    cfg.paths.ckpt_dir,
                    n_landmark=cfg.embeddings.phate.n_landmark,
                    knn=knn, decay=decay, gamma=gamma,
                    cache_dir=cfg.paths.laplacian_cache_dir if cfg.project.caching else None
                )

                for t, emb in tqdm.tqdm(zip(cfg.hyperparameters.ts, embeddings_list), desc=f"Metrics knn={knn}"):
                    compute_and_append_metrics(
                        "phate", emb, pca_input, metadata, cfg.data.admixtures_k, admixture_ratios_list,
                        {"gamma": gamma, "decay": decay, "knn": knn, "t": t}, phate_operator, results
                    )
                embeddings_list_k.append(embeddings_list)

            # Save plots if enabled
            if cfg.project.plotting:
                plot_phate_results(
                    embeddings_list_k, metadata, cfg.hyperparameters.ts, cfg.hyperparameters.knns,
                    "knn", cmap, Path(cfg.paths.plot_dir) / f"results_gamma={gamma}_decay={decay}.png"
                )

    # Save metrics to CSV
    logger.info("Saving results to CSV...")
    results_df = pd.DataFrame(results)
    results_file = Path(cfg.paths.output_dir) / "results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()