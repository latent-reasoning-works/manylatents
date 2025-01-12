import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import tqdm
from omegaconf import DictConfig, OmegaConf

from src.utils.metrics import compute_or_load_phate, compute_tsne
from src.utils.plotting import plot_phate_results
from src.utils.utils import compute_and_append_metrics, load_data

# Set up logging and project details
PROJECT_NAME = Path(__file__).parent.name
REPO_ROOTDIR = Path(__file__).parent.parent
logger = logging.getLogger(__name__)

@hydra.main(
    config_path="src/configs", 
    config_name="config",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """
    Main entry point for the PHATE hyperparameter search script.
    """
    logger.info("Starting the PHATE hyperparameter search...")
    logger.info(f"Loaded configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create directories for outputs
    output_dir = Path(cfg.model.output_dir)
    model_dir = output_dir / "models"
    plot_dir = output_dir / "plots"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Load data
    admixtures_k = cfg.data.admixtures_k
    pca_input, metadata, fit_idx, transform_idx, admixture_ratios_list, cmap = load_data(
        admixtures_k, cfg.data.data_dir, cfg.data.admixture_dir
    )

    # Results container
    results = []

    # Compute PCA and t-SNE metrics
    logger.info("Computing PCA and t-SNE metrics...")
    compute_and_append_metrics(
        "pca (50D)", pca_input, pca_input, metadata, admixtures_k, admixture_ratios_list,
        {"gamma": "NA", "decay": "NA", "knn": "NA", "t": "NA"}, None, results
    )
    compute_and_append_metrics(
        "pca (2D)", pca_input[:, :2], pca_input, metadata, admixtures_k, admixture_ratios_list,
        {"gamma": "NA", "decay": "NA", "knn": "NA", "t": "NA"}, None, results
    )
    tsne_emb, tsne_obj = compute_tsne(pca_input, fit_idx, transform_idx, init="pca")
    compute_and_append_metrics(
        "t-SNE", tsne_emb, pca_input, metadata, admixtures_k, admixture_ratios_list,
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
                    model_dir, n_landmark=None, knn=knn, decay=decay, gamma=gamma
                )

                for t, emb in tqdm.tqdm(zip(cfg.hyperparameters.ts, embeddings_list), desc=f"Metrics knn={knn}"):
                    compute_and_append_metrics(
                        "phate", emb, pca_input, metadata, admixtures_k, admixture_ratios_list,
                        {"gamma": gamma, "decay": decay, "knn": knn, "t": t}, phate_operator, results
                    )
                embeddings_list_k.append(embeddings_list)

            # Save plots
            plot_phate_results(
                embeddings_list_k, metadata, cfg.hyperparameters.ts, cfg.hyperparameters.knns,
                "knn", cmap, plot_dir / f"results_gamma={gamma}_decay={decay}.png"
            )

    # Save metrics to CSV
    logger.info("Saving results to CSV...")
    results_df = pd.DataFrame(results)
    results_file = output_dir / "results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
