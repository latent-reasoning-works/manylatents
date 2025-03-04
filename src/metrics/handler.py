import hydra
from omegaconf import DictConfig, OmegaConf


class MetricsHandler:
    def __init__(self, metrics_config: DictConfig):
        """
        Instantiate or store each metric definition.
        `metrics_config` is a dict-like object with entries like:
            trustworthiness:
                _target_: sklearn.manifold.trustworthiness
                n_neighbors: 5
                metric: euclidean
        """
        if not OmegaConf.is_config(metrics_config):
            metrics_config = OmegaConf.create(metrics_config)
        self.metrics = metrics_config

    def compute_all(self, original, embedded) -> dict:
        """
        Compute all metrics using the provided keyword arguments.

        Args:
            kwargs: Arguments needed by the compute method of each metric, for example:
                    - original: the original data
                    - embedded: the computed embeddings
                    - labels: optional labels

        Returns:
            dict: A dictionary mapping each metric name to its computed value.
        """
        results = {}
        for name, cfg in self.metrics.items():
            results[name] = hydra.utils.call(cfg, X=original, X_embedded=embedded)
        return results
