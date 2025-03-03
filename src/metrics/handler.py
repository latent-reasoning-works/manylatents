from omegaconf import DictConfig, OmegaConf
import hydra.utils

class MetricsHandler:
    def __init__(self, metrics_config: DictConfig):
        """
        Instantiate all metrics defined in the configuration.

        Args:
            metrics_config (DictConfig): Hydra configuration for metrics.
        """
        if not OmegaConf.is_config(metrics_config):
            metrics_config = OmegaConf.create(metrics_config)
        self.metrics = {}
        for metric_name, cfg in metrics_config.items():
            # Each metric config should include a _target_ key pointing to the metric class.
            self.metrics[metric_name] = hydra.utils.instantiate(cfg)

    def compute_all(self, **kwargs) -> dict:
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
        for name, metric in self.metrics.items():
            results[name] = metric.compute(**kwargs)
        return results
