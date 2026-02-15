# manylatents/metrics/magnipy_metric.py
import numpy as np
from typing import Optional, Any
from magnipy.magnipy import Magnipy
from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric


@register_metric(
    aliases=["magnitude_dimension", "mag_dim"],
    default_params={"n_ts": 50, "log_scale": False, "scale_finding": "convergence", "target_prop": 0.95, "metric": "euclidean", "p": 2, "n_neighbors": 12, "method": "cholesky", "one_point_property": True, "perturb_singularities": True, "positive_magnitude": False, "exact": False},
    description="Magnitude-based effective dimensionality",
)
def MagnitudeDimension(
    embeddings: np.ndarray, 
    dataset: Any, 
    module: Optional[LatentModule] = None,
    n_ts: int = 50,
    log_scale: bool = False,
    scale_finding: str = "convergence",
    target_prop: float = 0.95,
    metric: str = "euclidean",
    p: int = 2,
    n_neighbors: int = 12,
    method: str = "cholesky",
    one_point_property: bool = True,
    perturb_singularities: bool = True,
    positive_magnitude: bool = False,
    exact: bool = False
) -> float:
    """
    Computes the maximum magnitude dimension from an embedding.
    This metric computes the magnitude on-the-fly for the given embedding.
    
    Args:
        embeddings (np.ndarray): The low-dimensional embedding produced by a LatentModule.
        dataset (Any): The dataset object.
        module (Optional[LatentModule]): The fitted LatentModule instance.
        n_ts (int): Number of scale values to use. Default: 50.
        log_scale (bool): Whether to use logarithmic scaling. Default: False.
        scale_finding (str): Method for finding scales ('convergence' or other). Default: 'convergence'.
        target_prop (float): Target proportion for convergence. Default: 0.95.
        metric (str): Distance metric to use. Default: 'euclidean'.
        p (int): Parameter for Minkowski metric. Default: 2.
        n_neighbors (int): Number of neighbors for k-NN based metrics. Default: 12.
        method (str): Method for matrix operations ('cholesky' or other). Default: 'cholesky'.
        one_point_property (bool): Whether to use one-point property. Default: True.
        perturb_singularities (bool): Whether to perturb singularities. Default: True.
        positive_magnitude (bool): Whether to enforce positive magnitude. Default: False.
        exact (bool): Whether to use exact computation for magnitude dimension. Default: False.
        
    Returns:
        float: The maximum value of the magnitude dimension profile.
    """
    if embeddings is None or embeddings.size == 0:
        return np.nan
        
    # Instantiate Magnipy with the embedding and desired parameters.
    # Note: this recomputes the distance matrix every time this metric is called.
    mag_obj = Magnipy(
        X=embeddings,
        n_ts=n_ts,
        log_scale=log_scale,
        scale_finding=scale_finding,
        target_prop=target_prop,
        metric=metric,
        p=p,
        n_neighbors=n_neighbors,
        method=method,
        one_point_property=one_point_property,
        perturb_singularities=perturb_singularities,
        positive_magnitude=positive_magnitude
    )
    
    # Compute the magnitude dimension profile and find the maximum dimension.
    mag_dim_profile, _ = mag_obj.get_magnitude_dimension_profile(exact=exact)
    
    if mag_dim_profile is None or mag_dim_profile.size == 0:
        return np.nan
        
    return float(np.max(mag_dim_profile))