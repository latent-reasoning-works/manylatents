"""Execute the metric/algorithm configs whose CI failures escaped composition tests.

These use the smoke sweep's 100-point, 50-dimensional Swiss roll through run(),
which shares run_experiment/evaluate with the CLI. Workers are disabled for the
sandbox; seed is fixed so the matrix evidence is reproducible.
"""
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import manylatents.configs  # noqa: F401 — register Hydra schema
from manylatents.api import run
from manylatents.metrics.diffusion_spectral_entropy import DiffusionSpectralEntropy
from manylatents.utils.exceptions import MeasurementUnavailable


def _smoke_config(metric, algorithm="pca", neighborhood_size=None):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "manylatents/configs"),
        version_base="1.3",
    ):
        return compose(config_name="config", overrides=[
            f"algorithms/latent={algorithm}", "data=swissroll",
            "data.n_distributions=5", "data.n_points_per_distribution=20",
            "data.rotate_to_dim=50", "data.num_workers=0", "seed=42",
            f"metrics={metric}", "callbacks/embedding=minimal", "logger=none",
            f"neighborhood_size={neighborhood_size if neighborhood_size is not None else 'null'}",
        ])


@pytest.fixture
def smoke_data_kwargs():
    cfg = _smoke_config("dse_knn")
    kwargs = OmegaConf.to_container(cfg.data, resolve=True)
    del kwargs["_target_"]
    del kwargs["random_state"]  # supplied by run(seed=42)
    return kwargs


@pytest.mark.parametrize("k", [None, 7])
def test_dse_knn_smoke_config_executes(smoke_data_kwargs, k):
    cfg = _smoke_config("dse_knn", neighborhood_size=k)
    assert cfg.metrics.dse_knn.k == k
    result = run(
        data="swissroll", data_kwargs=smoke_data_kwargs, seed=42,
        algorithm=instantiate(cfg.algorithms.latent), metrics=OmegaConf.to_container(cfg.metrics, resolve=True),
    )
    assert result["embeddings"].shape == (100, 2)
    scores = result["scores"]
    assert len(scores) == len(cfg.metrics.dse_knn.t_high) == 5
    expected = [
        DiffusionSpectralEntropy(
            result["embeddings"], k=15 if k is None else k,
            output_mode="eigenvalue_count", t_high=t,
        )
        for t in cfg.metrics.dse_knn.t_high
    ]
    np.testing.assert_allclose(list(scores.values()), expected)
    assert np.all(np.isfinite(expected))


def test_mismatch_smoke_pca_refuses_signed_covariance(smoke_data_kwargs):
    cfg = _smoke_config("mismatch_ratio")
    module = instantiate(cfg.algorithms.latent)
    with pytest.raises(MeasurementUnavailable, match="finite nonnegative weights"):
        run(
            data="swissroll", data_kwargs=smoke_data_kwargs, seed=42,
            algorithm=module, metrics=OmegaConf.to_container(cfg.metrics, resolve=True),
        )

    # Evidence from the actual fitted smoke module, not a fake affinity.
    W = module.affinity()
    centered = module._fit_data - module.model.mean_
    np.testing.assert_allclose(W, centered @ centered.T / 99)
    assert W.shape == (100, 100)
    assert np.all(np.isfinite(W))
    assert np.any(W < 0)
    np.testing.assert_allclose(W.sum(axis=1), 0, atol=1e-5)
    assert np.any(module.affinity(ignore_diagonal=True, use_symmetric=False) < 0)


def test_mismatch_smoke_phate_measures_real_neighborhoods(smoke_data_kwargs):
    cfg = _smoke_config("mismatch_ratio", algorithm="phate")
    module = instantiate(cfg.algorithms.latent)
    result = run(
        data="swissroll", data_kwargs=smoke_data_kwargs, seed=42,
        algorithm=module, metrics=OmegaConf.to_container(cfg.metrics, resolve=True),
    )
    W = module.affinity()
    assert W.shape == (100, 100)
    assert np.all(np.isfinite(W)) and np.all(W >= 0)
    np.testing.assert_allclose(W.sum(axis=1), 1, atol=1e-12)
    W = module.affinity(ignore_diagonal=True, use_symmetric=False)
    probabilities = W / W.sum(axis=1, keepdims=True)
    expected_keff = 1 / (probabilities ** 2).sum(axis=1)
    scores = result["scores"]
    np.testing.assert_allclose(scores["mismatch_ratio.k_eff"], expected_keff)
    np.testing.assert_allclose(
        scores["mismatch_ratio.v"], expected_keff / scores["mismatch_ratio.k_star"],
    )
    assert np.all(np.isfinite(scores["mismatch_ratio.v"]))
    assert np.all((expected_keff >= 1) & (expected_keff <= 99))
