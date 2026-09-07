"""Analytic evidence and refusal contracts; no training needed for rank fixtures."""
import numpy as np
import pytest

from manylatents.metrics.score_jacobian_id import (
    FLIPD, ScoreDimensionEvidence, ScoreJacobianID, _largest_gap_cut,
)
from manylatents.utils.exceptions import MeasurementUnavailable


class _AnalyticPancake:
    def __init__(self, D=4, m=4, var=25.0, sigma=0.05):
        self.sigma_min = sigma
        self._is_fitted = True
        self.variance = np.array([var] * m + [0.] * (D - m), np.float32)
        self.probed_sigmas = []

    def _standardize(self, X):
        return np.asarray(X, np.float32)

    def score(self, Z, sigma, standardized=True):
        self.probed_sigmas.append(sigma)
        return -np.asarray(Z, np.float32) / (self.variance + sigma**2)

    def score_tensor(self, Z, sigma):
        import torch
        return -Z / (torch.as_tensor(self.variance, device=Z.device) + sigma**2)


def _legacy_values(module, X):
    """Reproduce bad behaviour only as a diagnostic, never via the metric."""
    evidence = ScoreDimensionEvidence(module, X, 0.05, 128, 0)
    return np.array([X.shape[1] - _largest_gap_cut(sv, 1e-2) for sv in evidence.singular_values])


@pytest.mark.parametrize("n,expected", [(1, 2), (16, 1.625)])
def test_legacy_origin_failure_evidence_and_refusal(n, expected):
    X = np.zeros((n, 4), np.float32)
    for rank in (0, 4):
        module = _AnalyticPancake(m=rank)
        assert _legacy_values(module, X).mean() == expected
        with pytest.raises(MeasurementUnavailable, match="unsound"):
            ScoreJacobianID(X, module=module, estimator_policy="largest_log_gap")


def test_legacy_gaussian_draws_are_not_repeated_origins():
    X = np.random.default_rng(0).normal(0, 5, (8, 4)).astype(np.float32)
    np.testing.assert_array_equal(_legacy_values(_AnalyticPancake(), X), 3)


@pytest.mark.parametrize("d", [2, 4, 10])
def test_legacy_cut_cannot_express_endpoints(d):
    for sv in (np.ones(d), np.geomspace(100, 1e-6, d)):
        assert 0 < d - _largest_gap_cut(sv, 1e-2) < d


@pytest.mark.parametrize("rank", range(5))
def test_no_policy_refuses_every_analytic_rank(rank):
    with pytest.raises(MeasurementUnavailable, match="caller-supplied estimator_policy"):
        ScoreJacobianID(np.zeros((16, 4)), module=_AnalyticPancake(m=rank))


@pytest.mark.parametrize("rank", range(5))
@pytest.mark.parametrize("exact_max_dim", [0, 32])
def test_flipd_all_analytic_ranks_across_scales(rank, exact_max_dim):
    # The callback is a fixture-specific finite-scale readout, not a shipped
    # dimension/scale policy. Verify every scale before choosing one explicitly.
    scales = np.array([0.2, 0.1, 0.05])
    expected = rank * 25 / (25 + scales**2)

    def read_fixture(values, supplied_scales):
        np.testing.assert_array_equal(supplied_scales, scales)
        np.testing.assert_allclose(values, np.tile(expected, (2, 1)), atol=1e-6)
        return values[:, -1]

    result = ScoreJacobianID(
        np.zeros((2, 4)), module=_AnalyticPancake(m=rank), return_per_sample=True,
        estimator_policy=FLIPD(scales, read_fixture, exact_max_dim=exact_max_dim, trace_samples=4),
    )
    np.testing.assert_allclose(result, expected[-1], atol=1e-6)
    assert abs(result.mean() - rank) < 0.0005


def test_flipd_includes_score_norm_away_from_origin():
    X = np.array([[1., 2., 3., 4.]], np.float32)
    sigma = 0.2
    covariance = 25 + sigma**2
    expected = 4 + sigma**2 * (-4 / covariance + (X**2).sum() / covariance**2)
    result = ScoreJacobianID(X, module=_AnalyticPancake(), estimator_policy=FLIPD([sigma], lambda values, scales: values[:, 0]))
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("policy", ["flipd", FLIPD([0.05], None)])
def test_flipd_requires_scale_interpretation(policy):
    with pytest.raises(MeasurementUnavailable, match="scale_policy"):
        ScoreJacobianID(np.zeros((1, 4)), module=_AnalyticPancake(), estimator_policy=policy)


@pytest.mark.parametrize("scales", [[], [0], [-1], [np.nan], [[0.05]]])
def test_flipd_requires_valid_explicit_scales(scales):
    with pytest.raises(MeasurementUnavailable, match="positive noise scales"):
        ScoreJacobianID(np.zeros((1, 4)), module=_AnalyticPancake(), estimator_policy=FLIPD(scales, lambda v, s: v[:, 0]))


@pytest.mark.parametrize("per_sample", [False, True])
def test_empty_cohort_refuses(per_sample):
    with pytest.raises(MeasurementUnavailable, match="nonempty"):
        ScoreJacobianID(np.zeros((0, 4)), module=_AnalyticPancake(), estimator_policy=lambda e: np.zeros(0), return_per_sample=per_sample)


@pytest.mark.parametrize("sigma,expected", [(None, 0.17), (0.03, 0.03)])
def test_spectral_policy_receives_model_noise_floor_or_override(sigma, expected):
    module = _AnalyticPancake(sigma=0.17)

    def owner_policy(evidence):
        assert evidence.sigma == expected
        assert evidence.ambient_dim == 4
        assert evidence.singular_values.shape == (3, 4)
        assert evidence.singular_values is evidence.singular_values
        # Stub policy proves aggregation/plumbing and lack of endpoint clipping.
        return np.array([0., 2., 4.])

    assert ScoreJacobianID(np.zeros((3, 4)), module=module, sigma=sigma, estimator_policy=owner_policy) == 2
    assert module.probed_sigmas == [expected]


def test_requires_fitted_module():
    with pytest.raises(MeasurementUnavailable, match="fitted"):
        ScoreJacobianID(np.zeros((1, 4)), estimator_policy=lambda e: np.zeros(1))


@pytest.mark.parametrize("result", [np.array([np.nan]), np.array([np.inf]), 2., np.zeros((1, 1))])
def test_policy_invalid_output_refuses(result):
    with pytest.raises(MeasurementUnavailable, match="one finite dimension"):
        ScoreJacobianID(np.zeros((1, 4)), module=_AnalyticPancake(), estimator_policy=lambda e: result)


def test_policy_refusal_propagates():
    def policy(evidence):
        raise MeasurementUnavailable("No stable scale regime in this cohort.")
    with pytest.raises(MeasurementUnavailable, match="stable scale regime"):
        ScoreJacobianID(np.zeros((1, 4)), module=_AnalyticPancake(), estimator_policy=policy)


@pytest.mark.parametrize("broken", [None, lambda z, sigma: z.detach(), lambda z, sigma: z.detach().requires_grad_()])
def test_flipd_requires_connected_differentiable_score(broken):
    module = _AnalyticPancake()
    module.score_tensor = broken
    with pytest.raises(MeasurementUnavailable, match="differentiable|disconnected"):
        ScoreJacobianID(np.zeros((1, 4)), module=module, estimator_policy=FLIPD([0.05], lambda v, s: v[:, 0]))


def test_fitted_score_diffusion_flipd_end_to_end():
    import torch
    from manylatents.algorithms.generative.score_diffusion import ScoreDiffusionModule

    X = np.random.default_rng(4).normal(size=(12, 4)).astype(np.float32)
    model = ScoreDiffusionModule(hidden=8, depth=1, n_fourier=2, epochs=1, device="cpu").fit(X)
    Z = model._standardize(X[:2])
    z = torch.tensor(Z, requires_grad=True)
    sigma = 0.1
    score = model.score_tensor(z, sigma)
    np.testing.assert_allclose(score.detach(), model.score(Z, sigma), rtol=1e-6, atol=1e-6)
    jacobian = torch.autograd.functional.jacobian(lambda t: model.score_tensor(t, sigma), z)
    expected = np.array([
        4 + sigma**2 * (float(jacobian[i, :, i, :].trace()) + float(score[i].detach().square().sum()))
        for i in range(2)
    ])
    model.net.zero_grad(set_to_none=True)
    with torch.no_grad():
        result = ScoreJacobianID(X[:2], module=model, return_per_sample=True, estimator_policy=FLIPD([sigma], lambda v, s: v[:, 0]))
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert all(p.grad is None for p in model.net.parameters())


def test_config_and_registry_refuse_without_policy():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from pathlib import Path
    from manylatents.metrics.registry import get_metric

    path = Path(__file__).parents[1] / "manylatents/configs/metrics/score_jacobian_id.yaml"
    config = OmegaConf.load(path).score_jacobian_id
    del config["at"]
    for metric in (instantiate(config), get_metric("score_id")):
        with pytest.raises(MeasurementUnavailable, match="estimator_policy"):
            metric(np.zeros((1, 4)), module=_AnalyticPancake())


def test_flipd_stochastic_trace_off_diagonal_is_seeded_and_converges():
    import torch

    class CorrelatedScore(_AnalyticPancake):
        def score_tensor(self, Z, sigma):
            precision = torch.eye(4, dtype=Z.dtype)
            precision[0, 1] = precision[1, 0] = 0.4
            return -Z @ precision

    policy = FLIPD([0.5], lambda values, scales: values[:, 0], exact_max_dim=0, trace_samples=1024)
    kwargs = dict(module=CorrelatedScore(), estimator_policy=policy, seed=12)
    first = ScoreJacobianID(np.zeros((1, 4)), **kwargs)
    assert ScoreJacobianID(np.zeros((1, 4)), **kwargs) == first
    assert first == pytest.approx(3., abs=0.02)
    second = ScoreJacobianID(np.zeros((1, 4)), **{**kwargs, "seed": 13})
    assert second != first


def test_policy_results_are_not_clipped():
    result = ScoreJacobianID(np.zeros((2, 4)), module=_AnalyticPancake(),
                            estimator_policy=lambda e: np.array([-0.01, 4.01]), return_per_sample=True)
    np.testing.assert_array_equal(result, [-0.01, 4.01])


def test_flipd_recovers_both_endpoints_the_gap_rule_cannot_express():
    """THE REASON THIS METRIC CHANGED, pinned so it cannot quietly regress.

    `_largest_gap_cut` always takes an INTERNAL gap, so with K >= D and D > 1 it
    can only return 1..D-1 — dimension 0 and dimension D are not expressible by
    construction, and clipping afterwards cannot restore them. Worse, at the
    origin these two score fields differ only by a positive scalar, which shifts
    every log singular value equally and leaves every log gap unchanged: measured
    on this fixture the old rule returned 1.625 for BOTH, a point mass and a
    4-dimensional Gaussian.

    FLIPD reads dimension from behaviour across noise scales instead of forcing a
    cut, so the magnitude information the gap rule discards is exactly what it
    uses. The tolerance is loose on purpose: the analytic value for the Gaussian
    is D - D*sigma^2/(var + sigma^2), which is 3.9996 here rather than 4, and
    pinning 4 exactly would be asserting something false.
    """
    sigma = 0.05
    at_origin = np.zeros((1, 4), np.float32)
    policy = FLIPD([sigma], lambda values, scales: values[:, 0])

    point_mass = _AnalyticPancake(D=4, m=0, var=0.0, sigma=sigma)
    gaussian = _AnalyticPancake(D=4, m=4, var=25.0, sigma=sigma)

    got_point = float(np.asarray(ScoreJacobianID(
        at_origin, module=point_mass, return_per_sample=True,
        estimator_policy=policy)).ravel()[0])
    got_gauss = float(np.asarray(ScoreJacobianID(
        at_origin, module=gaussian, return_per_sample=True,
        estimator_policy=policy)).ravel()[0])

    assert got_point == pytest.approx(0.0, abs=1e-3), "a point mass is 0-dimensional"
    assert got_gauss == pytest.approx(4.0 - 4.0 * sigma**2 / (25.0 + sigma**2), abs=1e-3)
    assert abs(got_gauss - got_point) > 3.0, (
        "the two must be distinguishable; the largest-log-gap rule gave both 1.625")
