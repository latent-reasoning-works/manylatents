"""Spearman correlation after removing the rank-linear effect of covariates."""
import numpy as np
import pytest
from scipy.stats import spearmanr

from manylatents.utils.exceptions import MeasurementUnavailable
from manylatents.utils.stats import partial_spearman


def test_without_covariates_equals_spearman():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = x + rng.normal(size=300)
    result = partial_spearman(x, y, np.empty((300, 0)))
    expected = spearmanr(x, y)
    assert result["rho"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue, rel=1e-6)
    assert result["n"] == 300 and result["dof"] == 298


def test_shared_cause_is_removed():
    rng = np.random.default_rng(1)
    z = rng.normal(size=2000)
    x = z + 0.5 * rng.normal(size=2000)
    y = z + 0.5 * rng.normal(size=2000)
    assert spearmanr(x, y).statistic > 0.6
    result = partial_spearman(x, y, z)
    assert abs(result["rho"]) < 0.1
    assert result["dof"] == 1997


def test_direct_dependence_survives():
    rng = np.random.default_rng(2)
    z = rng.normal(size=2000)
    x = z + rng.normal(size=2000)
    y = x + z + rng.normal(size=2000)
    result = partial_spearman(x, y, z[:, None])
    assert result["rho"] > 0.3
    assert result["p_value"] < 1e-10


def test_monotone_transforms_do_not_change_the_result():
    rng = np.random.default_rng(3)
    z = rng.normal(size=500)
    x = z + rng.normal(size=500)
    y = x + rng.normal(size=500)
    a = partial_spearman(x, y, z)
    b = partial_spearman(np.exp(x), y ** 3, np.arctan(z))
    assert a["rho"] == pytest.approx(b["rho"])


@pytest.mark.parametrize("bad", ["nan", "length", "explained", "too_few"])
def test_invalid_input_is_refused(bad):
    rng = np.random.default_rng(4)
    x, y, z = rng.normal(size=50), rng.normal(size=50), rng.normal(size=50)
    if bad == "nan":
        z[3] = np.nan
    elif bad == "length":
        z = z[:-1]
    elif bad == "explained":
        x = 2.0 * z            # x is a monotone function of the covariate
    else:
        x, y, z = x[:3], y[:3], z[:3]
    with pytest.raises((MeasurementUnavailable, ValueError)):
        partial_spearman(x, y, z)
