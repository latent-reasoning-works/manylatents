"""Group-weighted average precision with a stratified bootstrap."""
import numpy as np
import pytest

from manylatents.metrics.grouped_average_precision import grouped_average_precision
from manylatents.utils.exceptions import MeasurementUnavailable

SCORES = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.6])
LABELS = np.array([1, 0, 1, 0, 0, 1])
GROUPS = np.array(["a", "a", "a", "a", "b", "b"])


def test_hand_computed_value():
    # group a: positives ranked 1st and 3rd -> (1/1 + 2/3) / 2 = 5/6, weight 4
    # group b: the positive is ranked 1st   -> 1.0,                   weight 2
    result = grouped_average_precision(SCORES, LABELS, GROUPS)
    assert result["per_group"] == pytest.approx({"a": 5 / 6, "b": 1.0})
    assert result["weights"] == {"a": 4, "b": 2}
    assert result["auprc"] == pytest.approx((4 * 5 / 6 + 2 * 1.0) / 6)
    assert result["se"] is None


def test_accepts_boolean_labels_and_is_order_invariant():
    order = np.array([5, 2, 0, 4, 1, 3])
    a = grouped_average_precision(SCORES, LABELS.astype(bool), GROUPS)
    b = grouped_average_precision(SCORES[order], LABELS[order], GROUPS[order])
    assert a["auprc"] == pytest.approx(b["auprc"])


def test_single_group_equals_plain_average_precision():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (rng.random(200) < 0.2).astype(int)
    result = grouped_average_precision(scores, labels, np.zeros(200, dtype=int))
    assert result["auprc"] == pytest.approx(average_precision_score(labels, scores))


def test_bootstrap_se_is_reproducible_and_shrinks_with_sample_size():
    def cohort(n, seed):
        rng = np.random.default_rng(seed)
        labels = (rng.random(n) < 0.2).astype(int)
        scores = labels + rng.normal(scale=1.0, size=n)
        return scores, labels, rng.integers(0, 4, size=n)

    small = grouped_average_precision(*cohort(200, 1), n_bootstrap=200,
                                      rng=np.random.default_rng(9))
    again = grouped_average_precision(*cohort(200, 1), n_bootstrap=200,
                                      rng=np.random.default_rng(9))
    large = grouped_average_precision(*cohort(5000, 1), n_bootstrap=200,
                                      rng=np.random.default_rng(9))
    assert small["se"] == again["se"]
    assert 0 < large["se"] < small["se"]


def test_group_with_one_class_is_refused_and_named():
    labels = np.array([1, 0, 1, 0, 0, 0])
    with pytest.raises(MeasurementUnavailable, match="b"):
        grouped_average_precision(SCORES, labels, GROUPS)


@pytest.mark.parametrize("bad", ["nan_score", "length", "non_binary", "no_rng"])
def test_invalid_input_is_refused(bad):
    scores, labels, groups, kwargs = SCORES.copy(), LABELS.copy(), GROUPS.copy(), {}
    if bad == "nan_score":
        scores[0] = np.nan
    elif bad == "length":
        groups = groups[:-1]
    elif bad == "non_binary":
        labels = np.array([2, 0, 1, 0, 0, 1])
    else:
        kwargs = {"n_bootstrap": 10}
    with pytest.raises((MeasurementUnavailable, ValueError)):
        grouped_average_precision(scores, labels, groups, **kwargs)


@pytest.mark.parametrize("n_bootstrap", [-1, 1, 2.5, True])
def test_bootstrap_requires_zero_or_at_least_two_integer_replicates(n_bootstrap):
    with pytest.raises(MeasurementUnavailable):
        grouped_average_precision(SCORES, LABELS, GROUPS, n_bootstrap=n_bootstrap, rng=np.random.default_rng(0))


def test_nonfinite_scores_identify_rows():
    scores = SCORES.copy()
    scores[2] = np.nan
    with pytest.raises(MeasurementUnavailable) as err:
        grouped_average_precision(scores, LABELS, GROUPS)
    assert err.value.indices.tolist() == [2]


def test_group_resampling_matches_a_direct_computation():
    # Resampling whole groups: each replicate is a weighted mean of the fixed
    # per-group values over a multiset of groups.
    result = grouped_average_precision(
        SCORES, LABELS, GROUPS, n_bootstrap=500, rng=np.random.default_rng(3),
        resample="groups",
    )
    rng = np.random.default_rng(3)
    ap = np.array([5 / 6, 1.0])
    weight = np.array([4.0, 2.0])
    replicates = []
    for _ in range(500):
        draw = rng.integers(0, 2, size=2)
        replicates.append((ap[draw] * weight[draw]).sum() / weight[draw].sum())
    assert result["se"] == pytest.approx(np.std(replicates, ddof=1))
    assert result["auprc"] == pytest.approx((4 * 5 / 6 + 2 * 1.0) / 6)


def test_group_resampling_differs_from_row_resampling():
    rng = np.random.default_rng(0)
    n = 3000
    groups = rng.integers(0, 6, size=n)
    labels = (rng.random(n) < 0.2).astype(int)
    # signal strength differs a lot between groups, so between-group spread
    # exceeds within-group sampling noise
    scores = labels * (groups / 2.0) + rng.normal(size=n)
    rows = grouped_average_precision(scores, labels, groups, n_bootstrap=300,
                                     rng=np.random.default_rng(1))
    whole = grouped_average_precision(scores, labels, groups, n_bootstrap=300,
                                      rng=np.random.default_rng(1), resample="groups")
    assert whole["auprc"] == rows["auprc"]
    assert whole["se"] > 2 * rows["se"]


def test_group_resampling_needs_two_groups_and_a_known_mode():
    scores, labels = np.array([0.9, 0.1, 0.8, 0.2]), np.array([1, 0, 1, 0])
    with pytest.raises(MeasurementUnavailable, match="two groups"):
        grouped_average_precision(scores, labels, np.zeros(4, dtype=int), n_bootstrap=10,
                                  rng=np.random.default_rng(0), resample="groups")
    with pytest.raises(ValueError, match="resample"):
        grouped_average_precision(SCORES, LABELS, GROUPS, n_bootstrap=10,
                                  rng=np.random.default_rng(0), resample="blocks")
