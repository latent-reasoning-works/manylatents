"""Group-weighted average precision with a stratified bootstrap."""
import numpy as np
import pytest

from manylatents.metrics.grouped_average_precision import grouped_average_precision
from manylatents.metrics import grouped_average_precision_difference
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


CLUSTERS = np.array(["x", "x", "y", "y", "z", "z"])


def test_difference_observed_values_and_optional_clusters():
    result = grouped_average_precision_difference(
        SCORES, -SCORES, LABELS, GROUPS, n_bootstrap=20, rng=np.random.default_rng(0)
    )
    a = grouped_average_precision(SCORES, LABELS, GROUPS)
    b = grouped_average_precision(-SCORES, LABELS, GROUPS)
    assert result["auprc_a"] == a["auprc"]
    assert result["auprc_b"] == b["auprc"]
    assert result["difference"] == a["auprc"] - b["auprc"]
    assert result["groups_ahead"] == 2
    assert result["n_groups"] == 2
    assert result["ci95_clusters"] is None
    assert result["n_clusters"] is None


def test_difference_identical_scores_are_exactly_zero():
    result = grouped_average_precision_difference(
        SCORES, SCORES, LABELS, GROUPS, CLUSTERS, 30, np.random.default_rng(0)
    )
    assert result["difference"] == 0
    assert result["ci95_rows"] == [0, 0]
    assert result["ci95_clusters"] == [0, 0]
    assert result["groups_ahead"] == 0
    assert result["n_clusters"] == 3


def test_difference_better_score_and_seed_determinism():
    def compute():
        return grouped_average_precision_difference(
            LABELS, -LABELS, LABELS, GROUPS, CLUSTERS, 50, np.random.default_rng(9)
        )

    result = compute()
    assert result == compute()
    assert result["ci95_rows"][0] > 0
    assert result["ci95_clusters"][0] > 0


def test_difference_cluster_interval_wider_for_near_duplicate_rows():
    rng = np.random.default_rng(7)
    clusters = np.repeat(np.arange(12), 40)
    labels = np.tile(np.repeat([0, 1], 20), 12)
    # Each cluster repeats one pair of scores with tiny perturbations.
    strength = np.repeat(np.linspace(-2, 2, 12), 40)
    a = labels * strength + rng.normal(scale=0.001, size=labels.size)
    b = np.zeros(labels.size)
    result = grouped_average_precision_difference(
        a, b, labels, np.zeros(labels.size), clusters, 250, np.random.default_rng(3)
    )
    assert np.diff(result["ci95_clusters"])[0] > 2 * np.diff(result["ci95_rows"])[0]


def test_difference_tied_bootstraps_match_sklearn_with_variable_cluster_sizes():
    from sklearn.metrics import average_precision_score

    labels = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    groups = np.array([0] * 6 + [1] * 4)
    clusters = np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3])
    a = np.array([1, 1, 0, 1, 1, 0, 2, 2, 0, 1])
    b = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
    n = 40
    result = grouped_average_precision_difference(
        a, b, labels, groups, clusters, n, np.random.default_rng(5)
    )
    rng = np.random.default_rng(5)
    for mode in ("rows", "clusters"):
        differences = []
        for _ in range(n):
            samples = []
            for group in np.unique(groups):
                rows = np.flatnonzero(groups == group)
                if mode == "rows":
                    samples.append(np.concatenate([
                        rng.choice(rows[labels[rows] == cls], size=np.sum(labels[rows] == cls))
                        for cls in (0, 1)
                    ]))
                else:
                    keys = np.unique(clusters[rows])
                    drawn = rng.choice(keys, size=len(keys))
                    samples.append(np.concatenate([rows[clusters[rows] == key] for key in drawn]))
            values = [
                sum(len(rows) * average_precision_score(labels[rows], score[rows])
                    for rows in samples) / sum(map(len, samples))
                for score in (a, b)
            ]
            differences.append(values[0] - values[1])
        assert result[f"ci95_{mode}"] == pytest.approx(np.quantile(differences, [0.025, 0.975]))


@pytest.mark.parametrize("n_bootstrap", [0, 1, -1, 2.5, True, np.bool_(False)])
def test_difference_refuses_invalid_bootstrap_count(n_bootstrap):
    with pytest.raises(MeasurementUnavailable, match="n_bootstrap"):
        grouped_average_precision_difference(
            SCORES, SCORES, LABELS, GROUPS, n_bootstrap=n_bootstrap, rng=np.random.default_rng(0)
        )


@pytest.mark.parametrize("rng", [None, 42, np.random.RandomState(0)])
def test_difference_requires_generator(rng):
    with pytest.raises(MeasurementUnavailable, match="Generator"):
        grouped_average_precision_difference(SCORES, SCORES, LABELS, GROUPS, rng=rng)


@pytest.mark.parametrize("which", ["a", "b"])
def test_difference_refuses_nonfinite_scores(which):
    bad = SCORES.copy()
    bad[2] = np.nan
    a, b = (bad, SCORES) if which == "a" else (SCORES, bad)
    with pytest.raises(MeasurementUnavailable) as err:
        grouped_average_precision_difference(a, b, LABELS, GROUPS, rng=np.random.default_rng(0))
    assert err.value.indices.tolist() == [2]


@pytest.mark.parametrize("clusters, message", [
    (["x", "x", "y", "y", "x", "x"], "spanning groups:.*x"),
    (["positive", "negative", "positive", "negative", "z", "z"], "without both classes:.*negative.*positive"),
    ([0, 0, 1, 1, np.nan, np.nan], "missing keys"),
])
def test_difference_refuses_and_names_invalid_clusters(clusters, message):
    with pytest.raises(MeasurementUnavailable, match=message):
        grouped_average_precision_difference(
            SCORES, SCORES, LABELS, GROUPS, clusters, rng=np.random.default_rng(0)
        )


@pytest.mark.parametrize("labels", [[1, 0, 1, 0, 0, 0], [2, 0, 1, 0, 0, 1]])
def test_difference_reuses_label_and_group_validation(labels):
    with pytest.raises(MeasurementUnavailable):
        grouped_average_precision_difference(SCORES, SCORES, labels, GROUPS, rng=np.random.default_rng(0))


def test_difference_refuses_misaligned_clusters():
    with pytest.raises(ValueError, match="clusters"):
        grouped_average_precision_difference(
            SCORES, SCORES, LABELS, GROUPS, CLUSTERS[:-1], rng=np.random.default_rng(0)
        )
