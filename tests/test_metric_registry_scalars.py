"""`_to_scalar` chooses, and cross-modal metrics refuse what they cannot measure."""
import numpy as np
import pytest

from manylatents.metrics.registry import _to_scalar, get_metric, list_metrics


def test_to_scalar_honours_the_declared_key():
    assert _to_scalar({"alpha": 1.0, "mean_residual": 2.0}, "mean_residual") == 2.0


def test_to_scalar_falls_back_only_when_unambiguous():
    """One numeric value means there is nothing to choose between."""
    assert _to_scalar({"only": 3.0, "note": "text"}) == 3.0


def test_to_scalar_refuses_to_guess():
    """Two numeric keys and no declaration used to return whichever was written first.

    That is how `connected_components` recorded 66.667 — the mean component SIZE — for a
    3-component graph, under a description promising the count.
    """
    with pytest.raises(ValueError, match="Ambiguous"):
        _to_scalar({"a": 1.0, "b": 2.0})


def test_dict_metrics_declare_their_scalar():
    """Every dict-returning metric reachable headless must resolve without guessing."""
    for name, expected in [("shepard_residual", "mean_residual"),
                           ("topology_descriptor", "effective_dim"),
                           ("loglog_consistency", "mean_r_squared"),
                           ("mismatch_ratio", "mean_v"),
                           ("outlier_score", "mean")]:
        assert get_metric(name).scalar_key == expected, name


def test_cross_modal_metrics_refuse_a_single_array():
    """They compare two or more embeddings; handed one they returned exactly 1.0.

    A declared suite then records a perfect score for a real embedding, a graph, a label
    column and iid noise alike, and nothing downstream can tell a constant from a
    measurement.
    """
    x = np.random.default_rng(0).standard_normal((40, 3))
    aliases = ["cka_linear", "CKA", "neighborhood_jaccard", "cross_modal_overlap",
               "lid_rank_agreement", "rank_correlation", "alignment", "modal_alignment"]
    for alias in [a for a in aliases if a in list_metrics()]:
        with pytest.raises(ValueError):
            get_metric(alias)(embeddings=x, dataset=None, module=None, cache=None)


def test_cross_modal_dict_path_still_measures():
    a = np.random.default_rng(1).standard_normal((40, 3))
    b = np.random.default_rng(2).standard_normal((40, 3))
    same = get_metric("cka_linear")(embeddings={"a": a, "b": a.copy()},
                                    dataset=None, module=None, cache=None)
    diff = get_metric("cka_linear")(embeddings={"a": a, "b": b},
                                    dataset=None, module=None, cache=None)
    assert float(next(iter(same.values()))) == pytest.approx(1.0, abs=1e-6)
    assert float(next(iter(diff.values()))) < 0.3
