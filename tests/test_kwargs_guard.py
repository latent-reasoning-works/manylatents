"""Tests for route_kwargs and its use in UMAPModule (issue #289)."""
import numpy as np
import pytest
import torch

from manylatents.utils.kwargs import accepted_params, route_kwargs


def _target(alpha=1, beta=2, *, gamma=3, **kwargs):
    return alpha, beta, gamma


def test_accepted_params_ignores_var_keyword():
    # gamma/alpha/beta accepted; the function's own **kwargs is NOT a licence to pass anything
    assert accepted_params(_target) == {"alpha", "beta", "gamma"}


def test_route_forwards_accepted_and_drops_nothing_silently():
    out = route_kwargs(_target, {"alpha": 10, "gamma": 30})
    assert out == {"alpha": 10, "gamma": 30}


def test_route_rejects_unknown_strict_with_hint():
    with pytest.raises(TypeError, match="unexpected parameter"):
        route_kwargs(_target, {"gama": 30})            # typo of gamma
    with pytest.raises(TypeError, match="gamma"):        # did-you-mean names the real param
        route_kwargs(_target, {"gama": 30})


def test_route_warns_when_not_strict():
    with pytest.warns(UserWarning, match="unexpected parameter"):
        out = route_kwargs(_target, {"nope": 1}, strict=False)
    assert out == {}                                     # unknown not forwarded even when warning


def test_route_allow_list():
    out = route_kwargs(_target, {"passthrough": 1}, allow=("passthrough",))
    assert out == {"passthrough": 1}


def test_route_empty():
    assert route_kwargs(_target, {}) == {}


# --- UMAPModule integration: the exact hole the guard closes ---

def test_umap_forwards_valid_downstream_param():
    """A valid umap-learn param not explicitly declared on UMAPModule (spread) is forwarded, not rejected."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(n_components=2, random_state=42, n_neighbors=5, n_epochs=10, spread=3.0)
    assert m.model.spread == 3.0

    x = torch.randn(80, 10, generator=torch.Generator().manual_seed(0))
    emb_default = UMAPModule(n_components=2, random_state=42, n_neighbors=10, n_epochs=50).fit_transform(x)
    emb_spread = UMAPModule(n_components=2, random_state=42, n_neighbors=10, n_epochs=50,
                            spread=5.0, min_dist=0.99).fit_transform(x)
    assert not np.allclose(emb_default, emb_spread, atol=1e-3), "spread should reach umap-learn and change the layout"


def test_umap_rejects_typo_with_hint():
    """A misspelled param raises with a did-you-mean, instead of silently defaulting."""
    from manylatents.algorithms.latent.umap import UMAPModule

    with pytest.raises(TypeError, match="min_dist"):
        UMAPModule(n_components=2, random_state=42, min_dsit=0.1)


def test_umap_rejects_wrong_module_param():
    """perplexity belongs to t-SNE, not UMAP — reject rather than swallow."""
    from manylatents.algorithms.latent.umap import UMAPModule

    with pytest.raises(TypeError, match="unexpected parameter"):
        UMAPModule(n_components=2, random_state=42, perplexity=30)
