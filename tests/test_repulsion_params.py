"""UMAP repulsion_strength and t-SNE late_exaggeration / dof: trace to the optimizer, and direction of effect.

Trace tests wrap the library function that consumes each parameter and record what it receives:
- umap-learn: ``UMAP._fit_embed_data`` -> ``simplicial_set_embedding(..., gamma=repulsion_strength, ...)``,
  and ``gamma`` scales the repulsive gradient in ``umap.layouts.optimize_layout_euclidean``.
- openTSNE: ``TSNEEmbedding.optimize(exaggeration=..., dof=...)`` -> ``gradient_descent`` multiplies P by the
  exaggeration and calls ``kl_divergence_bh`` / ``kl_divergence_fft`` with ``dof``.

Direction tests use four well-separated Gaussian clusters and the ratio
separation = mean distance between cluster centres / mean distance of a point to its cluster centre.
"""
import numpy as np
import pytest
import torch

from manylatents.algorithms.latent.tsne import TSNEModule
from manylatents.algorithms.latent.umap import UMAPModule

SEEDS = [0, 1, 2]


@pytest.fixture(scope="module")
def clusters():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(4, 10)) * 8
    y = np.repeat(np.arange(4), 100)
    x = torch.tensor(centres[y] + rng.normal(size=(400, 10)), dtype=torch.float32)
    return x, y


def separation(emb, y):
    emb = np.asarray(emb)
    cen = np.stack([emb[y == c].mean(0) for c in np.unique(y)])
    within = np.mean([np.linalg.norm(emb[y == c] - cen[c], axis=1).mean() for c in np.unique(y)])
    between = np.mean([np.linalg.norm(cen[a] - cen[b]) for a in range(len(cen)) for b in range(a + 1, len(cen))])
    return between / within


# --- trace: the value reaches the function that uses it --------------------------------------------

@pytest.mark.parametrize("value, expected", [(None, 1.0), (3.0, 3.0)])
def test_umap_repulsion_strength_reaches_simplicial_set_embedding(monkeypatch, clusters, value, expected):
    import umap.umap_ as umap_mod

    seen = []
    original = umap_mod.simplicial_set_embedding

    def spy(*args, **kwargs):
        seen.append(args[6])  # positional order in UMAP._fit_embed_data: data, graph, n_components, alpha, a, b, gamma, ...
        return original(*args, **kwargs)

    monkeypatch.setattr(umap_mod, "simplicial_set_embedding", spy)
    x, _ = clusters
    UMAPModule(random_state=0, neighborhood_size=15, n_epochs=20, repulsion_strength=value).fit_transform(x)
    assert seen == [expected]


@pytest.mark.parametrize("late_exaggeration, dof", [(None, 1.0), (4.0, 0.5)])
def test_tsne_parameters_reach_gradient(monkeypatch, clusters, late_exaggeration, dof):
    """Record dof and sum(P) at every gradient evaluation. P sums to 1, so sum(P) is the exaggeration in force."""
    import openTSNE.tsne as tsne_mod

    calls = []
    for name in ("kl_divergence_bh", "kl_divergence_fft"):
        original = getattr(tsne_mod, name)

        def spy(embedding, P, dof, *args, _original=original, **kwargs):
            calls.append((float(dof), float(P.sum())))
            return _original(embedding, P, dof, *args, **kwargs)

        monkeypatch.setattr(tsne_mod, name, spy)
    x, _ = clusters
    n_early, n_late = 20, 30
    TSNEModule(random_state=0, neighborhood_size=30, n_iter_early=n_early, n_iter_late=n_late,
               late_exaggeration=late_exaggeration, dof=dof).fit_transform(x)
    # Each phase runs n_iter gradient steps, then divides P back and evaluates the final error once (sum(P) = 1).
    assert len(calls) == n_early + 1 + n_late + 1
    early, late = calls[:n_early], calls[n_early + 1:n_early + 1 + n_late]
    assert all(d == dof for d, _ in calls)
    np.testing.assert_allclose([s for _, s in early], 12.0, rtol=1e-6)
    np.testing.assert_allclose([s for _, s in late], late_exaggeration or 1.0, rtol=1e-6)
    np.testing.assert_allclose([calls[n_early][1], calls[-1][1]], 1.0, rtol=1e-6)


# --- direction: each parameter moves the layout the way the method says it should -------------------

def test_umap_repulsion_strength_spreads_clusters(clusters):
    """Stronger repulsion pushes points apart inside a cluster, so separation falls as repulsion_strength rises.

    Only the ends of the range are compared: UMAP layouts differ across platforms (numba, BLAS), and the step
    0.25 -> 1 was not ordered on the CI runner for seed 0. Local ratios sep(0.25) / sep(4) were 1.29 to 1.91.
    """
    x, y = clusters
    for s in SEEDS:
        weak, strong = (separation(UMAPModule(random_state=s, neighborhood_size=15, n_epochs=200, repulsion_strength=r).fit_transform(x), y)
                        for r in (0.25, 4.0))
        assert weak > 1.15 * strong, f"seed {s}: repulsion 0.25 {weak:.2f}, repulsion 4 {strong:.2f}"


def test_tsne_heavy_tails_separate_clusters(clusters):
    """dof < 1 gives a heavier-tailed output kernel (Kobak et al. 2019), so clusters separate more."""
    x, y = clusters
    for s in SEEDS:
        base = separation(TSNEModule(random_state=s, neighborhood_size=30).fit_transform(x), y)
        heavy = separation(TSNEModule(random_state=s, neighborhood_size=30, dof=0.5).fit_transform(x), y)
        assert heavy > 1.5 * base, f"seed {s}: dof=1 {base:.2f}, dof=0.5 {heavy:.2f}"


def test_tsne_late_exaggeration_compacts_clusters(clusters):
    """Exaggeration multiplies the attraction, so each cluster contracts relative to the distance between clusters."""
    x, y = clusters
    for s in SEEDS:
        base = separation(TSNEModule(random_state=s, neighborhood_size=30).fit_transform(x), y)
        exag = separation(TSNEModule(random_state=s, neighborhood_size=30, late_exaggeration=4.0).fit_transform(x), y)
        assert exag > 2.0 * base, f"seed {s}: exaggeration 1 {base:.2f}, exaggeration 4 {exag:.2f}"


def test_run_dict_route_reaches_optimizers(monkeypatch, clusters):
    """The same parameters set through ``manylatents.api.run(algorithms={...})`` arrive at the optimizers."""
    import openTSNE.tsne as tsne_mod
    import umap.umap_ as umap_mod
    from manylatents.api import run

    gammas, tsne_calls = [], []
    original_sse = umap_mod.simplicial_set_embedding
    monkeypatch.setattr(umap_mod, "simplicial_set_embedding",
                        lambda *a, **k: (gammas.append(a[6]), original_sse(*a, **k))[1])
    original_bh = tsne_mod.kl_divergence_bh
    monkeypatch.setattr(tsne_mod, "kl_divergence_bh",
                        lambda e, P, dof, *a, **k: (tsne_calls.append((float(dof), float(P.sum()))), original_bh(e, P, dof, *a, **k))[1])
    x, _ = clusters
    x = x.numpy()
    run(input_data=x, seed=0, algorithms={"latent": {
        "_target_": "manylatents.algorithms.latent.umap.UMAPModule", "n_components": 2, "neighborhood_size": 15,
        "n_epochs": 20, "random_state": 0, "repulsion_strength": 2.5}})
    run(input_data=x, seed=0, algorithms={"latent": {
        "_target_": "manylatents.algorithms.latent.tsne.TSNEModule", "n_components": 2, "neighborhood_size": 30,
        "n_iter_early": 10, "n_iter_late": 10, "random_state": 0, "late_exaggeration": 3.0, "dof": 0.7}})
    assert gammas == [2.5]
    assert {d for d, _ in tsne_calls} == {0.7}
    np.testing.assert_allclose([s for _, s in tsne_calls[11:21]], 3.0, rtol=1e-6)


def test_tsne_transform_uses_fitted_dof(monkeypatch, clusters):
    """transform() of new points optimizes them with the dof the model was fitted with."""
    import openTSNE.tsne as tsne_mod

    x, _ = clusters
    m = TSNEModule(random_state=0, neighborhood_size=30, n_iter_early=10, n_iter_late=10, dof=0.6)
    m.fit(x[:300])
    dofs = []
    for name in ("kl_divergence_bh", "kl_divergence_fft"):
        original = getattr(tsne_mod, name)
        monkeypatch.setattr(tsne_mod, name, lambda e, P, dof, *a, _o=original, **k: (dofs.append(float(dof)), _o(e, P, dof, *a, **k))[1])
    m.transform(x[300:])
    assert dofs and set(dofs) == {0.6}


def test_umap_init_array_reaches_optimizer(monkeypatch, clusters):
    """An init array reaches simplicial_set_embedding; the default stays umap-learn's 'spectral'."""
    import umap.umap_ as umap_mod
    from umap import UMAP as SklearnUMAP

    assert UMAPModule(n_neighbors=5, n_epochs=10).model.init == SklearnUMAP().init == "spectral"
    x, _ = clusters
    start = np.random.default_rng(0).normal(size=(len(x), 2))
    seen = []
    original = umap_mod.simplicial_set_embedding
    monkeypatch.setattr(umap_mod, "simplicial_set_embedding", lambda *a, **k: (seen.append(np.array(k.get("init", a[9] if len(a) > 9 else None))), original(*a, **k))[1])
    m = UMAPModule(random_state=0, neighborhood_size=15, n_epochs=10, init=start)
    m.fit_transform(x)
    np.testing.assert_allclose(seen[0], start)
    np.testing.assert_allclose(m.model.init, start)


def test_umap_init_array_row_mismatch_raises(clusters):
    """An init array must have one row per fitted sample; fit_fraction < 1 with an n-row init fails early."""
    x, _ = clusters
    start = np.zeros((len(x), 2))
    with pytest.raises(ValueError, match="init array has"):
        UMAPModule(n_neighbors=5, n_epochs=10, fit_fraction=0.5, init=start).fit(x)
