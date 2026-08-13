"""Tests for API resolution without Hydra."""
import numpy as np
import pytest


def test_resolve_datamodule_by_name():
    """String name resolves via data registry, not Hydra."""
    from manylatents.api import _resolve_datamodule
    dm = _resolve_datamodule(data="swissroll")
    assert hasattr(dm, "setup")
    assert hasattr(dm, "train_dataloader")


def test_resolve_datamodule_from_array():
    """numpy array wraps in PrecomputedDataModule."""
    from manylatents.api import _resolve_datamodule
    arr = np.random.randn(50, 3).astype(np.float32)
    dm = _resolve_datamodule(input_data=arr)
    dm.setup()
    assert dm.train_dataset is not None


def test_resolve_datamodule_forwards_generator_kwargs():
    """Generator params reach the DataModule constructor.

    Regression: ``run()`` accepted them into ``**kwargs`` and never forwarded, so every
    parameterisation of a named dataset silently produced the DEFAULT dataset. Two configs
    that differ only in ``data_kwargs`` then yielded byte-identical embeddings, which reads
    downstream as zero within-group variance rather than as a dropped argument.
    """
    from manylatents.api import _resolve_datamodule
    dm = _resolve_datamodule(data="gaussian_blob", n_samples=250, centers=4, cluster_std=0.4)
    assert dm.n_samples == 250
    assert dm.centers == 4
    assert dm.cluster_std == 0.4


def test_resolve_datamodule_rejects_unknown_kwarg():
    """An unknown generator param raises rather than being dropped."""
    from manylatents.api import _resolve_datamodule
    with pytest.raises(TypeError, match="not_a_real_param"):
        _resolve_datamodule(data="gaussian_blob", not_a_real_param=1)


def test_run_data_kwargs_changes_the_data():
    """End to end: `data_kwargs` is the channel, and it actually changes the point cloud."""
    from manylatents.api import run
    small = run(data="gaussian_blob", algorithm="pca", data_kwargs={"n_samples": 120})
    large = run(data="gaussian_blob", algorithm="pca", data_kwargs={"n_samples": 480})
    assert np.asarray(small["embeddings"]).shape[0] == 120
    assert np.asarray(large["embeddings"]).shape[0] == 480


def test_fit_fraction_survives_the_full_mode_shortcut():
    """`fit_fraction < 1` must keep working through `run()`.

    Regression introduced by the row-alignment fix: the `mode='full'` shortcut called
    `fit_transform` once, and PHATE/TSNE override it to embed only the fitted subset — which
    then tripped the new row-cardinality postcondition and turned a shipped, documented
    parameter into a hard ValueError. Those modules keep the fit-then-transform path, where
    `transform` extends the embedding back over every row.
    """
    from manylatents.algorithms.latent.phate import PHATEModule
    from manylatents.algorithms.latent.tsne import TSNEModule
    from manylatents.api import run

    x = np.random.default_rng(0).standard_normal((120, 6)).astype(np.float32)
    for mod in (PHATEModule(fit_fraction=0.5), TSNEModule(fit_fraction=0.5)):
        out = run(input_data=x, algorithm=mod)
        assert np.asarray(out["embeddings"]).shape[0] == 120


def test_mds_fit_transform_then_transform_is_allowed():
    """The remedy the transductive error message prescribes must actually work.

    `MDSModule.fit_transform` overrides the base `fit(); transform()`, so it has to record the
    fit fingerprint itself — otherwise `fit_transform(X)` followed by `transform(X)` raised,
    telling the caller to call `fit_transform`.
    """
    from manylatents.algorithms.latent import get_algorithm

    x = np.random.default_rng(0).standard_normal((60, 5)).astype(np.float32)
    m = get_algorithm("mds")()
    first = m.fit_transform(x)
    again = m.transform(x)
    assert np.asarray(first).shape == np.asarray(again).shape


def test_reeb_graph_width_is_deterministic():
    """The embedding width must be a function of the inputs, not of an unseeded RNG.

    Materializing the filtration exposed gudhi's sparse-Rips approximation, whose RNG this
    code never seeded: the column count (one per Reeb node) varied 12/12/12/11 across calls at
    a fixed seed in one process. A feature matrix whose width is not reproducible cannot be
    stacked into a results table or chained into a next step.
    """
    from manylatents.api import run

    widths = {
        np.asarray(run(data="gaussian_blob", algorithms={"latent": "reeb_graph"}, seed=42,
                       data_kwargs={"n_samples": 80})["embeddings"]).shape[1]
        for _ in range(3)
    }
    assert len(widths) == 1, f"reeb_graph width varies across identical calls: {widths}"


def test_subset_view_refuses_unrealigned_attributes():
    """A Subset view must not hand metrics full-dataset arrays behind a passing guard.

    Metrics gate on `hasattr(dataset, 'get_gt_dists')` and `assert hasattr(...)`. If the view
    forwards those unsliced, the guard passes and the metric indexes N_full rows against
    N_subset embeddings — either an IndexError or, worse, a plausible number computed against
    the wrong ground truth. That is the exact bug class the view was added to fix.
    """
    from manylatents.experiment import _SubsetView

    class Base:
        data = np.arange(40).reshape(10, 4)

        def get_labels(self):
            return np.arange(10)

        def get_gt_dists(self):
            return np.arange(100).reshape(10, 10)

        def unrelated(self):
            return "not row-indexed"

    class FakeSubset:
        dataset = Base()
        indices = [1, 3, 5]

        def __len__(self):
            return 3

    v = _SubsetView(FakeSubset())
    assert v.data.shape == (3, 4)
    assert v.get_labels().tolist() == [1, 3, 5]
    assert v.get_gt_dists().shape == (3, 3)          # both axes
    with pytest.raises(AttributeError, match="not row-realigned"):
        v.unrelated


def test_fit_tensor_is_unshuffled_so_rows_align():
    """The recurring defect, end to end: fit order must match eval order.

    `run_experiment` sourced its fit array from `train_dataloader()` (which shuffles by
    default) and its eval array from `test_dataloader()` (which does not) — and in the default
    `mode='full'` those are the SAME rows. Every LatentModule was therefore fitted on a
    permutation of the array it was then asked to transform, so any module whose output is
    defined on its fit rows returned the right shape with the wrong pairing. Measured before
    the fix: `leiden` AMI -0.0090, `diffusion_map(mode='cluster')` ARI +0.0086.

    Note every datamodule's YAML says `shuffle_traindata: false` while its Python default says
    True, so the Hydra CLI never hit this and the programmatic API always did.
    """
    from sklearn.metrics import adjusted_mutual_info_score

    from manylatents.api import run
    out = run(data="gaussian_blob", algorithm="leiden",
              data_kwargs={"n_samples": 200, "n_features": 10, "centers": 4,
                           "shuffle_traindata": True})
    labels = np.asarray(out["label"]).ravel()
    pred = np.asarray(out["embeddings"]).ravel()
    assert adjusted_mutual_info_score(labels, pred) > 0.9


def test_row_cardinality_postcondition():
    """A latent module must emit one row per input row, whatever kind of output it is.

    This is the axis that catches a module returning stored fit-time state, and it needs no
    per-module cooperation and no declaration on the ABC.
    """
    import torch

    from manylatents.algorithms.latent.latent_module_base import LatentModule

    class Liar(LatentModule):
        """Returns a row count unrelated to its input — the shape every module that hands
        back stored fit-time state takes when the eval array is a different length."""

        def fit(self, x, y=None):
            self._is_fitted = True

        def transform(self, x):
            return torch.zeros((x.shape[0] + 5, 2))

        def fit_transform(self, x, y=None):
            self.fit(x, y)
            return self.transform(x)

    from manylatents.api import run
    with pytest.raises(ValueError, match="one row per input row"):
        run(data="gaussian_blob", algorithm=Liar(), data_kwargs={"n_samples": 60})


def test_unknown_constructor_kwarg_raises():
    """A misspelled parameter must not be indistinguishable from its default.

    `PCAModule(n_compnents=7).n_components` was 2, so two sweep arms differing only in a typo
    produced byte-identical results while looking like different configurations.
    """
    from manylatents.algorithms.latent import get_algorithm
    with pytest.raises(TypeError, match="n_compnents"):
        get_algorithm("pca")(n_compnents=7)


def test_algorithm_kwargs_reach_the_constructor():
    """`n_components` used to be dropped between `run` and the algorithm."""
    from manylatents.api import run
    out = run(data="gaussian_blob", algorithm="pca", n_components=5,
              data_kwargs={"n_samples": 100, "n_features": 10})
    assert np.asarray(out["embeddings"]).shape[1] == 5


def test_transductive_modules_do_not_return_stale_rows():
    """A transductive module must refuse to `transform` rows it was not fitted on.

    Regression, and the worst defect the component audit found because it is invisible:
    `LeidenModule.transform` returned `self._labels` regardless of its argument. Since
    `run_experiment` fits the (shuffled) train tensor and transforms the (unshuffled) test
    tensor, and every named datamodule defaults `shuffle_traindata=True`, the labels came back
    in fit order paired with test-order rows — a silent permutation with the correct shape and
    no exception. Measured before the fix: ARI against ground truth -0.0012 shuffled versus
    +1.0000 unshuffled.

    Raising NotImplementedError is sufficient: `run_experiment` already catches it and falls
    back to `fit_transform` on the array actually being embedded.
    """
    from manylatents.algorithms.latent import get_algorithm
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 4)).astype(np.float32)

    for name in ("leiden", "mds"):
        mod = get_algorithm(name)()
        mod.fit(x)
        same = mod.transform(x)                     # the fitted rows: allowed
        assert same.shape[0] == 60
        with pytest.raises(NotImplementedError):    # a PERMUTATION of them: refused
            mod.transform(x[rng.permutation(60)])


def test_leiden_labels_align_with_ground_truth_when_shuffled():
    """End to end: the shuffled train loader no longer permutes the output."""
    from sklearn.metrics import adjusted_rand_score

    from manylatents.api import run
    out = run(data="gaussian_blob", algorithm="leiden",
              data_kwargs={"n_samples": 300, "n_features": 10, "centers": 3,
                           "shuffle_traindata": True})
    labels = np.asarray(out["label"]).ravel()
    pred = np.asarray(out["embeddings"]).ravel()
    assert adjusted_rand_score(labels, pred) > 0.9


def test_reeb_graph_filtration_is_materialized():
    """The Vietoris-Rips 1- and 2-skeletons are non-empty.

    Regression: `filt = st.get_filtration()` bound a generator that three comprehensions then
    iterated; the first exhausted it, so `one_skel` and `two_skel` were ALWAYS empty and the
    Reeb graph had no real edges.
    """
    from manylatents.algorithms.latent.reeb_graph import _vietoris_rips
    from scipy.spatial.distance import squareform, pdist
    rng = np.random.default_rng(0)
    D = squareform(pdist(rng.standard_normal((80, 3))))
    _, (zero, one, two) = _vietoris_rips(D)
    assert len(zero) == 80
    assert len(one) > 0, "1-skeleton empty — the filtration generator was consumed"


def test_lightning_algorithms_are_listable():
    """The lightning group has a listing function at all.

    Regression: `algorithms/lightning/__init__.py` was a single comment, so
    `algorithms.latent.list_algorithms()` was the only enumeration available and read like a
    catalogue of the whole engine. A consumer building its catalog from that one call could
    not see mioflow, cflows, latent_ode or either reconstruction config — and concluded from
    its absence that archetypal analysis did not exist in manylatents.
    """
    from manylatents.algorithms.lightning import list_algorithms
    names = list_algorithms()
    assert "aanet_reconstruction" in names          # archetypal analysis
    assert "mioflow" in names and "cflows" in names
    assert "default" not in names                   # a composition stub, not an algorithm


def test_resolve_lightning_by_name():
    """A string under the `lightning` key resolves the packaged config.

    Regression: the registry lookup in `_resolve_algorithm` was gated on
    `algo_type == "latent"`, so every lightning string fell through to
    `ValueError: Algorithm '<name>' not found in registry`. Nothing in the lightning group was
    reachable by name through the public API — only a hand-written `_target_` dict worked,
    which required the caller to already know the network/loss/optimizer wiring that the
    packaged configs exist to express.
    """
    from lightning import LightningModule

    from manylatents.api import _resolve_datamodule, _resolve_algorithm
    dm = _resolve_datamodule(data="gaussian_blob", n_samples=60)
    algo = _resolve_algorithm(algorithms={"lightning": "ae_reconstruction"}, datamodule=dm)
    assert isinstance(algo, LightningModule)
    assert algo.datamodule is dm                    # `${data}` was overridden, not resolved


def test_resolve_lightning_unknown_name_still_raises():
    from manylatents.api import _resolve_algorithm
    with pytest.raises(ValueError, match="not found in registry"):
        _resolve_algorithm(algorithms={"lightning": "no_such_algorithm"})


def test_run_lightning_by_name_end_to_end():
    """A named lightning algorithm trains and returns embeddings through `run`."""
    from manylatents.api import run
    out = run(data="gaussian_blob", algorithms={"lightning": "ae_reconstruction"},
              data_kwargs={"n_samples": 60})
    assert np.asarray(out["embeddings"]).shape[0] > 0


def test_lightning_string_form_forwards_kwargs():
    """A declared parameter reaches the module on `algorithms={'lightning': ...}`.

    Regression: the lightning branch called `_instantiate_lightning(cfg, datamodule)` and
    never passed `**kwargs`, so the packaged yaml won every argument the caller named.
    Measured on the broken code, `algorithms={'lightning': 'mioflow'}` with
    `n_global_epochs=3, lambda_energy=0.5, n_bins=7` trained mioflow.yaml's 100 / 0.01 / 100.
    The two latent string forms above (api.py:231, api.py:274 — the two `algo_kwargs =
    dict(kwargs)` lines) each already forward, after
    two earlier rounds of exactly this bug — this key was the one added later and missed.
    """
    from manylatents.api import _resolve_algorithm, _resolve_datamodule
    dm = _resolve_datamodule(data="gaussian_blob", n_samples=60)
    algo = _resolve_algorithm(algorithms={"lightning": "mioflow"}, datamodule=dm,
                              n_global_epochs=3, lambda_energy=0.5, n_bins=7)
    assert (algo.n_global_epochs, algo.lambda_energy, algo.n_bins) == (3, 0.5, 7)


def test_lightning_unknown_kwarg_raises_rather_than_vanishing():
    """A misspelling must be distinguishable from a default, as it is for latent modules.

    `{'latent': 'pca'}` with a bogus key raises TypeError from latent_module_base.py:52;
    `{'lightning': 'mioflow'}` raised nothing at all and returned a default MIOFlow, so a
    typo and a deliberate default were the same observable.
    """
    from manylatents.api import _resolve_algorithm, _resolve_datamodule
    dm = _resolve_datamodule(data="gaussian_blob", n_samples=60)
    with pytest.raises(TypeError, match="not_a_real_param"):
        _resolve_algorithm(algorithms={"lightning": "mioflow"}, datamodule=dm,
                           not_a_real_param=1)


def test_lightning_nested_override_patches_the_node():
    """`network={'latent_dim': 4}` overrides one key and keeps the rest of the node.

    Replacing the node with the caller's bare dict would drop every sibling the packaged
    config exists to supply, and would break `setup()`, which reads
    `self.network_config.input_dim` by ATTRIBUTE (reconstruction.py:45).
    """
    from manylatents.api import _resolve_algorithm, _resolve_datamodule
    dm = _resolve_datamodule(data="gaussian_blob", n_samples=60)
    algo = _resolve_algorithm(algorithms={"lightning": "cflows"}, datamodule=dm,
                              network={"latent_dim": 4})
    assert algo.network_config.latent_dim == 4
    assert algo.network_config.hidden_dim == 128      # sibling survived the patch
    assert "_target_" in algo.network_config          # so did the class it names
    assert algo.network_config.input_dim is None      # attribute access, not dict indexing


def test_lightning_nested_unknown_key_raises():
    """A typo inside a nested node fails at resolve time, naming the class that refused it.

    Without this the merged node carries `ltent_dim` all the way to `setup()`, where
    `hydra_zen.instantiate` raises during `trainer.fit` — a full dataload later, and wrapped
    in an InstantiationException that names hydra rather than the caller's argument.
    """
    from manylatents.api import _resolve_algorithm, _resolve_datamodule
    dm = _resolve_datamodule(data="gaussian_blob", n_samples=60)
    with pytest.raises(TypeError, match="ltent_dim"):
        _resolve_algorithm(algorithms={"lightning": "cflows"}, datamodule=dm,
                           network={"ltent_dim": 4})


def test_lightning_nested_override_may_add_a_key_the_config_omits():
    """A key the packaged yaml never wrote is legal if the target class takes it.

    The check is against the `_target_`'s SIGNATURE, not against the keys the yaml happens
    to list: `cflows.yaml`'s optimizer node writes only `lr`, but `torch.optim.Adam` takes
    `weight_decay`, and rejecting it would make a routine override unreachable through the
    only form that patches (the `_target_` dict form replaces the whole node).
    """
    from manylatents.api import _resolve_algorithm, _resolve_datamodule
    dm = _resolve_datamodule(data="gaussian_blob", n_samples=60)
    algo = _resolve_algorithm(algorithms={"lightning": "cflows"}, datamodule=dm,
                              optimizer={"weight_decay": 1e-4})
    assert algo.optimizer_config.weight_decay == 1e-4
    assert algo.optimizer_config.lr == 0.001          # sibling survived
    assert algo.optimizer_config._partial_ is True    # and so did the meta-key


def test_run_lightning_override_changes_the_answer():
    """End-to-end: a declared parameter changes the result, not just the object.

    Cheapest observable, one training run: ae_reconstruction's embedding width IS
    `network.latent_dim`, which the packaged config sets to 50.
    """
    from manylatents.api import run
    out = run(data="gaussian_blob", algorithms={"lightning": "ae_reconstruction"},
              data_kwargs={"n_samples": 60}, network={"latent_dim": 3})
    assert np.asarray(out["embeddings"]).shape == (60, 3)


def test_lightning_seed_reaches_init_seed():
    """`seed=` reaches weight init on the lightning path, as it does on the latent one.

    Lightning modules seed their own weight init from `init_seed` in `configure_model`
    (reconstruction.py:66, mioflow.py:123, cflows.py:172), and they do it AFTER
    `experiment.py` calls `seed_everything(seed)` — so they OVERRIDE the global seed and
    `run(seed=7, algorithms={'lightning': ...})` initialised at 42 regardless.
    """
    from manylatents.api import _resolve_algorithm, _resolve_datamodule
    dm = _resolve_datamodule(data="gaussian_blob", n_samples=60)
    assert _resolve_algorithm(algorithms={"lightning": "mioflow"}, datamodule=dm,
                              seed=7).init_seed == 7
    # An explicit init_seed beats the run-wide seed.
    assert _resolve_algorithm(algorithms={"lightning": "mioflow"}, datamodule=dm,
                              seed=7, init_seed=11).init_seed == 11


def test_resolve_algorithm_by_name():
    """String name resolves via algorithm registry."""
    from manylatents.api import _resolve_algorithm
    from manylatents.algorithms.latent.latent_module_base import LatentModule
    algo = _resolve_algorithm(algorithm="pca")
    assert isinstance(algo, LatentModule)


def test_resolve_algorithm_instance_passthrough():
    """Pre-built instance passes through unchanged."""
    from manylatents.api import _resolve_algorithm
    from manylatents.algorithms.latent.pca import PCAModule
    mod = PCAModule(n_components=3)
    result = _resolve_algorithm(algorithm=mod)
    assert result is mod


def test_resolve_algorithm_dict_config():
    """Dict with _target_ still works (Hydra fallback)."""
    from manylatents.api import _resolve_algorithm
    algo = _resolve_algorithm(algorithms={
        "latent": {
            "_target_": "manylatents.algorithms.latent.pca.PCAModule",
            "n_components": 5,
        }
    })
    assert algo.n_components == 5


def test_resolve_algorithm_dict_string():
    """Dict with string value uses registry."""
    from manylatents.api import _resolve_algorithm
    from manylatents.algorithms.latent.pca import PCAModule
    algo = _resolve_algorithm(algorithms={"latent": "pca"})
    assert isinstance(algo, PCAModule)


def test_api_run_string_names():
    """Full run() with string names."""
    from manylatents.api import run
    result = run(
        data="swissroll",
        algorithm="pca",
        metrics=["FractalDimension"],
    )
    assert "embeddings" in result
    assert "scores" in result
    assert "FractalDimension" in result["scores"]
