"""The GRN head refuses to emit a graph until the time axis is declared.

``Cflows.extra_outputs()`` takes no arguments, so every knob it needs has to be a
constructor argument — that is why ``grn_gene_names`` already is one, and why
``grn_time_axis`` is one too. It has **no default**: a Granger test cannot tell a
measured time axis from an ordering derived from the data under test, and returns
a confident p-value either way (12/12 seeds on pure noise, per
``geomancer/docs/granger-verdict.md``).

The model genuinely cannot infer this. The integration grid is
``linspace(t_min, t_max, n_bins)`` over the *fit* timepoints — a reparametrisation
of whatever ``datamodule.time_tensor`` held, and a reparametrisation cannot create
measurement. So the run's configuration has to say.

These tests build the module and set the fit-population cache directly rather than
training: the guard is a wiring property and the flow's fidelity is irrelevant to
it. ``tests/test_cflows_grn_head.py`` covers the trained path.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("ot")
pytest.importorskip("torchdiffeq")
pytest.importorskip("statsmodels")

from manylatents.algorithms.lightning.cflows import Cflows  # noqa: E402
from manylatents.algorithms.lightning.losses.cflows import OTLoss  # noqa: E402
from manylatents.algorithms.lightning.networks.latent_ode import (  # noqa: E402
    LatentODENetwork,
)

N_GENES = 4
N_CELLS = 24
GENE_NAMES = ["R", "T", "N0", "N1"]


def _model(**grn_kwargs) -> Cflows:
    """A configured (untrained) Cflows with a populated fit cache."""
    net = LatentODENetwork(
        input_dim=N_GENES,
        latent_dim=4,
        hidden_dim=16,
        encoder_hidden_dims=[16],
        decoder_hidden_dims=[16],
        ode_n_layers=1,
        solver="euler",
        use_adjoint=False,
    )
    model = Cflows(
        network=net,
        optimizer=None,
        loss=OTLoss(which="emd"),
        datamodule=None,
        grn_gene_names=GENE_NAMES,
        grn_n_bins=64,
        grn_downsample=1,
        **grn_kwargs,
    )
    model.setup()
    rng = np.random.default_rng(0)
    model._fit_cells = torch.from_numpy(
        rng.standard_normal((N_CELLS, N_GENES)).astype(np.float32)
    )
    model._fit_times = torch.from_numpy(
        np.repeat([0.0, 1.0], N_CELLS // 2).astype(np.float32)
    )
    return model


# --------------------------------------------------------------------------- #
# the refusal
# --------------------------------------------------------------------------- #
def test_no_time_axis_stated_emits_nothing():
    """The default must be silence, not a graph."""
    assert _model().extra_outputs() == {}


def test_derived_time_axis_emits_nothing_by_default():
    assert _model(grn_time_axis="derived").extra_outputs() == {}


def test_bad_time_axis_value_emits_nothing():
    assert _model(grn_time_axis="pseudotime").extra_outputs() == {}


def test_refusal_is_logged_with_the_reason(caplog):
    """A silently empty dict would look identical to 'no fit yet'. The warning
    has to name the cause or the gap is undiagnosable from a run log."""
    import logging

    with caplog.at_level(logging.WARNING):
        _model().extra_outputs()
    text = caplog.text
    assert "refused" in text
    assert "time_axis" in text


# --------------------------------------------------------------------------- #
# the admission
# --------------------------------------------------------------------------- #
def test_measured_time_axis_emits_the_triple_plus_provenance():
    out = _model(grn_time_axis="measured").extra_outputs()
    assert set(out) == {"grn_edges", "grn_weights", "grn_node_ids", "grn_provenance"}

    edges, node_ids, weights = out["grn_edges"], out["grn_node_ids"], out["grn_weights"]
    assert np.issubdtype(edges.dtype, np.integer)
    assert np.issubdtype(node_ids.dtype, np.integer)
    assert np.issubdtype(weights.dtype, np.floating)
    assert weights.shape[0] == edges.shape[0]
    assert edges.max() < len(node_ids)          # edges index into node_ids
    assert node_ids.max() < len(GENE_NAMES)     # node_ids index into gene_names

    assert out["grn_provenance"][0] == "granger:time_axis=measured"


def test_forced_derived_run_is_marked_not_causal_in_the_output():
    """Methods work stays possible, but the graph carries the warning with it —
    it cannot be picked up later and read as a regulatory claim."""
    out = _model(
        grn_time_axis="derived", grn_allow_derived_time_axis=True
    ).extra_outputs()
    assert set(out) == {"grn_edges", "grn_weights", "grn_node_ids", "grn_provenance"}
    assert any("NOT-CAUSAL" in p for p in out["grn_provenance"])


# --------------------------------------------------------------------------- #
# the knobs are constructor arguments, reachable from a config
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,value",
    [
        ("grn_time_axis", "measured"),
        ("grn_allow_derived_time_axis", True),
        ("grn_n_top_genes", 3),
        ("grn_flavor", "variance"),
        ("grn_alpha", 0.05),
        ("grn_top_k", 5),
    ],
)
def test_grn_knobs_are_constructor_arguments(name, value):
    """extra_outputs() takes no arguments, so anything not on the constructor is
    unreachable on the flow path."""
    model = _model(**{name: value})
    assert getattr(model, name) == value


def test_unimplemented_maths_knobs_do_not_crash_the_run():
    """select_genes/threshold_edges raise NotImplementedError; the head's guard
    turns that into an empty dict rather than taking the run down."""
    assert _model(grn_time_axis="measured", grn_n_top_genes=2).extra_outputs() == {}
    assert _model(grn_time_axis="measured", grn_alpha=0.05).extra_outputs() == {}
    assert _model(grn_time_axis="measured", grn_top_k=2).extra_outputs() == {}
