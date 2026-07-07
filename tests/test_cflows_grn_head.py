"""End-to-end test for the Cflows Granger-GRN head.

Wires the causal-network output of :class:`Cflows`: decode the trained flow's
latent trajectory back to gene space, run the Granger estimator on it, and expose
it through ``extra_outputs()``.

Synthetic ground truth (a KNOWN lagged causal edge along the trajectory)
-----------------------------------------------------------------------
We build a gene-expression trajectory whose *differences* form a clean one-way
lag-1 VAR (mirroring ``tests/test_cflows_granger.make_integrated_pair`` so the
estimator's internal first-difference recovers it):

    dR_t = white noise                       # regulator R has a stochastic ramp
    dT_t = COUPLING * dR_{t-1} + eps_t        # target T follows R with a +lag
    dNi_t = white noise (noise genes)

with ``COUPLING > 0`` (activation), integrated (cumsum) to observed levels. Gene
0 is R, gene 1 is T, genes 2.. are noise. Cells at coarse timepoint ``k`` are
Gaussian samples around the trajectory mean at that time, carrying label ``k``.

Two assertions, and *which model each runs on* is called out explicitly:

  (A) TRAINED MODEL — structural / plumbing:
        * ``gene_trajectory(x0, t_grid).shape == (T, n_cells, n_genes)``
        * ``extra_outputs()`` runs and returns ``grn_edges`` / ``grn_weights`` /
          ``grn_node_ids`` with shapes valid for ``manykinds.SparseGraph``
          (int ``(E,2)`` edges, one aligned float weight per edge) — we build a
          real ``SparseGraph`` from them.
      The trained-model R->T weight is *printed* (informational).

  (B) DECODED GROUND-TRUTH TRAJECTORY — signed-edge recovery:
        The known R->T edge is recovered with a POSITIVE weight and is stronger
        (|weight|) than a matched noise->T edge, by feeding the hand-built
        ground-truth gene trajectory straight to ``granger_grn``.

Why the signed-edge assertion uses the ground-truth trajectory (fallback allowed
by the task): the flow integrates a *single deterministic* latent path per cell
with a smooth (dopri5) ODE, so the decoded population-mean trajectory is a smooth
interpolation of the coarse timepoint means. It cannot reproduce the *stochastic*
lag-1 increment structure that a lag-1 Granger test keys on — the information
needed to recover the signed edge with statistical power lives in the fine
random-walk increments, which are averaged/smoothed away by (i) a handful of CPU
epochs on a tiny model and (ii) the ODE's deterministic interpolation. So we
assert the estimator + decoding contract on the ideal decoded output, and assert
the head's plumbing (shapes, SparseGraph-validity, no crash) on the trained
model. The trained-model weight is printed so the decoding-fidelity gap is
visible rather than hidden.
"""
import functools

import numpy as np
import pytest
import torch

pytest.importorskip("ot")
pytest.importorskip("torchdiffeq")
pytest.importorskip("statsmodels")

from lightning.pytorch import Trainer, seed_everything  # noqa: E402

from manylatents.algorithms.cflows_granger import granger_grn  # noqa: E402
from manylatents.algorithms.lightning.cflows import Cflows  # noqa: E402
from manylatents.algorithms.lightning.losses.cflows import OTLoss  # noqa: E402
from manylatents.algorithms.lightning.networks.latent_ode import (  # noqa: E402
    LatentODENetwork,
)
from manylatents.data.precomputed_datamodule import PrecomputedDataModule  # noqa: E402

# --------------------------------------------------------------------------- #
# Synthetic ground truth
# --------------------------------------------------------------------------- #
R_GENE = 0            # regulator  (gene index 0)
T_GENE = 1            # target     (gene index 1)
N_NOISE = 3           # noise genes (indices 2, 3, 4)
N_GENES = 2 + N_NOISE
COUPLING = 0.9        # POSITIVE coupling -> activation edge R -> T
T_FINE = 300          # length of the fine ground-truth trajectory
K = 8                 # coarse (measured) timepoints used to train the flow
N_PER = 60            # cells per coarse timepoint
CELL_SIGMA = 0.10     # per-cell Gaussian jitter around the trajectory mean
LATENT_DIM = 8        # over-complete latent -> near-lossless autoencoder


def _fine_ground_truth(seed: int = 0) -> np.ndarray:
    """Return the ``[T_FINE, N_GENES]`` ground-truth trajectory (gene means).

    Differences form a one-way lag-1 VAR: dT_t = COUPLING * dR_{t-1} + eps, so
    R -> T holds with a POSITIVE coefficient and T -> R does not.
    """
    rng = np.random.default_rng(seed)
    dR = rng.standard_normal(T_FINE)
    eps = rng.standard_normal(T_FINE) * 1.0
    dT = np.zeros(T_FINE)
    for t in range(1, T_FINE):
        dT[t] = COUPLING * dR[t - 1] + eps[t]
    cols = [np.cumsum(dR), np.cumsum(dT)]
    for _ in range(N_NOISE):
        cols.append(np.cumsum(rng.standard_normal(T_FINE)))
    return np.column_stack(cols).astype(np.float32)  # [T_FINE, N_GENES]


def _make_cell_dataset(fine: np.ndarray, seed: int = 0):
    """Sample cells at K coarse timepoints around the fine trajectory means."""
    rng = np.random.default_rng(seed + 1)
    coarse_idx = np.linspace(0, T_FINE - 1, K).astype(int)
    xs, ts = [], []
    for k, ci in enumerate(coarse_idx):
        mean = fine[ci]  # [N_GENES]
        cloud = mean + CELL_SIGMA * rng.standard_normal((N_PER, N_GENES)).astype(np.float32)
        xs.append(cloud.astype(np.float32))
        ts.append(np.full(N_PER, float(k), dtype=np.float32))
    return np.concatenate(xs, 0), np.concatenate(ts, 0)


def _build_model(dm, seed: int = 42) -> Cflows:
    net = LatentODENetwork(
        input_dim=N_GENES,
        latent_dim=LATENT_DIM,
        hidden_dim=64,
        encoder_hidden_dims=[64, 64],
        decoder_hidden_dims=[64, 64],
        ode_n_layers=2,
        solver="dopri5",
        rtol=1e-4,
        atol=1e-4,
        use_adjoint=False,
    )
    return Cflows(
        network=net,
        optimizer=functools.partial(torch.optim.Adam, lr=1e-2),
        loss=OTLoss(which="emd"),
        datamodule=dm,
        init_seed=seed,
        integration_times=[0.0, 1.0],
        lambda_density=1.0,
        lambda_energy=0.0,
        # GRN head: gene names R, T, N0.. ; all genes both regulators & targets.
        grn_gene_names=["R", "T"] + [f"N{i}" for i in range(N_NOISE)],
        grn_n_bins=64,
        grn_downsample=1,
        grn_direction="forward",
    )


def _weights_by_direction(edges, node_ids, weights, gene_names):
    """Map (regulator_name, target_name) -> signed weight."""
    pos_to_gene = {pos: gene_names[node_ids[pos]] for pos in range(len(node_ids))}
    return {
        (pos_to_gene[a], pos_to_gene[b]): w
        for (a, b), w in zip(edges.tolist(), weights.tolist())
    }


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #
def test_cflows_grn_head_end_to_end():
    seed_everything(0, workers=True)
    gene_names = ["R", "T"] + [f"N{i}" for i in range(N_NOISE)]

    fine = _fine_ground_truth(seed=0)
    X, t = _make_cell_dataset(fine, seed=0)
    dm = PrecomputedDataModule(data=X, time=t, batch_size=len(X), shuffle_traindata=False)

    model = _build_model(dm)

    # ---- extra_outputs() before any fit -> {} (guarded) ------------------- #
    assert model.extra_outputs() == {}, "extra_outputs() must be empty before fit"

    trainer = Trainer(
        max_epochs=60,
        accelerator="cpu",  # MPS + torchdiffeq float64 fails on this box
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dm)
    model.eval()

    # ================================================================== #
    # (A) TRAINED MODEL — structural / plumbing
    # ================================================================== #
    x0 = torch.from_numpy(X[t == 0.0])            # cells at the first timepoint
    n_bins = 50
    t_grid = torch.linspace(0.0, float(K - 1), n_bins)
    with torch.no_grad():
        traj = model.gene_trajectory(x0, t_grid)
    print(f"\n[grn-head] gene_trajectory shape = {tuple(traj.shape)}  "
          f"(expected ({n_bins}, {x0.shape[0]}, {N_GENES}))")
    assert traj.shape == (n_bins, x0.shape[0], N_GENES)

    out = model.extra_outputs()
    assert set(out) == {"grn_edges", "grn_weights", "grn_node_ids"}, out.keys()
    edges, weights, node_ids = out["grn_edges"], out["grn_weights"], out["grn_node_ids"]

    # Shapes valid for manykinds.SparseGraph.
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert np.issubdtype(edges.dtype, np.integer)
    assert np.issubdtype(node_ids.dtype, np.integer)
    assert node_ids.ndim == 1
    assert weights.shape[0] == edges.shape[0]
    assert np.issubdtype(weights.dtype, np.floating)
    if edges.shape[0]:
        assert edges.min() >= 0 and edges.max() < len(node_ids)

    # Actually construct the SparseGraph (the real downstream contract).
    from manykinds import SparseGraph

    graph = SparseGraph(edges=edges, node_ids=node_ids, edge_weights=weights)
    graph.validate()
    print(f"[grn-head] trained-model GRN: {edges.shape[0]} edges over "
          f"{len(node_ids)} nodes -> SparseGraph OK")

    # Informational: trained-model R->T weight vs a noise->T weight.
    if edges.shape[0]:
        tw = _weights_by_direction(edges, node_ids, weights, gene_names)
        rt = tw.get(("R", "T"))
        nt = tw.get(("N0", "T"))
        print(f"[grn-head] TRAINED-MODEL  R->T={rt}   N0->T={nt}  (informational)")

    # ================================================================== #
    # (B) DECODED GROUND-TRUTH TRAJECTORY — signed-edge recovery
    # ================================================================== #
    # Feed the hand-built ground-truth gene trajectory straight to granger_grn
    # (this is exactly the estimator the head calls, on the ideal decoded output).
    gt_edges, gt_nodes, gt_weights = granger_grn(fine, gene_names, downsample=1)
    gtw = _weights_by_direction(gt_edges, gt_nodes, gt_weights, gene_names)
    rt = gtw[("R", "T")]
    tr = gtw[("T", "R")]           # reverse edge (should be weak)
    noise_to_T = [gtw[(f"N{i}", "T")] for i in range(N_NOISE)]
    strongest_noise = max(noise_to_T, key=abs)
    print(f"[grn-head] GROUND-TRUTH   R->T={rt:.4f}   T->R={tr:.4f}   "
          f"max|Ni->T|={strongest_noise:.4f}")

    # The known R->T edge: present, POSITIVE (activation), and strongest into T.
    assert ("R", "T") in gtw
    assert rt > 4.0, f"R->T should be strong & positive, got {rt}"
    assert abs(rt) > abs(strongest_noise), (
        f"R->T (|{rt:.3f}|) must beat matched noise->T (|{strongest_noise:.3f}|)"
    )
    assert abs(rt) > abs(tr), f"edge must be DIRECTED: |R->T|={abs(rt):.3f} vs |T->R|={abs(tr):.3f}"
    print("[grn-head] signed-edge recovery asserted on the DECODED GROUND-TRUTH "
          "trajectory; plumbing asserted on the TRAINED model.")
