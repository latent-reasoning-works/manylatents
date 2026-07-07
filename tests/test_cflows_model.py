"""End-to-end correctness test for the Cflows trajectory LightningModule.

Builds a toy dataset where a Gaussian blob *drifts* at constant velocity over
K=4 timepoints, trains ``Cflows`` a handful of epochs, and proves:

  1. Training loss decreases (it learns something).
  2. ``encode(X)`` returns ``(n, latent_dim)``.
  3. Drift recovery: integrating the trained flow from the ``t_0`` cells forward
     to ``t_{K-1}`` moves the cloud toward the true late distribution — its OT
     distance to the real ``t_{K-1}`` cloud is smaller than the un-integrated
     ``t_0`` cloud's OT distance to it. (The real correctness check: the flow
     learned the known drift.)
"""
import functools

import numpy as np
import pytest
import torch

pytest.importorskip("ot")
pytest.importorskip("torchdiffeq")

from lightning.pytorch import Trainer, seed_everything  # noqa: E402
from lightning.pytorch.callbacks import Callback  # noqa: E402

from manylatents.algorithms.lightning.cflows import Cflows  # noqa: E402
from manylatents.algorithms.lightning.losses.cflows import OTLoss  # noqa: E402
from manylatents.algorithms.lightning.networks.latent_ode import (  # noqa: E402
    LatentODENetwork,
)
from manylatents.data.precomputed_datamodule import PrecomputedDataModule  # noqa: E402


# --------------------------------------------------------------------------- #
# Toy drifting-blob dataset
# --------------------------------------------------------------------------- #
K = 4                       # number of timepoints (0, 1, 2, 3)
N_PER = 150                 # cells per timepoint
DIM = 2                     # feature dimension
VELOCITY = np.array([4.0, 0.0], dtype=np.float32)  # constant per-step drift
SIGMA = 0.3                 # blob std
LATENT_DIM = 8              # over-complete latent so the AE is near-lossless


def _make_drift_dataset(seed: int = 0):
    """Blob at timepoint k ~ N(k * VELOCITY, SIGMA^2 I)."""
    rng = np.random.default_rng(seed)
    xs, ts = [], []
    for k in range(K):
        center = k * VELOCITY
        cloud = center + SIGMA * rng.standard_normal((N_PER, DIM)).astype(np.float32)
        xs.append(cloud)
        ts.append(np.full(N_PER, float(k), dtype=np.float32))
    X = np.concatenate(xs, axis=0)
    t = np.concatenate(ts, axis=0)
    return X, t


class _LossRecorder(Callback):
    """Record the epoch-level ``train_loss`` after each training epoch."""

    def __init__(self):
        self.losses: list[float] = []

    def on_train_epoch_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get("train_loss")
        if val is not None:
            self.losses.append(float(val))


def _build_model(dm, seed: int = 42):
    net = LatentODENetwork(
        input_dim=DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=64,
        encoder_hidden_dims=[64, 64],
        decoder_hidden_dims=[64, 64],
        ode_n_layers=2,
        solver="dopri5",
        rtol=1e-4,
        atol=1e-4,
        use_adjoint=False,  # plain odeint: simpler/faster for this tiny problem
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
    )


def test_cflows_learns_known_drift(capsys):
    seed_everything(0, workers=True)

    X, t = _make_drift_dataset(seed=0)
    # One batch holds every timepoint so grouping-by-time is complete each step.
    dm = PrecomputedDataModule(data=X, time=t, batch_size=len(X), shuffle_traindata=False)

    model = _build_model(dm)
    recorder = _LossRecorder()
    trainer = Trainer(
        max_epochs=150,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[recorder],
    )
    trainer.fit(model, datamodule=dm)

    # ---- (1) training loss went down ----
    assert len(recorder.losses) >= 2, "expected per-epoch losses to be recorded"
    first, last = recorder.losses[0], recorder.losses[-1]
    print(f"\n[cflows] train_loss  first={first:.4f}  last={last:.4f}")
    assert last < first, f"loss did not decrease: first={first}, last={last}"

    # ---- (2) encode returns (n, latent_dim) ----
    Xt = torch.from_numpy(X)
    model.eval()
    with torch.no_grad():
        emb = model.encode(Xt)
    print(f"[cflows] encode(X).shape = {tuple(emb.shape)}")
    assert emb.shape == (len(X), LATENT_DIM)

    # ---- (3) drift recovery ----
    x0 = torch.from_numpy(X[t == 0.0])          # cells at t_0
    x_last = torch.from_numpy(X[t == float(K - 1)])  # true cells at t_{K-1}
    ot = OTLoss(which="emd")
    with torch.no_grad():
        x_pred = model.integrate(x0, t_start=0.0, t_end=float(K - 1))
        d_raw = ot(x0, x_last).item()           # un-integrated t_0 -> t_{K-1}
        d_flow = ot(x_pred, x_last).item()       # flowed t_0 -> t_{K-1}
    print(
        f"[cflows] drift OT:  raw(t0->t{K-1})={d_raw:.4f}  "
        f"flow(t0->t{K-1})={d_flow:.4f}  (ideal ~= {float(np.sum((VELOCITY*(K-1))**2)):.1f} vs ~0)"
    )
    assert d_flow < d_raw, (
        f"flow did not move cells toward t_{K-1}: raw={d_raw}, flow={d_flow}"
    )
