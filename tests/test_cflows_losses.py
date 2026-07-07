"""Correctness tests for the MIOFlow-family trajectory losses.

These losses are the correctness anchors of the trajectory pipeline, so the
assertions pin exact numeric identities (independent recomputes), not just
shapes/signs.
"""
import pytest
import torch

ot = pytest.importorskip("ot")

from manylatents.algorithms.lightning.losses.cflows import (  # noqa: E402
    DensityLoss,
    MMDLoss,
    OTLoss,
    energy,
)


torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# OTLoss
# --------------------------------------------------------------------------- #
class TestOTLoss:
    def test_zero_on_identical(self):
        X = torch.randn(24, 6)
        loss = OTLoss()(X, X)
        assert loss.shape == ()
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_constant_shift_equals_norm_squared(self):
        """For Y = X + v, OT cost == ||v||^2 (optimal map is the translation)."""
        X = torch.randn(40, 5)
        v = torch.tensor([0.3, -0.7, 1.1, 0.2, -0.4])
        Y = X + v
        expected = (v**2).sum().item()

        loss = OTLoss()(X, Y).item()
        assert loss == pytest.approx(expected, rel=1e-5, abs=1e-6)

    def test_matches_independent_emd2(self):
        """OTLoss == ot.emd2(unif, unif, cdist(X,Y)**2) recomputed independently."""
        X = torch.randn(30, 4)
        Y = torch.randn(30, 4) + 0.5

        n = X.shape[0]
        a = torch.full((n,), 1.0 / n, dtype=X.dtype)
        b = torch.full((n,), 1.0 / n, dtype=Y.dtype)
        M = torch.cdist(X, Y) ** 2
        ref = ot.emd2(a, b, M)
        ref = ref.item() if hasattr(ref, "item") else float(ref)

        loss = OTLoss()(X, Y).item()
        assert loss == pytest.approx(ref, rel=1e-5, abs=1e-6)

    def test_plan_is_detached(self):
        """The transport plan carries no gradient; grad flows only through M."""
        X = torch.randn(12, 3, requires_grad=True)
        Y = torch.randn(12, 3)
        loss, plan = OTLoss()(X, Y, return_plan=True)
        assert not plan.requires_grad
        loss.backward()
        assert X.grad is not None

    def test_source_mass_override_renormalises(self):
        """source_mass is renormalised to sum to 1.

        An equal-but-unnormalised mass reduces to the uniform marginal, so the
        identity coupling on identical clouds stays feasible and the cost is ~0
        (this pins the renormalisation, not just that a number comes out)."""
        X = torch.randn(16, 4)
        mass = torch.full((16,), 2.0)  # -> 1/16 uniform after renormalisation
        loss = OTLoss()(X, X, source_mass=mass)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_source_mass_override_changes_cost(self):
        """A non-uniform source marginal changes the transport cost vs uniform."""
        X = torch.randn(16, 4)
        Y = torch.randn(16, 4)
        mass = torch.rand(16) + 0.1
        base = OTLoss()(X, Y).item()
        weighted = OTLoss()(X, Y, source_mass=mass).item()
        assert weighted != pytest.approx(base, abs=1e-6)

    def test_sinkhorn_solver_runs(self):
        X = torch.randn(20, 4)
        Y = torch.randn(20, 4)
        loss = OTLoss(which="sinkhorn")(X, Y)
        assert loss.item() >= 0.0

    def test_unbalanced_solver_runs(self):
        X = torch.randn(20, 4)
        Y = torch.randn(20, 4)
        loss = OTLoss(which="unbalanced")(X, Y)
        assert loss.item() >= 0.0


# --------------------------------------------------------------------------- #
# DensityLoss
# --------------------------------------------------------------------------- #
class TestDensityLoss:
    def test_zero_when_within_hinge(self):
        """Zero when every pred is within hinge of >= top_k targets."""
        top_k = 5
        # 8 target points at the origin (all coincident) -> every pred has
        # >= top_k targets at distance 0 < hinge.
        target = torch.zeros(8, 3)
        pred = torch.zeros(10, 3)
        loss = DensityLoss(hinge_value=0.01)(pred, target, top_k=top_k)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_positive_when_far(self):
        """Positive when preds are far from all targets."""
        target = torch.zeros(8, 3)
        pred = torch.full((10, 3), 10.0)
        loss = DensityLoss(hinge_value=0.01)(pred, target, top_k=5)
        assert loss.item() > 0.0

    def test_matches_manual_formula(self):
        top_k = 3
        hinge = 0.05
        pred = torch.randn(15, 4)
        target = torch.randn(20, 4)

        c = torch.cdist(pred, target)
        vals, _ = torch.topk(c, top_k, dim=1, largest=False, sorted=False)
        manual = torch.clamp(vals - hinge, min=0.0).mean().item()

        loss = DensityLoss(hinge_value=hinge)(pred, target, top_k=top_k).item()
        assert loss == pytest.approx(manual, rel=1e-6, abs=1e-7)


# --------------------------------------------------------------------------- #
# MMDLoss
# --------------------------------------------------------------------------- #
class TestMMDLoss:
    def test_zero_on_identical(self):
        X = torch.randn(30, 5)
        loss = MMDLoss()(X, X)
        assert loss.shape == ()
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_on_separated_clouds(self):
        X = torch.randn(30, 5)
        Y = torch.randn(30, 5) + 20.0
        loss = MMDLoss()(X, Y)
        assert loss.item() > 0.0


# --------------------------------------------------------------------------- #
# energy helper
# --------------------------------------------------------------------------- #
class TestEnergy:
    def test_equals_manual_sum(self):
        norms = [torch.tensor(float(i) ** 2) for i in range(1, 6)]  # [1,4,9,16,25]
        manual = sum(n.item() for n in norms)  # 55
        assert energy(norms).item() == pytest.approx(manual)

    def test_sums_per_sample_norms(self):
        # each step contributes a per-sample squared-norm vector
        norms = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.5, 0.5, 1.0])]
        manual = torch.stack(norms).sum().item()  # 8.0
        assert energy(norms).item() == pytest.approx(manual)

    def test_empty_is_zero(self):
        assert energy([]).item() == pytest.approx(0.0)
