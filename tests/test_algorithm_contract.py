"""What `run_experiment()` requires of an algorithm, asserted structurally.

The engine's contract with an algorithm has two halves and they behave differently.

**The fit dispatch is bound to two base classes.** `experiment.py` branches
`isinstance(algorithm, LatentModule)` / `elif isinstance(algorithm, LightningModule)`. Until
#299 there was no `else`: an algorithm outside both fell through, `latents` stayed None, the
`if latents is not None` guard skipped packaging, and the call returned a result with no
embeddings and no error. Reachable by any second substrate — a JAX module hits it on run one.

**The capability hooks are NOT bound to anything.** All four are `hasattr`, never `isinstance`,
so an algorithm satisfies them by *having the methods*. That is worth pinning, because the
obvious way to "declare" them — a `MLLightningModule` mixin — would bind them to an inheritance
chain and force a parallel base for every future framework, which is the one thing the current
shape does not require.

Structural, and deliberately so: this mirrors
`manylatents-omics/tests/kinds/test_kind_protocol_conformance.py`, which asserts a consumer's
protocol without importing the consumer, so CI catches drift with no cross-repo dependency.
"""
import pytest
import torch
from lightning import LightningModule

from manylatents.experiment import run_experiment

#: What the engine looks for on an algorithm, with what its ABSENCE means. Two of the four can
#: have a neutral default and two cannot — `encode` absent means "use fit/transform", and
#: `test_step` absent means "skip evaluation entirely", so a base class supplying either would
#: change control flow rather than fill a gap.
ENGINE_HOOKS = {
    "fit_fraction":  "fit all rows        (experiment.py:355, getattr default 1.0)",
    "encode":        "use fit/transform   (experiment.py:412)",
    "extra_outputs": "nothing extra       (experiment.py:442)",
}

#: `test_step` is the fourth thing the engine checks and it is NOT in the table above, because
#: its check is vacuous. `experiment.py:542` guards `_evaluate_lightningmodule` with
#: `if not hasattr(algorithm, "test_step"): skip` — but that function is only ever called from
#: inside the `isinstance(algorithm, LightningModule)` branch, and `LightningModule` DEFINES
#: `test_step` on the base class. So the guard is True for every algorithm that can reach it and
#: the skip can never fire. Measured: `hasattr(LightningModule(), "test_step")` is True, while
#: `encode` is False. Recorded here rather than fixed — whether the guard should test for an
#: OVERRIDE, or go, is a behaviour change.
VACUOUS_HOOK = "test_step"


class _NotAnAlgorithm:
    """Neither a LatentModule nor a LightningModule — a stand-in for a future substrate."""


def _dm():
    """A real datamodule: the dispatch is reached only after `datamodule.setup()`, so a stub
    would fail earlier and the test would pass for the wrong reason."""
    import numpy as np

    from manylatents.data.precomputed_datamodule import PrecomputedDataModule

    return PrecomputedDataModule(data=np.zeros((6, 3), dtype="float32"), batch_size=6,
                                 shuffle_traindata=False)


def test_an_unrecognised_algorithm_is_refused_by_name():
    """#299. The failure this replaces was silent: `latents` stayed None, the
    `if latents is not None` guard skipped packaging, and the call returned a result with no
    embeddings and no error — indistinguishable from an algorithm that ran and produced
    nothing."""
    with pytest.raises(TypeError, match="neither a LatentModule .* nor a LightningModule"):
        run_experiment(datamodule=_dm(), algorithm=_NotAnAlgorithm(), trainer=None)


def test_the_message_names_what_was_passed():
    """A caller who handed over the wrong object needs to know which object."""
    with pytest.raises(TypeError, match="_NotAnAlgorithm"):
        run_experiment(datamodule=_dm(), algorithm=_NotAnAlgorithm(), trainer=None)


def test_the_vacuous_hook_stays_measured_rather_than_assumed():
    """Pins the finding above so it is not re-derived. If `LightningModule` ever stops defining
    `test_step`, `experiment.py:542`'s guard becomes live and this fails — which is the moment
    to decide what it should do."""
    assert hasattr(LightningModule(), VACUOUS_HOOK)
    assert not hasattr(LightningModule(), "encode")


@pytest.mark.parametrize("hook", sorted(ENGINE_HOOKS))
def test_every_engine_hook_is_optional_and_structural(hook):
    """None of the four may become required, and none may be reachable only by inheritance.

    A bare `LightningModule` implements none of them and must still be a legal algorithm — which
    is what makes `distillation` and `phase1_align` (which define none of the four today) run at
    all. If this starts failing because a hook moved onto a base class, the contract has stopped
    being structural and a second framework now needs a parallel base to satisfy it.
    """
    assert not hasattr(LightningModule(), hook), (
        f"{hook!r} is now supplied by LightningModule's own hierarchy, so its absence — "
        f"{ENGINE_HOOKS[hook]} — can no longer be expressed")


def test_a_plain_object_can_satisfy_the_hooks_without_inheriting_anything():
    """The property a JAX port depends on: the hooks are duck-typed, so conformance costs no
    base class. Asserted on an object that inherits from nothing at all."""

    class Duck:
        fit_fraction = 1.0

        def encode(self, x):
            return torch.zeros(len(x), 2)

        def extra_outputs(self):
            return {}

        def test_step(self, batch, batch_idx):
            return {}

    duck = Duck()
    for hook in (*ENGINE_HOOKS, VACUOUS_HOOK):
        assert hasattr(duck, hook), hook
