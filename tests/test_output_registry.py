"""The generic-output registry — the collect that used to be inherited.

Every assertion here was previously either impossible to state (the collect was a
method body, reachable only by classes that inherited it) or stated indirectly
through ``LatentModule.extra_outputs()``. The point of the registry is that an
extractor is generic over *anything* that stores the attribute, so a LightningModule
that computes trajectories (MIOFlow, #295) is reached by the same code path as a
LatentModule that computes an affinity.
"""
import inspect

import numpy as np
import pytest
import torch

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.outputs import (
    adjacency,
    affinity,
    collect_outputs,
    collect_registered_outputs,
    get_output_registry,
    kernel,
    list_outputs,
    trajectories,
)


class _Bare:
    """An object with none of the generic hooks. Not a LatentModule on purpose —
    the registry must not require inheriting anything."""


class _MinimalModule(LatentModule):
    def fit(self, x, y=None):
        self._is_fitted = True

    def transform(self, x):
        return x[:, : self.n_components]


# --------------------------------------------------------------- enumeration --


def test_registry_is_enumerable_without_fitting_anything():
    """The whole reason for a registry: 'what can an algorithm emit' is answerable
    from the registry, not from a fitted object."""
    assert list_outputs() == ["adjacency", "affinity", "kernel", "trajectories"]


def test_registry_specs_carry_their_requirements():
    reg = get_output_registry()
    assert reg["affinity"].requires == ("affinity",)
    assert reg["trajectories"].description
    # a copy — mutating what a caller got back must not corrupt the registry
    reg.pop("affinity")
    assert "affinity" in get_output_registry()


# ------------------------------------------------------------------- absence --


@pytest.mark.parametrize("extractor", [trajectories, affinity, adjacency, kernel])
def test_extractor_is_empty_when_the_hook_is_absent(extractor):
    assert extractor(_Bare()) == {}


def test_unfitted_module_yields_nothing():
    """RuntimeError is 'not fitted yet', not 'broken' — swallowed, as the inherited
    collect at latent_module_base.py swallowed it."""
    from manylatents.algorithms.latent import PCAModule

    assert collect_registered_outputs(PCAModule(n_components=2)) == {}


# ---------------------------------------------------------------- guard rails --


def test_non_callable_attribute_is_not_an_output():
    """Someone will store an array under a hook name. Calling it would TypeError."""
    obj = _Bare()
    obj.kernel = np.eye(3)
    assert kernel(obj) == {}


def test_accessor_that_requires_an_argument_is_skipped():
    """``gpu_phate_local.PHATE`` sets ``self.affinity = PHATEAffinity(...)``, a
    callable instance that takes X (gpu_phate_local.py:367). 'getattr the name and
    call it' is not safe without a bind check."""

    class _NeedsX:
        def affinity(self, X):
            raise AssertionError("must not be called")

    assert affinity(_NeedsX()) == {}


def test_nn_module_layer_is_not_called():
    """A LightningModule may own a LAYER named ``kernel``; calling it runs a forward
    pass, which is not reading an output."""

    class _HasLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kernel = torch.nn.Linear(3, 3)

    assert kernel(_HasLayer()) == {}


def test_property_raising_before_fit_yields_nothing():
    class _Unfitted:
        @property
        def trajectories(self):
            raise RuntimeError("not fitted")

    assert trajectories(_Unfitted()) == {}


def test_a_real_bug_inside_an_accessor_still_propagates():
    """Guard by PRECONDITION, never by a blanket ``except``. A TypeError raised
    inside a genuine ``kernel()`` is a defect, and recording it as 'this algorithm
    has no kernel' is how a broken accessor ships unnoticed."""

    class _Broken:
        def kernel(self):
            raise TypeError("genuine bug")

    with pytest.raises(TypeError, match="genuine bug"):
        collect_outputs(_Broken())


# ------------------------------------------------------------------ contents --


def test_tensor_trajectories_become_numpy():
    class _Traj:
        trajectories = torch.randn(5, 10, 2)

    out = trajectories(_Traj())
    assert isinstance(out["trajectories"], np.ndarray)
    assert out["trajectories"].shape == (5, 10, 2)


def test_lightning_style_trajectories_are_reachable():
    """#295 in one assertion: MIOFlow stores ``.trajectories`` on a LightningModule
    and therefore never inherited the collect. An extractor does not care."""

    class _Lit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._trajectories = torch.randn(4, 6, 2)

        @property
        def trajectories(self):
            return self._trajectories

    assert collect_registered_outputs(_Lit())["trajectories"].shape == (4, 6, 2)


# -------------------------------------------------------------------- merge --


def test_collect_outputs_merges_registry_and_algorithm_specific():
    class _Both:
        def affinity(self):
            return np.eye(4)

        def extra_outputs(self):
            return {"mismatch_ratio": 0.25}

    out = collect_outputs(_Both())
    assert out["affinity"].shape == (4, 4)
    assert out["mismatch_ratio"] == 0.25


def test_algorithm_specific_wins_a_collision():
    """A method that names a key has made a decision about it; a generic extractor
    has not."""

    class _Collides:
        def kernel(self):
            return np.zeros((2, 2))

        def extra_outputs(self):
            return {"kernel": "mine"}

    assert collect_outputs(_Collides())["kernel"] == "mine"


def test_base_extra_outputs_is_marked_as_the_registry_default():
    """The marker is what stops ``collect_outputs`` running the whole registry pass
    a second time through the inherited shim."""
    assert getattr(LatentModule.extra_outputs, "_manylatents_registry_default", False)
    assert getattr(_MinimalModule().extra_outputs, "_manylatents_registry_default", False)


def test_subclass_override_is_not_marked():
    from manylatents.algorithms.latent import PCAModule

    assert not getattr(
        PCAModule.extra_outputs, "_manylatents_registry_default", False
    )


def test_collect_outputs_does_not_run_the_generic_pass_twice():
    """Measured: one generic collect on PCAModule at N=3000 is ~10 ms and an NxN
    allocation, and ``PCAModule.affinity()`` already calls ``kernel()`` internally.
    Two calls is the floor (registry ``kernel`` + ``affinity``->``kernel``); four
    means the inherited shim ran as well."""
    from manylatents.algorithms.latent import PCAModule

    m = PCAModule(n_components=2)
    m.fit(torch.randn(30, 5))

    calls = []
    real = m.kernel

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    m.kernel = counting
    out = collect_outputs(m)

    assert len(calls) == 2, f"generic pass ran more than once: {len(calls)} kernel() calls"
    assert {"affinity", "kernel"} <= set(out)


def test_extractors_take_exactly_one_argument():
    """Registration contract: an extractor is ``fn(algorithm) -> dict``."""
    for spec in get_output_registry().values():
        assert len(inspect.signature(spec.func).parameters) == 1
