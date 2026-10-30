"""Name lookup of extension datamodules through the ``manylatents.datamodules`` entry points.

No test installs a second distribution: ``importlib.metadata.entry_points`` is patched to
return a fake group whose entry points point at classes in this module.
"""
import importlib.metadata
import sys

import pytest

from manylatents.data import SwissRollDataModule, get_datamodule, list_datamodules

GROUP = "manylatents.datamodules"

# The module name under which pytest imported this file. An entry point that names it loads
# this very module, so ``isinstance`` checks see the same ``FakeDM`` class.
FAKE_VALUE = f"{__name__}:FakeDM"


class FakeDM:
    """Stands in for a datamodule that an extension package ships."""

    def __init__(self, dataset_id: str, random_state: int = 0):
        self.dataset_id = dataset_id
        self.random_state = random_state


def _patch_group(monkeypatch, entries):
    """Make ``entry_points(group=GROUP)`` return ``entries`` (name -> value)."""
    eps = [importlib.metadata.EntryPoint(name=n, value=v, group=GROUP) for n, v in entries.items()]
    real = importlib.metadata.entry_points

    def fake_entry_points(**params):
        if params.get("group") == GROUP:
            return importlib.metadata.EntryPoints(eps)
        return real(**params)

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


@pytest.fixture
def fake_eps(monkeypatch):
    _patch_group(monkeypatch, {"fake_dm": FAKE_VALUE})


def test_entry_point_resolves(fake_eps):
    dm = get_datamodule("fake_dm", dataset_id="X", random_state=7)
    assert isinstance(dm, FakeDM)
    assert dm.dataset_id == "X"
    assert dm.random_state == 7


def test_normalized_name(fake_eps):
    dm = get_datamodule("Fake-DM", dataset_id="X")
    assert isinstance(dm, FakeDM)


def test_core_name_not_shadowed(monkeypatch):
    _patch_group(monkeypatch, {"fake_dm": FAKE_VALUE, "swissroll": FAKE_VALUE})
    dm = get_datamodule("swissroll")
    assert isinstance(dm, SwissRollDataModule)


def test_broken_entry_point_reported(monkeypatch):
    _patch_group(monkeypatch, {"fake_dm": "not_installed_pkg.mod:DM"})
    with pytest.raises(ImportError) as info:
        get_datamodule("fake_dm")
    assert "fake_dm" in str(info.value)
    assert "not_installed_pkg.mod:DM" in str(info.value)
    assert info.value.__cause__ is not None


def test_unknown_name_lists_entry_points(fake_eps):
    with pytest.raises(ValueError, match="fake_dm"):
        get_datamodule("nope")


def test_list_includes_entry_points(fake_eps):
    names = list_datamodules()
    assert "fake_dm" in names
    assert "swissroll" in names
    assert names == sorted(names)


def test_declared_random_state_wins(fake_eps, monkeypatch):
    import manylatents.data as data_pkg
    from manylatents.api import _resolve_datamodule

    calls = []
    real = data_pkg.get_datamodule

    def counting_get_datamodule(name, **kwargs):
        calls.append(kwargs)
        return real(name, **kwargs)

    monkeypatch.setattr(data_pkg, "get_datamodule", counting_get_datamodule)
    # The old call ``get_datamodule(data, random_state=seed, **kwargs)`` failed with a
    # duplicate-keyword TypeError before it reached ``get_datamodule``, so the call count
    # alone cannot see the retry. Record the exceptions raised in this frame as well.
    raised = []

    def tracer(frame, event, arg):
        if frame.f_code is _resolve_datamodule.__code__:
            def local(frame, event, arg):
                if event == "exception":
                    raised.append(arg[0])
                return local
            return local
        return None

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        dm = _resolve_datamodule(
            data="fake_dm", seed=101, dataset_id="X", random_state=20261003
        )
    finally:
        sys.settrace(previous)
    assert dm.random_state == 20261003
    assert len(calls) == 1
    assert raised == [], f"the declared random_state needed a retry: {raised}"


def test_run_seed_is_default(fake_eps):
    from manylatents.api import _resolve_datamodule

    dm = _resolve_datamodule(data="fake_dm", seed=101, dataset_id="X")
    assert dm.random_state == 101


def test_wrong_kwarg_still_raises(fake_eps):
    from manylatents.api import _resolve_datamodule

    with pytest.raises(TypeError, match="n_cels_max"):
        _resolve_datamodule(data="fake_dm", seed=101, dataset_id="X", n_cels_max=5)
