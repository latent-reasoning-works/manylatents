"""The ``graph`` output: a node/edge view of an (M, M) adjacency with node coordinates."""
import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from manylatents.outputs import (
    collect_outputs,
    get_output_registry,
    graph,
    list_outputs,
)

OUTPUTS_PY = Path(__file__).resolve().parents[1] / "manylatents" / "outputs.py"


def _make_swissroll(n=200, seed=42):
    """Generate a simple Swiss roll dataset (copied from tests/test_reeb_graph.py)."""
    rng = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.rand(n))
    x = t * np.cos(t)
    y = 30 * rng.rand(n)
    z = t * np.sin(t)
    return np.column_stack([x, y, z])


@pytest.fixture(scope="module")
def fitted_reeb():
    pytest.importorskip("ripser")
    pytest.importorskip("gudhi")
    from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

    m = ReebGraphModule(n_bins=5)
    m.fit(torch.from_numpy(_make_swissroll(n=200, seed=42)).float())
    return m


class _CountingAdjacency:
    """An adjacency() with no node_coordinates; counts its calls."""

    def __init__(self):
        self.calls = 0

    def adjacency(self):
        self.calls += 1
        return np.eye(3)


class _RaisingCoordinates:
    def adjacency(self):
        return np.eye(3)

    @property
    def node_coordinates(self):
        raise RuntimeError("not fitted")


class _Fixed:
    def __init__(self, A, coords):
        self._A = A
        self.node_coordinates = coords

    def adjacency(self):
        return self._A


def _A4():
    A = np.zeros((4, 4))
    A[0, 1] = A[1, 0] = 1.0
    A[1, 2] = A[2, 1] = 1.0
    A[2, 3] = A[3, 2] = 0.5
    return A


def test_list_outputs_has_graph():
    assert list_outputs() == ["adjacency", "affinity", "graph", "kernel", "trajectories"]
    assert get_output_registry()["graph"].requires == ("adjacency",)


def test_outputs_imports_no_manylatents():
    assert OUTPUTS_PY.is_file()
    tree = ast.parse(OUTPUTS_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names = ["manylatents (relative import)"]
            else:
                names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("manylatents"), name


def test_reeb_swissroll_graph(fitted_reeb):
    m = fitted_reeb
    g = graph(m)["graph"]
    assert set(g) == {"edges", "edge_weights", "node_ids", "node_coordinates"}
    assert g["edges"].dtype == np.int64
    assert g["edges"].shape == (m.structural_summary["n_edges"], 2)
    assert (g["edges"][:, 0] <= g["edges"][:, 1]).all()
    np.testing.assert_array_equal(g["node_ids"], np.arange(m.structural_summary["n_nodes"]))
    np.testing.assert_array_equal(g["node_coordinates"], m.node_coordinates)
    assert (g["edge_weights"] == 1.0).all()


def test_edges_match_adjacency():
    g = graph(_Fixed(_A4(), np.zeros((4, 2))))["graph"]
    np.testing.assert_array_equal(g["edges"], [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(g["edge_weights"], [1.0, 1.0, 0.5])
    np.testing.assert_array_equal(g["node_ids"], [0, 1, 2, 3])
    assert g["edges"].dtype == np.int64
    assert g["edge_weights"].dtype == np.float64
    assert g["node_ids"].dtype == np.int64
    assert g["node_coordinates"].dtype == np.float64


def test_collected_by_engine(fitted_reeb):
    keys = set(collect_outputs(fitted_reeb))
    assert {"adjacency", "graph", "node_coordinates", "structural_summary"} <= keys


def test_declines_without_coordinates():
    stub = _CountingAdjacency()
    assert graph(stub) == {}
    assert stub.calls == 0


def test_declines_when_coordinates_raise():
    assert graph(_RaisingCoordinates()) == {}


def test_unfitted_reeb_declines():
    pytest.importorskip("ripser")
    from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

    assert graph(ReebGraphModule()) == {}


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match=r"(?s)(4.*3|3.*4)"):
        graph(_Fixed(_A4(), np.zeros((3, 2))))
