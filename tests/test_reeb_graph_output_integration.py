"""Reeb cell cap + graph output + SaveOutputs graph files, end to end."""
import numpy as np
import pandas as pd
import pytest
import torch

ripser = pytest.importorskip("ripser")
gudhi = pytest.importorskip("gudhi")

from manylatents.algorithms.latent.reeb_graph import ReebGraphModule
from manylatents.callbacks.embedding.save_outputs import SaveOutputs
from manylatents.outputs import collect_outputs


def _make_swissroll(n=200, seed=42):
    """Generate a simple Swiss roll dataset (copied from tests/test_reeb_graph.py)."""
    rng = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.rand(n))
    x = t * np.cos(t)
    y = 30 * rng.rand(n)
    z = t * np.sin(t)
    return np.column_stack([x, y, z])


def test_collected_outputs_include_subsample_indices():
    m = ReebGraphModule(max_N=150, n_bins=5)
    m.fit(torch.from_numpy(_make_swissroll(n=200, seed=42)).float())
    keys = set(collect_outputs(m))
    assert {
        "adjacency", "graph", "node_coordinates", "structural_summary", "subsample_indices"
    } <= keys


def test_swissroll_round_trip(tmp_path):
    from manylatents.api import run

    result = run(data="swissroll", algorithms={"latent": "reeb_graph"}, max_N=300, n_bins=5)
    saver = SaveOutputs(
        save_dir=str(tmp_path), save_format="csv", use_timestamp=False, experiment_name="t"
    )
    out = saver.on_latent_end(dataset=None, embeddings=result)

    nodes = pd.read_csv(out["graph_nodes_path"])
    edges = pd.read_csv(out["graph_edges_path"])
    coords = result["graph"]["node_coordinates"]
    assert coords.shape[1] == 3
    np.testing.assert_allclose(
        nodes[["x0", "x1", "x2"]].to_numpy(), coords, rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(
        edges[["source", "target"]].to_numpy(), result["graph"]["edges"]
    )
