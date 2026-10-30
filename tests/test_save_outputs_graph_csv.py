"""SaveOutputs writes the graph output as nodes/edges CSV files and a SparseGraph archive."""
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

from manylatents.callbacks.embedding.save_outputs import SaveOutputs

LOGGER_NAME = "manylatents.callbacks.embedding.save_outputs"


def _g3():
    return {
        "edges": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "edge_weights": np.array([1.0, 0.5]),
        "node_ids": np.arange(3, dtype=np.int64),
        "node_coordinates": np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
    }


def _small():
    return {"embeddings": np.zeros((5, 1)), "graph": _g3()}


def _saver(tmp_path, **kw):
    kw.setdefault("save_format", "csv")
    return SaveOutputs(
        save_dir=str(tmp_path), use_timestamp=False, experiment_name="t", **kw
    )


def test_column_order(tmp_path):
    _saver(tmp_path).on_latent_end(dataset=None, embeddings=_small())
    nodes = (tmp_path / "embeddings_t_nodes.csv").read_text().splitlines()
    edges = (tmp_path / "embeddings_t_edges.csv").read_text().splitlines()
    assert nodes[0] == "node_id,x0,x1"
    assert len(nodes) == 4
    assert edges[0] == "source,target,weight"
    assert len(edges) == 3


def test_column_dtypes(tmp_path):
    _saver(tmp_path).on_latent_end(dataset=None, embeddings=_small())
    nodes = pd.read_csv(tmp_path / "embeddings_t_nodes.csv")
    edges = pd.read_csv(tmp_path / "embeddings_t_edges.csv")
    for col in ("node_id",):
        assert nodes[col].dtype == np.int64
    for col in ("source", "target"):
        assert edges[col].dtype == np.int64
    for col in ("x0", "x1"):
        assert nodes[col].dtype == np.float64
    assert edges["weight"].dtype == np.float64
    assert edges["weight"].tolist() == [1.0, 0.5]


def test_written_without_additional_outputs(tmp_path):
    _saver(tmp_path, save_additional_outputs=False).on_latent_end(
        dataset=None, embeddings=_small()
    )
    assert (tmp_path / "embeddings_t_nodes.csv").is_file()
    assert (tmp_path / "embeddings_t_edges.csv").is_file()


def test_no_graph_no_files(tmp_path):
    _saver(tmp_path).on_latent_end(dataset=None, embeddings={"embeddings": np.zeros((5, 1))})
    for pattern in ("*_nodes.csv", "*_edges.csv", "*_graph.npz"):
        assert list(tmp_path.glob(pattern)) == []


def test_archive_loads_back(tmp_path):
    manykinds = pytest.importorskip("manykinds")
    _saver(tmp_path).on_latent_end(dataset=None, embeddings=_small())
    sg = manykinds.SparseGraph.load(tmp_path / "embeddings_t_graph.npz")
    g = _g3()
    np.testing.assert_array_equal(sg.edges, g["edges"])
    np.testing.assert_array_equal(sg.node_ids, g["node_ids"])
    np.testing.assert_array_equal(sg.edge_weights, g["edge_weights"])
    assert tuple(sg.provenance) == ("manylatents.outputs.graph",)


def test_manykinds_missing(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "manykinds", None)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        _saver(tmp_path).on_latent_end(dataset=None, embeddings=_small())
    assert (tmp_path / "embeddings_t_nodes.csv").is_file()
    assert (tmp_path / "embeddings_t_edges.csv").is_file()
    assert list(tmp_path.glob("*.npz")) == []
    hits = [
        r for r in caplog.records
        if r.name == LOGGER_NAME and r.levelno == logging.WARNING
        and "manykinds" in r.getMessage()
    ]
    assert len(hits) == 1


def test_csv_mode_skips_graph_json(tmp_path):
    _saver(tmp_path, save_additional_outputs=True).on_latent_end(
        dataset=None, embeddings=_small()
    )
    assert (tmp_path / "embeddings_t_nodes.csv").is_file()
    assert (tmp_path / "embeddings_t_edges.csv").is_file()
    assert not (tmp_path / "embeddings_t_graph.json").exists()


def test_npy_mode_unchanged(tmp_path):
    _saver(tmp_path, save_format="npy", save_additional_outputs=True).on_latent_end(
        dataset=None, embeddings=_small()
    )
    assert (tmp_path / "embeddings_t_graph.json").is_file()
    for pattern in ("*_nodes.csv", "*_edges.csv", "*_graph.npz"):
        assert list(tmp_path.glob(pattern)) == []


def test_paths_in_callback_outputs(tmp_path):
    pytest.importorskip("manykinds")
    out = _saver(tmp_path).on_latent_end(dataset=None, embeddings=_small())
    for key in ("graph_nodes_path", "graph_edges_path", "graph_npz_path"):
        assert key in out
        assert os.path.isfile(out[key])


def test_no_npz_path_without_manykinds(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "manykinds", None)
    out = _saver(tmp_path).on_latent_end(dataset=None, embeddings=_small())
    assert "graph_nodes_path" in out
    assert "graph_edges_path" in out
    assert "graph_npz_path" not in out
