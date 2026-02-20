# Embedding Artifact Detection Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect when dimensionality reduction creates biological artifacts by comparing DE results across embedding parameter settings.

**Architecture:** Three-phase pipeline. Phase 1 is a manual exploratory script (mbyl_for_practitioners) that gates Phases 2-3. Phase 2 adds Leiden clustering to manylatents core. Phase 3 adds DE + complement set analysis to manylatents-omics. Each feature phase gets its own PR.

**Tech Stack:** scanpy (DE, clustering), leidenalg + python-igraph (Leiden), anndata (data format), matplotlib (plots)

---

## Phase 1: Manual Check Script

This runs from the omics venv since it needs scanpy. Output goes to `mbyl_for_practitioners/notes/`.

### Task 1: Install leidenalg in omics venv and inspect data

**Files:**
- Modify: `/network/scratch/c/cesar.valdez/lrw/omics/pyproject.toml` (add `leidenalg` to singlecell extra)

**Step 1: Add leidenalg to omics singlecell extra**

In `/network/scratch/c/cesar.valdez/lrw/omics/pyproject.toml`, change:
```toml
singlecell = ["anndata>=0.11.3", "python-igraph>=1.0.0", "scanpy>=1.11.5"]
```
to:
```toml
singlecell = ["anndata>=0.11.3", "leidenalg>=0.10", "python-igraph>=1.0.0", "scanpy>=1.11.5"]
```

**Step 2: Sync the omics venv**

Run from `/network/scratch/c/cesar.valdez/lrw/omics/`:
```bash
uv sync --extra singlecell
```
Expected: resolves and installs leidenalg.

**Step 3: Verify imports**

```bash
.venv/bin/python -c "import scanpy, leidenalg, igraph; print('OK')"
```
Expected: `OK`

**Step 4: Inspect the h5ad data**

Run a quick Python snippet from the omics venv:
```python
import scanpy as sc
adata = sc.read_h5ad("/network/scratch/c/cesar.valdez/lrw/manylatents/data/scRNAseq/EBT_2k_hvg.h5ad")
print(f"Shape: {adata.shape}")
print(f"obs columns: {list(adata.obs.columns)}")
print(f"var columns: {list(adata.var.columns)}")
print(f"var_names[:10]: {list(adata.var_names[:10])}")
print(f"obsm keys: {list(adata.obsm.keys())}")
print(f"X dtype: {adata.X.dtype}, min: {adata.X.min():.2f}, max: {adata.X.max():.2f}")
print(f"Has raw: {adata.raw is not None}")
```

Log the output. Key questions: Are gene names symbols or Ensembl IDs? Is X already log-normalized? Is there a PCA in obsm?

**Step 5: Commit the pyproject.toml change**

```bash
cd /network/scratch/c/cesar.valdez/lrw/omics
git add pyproject.toml
git commit -m "deps: add leidenalg to singlecell extra"
```

---

### Task 2: Write the embedding artifact check script

**Files:**
- Create: `/network/scratch/c/cesar.valdez/mbyl_for_practitioners/experiments/scripts/embedding_artifact_check.py`

This script does the full Phase 1 pipeline. It must be run with the omics venv python.

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Embedding artifact detection: manual check on embryoid body data.

Compares DE results from two UMAP parameter settings to find genes
that are artifacts of the embedding rather than real biology.

Run from mbyl_for_practitioners/:
    /network/scratch/c/cesar.valdez/lrw/omics/.venv/bin/python \
        experiments/scripts/embedding_artifact_check.py
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Config ---
DATA_PATH = "/network/scratch/c/cesar.valdez/lrw/manylatents/data/scRNAseq/EBT_2k_hvg.h5ad"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "notes" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SETTING_A = {"n_neighbors": 5, "min_dist": 0.01, "label": "fragmenting"}
SETTING_B = {"n_neighbors": 50, "min_dist": 0.5, "label": "smooth"}
LEIDEN_RESOLUTION = 0.5
DE_METHOD = "wilcoxon"
P_THRESHOLD = 0.05
LFC_THRESHOLD = 1.0
N_TOP_GENES = 200


def load_data():
    """Load and inspect the embryoid body data."""
    adata = sc.read_h5ad(DATA_PATH)
    print(f"Shape: {adata.shape}")
    print(f"obs columns: {list(adata.obs.columns)}")
    print(f"var_names[:10]: {list(adata.var_names[:10])}")
    print(f"X dtype: {adata.X.dtype}")

    # Check if already log-normalized by looking at value range
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    xmax = x.max()
    print(f"X range: [{x.min():.2f}, {xmax:.2f}]")

    # If raw counts (integers, large max), normalize
    if xmax > 20:
        print("Data appears to be raw counts. Normalizing...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("Data appears already log-normalized.")

    # PCA if not present
    if "X_pca" not in adata.obsm:
        print("Computing PCA...")
        sc.tl.pca(adata, n_comps=50)

    return adata


def embed_and_cluster(adata, setting, key_suffix):
    """Build kNN graph, embed with UMAP, cluster with Leiden."""
    print(f"\n--- Setting: {setting['label']} (k={setting['n_neighbors']}, min_dist={setting['min_dist']}) ---")

    sc.pp.neighbors(adata, n_neighbors=setting["n_neighbors"], use_rep="X_pca")
    sc.tl.umap(adata, min_dist=setting["min_dist"])

    # Save embedding
    umap_key = f"X_umap_{key_suffix}"
    adata.obsm[umap_key] = adata.obsm["X_umap"].copy()

    # Cluster on this graph
    leiden_key = f"leiden_{key_suffix}"
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added=leiden_key)

    n_clusters = adata.obs[leiden_key].nunique()
    print(f"  Clusters: {n_clusters}")
    print(f"  Cluster sizes: {adata.obs[leiden_key].value_counts().to_dict()}")

    return umap_key, leiden_key, n_clusters


def run_de(adata, leiden_key, de_key):
    """Run Wilcoxon rank-sum DE and extract significant genes."""
    print(f"\n--- DE on {leiden_key} ---")
    sc.tl.rank_genes_groups(adata, groupby=leiden_key, method=DE_METHOD,
                            key_added=de_key, n_genes=N_TOP_GENES)

    # Extract results into a tidy dataframe
    result = adata.uns[de_key]
    groups = result["names"].dtype.names
    rows = []
    for group in groups:
        for i in range(len(result["names"][group])):
            gene = result["names"][group][i]
            pval_adj = result["pvals_adj"][group][i]
            lfc = result["logfoldchanges"][group][i]
            score = result["scores"][group][i]
            rows.append({
                "gene": gene,
                "cluster": group,
                "pval_adj": pval_adj,
                "logfoldchange": lfc,
                "score": score,
            })

    df = pd.DataFrame(rows)
    sig = df[(df["pval_adj"] < P_THRESHOLD) & (df["logfoldchange"].abs() > LFC_THRESHOLD)]
    sig_genes = set(sig["gene"].unique())

    print(f"  Total DE genes tested: {df['gene'].nunique()}")
    print(f"  Significant (padj<{P_THRESHOLD}, |lfc|>{LFC_THRESHOLD}): {len(sig_genes)}")

    return sig_genes, sig, df


def compute_complement(genes_a, genes_b, df_a, df_b):
    """Compute complement sets."""
    robust = genes_a & genes_b
    artifacts = genes_a - genes_b
    missed = genes_b - genes_a

    results = {
        "n_genes_a": len(genes_a),
        "n_genes_b": len(genes_b),
        "n_robust": len(robust),
        "n_artifacts": len(artifacts),
        "n_missed": len(missed),
        "jaccard": len(robust) / len(genes_a | genes_b) if (genes_a | genes_b) else 0,
        "robust": sorted(robust),
        "artifacts": sorted(artifacts),
        "missed": sorted(missed),
    }

    print(f"\n=== COMPLEMENT SET RESULTS ===")
    print(f"  |genes_A| (fragmenting): {results['n_genes_a']}")
    print(f"  |genes_B| (smooth):      {results['n_genes_b']}")
    print(f"  |robust| (A ∩ B):        {results['n_robust']}")
    print(f"  |artifacts| (A \\ B):     {results['n_artifacts']}")
    print(f"  |missed| (B \\ A):        {results['n_missed']}")
    print(f"  Jaccard similarity:       {results['jaccard']:.3f}")

    if results["n_artifacts"] > 0:
        print(f"\n  Artifact genes: {results['artifacts'][:20]}{'...' if len(results['artifacts']) > 20 else ''}")

    return results


def plot_embeddings(adata, umap_a, leiden_a, umap_b, leiden_b):
    """Side-by-side UMAP plots colored by cluster assignments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, umap_key, leiden_key, title in [
        (axes[0], umap_a, leiden_a, f"Setting A: fragmenting (k=5, min_dist=0.01)"),
        (axes[1], umap_b, leiden_b, f"Setting B: smooth (k=50, min_dist=0.5)"),
    ]:
        coords = adata.obsm[umap_key]
        clusters = adata.obs[leiden_key].astype(int)
        n_clusters = clusters.nunique()
        cmap = plt.cm.get_cmap("tab20", n_clusters)

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap=cmap,
                             s=3, alpha=0.7, rasterized=True)
        ax.set_title(f"{title}\n{n_clusters} clusters")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_aspect("equal")

    plt.tight_layout()
    path = OUTPUT_DIR / "embedding_artifact_umap_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved UMAP comparison: {path}")
    plt.close(fig)


def plot_complement_venn(results):
    """Simple bar chart of complement set sizes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Robust\n(A ∩ B)", "Artifacts\n(A \\ B)", "Missed\n(B \\ A)"]
    values = [results["n_robust"], results["n_artifacts"], results["n_missed"]]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Number of DE genes")
    ax.set_title(f"Complement Set Analysis\nJaccard = {results['jaccard']:.3f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = OUTPUT_DIR / "embedding_artifact_complement_set.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved complement set plot: {path}")
    plt.close(fig)


def write_note(adata, results, n_clusters_a, n_clusters_b):
    """Write the results as a dated markdown note."""
    note_path = OUTPUT_DIR.parent / "2026-02-19_embedding_artifact_detection.md"

    artifact_genes_str = ", ".join(f"`{g}`" for g in results["artifacts"][:50])
    robust_genes_str = ", ".join(f"`{g}`" for g in results["robust"][:30])

    note = f"""# Embedding Artifact Detection — Embryoid Body

*February 19, 2026*

## Setup

- **Data**: `EBT_2k_hvg.h5ad` — {adata.shape[0]} cells, {adata.shape[1]} HVGs
- **Setting A** (fragmenting): UMAP k=5, min_dist=0.01 → {n_clusters_a} Leiden clusters
- **Setting B** (smooth): UMAP k=50, min_dist=0.5 → {n_clusters_b} Leiden clusters
- **DE**: Wilcoxon rank-sum, padj < {P_THRESHOLD}, |log2FC| > {LFC_THRESHOLD}
- **Leiden resolution**: {LEIDEN_RESOLUTION}

![UMAP comparison](figures/embedding_artifact_umap_comparison.png)

## Complement Set Results

| Metric | Count |
|--------|-------|
| \\|genes_A\\| (fragmenting) | {results['n_genes_a']} |
| \\|genes_B\\| (smooth) | {results['n_genes_b']} |
| \\|robust\\| (A ∩ B) | {results['n_robust']} |
| \\|artifacts\\| (A \\\\ B) | {results['n_artifacts']} |
| \\|missed\\| (B \\\\ A) | {results['n_missed']} |
| Jaccard similarity | {results['jaccard']:.3f} |

![Complement set](figures/embedding_artifact_complement_set.png)

## Artifact Genes (A \\\\ B)

{artifact_genes_str if results['n_artifacts'] > 0 else "**None found.** The complement set is empty — embedding parameters did not affect DE results on this dataset."}

## Robust Genes (A ∩ B)

{robust_genes_str}{'...' if len(results['robust']) > 30 else ''}

## Interpretation

{"The non-empty complement set confirms that embedding parameter choices create spurious DE genes. These 'artifact' genes appear as markers only because the fragmenting setting (k=5) splits the continuous trajectory into discrete blobs. Proceed to Phases 2-3." if results['n_artifacts'] > 10 else ""}{"The complement set is near-empty. Embedding parameters have minimal effect on downstream DE for this dataset. The paper thesis needs revisiting." if results['n_artifacts'] <= 5 else ""}
"""
    note_path.write_text(note)
    print(f"\nSaved note: {note_path}")

    # Also save raw results as JSON for downstream use
    json_path = OUTPUT_DIR / "embedding_artifact_results.json"
    serializable = {k: v if not isinstance(v, set) else sorted(v) for k, v in results.items()}
    json_path.write_text(json.dumps(serializable, indent=2))
    print(f"Saved JSON results: {json_path}")


def main():
    adata = load_data()

    # Setting A: fragmenting
    umap_a, leiden_a, n_clusters_a = embed_and_cluster(adata, SETTING_A, "frag")
    genes_a, sig_a, df_a = run_de(adata, leiden_a, "de_frag")

    # Setting B: smooth
    umap_b, leiden_b, n_clusters_b = embed_and_cluster(adata, SETTING_B, "smooth")
    genes_b, sig_b, df_b = run_de(adata, leiden_b, "de_smooth")

    # Complement set
    results = compute_complement(genes_a, genes_b, sig_a, sig_b)

    # Plots
    plot_embeddings(adata, umap_a, leiden_a, umap_b, leiden_b)
    plot_complement_venn(results)

    # Note
    write_note(adata, results, n_clusters_a, n_clusters_b)

    print("\n=== DONE ===")
    return results


if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

```bash
cd /network/scratch/c/cesar.valdez/mbyl_for_practitioners
/network/scratch/c/cesar.valdez/lrw/omics/.venv/bin/python \
    experiments/scripts/embedding_artifact_check.py
```

Expected: prints data inspection, cluster counts, DE gene counts, complement set numbers. Saves plots + note + JSON.

**Step 3: Inspect results and log to MEMORY.md**

Read the terminal output and the generated note. Log the key numbers (|artifacts|, |robust|, etc.) to MEMORY.md.

**Gate decision**: If |artifacts| > 10, proceed to Tasks 3+. If ≈ 0, stop and report back.

---

## Phase 2: Leiden Clustering Module (manylatents core)

### Task 3: Create feature branch for Leiden clustering

**Step 1: Create and checkout branch**

```bash
cd /network/scratch/c/cesar.valdez/lrw/manylatents
git checkout -b feat/leiden-clustering
```

---

### Task 4: Add optional clustering dependencies

**Files:**
- Modify: `/network/scratch/c/cesar.valdez/lrw/manylatents/pyproject.toml`

**Step 1: Add clustering extra to pyproject.toml**

Find the `[project.optional-dependencies]` section. Add after the existing `torchdr` line:

```toml
clustering = ["leidenalg>=0.10", "python-igraph>=1.0"]
```

Also add `clustering` to the `all` extra:

```toml
all = ["manylatents[tracking,hf,dynamics,transport,topology,cluster,torchdr,clustering]"]
```

**Step 2: Sync**

```bash
cd /network/scratch/c/cesar.valdez/lrw/manylatents
uv sync --extra clustering
```

**Step 3: Verify**

```bash
.venv/bin/python -c "import leidenalg, igraph; print('OK')"
```
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add optional clustering extra (leidenalg, python-igraph)"
```

---

### Task 5: Write failing test for LeidenClustering

**Files:**
- Create: `/network/scratch/c/cesar.valdez/lrw/manylatents/tests/test_clustering.py`

**Step 1: Write the test**

```python
"""Tests for LeidenClustering analysis module."""
import numpy as np
import pytest

leidenalg = pytest.importorskip("leidenalg")


def _make_blobs(n_per_cluster=100, n_clusters=5, n_features=10, seed=42):
    """Generate well-separated Gaussian blobs."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, n_features) * 10
    data = np.vstack([
        centers[i] + rng.randn(n_per_cluster, n_features) * 0.3
        for i in range(n_clusters)
    ])
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    return data, labels


class TestLeidenClustering:
    def test_fit_returns_labels(self):
        """fit() returns integer cluster labels with correct shape."""
        from manylatents.analysis.clustering import LeidenClustering

        data, _ = _make_blobs()
        lc = LeidenClustering(resolution=0.5, n_neighbors=15, random_state=42)
        labels = lc.fit(data)

        assert isinstance(labels, np.ndarray)
        assert labels.shape == (data.shape[0],)
        assert labels.dtype in (np.int32, np.int64)

    def test_finds_approximately_correct_clusters(self):
        """On well-separated blobs, Leiden finds ~5 clusters."""
        from manylatents.analysis.clustering import LeidenClustering

        data, _ = _make_blobs(n_clusters=5)
        lc = LeidenClustering(resolution=0.5, n_neighbors=15, random_state=42)
        labels = lc.fit(data)

        n_clusters = len(np.unique(labels))
        assert 3 <= n_clusters <= 8, f"Expected ~5 clusters, got {n_clusters}"

    def test_fit_from_graph(self):
        """fit_from_graph() accepts a sparse adjacency matrix."""
        from manylatents.analysis.clustering import LeidenClustering
        from sklearn.neighbors import kneighbors_graph

        data, _ = _make_blobs()
        adj = kneighbors_graph(data, n_neighbors=15, mode="connectivity")

        lc = LeidenClustering(random_state=42)
        labels = lc.fit_from_graph(adj)

        assert isinstance(labels, np.ndarray)
        assert labels.shape == (data.shape[0],)

    def test_deterministic(self):
        """Same random_state produces same labels."""
        from manylatents.analysis.clustering import LeidenClustering

        data, _ = _make_blobs()
        lc1 = LeidenClustering(random_state=42)
        lc2 = LeidenClustering(random_state=42)

        labels1 = lc1.fit(data)
        labels2 = lc2.fit(data)
        np.testing.assert_array_equal(labels1, labels2)

    def test_resolution_affects_cluster_count(self):
        """Higher resolution produces more clusters."""
        from manylatents.analysis.clustering import LeidenClustering

        data, _ = _make_blobs(n_clusters=5)

        lc_low = LeidenClustering(resolution=0.1, n_neighbors=15, random_state=42)
        lc_high = LeidenClustering(resolution=2.0, n_neighbors=15, random_state=42)

        n_low = len(np.unique(lc_low.fit(data)))
        n_high = len(np.unique(lc_high.fit(data)))

        assert n_high >= n_low, f"High res ({n_high}) should have >= clusters than low res ({n_low})"
```

**Step 2: Run the test to verify it fails**

```bash
cd /network/scratch/c/cesar.valdez/lrw/manylatents
uv run pytest tests/test_clustering.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'manylatents.analysis'`

---

### Task 6: Implement LeidenClustering

**Files:**
- Create: `/network/scratch/c/cesar.valdez/lrw/manylatents/manylatents/analysis/__init__.py`
- Create: `/network/scratch/c/cesar.valdez/lrw/manylatents/manylatents/analysis/clustering.py`

**Step 1: Create `__init__.py`**

```python
"""Post-embedding analysis tools (clustering, etc.)."""
```

**Step 2: Create `clustering.py`**

```python
"""Leiden community detection for embedding analysis."""
import numpy as np
import scipy.sparse


class LeidenClustering:
    """Cluster embeddings or kNN graphs using the Leiden algorithm.

    Parameters
    ----------
    resolution : float
        Leiden resolution parameter. Higher values produce more clusters.
    n_neighbors : int
        Number of neighbors for kNN graph construction (used by fit()).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, resolution: float = 0.5, n_neighbors: int = 15, random_state: int = 42):
        self.resolution = resolution
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, embedding: np.ndarray) -> np.ndarray:
        """Build kNN graph on embedding, run Leiden, return cluster labels.

        Parameters
        ----------
        embedding : np.ndarray
            (n_samples, n_features) array.

        Returns
        -------
        np.ndarray
            Integer cluster labels, shape (n_samples,).
        """
        from sklearn.neighbors import kneighbors_graph

        adj = kneighbors_graph(
            embedding, n_neighbors=self.n_neighbors, mode="connectivity", include_self=False
        )
        # Symmetrize
        adj = adj + adj.T
        adj[adj > 1] = 1

        return self.fit_from_graph(adj)

    def fit_from_graph(self, adjacency: scipy.sparse.spmatrix) -> np.ndarray:
        """Run Leiden on a precomputed adjacency matrix.

        Parameters
        ----------
        adjacency : scipy.sparse.spmatrix
            Sparse adjacency matrix (n_samples, n_samples).

        Returns
        -------
        np.ndarray
            Integer cluster labels, shape (n_samples,).
        """
        import igraph as ig
        import leidenalg

        adj_coo = scipy.sparse.coo_matrix(adjacency)
        edges = list(zip(adj_coo.row.tolist(), adj_coo.col.tolist()))
        weights = adj_coo.data.tolist()

        g = ig.Graph(n=adjacency.shape[0], edges=edges, directed=False)
        g.es["weight"] = weights
        g.simplify(combine_edges="max")

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            seed=self.random_state,
            weights="weight",
        )

        return np.array(partition.membership, dtype=np.int64)
```

**Step 3: Run tests to verify they pass**

```bash
cd /network/scratch/c/cesar.valdez/lrw/manylatents
uv run pytest tests/test_clustering.py -v
```
Expected: all 5 tests PASS.

**Step 4: Commit**

```bash
git add manylatents/analysis/__init__.py manylatents/analysis/clustering.py tests/test_clustering.py
git commit -m "feat: add LeidenClustering analysis module"
```

---

### Task 7: Add Hydra config for Leiden

**Files:**
- Create: `/network/scratch/c/cesar.valdez/lrw/manylatents/manylatents/configs/analysis/leiden.yaml`

**Step 1: Create config directory and YAML**

```yaml
_target_: manylatents.analysis.clustering.LeidenClustering
resolution: 0.5
n_neighbors: 15
random_state: 42
```

**Step 2: Commit**

```bash
git add manylatents/configs/analysis/leiden.yaml
git commit -m "config: add Hydra config for Leiden clustering"
```

---

### Task 8: Run full test suite and create PR

**Step 1: Run full test suite**

```bash
cd /network/scratch/c/cesar.valdez/lrw/manylatents
uv run pytest tests/ -x -q
```
Expected: all tests pass (existing + new clustering tests).

**Step 2: Push and create PR**

```bash
git push -u origin feat/leiden-clustering
```

Create PR with title: "feat: add Leiden clustering analysis module"

Body:
```
## Summary
- Adds `manylatents.analysis.clustering.LeidenClustering` for post-embedding cluster analysis
- `fit(embedding)` builds kNN graph + runs Leiden, `fit_from_graph(adjacency)` accepts precomputed graphs
- Optional deps: `leidenalg`, `python-igraph` under `[clustering]` extra
- Hydra config at `analysis/leiden.yaml`

## Test plan
- [ ] `tests/test_clustering.py` — 5 tests covering fit, fit_from_graph, determinism, resolution effect
- [ ] Full test suite passes with no regressions
```

---

## Phase 3: DE + Complement Set Analysis (manylatents-omics)

### Task 9: Create feature branch in omics repo

**Step 1: Create branch**

```bash
cd /network/scratch/c/cesar.valdez/lrw/omics
git checkout -b feat/embedding-audit
```

---

### Task 10: Write failing tests for DifferentialExpression

**Files:**
- Create: `/network/scratch/c/cesar.valdez/lrw/omics/tests/singlecell/__init__.py`
- Create: `/network/scratch/c/cesar.valdez/lrw/omics/tests/singlecell/test_differential_expression.py`

**Step 1: Write the test**

```python
"""Tests for DifferentialExpression wrapper."""
import numpy as np
import pytest

sc = pytest.importorskip("scanpy")


def _make_test_adata(n_cells=200, n_genes=50, n_clusters=3, seed=42):
    """Create a minimal AnnData with cluster assignments for DE testing."""
    rng = np.random.RandomState(seed)
    # Create data where cluster 0 has high expression of genes 0-4
    X = rng.randn(n_cells, n_genes).astype(np.float32)
    labels = np.repeat(np.arange(n_clusters), n_cells // n_clusters)
    # Boost genes 0-4 in cluster 0
    cluster_0_mask = labels == 0
    X[cluster_0_mask, :5] += 5.0

    import anndata
    adata = anndata.AnnData(
        X=X,
        obs={"cluster": labels.astype(str)},
        var={"gene_name": [f"gene_{i}" for i in range(n_genes)]},
    )
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    return adata


class TestDifferentialExpression:
    def test_run_returns_dataframe(self):
        """run() returns a DataFrame with expected columns."""
        from manylatents.singlecell.analysis.differential_expression import DifferentialExpression

        adata = _make_test_adata()
        de = DifferentialExpression(method="wilcoxon")
        df = de.run(adata, groupby="cluster")

        assert "gene" in df.columns
        assert "cluster" in df.columns
        assert "pval_adj" in df.columns
        assert "logfoldchange" in df.columns
        assert len(df) > 0

    def test_get_significant_genes(self):
        """get_significant_genes() returns a set of gene names."""
        from manylatents.singlecell.analysis.differential_expression import DifferentialExpression

        adata = _make_test_adata()
        de = DifferentialExpression(method="wilcoxon", p_threshold=0.05, lfc_threshold=0.5)
        de.run(adata, groupby="cluster")
        genes = de.get_significant_genes(adata)

        assert isinstance(genes, set)
        # The boosted genes should be significant
        assert len(genes) > 0

    def test_boosted_genes_detected(self):
        """DE detects the genes we artificially boosted in cluster 0."""
        from manylatents.singlecell.analysis.differential_expression import DifferentialExpression

        adata = _make_test_adata(n_cells=300)
        de = DifferentialExpression(method="wilcoxon", p_threshold=0.05, lfc_threshold=1.0)
        de.run(adata, groupby="cluster")
        genes = de.get_significant_genes(adata)

        boosted = {f"gene_{i}" for i in range(5)}
        assert boosted & genes, f"Expected some of {boosted} in significant genes {genes}"
```

**Step 2: Run to verify failure**

```bash
cd /network/scratch/c/cesar.valdez/lrw/omics
uv run pytest tests/singlecell/test_differential_expression.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'manylatents.singlecell.analysis'`

---

### Task 11: Implement DifferentialExpression

**Files:**
- Create: `/network/scratch/c/cesar.valdez/lrw/omics/manylatents/singlecell/analysis/__init__.py`
- Create: `/network/scratch/c/cesar.valdez/lrw/omics/manylatents/singlecell/analysis/differential_expression.py`

**Step 1: Create `__init__.py`**

```python
"""Single-cell analysis tools: DE, complement sets, embedding audits."""
```

**Step 2: Create `differential_expression.py`**

```python
"""Differential expression wrapper around scanpy."""
import pandas as pd
import scanpy as sc


class DifferentialExpression:
    """Run DE on an AnnData object given cluster assignments.

    Parameters
    ----------
    method : str
        Statistical test: 'wilcoxon', 't-test', 't-test_overestim_var'.
    p_threshold : float
        Adjusted p-value cutoff for significance.
    lfc_threshold : float
        Minimum absolute log-fold-change for significance.
    n_genes : int
        Number of top genes to test per cluster.
    """

    def __init__(self, method: str = "wilcoxon", p_threshold: float = 0.05,
                 lfc_threshold: float = 1.0, n_genes: int = 200):
        self.method = method
        self.p_threshold = p_threshold
        self.lfc_threshold = lfc_threshold
        self.n_genes = n_genes
        self._key = None

    def run(self, adata: sc.AnnData, groupby: str, key_added: str = "de") -> pd.DataFrame:
        """Run rank_genes_groups, return tidy DataFrame.

        Parameters
        ----------
        adata : sc.AnnData
            Annotated data matrix. Modified in-place (results stored in adata.uns).
        groupby : str
            Column in adata.obs to group by.
        key_added : str
            Key for storing results in adata.uns.

        Returns
        -------
        pd.DataFrame
            Columns: gene, cluster, pval_adj, logfoldchange, score.
        """
        self._key = key_added
        sc.tl.rank_genes_groups(adata, groupby=groupby, method=self.method,
                                key_added=key_added, n_genes=self.n_genes)

        result = adata.uns[key_added]
        groups = result["names"].dtype.names
        rows = []
        for group in groups:
            for i in range(len(result["names"][group])):
                rows.append({
                    "gene": result["names"][group][i],
                    "cluster": group,
                    "pval_adj": result["pvals_adj"][group][i],
                    "logfoldchange": result["logfoldchanges"][group][i],
                    "score": result["scores"][group][i],
                })

        self._df = pd.DataFrame(rows)
        return self._df

    def get_significant_genes(self, adata: sc.AnnData, key: str = None) -> set:
        """Extract set of significant gene names from a completed DE run.

        Parameters
        ----------
        adata : sc.AnnData
            Must have DE results in adata.uns (from a prior run() call).
        key : str, optional
            Key in adata.uns. Defaults to the key used in the last run() call.

        Returns
        -------
        set
            Gene names passing both p-value and log-fold-change thresholds.
        """
        if self._df is None:
            raise RuntimeError("Call run() before get_significant_genes()")

        sig = self._df[
            (self._df["pval_adj"] < self.p_threshold)
            & (self._df["logfoldchange"].abs() > self.lfc_threshold)
        ]
        return set(sig["gene"].unique())
```

**Step 3: Run tests**

```bash
cd /network/scratch/c/cesar.valdez/lrw/omics
uv run pytest tests/singlecell/test_differential_expression.py -v
```
Expected: all 3 tests PASS.

**Step 4: Commit**

```bash
git add manylatents/singlecell/analysis/__init__.py manylatents/singlecell/analysis/differential_expression.py tests/singlecell/__init__.py tests/singlecell/test_differential_expression.py
git commit -m "feat: add DifferentialExpression wrapper for scanpy DE"
```

---

### Task 12: Write failing tests for ComplementSetAnalysis

**Files:**
- Create: `/network/scratch/c/cesar.valdez/lrw/omics/tests/singlecell/test_complement_set.py`

**Step 1: Write the test**

```python
"""Tests for ComplementSetAnalysis."""
import pytest


class TestComplementSetAnalysis:
    def test_basic_set_operations(self):
        """Correctly computes intersection, difference, Jaccard."""
        from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis

        csa = ComplementSetAnalysis()
        genes_a = {"A", "B", "C", "D"}
        genes_b = {"C", "D", "E", "F"}

        result = csa.compare(genes_a, genes_b)

        assert result["robust"] == {"C", "D"}
        assert result["artifacts"] == {"A", "B"}
        assert result["missed"] == {"E", "F"}
        assert result["n_robust"] == 2
        assert result["n_artifacts"] == 2
        assert result["n_missed"] == 2
        assert abs(result["jaccard"] - 2 / 6) < 1e-9

    def test_empty_complement(self):
        """When sets are identical, artifacts and missed are empty."""
        from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis

        csa = ComplementSetAnalysis()
        genes = {"A", "B", "C"}
        result = csa.compare(genes, genes)

        assert result["n_artifacts"] == 0
        assert result["n_missed"] == 0
        assert result["jaccard"] == 1.0

    def test_disjoint_sets(self):
        """When sets are disjoint, robust is empty, Jaccard is 0."""
        from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis

        csa = ComplementSetAnalysis()
        result = csa.compare({"A", "B"}, {"C", "D"})

        assert result["n_robust"] == 0
        assert result["jaccard"] == 0.0

    def test_empty_inputs(self):
        """Handles empty gene sets gracefully."""
        from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis

        csa = ComplementSetAnalysis()
        result = csa.compare(set(), set())

        assert result["n_robust"] == 0
        assert result["jaccard"] == 0.0
```

**Step 2: Run to verify failure**

```bash
cd /network/scratch/c/cesar.valdez/lrw/omics
uv run pytest tests/singlecell/test_complement_set.py -v
```
Expected: FAIL with import error.

---

### Task 13: Implement ComplementSetAnalysis

**Files:**
- Create: `/network/scratch/c/cesar.valdez/lrw/omics/manylatents/singlecell/analysis/complement_set.py`

**Step 1: Write the implementation**

```python
"""Compare DE gene lists across embedding parameter settings."""
import pandas as pd


class ComplementSetAnalysis:
    """Compare DE results from two parameter settings.

    Computes the complement set to identify genes that are artifacts
    of embedding parameters versus robust biological signals.
    """

    def compare(self, genes_a: set, genes_b: set,
                df_a: pd.DataFrame = None, df_b: pd.DataFrame = None) -> dict:
        """Compute complement sets with optional per-gene stats.

        Parameters
        ----------
        genes_a : set
            Significant DE genes from setting A (e.g. fragmenting).
        genes_b : set
            Significant DE genes from setting B (e.g. smooth).
        df_a : pd.DataFrame, optional
            Full DE results from setting A (for per-gene stats on artifacts).
        df_b : pd.DataFrame, optional
            Full DE results from setting B.

        Returns
        -------
        dict
            Keys: robust, artifacts, missed (sets), n_robust, n_artifacts,
            n_missed (ints), jaccard (float), artifact_genes_df (DataFrame or None).
        """
        robust = genes_a & genes_b
        artifacts = genes_a - genes_b
        missed = genes_b - genes_a
        union = genes_a | genes_b
        jaccard = len(robust) / len(union) if union else 0.0

        artifact_df = None
        if df_a is not None and len(artifacts) > 0:
            artifact_df = df_a[df_a["gene"].isin(artifacts)].copy()

        return {
            "robust": robust,
            "artifacts": artifacts,
            "missed": missed,
            "n_robust": len(robust),
            "n_artifacts": len(artifacts),
            "n_missed": len(missed),
            "jaccard": jaccard,
            "artifact_genes_df": artifact_df,
        }
```

**Step 2: Run tests**

```bash
cd /network/scratch/c/cesar.valdez/lrw/omics
uv run pytest tests/singlecell/test_complement_set.py -v
```
Expected: all 4 tests PASS.

**Step 3: Commit**

```bash
git add manylatents/singlecell/analysis/complement_set.py tests/singlecell/test_complement_set.py
git commit -m "feat: add ComplementSetAnalysis for comparing DE gene lists"
```

---

### Task 14: Write failing test for EmbeddingAudit

**Files:**
- Create: `/network/scratch/c/cesar.valdez/lrw/omics/tests/singlecell/test_embedding_audit.py`

**Step 1: Write the test**

```python
"""Tests for EmbeddingAudit end-to-end pipeline."""
import numpy as np
import pytest

sc = pytest.importorskip("scanpy")
pytest.importorskip("leidenalg")


def _make_trajectory_adata(n_cells=300, n_genes=100, seed=42):
    """Create synthetic trajectory data where fragmenting settings produce artifacts."""
    rng = np.random.RandomState(seed)

    # Simulate a 1D trajectory (pseudotime) embedded in gene space
    pseudotime = np.linspace(0, 1, n_cells)

    # Base expression: smooth gradients along pseudotime
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for g in range(n_genes):
        freq = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        X[:, g] = np.sin(freq * pseudotime * 2 * np.pi + phase) + rng.randn(n_cells) * 0.3

    import anndata
    adata = anndata.AnnData(
        X=X,
        var={"gene_name": [f"gene_{i}" for i in range(n_genes)]},
    )
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Add PCA
    sc.tl.pca(adata, n_comps=min(50, n_genes - 1))

    return adata


class TestEmbeddingAudit:
    def test_run_returns_complement_results(self):
        """run() returns dict with complement set keys."""
        from manylatents.singlecell.analysis.embedding_audit import EmbeddingAudit

        adata = _make_trajectory_adata()
        audit = EmbeddingAudit(
            setting_a={"n_neighbors": 5, "min_dist": 0.01},
            setting_b={"n_neighbors": 30, "min_dist": 0.5},
            leiden_resolution=0.5,
        )
        results = audit.run(adata)

        assert "n_robust" in results
        assert "n_artifacts" in results
        assert "n_missed" in results
        assert "jaccard" in results
        assert "setting_a_clusters" in results
        assert "setting_b_clusters" in results

    def test_run_preserves_embeddings(self):
        """run() stores both embeddings in adata.obsm."""
        from manylatents.singlecell.analysis.embedding_audit import EmbeddingAudit

        adata = _make_trajectory_adata()
        audit = EmbeddingAudit(
            setting_a={"n_neighbors": 5, "min_dist": 0.01},
            setting_b={"n_neighbors": 30, "min_dist": 0.5},
        )
        audit.run(adata)

        assert "X_umap_setting_a" in adata.obsm
        assert "X_umap_setting_b" in adata.obsm
```

**Step 2: Run to verify failure**

```bash
cd /network/scratch/c/cesar.valdez/lrw/omics
uv run pytest tests/singlecell/test_embedding_audit.py -v
```
Expected: FAIL with import error.

---

### Task 15: Implement EmbeddingAudit

**Files:**
- Create: `/network/scratch/c/cesar.valdez/lrw/omics/manylatents/singlecell/analysis/embedding_audit.py`

**Step 1: Write the implementation**

```python
"""End-to-end embedding fidelity audit: embed -> cluster -> DE -> compare."""
import logging

import matplotlib
import numpy as np
import scanpy as sc

from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis
from manylatents.singlecell.analysis.differential_expression import DifferentialExpression

logger = logging.getLogger(__name__)


class EmbeddingAudit:
    """Audit embedding fidelity by comparing DE results across parameter settings.

    Parameters
    ----------
    setting_a : dict
        UMAP parameters for the first setting (e.g. fragmenting).
        Keys: n_neighbors, min_dist.
    setting_b : dict
        UMAP parameters for the second setting (e.g. smooth).
    leiden_resolution : float
        Resolution for Leiden clustering.
    de_method : str
        Statistical test for DE.
    de_p_threshold : float
        Adjusted p-value cutoff.
    de_lfc_threshold : float
        Log-fold-change cutoff.
    """

    def __init__(self, setting_a: dict, setting_b: dict,
                 leiden_resolution: float = 0.5,
                 de_method: str = "wilcoxon",
                 de_p_threshold: float = 0.05,
                 de_lfc_threshold: float = 1.0):
        self.setting_a = setting_a
        self.setting_b = setting_b
        self.leiden_resolution = leiden_resolution
        self.de = DifferentialExpression(
            method=de_method, p_threshold=de_p_threshold, lfc_threshold=de_lfc_threshold
        )
        self.csa = ComplementSetAnalysis()

    def _embed_and_cluster(self, adata: sc.AnnData, setting: dict, suffix: str):
        """Build kNN graph, UMAP embed, Leiden cluster."""
        use_rep = "X_pca" if "X_pca" in adata.obsm else "X"
        sc.pp.neighbors(adata, n_neighbors=setting["n_neighbors"], use_rep=use_rep)
        sc.tl.umap(adata, min_dist=setting.get("min_dist", 0.1))

        umap_key = f"X_umap_{suffix}"
        adata.obsm[umap_key] = adata.obsm["X_umap"].copy()

        leiden_key = f"leiden_{suffix}"
        sc.tl.leiden(adata, resolution=self.leiden_resolution, key_added=leiden_key)

        n_clusters = adata.obs[leiden_key].nunique()
        logger.info(f"Setting {suffix}: {n_clusters} clusters (k={setting['n_neighbors']})")

        return leiden_key, n_clusters

    def run(self, adata: sc.AnnData) -> dict:
        """Full pipeline: embed -> cluster -> DE -> compare.

        Parameters
        ----------
        adata : sc.AnnData
            Annotated data matrix. Modified in-place.

        Returns
        -------
        dict
            Complement set results plus metadata about both settings.
        """
        # Setting A
        leiden_a, n_clusters_a = self._embed_and_cluster(adata, self.setting_a, "setting_a")
        df_a = self.de.run(adata, groupby=leiden_a, key_added="de_setting_a")
        genes_a = self.de.get_significant_genes(adata)

        # Setting B
        leiden_b, n_clusters_b = self._embed_and_cluster(adata, self.setting_b, "setting_b")
        df_b = self.de.run(adata, groupby=leiden_b, key_added="de_setting_b")
        genes_b = self.de.get_significant_genes(adata)

        # Compare
        results = self.csa.compare(genes_a, genes_b, df_a=df_a, df_b=df_b)
        results["setting_a_clusters"] = n_clusters_a
        results["setting_b_clusters"] = n_clusters_b
        results["setting_a"] = self.setting_a
        results["setting_b"] = self.setting_b

        return results

    def plot_comparison(self, adata: sc.AnnData, results: dict,
                        save_path: str = None):
        """Side-by-side embeddings colored by clusters, with complement set summary.

        Parameters
        ----------
        adata : sc.AnnData
            Must have obsm keys X_umap_setting_a and X_umap_setting_b.
        results : dict
            Output from run().
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for ax, suffix, title in [
            (axes[0], "setting_a", f"Setting A (k={self.setting_a['n_neighbors']})"),
            (axes[1], "setting_b", f"Setting B (k={self.setting_b['n_neighbors']})"),
        ]:
            coords = adata.obsm[f"X_umap_{suffix}"]
            clusters = adata.obs[f"leiden_{suffix}"].astype(int)
            n_clusters = clusters.nunique()
            cmap = plt.cm.get_cmap("tab20", n_clusters)
            ax.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap=cmap,
                       s=3, alpha=0.7, rasterized=True)
            ax.set_title(f"{title}\n{n_clusters} clusters")
            ax.set_aspect("equal")

        # Complement set bar chart
        ax = axes[2]
        categories = ["Robust\n(A∩B)", "Artifacts\n(A\\B)", "Missed\n(B\\A)"]
        values = [results["n_robust"], results["n_artifacts"], results["n_missed"]]
        colors = ["#2ecc71", "#e74c3c", "#3498db"]
        bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("DE genes")
        ax.set_title(f"Jaccard = {results['jaccard']:.3f}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
```

**Step 2: Run tests**

```bash
cd /network/scratch/c/cesar.valdez/lrw/omics
uv run pytest tests/singlecell/test_embedding_audit.py -v
```
Expected: both tests PASS.

**Step 3: Commit**

```bash
git add manylatents/singlecell/analysis/embedding_audit.py tests/singlecell/test_embedding_audit.py
git commit -m "feat: add EmbeddingAudit end-to-end pipeline"
```

---

### Task 16: Run full omics test suite and create PR

**Step 1: Run tests**

```bash
cd /network/scratch/c/cesar.valdez/lrw/omics
uv run pytest tests/singlecell/ -v
```
Expected: all singlecell tests pass.

**Step 2: Push and create PR**

```bash
git push -u origin feat/embedding-audit
```

Create PR with title: "feat: add embedding audit pipeline (DE + complement set analysis)"

Body:
```
## Summary
- `DifferentialExpression`: wraps scanpy `rank_genes_groups` with tidy DataFrame output
- `ComplementSetAnalysis`: compares DE gene lists across settings (robust/artifacts/missed + Jaccard)
- `EmbeddingAudit`: end-to-end pipeline — embed → cluster → DE → compare
- Adds `leidenalg` to `singlecell` extra

## Test plan
- [ ] `tests/singlecell/test_differential_expression.py` — 3 tests
- [ ] `tests/singlecell/test_complement_set.py` — 4 tests
- [ ] `tests/singlecell/test_embedding_audit.py` — 2 tests
- [ ] No regressions in existing tests
```

---

## Task Dependency Summary

```
Task 1 (install deps, inspect data) → Task 2 (write script, run Phase 1)
                                        ↓ (gate: |artifacts| > 10?)
Task 3 (branch) → Task 4 (deps) → Task 5 (test) → Task 6 (impl) → Task 7 (config) → Task 8 (PR)
Task 9 (branch) → Task 10 (test) → Task 11 (impl) → Task 12 (test) → Task 13 (impl) → Task 14 (test) → Task 15 (impl) → Task 16 (PR)
```

Phase 2 (Tasks 3-8) and Phase 3 (Tasks 9-16) can be done in parallel after Phase 1 passes the gate.
