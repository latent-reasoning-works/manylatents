"""Tests for PrecomputedDataModule metadata_path support."""
import numpy as np
import pandas as pd
import pytest

from manylatents.data.precomputed_datamodule import PrecomputedDataModule


class TestMetadataPath:
    """Tests for metadata_path parameter support."""

    def test_load_metadata_parquet(self, tmp_path):
        """PrecomputedDataModule loads metadata from parquet file."""
        # Create test embeddings
        embeddings = np.random.randn(100, 64).astype(np.float32)
        emb_path = tmp_path / "embeddings.npy"
        np.save(emb_path, embeddings)

        # Create test metadata
        metadata = pd.DataFrame({
            "sample_id": [f"sample_{i}" for i in range(100)],
            "label": np.random.randint(0, 2, 100),
            "score": np.random.randn(100),
        })
        meta_path = tmp_path / "metadata.parquet"
        metadata.to_parquet(meta_path, index=False)

        # Load with metadata_path
        dm = PrecomputedDataModule(
            path=str(emb_path),
            metadata_path=str(meta_path),
        )
        dm.setup()

        # Verify metadata loaded
        assert dm.metadata is not None
        assert len(dm.metadata) == 100
        assert "sample_id" in dm.metadata.columns
        assert "label" in dm.metadata.columns

    def test_load_metadata_csv(self, tmp_path):
        """PrecomputedDataModule loads metadata from CSV file."""
        # Create test embeddings
        embeddings = np.random.randn(50, 32).astype(np.float32)
        emb_path = tmp_path / "embeddings.npy"
        np.save(emb_path, embeddings)

        # Create test metadata as CSV
        metadata = pd.DataFrame({
            "variant_id": [f"var_{i}" for i in range(50)],
            "pathogenicity": ["benign", "pathogenic"] * 25,
        })
        meta_path = tmp_path / "metadata.csv"
        metadata.to_csv(meta_path, index=False)

        dm = PrecomputedDataModule(
            path=str(emb_path),
            metadata_path=str(meta_path),
        )
        dm.setup()

        assert dm.metadata is not None
        assert len(dm.metadata) == 50
        assert "pathogenicity" in dm.metadata.columns

    def test_get_metadata_column(self, tmp_path):
        """get_metadata_column returns specific column as array."""
        embeddings = np.random.randn(20, 16).astype(np.float32)
        emb_path = tmp_path / "embeddings.npy"
        np.save(emb_path, embeddings)

        metadata = pd.DataFrame({
            "gene": ["BRCA1"] * 10 + ["BRCA2"] * 10,
            "position": list(range(20)),
        })
        meta_path = tmp_path / "metadata.parquet"
        metadata.to_parquet(meta_path, index=False)

        dm = PrecomputedDataModule(
            path=str(emb_path),
            metadata_path=str(meta_path),
        )
        dm.setup()

        genes = dm.get_metadata_column("gene")
        assert len(genes) == 20
        assert genes[0] == "BRCA1"
        assert genes[15] == "BRCA2"

    def test_metadata_none_when_no_path(self, tmp_path):
        """metadata is None when metadata_path not provided."""
        embeddings = np.random.randn(10, 8).astype(np.float32)
        emb_path = tmp_path / "embeddings.npy"
        np.save(emb_path, embeddings)

        dm = PrecomputedDataModule(path=str(emb_path))
        dm.setup()

        assert dm.metadata is None

    def test_metadata_length_mismatch_warns(self, tmp_path):
        """Warning when metadata length doesn't match embeddings."""
        embeddings = np.random.randn(100, 64).astype(np.float32)
        emb_path = tmp_path / "embeddings.npy"
        np.save(emb_path, embeddings)

        # Metadata with wrong length
        metadata = pd.DataFrame({"id": range(50)})  # 50 != 100
        meta_path = tmp_path / "metadata.parquet"
        metadata.to_parquet(meta_path, index=False)

        dm = PrecomputedDataModule(
            path=str(emb_path),
            metadata_path=str(meta_path),
        )

        with pytest.warns(UserWarning, match="Metadata length.*does not match"):
            dm.setup()

    def test_get_metadata_column_raises_when_no_metadata(self, tmp_path):
        """get_metadata_column raises ValueError when no metadata loaded."""
        embeddings = np.random.randn(10, 8).astype(np.float32)
        emb_path = tmp_path / "embeddings.npy"
        np.save(emb_path, embeddings)

        dm = PrecomputedDataModule(path=str(emb_path))
        dm.setup()

        with pytest.raises(ValueError, match="No metadata loaded"):
            dm.get_metadata_column("some_column")

    def test_get_metadata_column_raises_for_missing_column(self, tmp_path):
        """get_metadata_column raises ValueError for missing column."""
        embeddings = np.random.randn(10, 8).astype(np.float32)
        emb_path = tmp_path / "embeddings.npy"
        np.save(emb_path, embeddings)

        metadata = pd.DataFrame({"existing_col": range(10)})
        meta_path = tmp_path / "metadata.parquet"
        metadata.to_parquet(meta_path, index=False)

        dm = PrecomputedDataModule(
            path=str(emb_path),
            metadata_path=str(meta_path),
        )
        dm.setup()

        with pytest.raises(ValueError, match="Column 'missing_col' not found"):
            dm.get_metadata_column("missing_col")
