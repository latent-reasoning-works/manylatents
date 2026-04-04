"""Tests for HFTextDataModule local loading and scale controls."""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from manylatents.data.text import HFTextDataModule


class TestLocalLoading:
    """Test _load_local() with mock jsonl files."""

    @pytest.fixture
    def mock_pile_dir(self, tmp_path):
        """Create a mock Pile-like directory with jsonl files."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()

        # Create 3 small shard files
        for i in range(3):
            shard = train_dir / f"{i:02d}.jsonl"
            lines = [
                json.dumps({"text": f"Shard {i} example {j}.", "meta": {"pile_set_name": "test"}})
                for j in range(20)
            ]
            shard.write_text("\n".join(lines))

        # Create val file
        val_file = tmp_path / "val.jsonl"
        lines = [
            json.dumps({"text": f"Val example {j}.", "meta": {"pile_set_name": "test"}})
            for j in range(50)
        ]
        val_file.write_text("\n".join(lines))

        return tmp_path

    def test_load_local_basic(self, mock_pile_dir):
        """Local mode loads train and val from jsonl files."""
        dm = HFTextDataModule(
            data_dir=str(mock_pile_dir),
            train_files="train/*.jsonl",
            val_files="val.jsonl",
            tokenizer_name="gpt2",
            max_length=32,
            batch_size=4,
            probe_n_samples=8,
            seed=42,
        )
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0

    def test_num_shards(self, mock_pile_dir):
        """num_shards=2 loads only 2 of 3 available shards."""
        dm_all = HFTextDataModule(
            data_dir=str(mock_pile_dir),
            train_files="train/*.jsonl",
            val_files="val.jsonl",
            tokenizer_name="gpt2",
            max_length=32,
            batch_size=4,
            probe_n_samples=8,
        )
        dm_all.setup()
        n_all = len(dm_all.train_dataset)

        dm_2 = HFTextDataModule(
            data_dir=str(mock_pile_dir),
            train_files="train/*.jsonl",
            val_files="val.jsonl",
            tokenizer_name="gpt2",
            max_length=32,
            batch_size=4,
            probe_n_samples=8,
            num_shards=2,
        )
        dm_2.setup()
        n_2 = len(dm_2.train_dataset)

        assert n_2 < n_all

    def test_max_train_samples(self, mock_pile_dir):
        """max_train_samples caps the training set."""
        dm = HFTextDataModule(
            data_dir=str(mock_pile_dir),
            train_files="train/*.jsonl",
            val_files="val.jsonl",
            tokenizer_name="gpt2",
            max_length=32,
            batch_size=4,
            probe_n_samples=8,
            max_train_samples=10,
        )
        dm.setup()
        # 10 max, but some may be empty after filtering
        assert len(dm.train_dataset) <= 10

    def test_max_val_samples(self, mock_pile_dir):
        """max_val_samples caps the validation set."""
        dm = HFTextDataModule(
            data_dir=str(mock_pile_dir),
            train_files="train/*.jsonl",
            val_files="val.jsonl",
            tokenizer_name="gpt2",
            max_length=32,
            batch_size=4,
            probe_n_samples=8,
            max_val_samples=5,
        )
        dm.setup()
        assert len(dm.val_dataset) <= 5

    def test_text_column(self, mock_pile_dir):
        """Custom text_column reads from the correct field."""
        custom_dir = mock_pile_dir / "custom"
        custom_dir.mkdir()
        train_dir = custom_dir / "train"
        train_dir.mkdir()
        shard = train_dir / "00.jsonl"
        lines = [
            json.dumps({"content": f"Custom col example {j}."})
            for j in range(20)
        ]
        shard.write_text("\n".join(lines))
        val = custom_dir / "val.jsonl"
        lines = [
            json.dumps({"content": f"Custom col val {j}."})
            for j in range(10)
        ]
        val.write_text("\n".join(lines))

        dm = HFTextDataModule(
            data_dir=str(custom_dir),
            train_files="train/*.jsonl",
            val_files="val.jsonl",
            text_column="content",
            tokenizer_name="gpt2",
            max_length=32,
            batch_size=4,
            probe_n_samples=5,
        )
        dm.setup()
        assert len(dm.train_dataset) > 0
        item = dm.train_dataset[0]
        assert "input_ids" in item

    def test_missing_train_files_raises(self, tmp_path):
        """FileNotFoundError when no train files match the glob."""
        (tmp_path / "val.jsonl").write_text(json.dumps({"text": "hi"}))
        dm = HFTextDataModule(
            data_dir=str(tmp_path),
            train_files="train/*.jsonl",
            val_files="val.jsonl",
            tokenizer_name="gpt2",
        )
        with pytest.raises(FileNotFoundError, match="No train files"):
            dm.setup()

    def test_missing_val_files_raises(self, tmp_path):
        """FileNotFoundError when no val files match the glob."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "00.jsonl").write_text(json.dumps({"text": "hi"}))
        dm = HFTextDataModule(
            data_dir=str(tmp_path),
            train_files="train/*.jsonl",
            val_files="val.jsonl",
            tokenizer_name="gpt2",
        )
        with pytest.raises(FileNotFoundError, match="No val files"):
            dm.setup()


class TestHubLoadingUnchanged:
    """Verify hub loading still works after refactor."""

    def test_default_hub_mode(self):
        """data_dir=None means hub mode (existing behavior)."""
        dm = HFTextDataModule(
            tokenizer_name="gpt2",
            max_length=32,
            batch_size=4,
            probe_n_samples=8,
        )
        assert dm.data_dir is None
        assert dm.dataset_name == "wikitext"

    def test_backward_compat_alias(self):
        """TextDataModule alias still works."""
        from manylatents.data.text import TextDataModule
        dm = TextDataModule(tokenizer_name="gpt2")
        assert isinstance(dm, HFTextDataModule)
