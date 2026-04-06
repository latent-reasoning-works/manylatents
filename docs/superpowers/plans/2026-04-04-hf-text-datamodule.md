# HFTextDataModule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `TextDataModule` → `HFTextDataModule` and add local `.jsonl.zst` shard loading so the Pythia probe can use The Pile from the Mila cluster.

**Architecture:** One class, two source modes. When `data_dir` is set, loads local files via `load_dataset("json", data_files=...)`. When `data_dir` is None, loads from HF Hub (existing behavior). Everything downstream of the HF Dataset dict is unchanged.

**Tech Stack:** HuggingFace `datasets`, `transformers`, PyTorch Lightning, Hydra

**Spec:** `docs/superpowers/specs/2026-04-04-hf-text-datamodule-design.md`

---

### Task 1: Rename TextDataModule → HFTextDataModule + backward compat alias

**Files:**
- Modify: `manylatents/data/text.py:13,32` (class renames)
- Modify: `manylatents/lightning/callbacks/activation_tracker.py:142` (error message)
- Modify: `docs/probing.md:106` (doc reference)

- [ ] **Step 1: Rename the dataclass and class in text.py**

In `manylatents/data/text.py`, make these changes:

Line 13 — rename the dataclass:
```python
# old:
class TextDataConfig:
# new:
class HFTextDataConfig:
```

Line 32 — rename the class:
```python
# old:
class TextDataModule(LightningDataModule):
# new:
class HFTextDataModule(LightningDataModule):
```

Add backward-compat aliases at the very end of the file (after the `TokenizedDataset` class, around line 183):
```python
# Backward compatibility aliases
TextDataModule = HFTextDataModule
TextDataConfig = HFTextDataConfig
```

- [ ] **Step 2: Update error message in activation_tracker.py**

In `manylatents/lightning/callbacks/activation_tracker.py` line 142, change:
```python
# old:
                    "with probe_dataloader() (e.g., TextDataModule)."
# new:
                    "with probe_dataloader() (e.g., HFTextDataModule)."
```

- [ ] **Step 3: Update doc reference in probing.md**

In `docs/probing.md` line 106, change:
```markdown
<!-- old: -->
Probing uses a fixed subset of data for consistent comparisons across training. The `TextDataModule` provides this via `probe_dataloader()`:
<!-- new: -->
Probing uses a fixed subset of data for consistent comparisons across training. The `HFTextDataModule` provides this via `probe_dataloader()`:
```

- [ ] **Step 4: Run existing tests to verify nothing breaks**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_config_composition.py -x -q`

Expected: All pass (wikitext.yaml still works because its `_target_` hasn't changed yet — that's Task 4).

- [ ] **Step 5: Commit**

```bash
git add manylatents/data/text.py manylatents/lightning/callbacks/activation_tracker.py docs/probing.md
git commit -m "refactor: rename TextDataModule → HFTextDataModule with backward compat alias"
```

---

### Task 2: Add local loading parameters and `_load_local()` / `_load_hub()` methods

**Files:**
- Modify: `manylatents/data/text.py` (HFTextDataModule `__init__`, new methods, `setup()` refactor)
- Test: `tests/test_hf_text_datamodule.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hf_text_datamodule.py`:

```python
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
        # Create a file with a different column name
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_hf_text_datamodule.py -x -v 2>&1 | head -40`

Expected: Failures — `HFTextDataModule.__init__()` does not accept `data_dir`, `num_shards`, etc.

- [ ] **Step 3: Implement the changes in text.py**

Replace the `__init__` method of `HFTextDataModule` (lines 39-63) with:

```python
    def __init__(
        self,
        # Source: hub (existing)
        dataset_name: str | None = "wikitext",
        dataset_config: str | None = "wikitext-2-raw-v1",
        # Source: local
        data_dir: str | None = None,
        train_files: str = "train/*.jsonl.zst",
        val_files: str = "val.jsonl.zst",
        text_column: str = "text",
        # Scale control
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        num_shards: int | None = None,
        # Existing (unchanged)
        tokenizer_name: str = "gpt2",
        max_length: int = 128,
        batch_size: int = 8,
        num_workers: int = 0,
        probe_n_samples: int = 512,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.data_dir = data_dir
        self.train_files = train_files
        self.val_files = val_files
        self.text_column = text_column
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.num_shards = num_shards
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.probe_n_samples = probe_n_samples
        self.seed = seed

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.probe_dataset = None
```

Add two private methods after `__init__` (before `prepare_data`):

```python
    def _load_hub(self):
        """Load dataset from HuggingFace Hub."""
        from datasets import load_dataset
        return load_dataset(self.dataset_name, self.dataset_config)

    def _load_local(self):
        """Load dataset from local jsonl/jsonl.zst shards."""
        from datasets import load_dataset
        from glob import glob
        from pathlib import Path

        base = Path(self.data_dir)
        train_paths = sorted(glob(str(base / self.train_files)))
        val_paths = sorted(glob(str(base / self.val_files)))

        if self.num_shards is not None:
            train_paths = train_paths[:self.num_shards]

        if not train_paths:
            raise FileNotFoundError(f"No train files matching {base / self.train_files}")
        if not val_paths:
            raise FileNotFoundError(f"No val files matching {base / self.val_files}")

        # Resolve symlinks (git-annex on Mila stores data behind symlinks)
        train_paths = [str(Path(p).resolve()) for p in train_paths]
        val_paths = [str(Path(p).resolve()) for p in val_paths]

        return load_dataset("json", data_files={
            "train": train_paths,
            "validation": val_paths,
        })
```

Replace `prepare_data` (lines 65-69):

```python
    def prepare_data(self):
        """Download dataset and tokenizer (hub mode only)."""
        if self.data_dir is None:
            from datasets import load_dataset
            load_dataset(self.dataset_name, self.dataset_config)
        AutoTokenizer.from_pretrained(self.tokenizer_name)
```

Replace `setup` (lines 71-109):

```python
    def setup(self, stage: Optional[str] = None):
        """Tokenize and prepare datasets."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.data_dir is not None:
            dataset = self._load_local()
        else:
            dataset = self._load_hub()

        # Apply sample caps
        if self.max_train_samples is not None:
            n = min(self.max_train_samples, len(dataset["train"]))
            dataset["train"] = dataset["train"].select(range(n))
        if self.max_val_samples is not None:
            n = min(self.max_val_samples, len(dataset["validation"]))
            dataset["validation"] = dataset["validation"].select(range(n))

        text_column = self.text_column

        def tokenize_fn(examples):
            texts = [t for t in examples[text_column] if t.strip()]
            if not texts:
                return {"input_ids": [], "attention_mask": [], "labels": []}

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        self.train_dataset = TokenizedDataset(
            dataset["train"], tokenize_fn, self.max_length, self.text_column
        )
        self.val_dataset = TokenizedDataset(
            dataset["validation"], tokenize_fn, self.max_length, self.text_column
        )

        # Create fixed probe subset
        generator = torch.Generator().manual_seed(self.seed)
        n_probe = min(self.probe_n_samples, len(self.val_dataset))
        indices = torch.randperm(len(self.val_dataset), generator=generator)[:n_probe]
        self.probe_dataset = Subset(self.val_dataset, indices.tolist())
```

- [ ] **Step 4: Update `TokenizedDataset` to accept `text_column`**

Replace the `TokenizedDataset.__init__` (lines 157-165) and `__getitem__` (lines 172-182):

```python
class TokenizedDataset(Dataset):
    """Lazily tokenized dataset wrapper."""

    def __init__(self, hf_dataset, tokenize_fn, max_length: int, text_column: str = "text"):
        self.hf_dataset = hf_dataset
        self.tokenize_fn = tokenize_fn
        self.max_length = max_length
        self.text_column = text_column
        self._cache = {}

        # Pre-filter to non-empty texts
        self.valid_indices = [
            i for i, ex in enumerate(hf_dataset)
            if ex[self.text_column].strip()
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        if real_idx not in self._cache:
            example = self.hf_dataset[real_idx]
            tokenized = self.tokenize_fn({self.text_column: [example[self.text_column]]})
            self._cache[real_idx] = {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "labels": tokenized["labels"][0],
            }
        return self._cache[real_idx]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_hf_text_datamodule.py -x -v`

Expected: All pass.

- [ ] **Step 6: Run existing tests for regression**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_config_composition.py -x -q`

Expected: All pass (wikitext.yaml still targets `manylatents.data.text.TextDataModule` which is now an alias).

- [ ] **Step 7: Commit**

```bash
git add manylatents/data/text.py tests/test_hf_text_datamodule.py
git commit -m "feat: add local shard loading to HFTextDataModule (data_dir, num_shards, max_samples)"
```

---

### Task 3: Config files — wikitext.yaml update, pile.yaml, experiment update

**Files:**
- Modify: `manylatents/configs/data/wikitext.yaml` (update `_target_`)
- Create: `manylatents/configs/data/pile.yaml`
- Modify: `manylatents/configs/experiment/representation_probe_pythia.yaml` (switch to pile data)

- [ ] **Step 1: Update wikitext.yaml `_target_`**

In `manylatents/configs/data/wikitext.yaml` line 5, change:
```yaml
# old:
_target_: manylatents.data.text.TextDataModule
# new:
_target_: manylatents.data.text.HFTextDataModule
```

- [ ] **Step 2: Create pile.yaml**

Create `manylatents/configs/data/pile.yaml`:
```yaml
# The Pile — Pythia's training data, loaded from Mila cluster local shards
# Uses jsonl.zst format; num_shards/max_train_samples control scale
_target_: manylatents.data.text.HFTextDataModule
dataset_name: null
dataset_config: null
data_dir: "/network/datasets/pile"
train_files: "train/*.jsonl.zst"
val_files: "val.jsonl.zst"
text_column: "text"
tokenizer_name: "EleutherAI/pythia-70m"
max_length: 128
batch_size: 8
num_workers: 0
probe_n_samples: 512
num_shards: 1
max_train_samples: 50000
seed: ${seed}
```

- [ ] **Step 3: Update representation_probe_pythia.yaml**

Replace the full content of `manylatents/configs/experiment/representation_probe_pythia.yaml`:

```yaml
# @package _global_
#
# End-to-end experiment for representation probing on Pythia models.
# Uses The Pile (Pythia's training data) from Mila cluster local shards.
#
# Usage:
#   python -m manylatents.main experiment=representation_probe_pythia
#   python -m manylatents.main experiment=representation_probe_pythia \
#       algorithms.lightning.config.model_name_or_path=/network/weights/pythia/pythia-410m \
#       data.tokenizer_name=EleutherAI/pythia-410m
#
name: representation_probe_pythia

defaults:
  - override /data: pile
  - override /algorithms/lightning: hf_trainer
  - override /trainer: default
  - override /callbacks/trainer: probe_pythia
  - _self_

seed: 42
project: representation_probe
debug: false

# --- Data Configuration ---
data:
  probe_n_samples: 512

# --- Model Configuration ---
# Weights are local on Mila cluster; tokenizer loaded from HuggingFace hub
algorithms:
  lightning:
    config:
      model_name_or_path: "/network/weights/pythia/pythia-70m/step143000"
      output_hidden_states: true
      learning_rate: 2e-5
      weight_decay: 0.01
      warmup_steps: 100

# --- Trainer Configuration ---
trainer:
  max_epochs: 3
  precision: bf16-mixed
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4
  val_check_interval: 0.5
  log_every_n_steps: 50
```

- [ ] **Step 4: Run config composition tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_config_composition.py -x -q`

Expected: All pass, including new `pile` config.

- [ ] **Step 5: Commit**

```bash
git add manylatents/configs/data/wikitext.yaml manylatents/configs/data/pile.yaml manylatents/configs/experiment/representation_probe_pythia.yaml
git commit -m "config: add pile.yaml data config, update pythia experiment to use Pile"
```

---

### Task 4: Final verification

**Files:** None (test-only)

- [ ] **Step 1: Run full test suite**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/ -x -q`

Expected: All pass.

- [ ] **Step 2: Run callback tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest manylatents/callbacks/tests/ -x -q`

Expected: All pass.

- [ ] **Step 3: Run new HFTextDataModule tests specifically**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_hf_text_datamodule.py -v`

Expected: All 9 tests pass:
- `test_load_local_basic` — PASS
- `test_num_shards` — PASS
- `test_max_train_samples` — PASS
- `test_max_val_samples` — PASS
- `test_text_column` — PASS
- `test_missing_train_files_raises` — PASS
- `test_missing_val_files_raises` — PASS
- `test_default_hub_mode` — PASS
- `test_backward_compat_alias` — PASS

- [ ] **Step 4: Verify config composition includes pile**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_config_composition.py -k pile -v`

Expected: pile config resolves without errors.
