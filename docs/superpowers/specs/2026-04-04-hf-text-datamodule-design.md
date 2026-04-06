# HFTextDataModule: Unified Hub + Local Text Data Loading

**Date:** 2026-04-04
**Status:** Design approved
**Scope:** manylatents core (public repo)

## Problem

`TextDataModule` loads text exclusively from HuggingFace Hub via `load_dataset(name, config)`. The Pythia representation probe needs The Pile — Pythia's actual training data — which sits on the Mila cluster as local `.jsonl.zst` shards. Using WikiText (the current default) means probing on OOD text, confounding interpretability results.

The Pile on Mila:
- **Train:** 30 shards (`train/00.jsonl.zst` ... `train/29.jsonl.zst`), ~14 GB each, 427 GB total
- **Val:** `val.jsonl.zst`, 214,670 examples, ~450 MB compressed
- **Format:** `{"text": "...", "meta": {"pile_set_name": "..."}}` — same `text` key as WikiText
- **Storage:** git-annex symlinks (resolved, readable)

## Design

### Rename: `TextDataModule` → `HFTextDataModule`

The current name is misleading. The module is HF-ecosystem end-to-end: HF `datasets` for loading, HF `AutoTokenizer` for tokenization, HF-format output (`input_ids`, `attention_mask`, `labels`). The rename makes this explicit.

File rename: `manylatents/data/text.py` stays (module path unchanged), class name changes. Config `_target_` paths update accordingly.

### Two source modes, one class

The tokenization → batching → probe subset pipeline is identical regardless of where the raw text comes from. Only the `load_dataset()` call differs:

```python
# Hub mode (current behavior):
dataset = load_dataset(self.dataset_name, self.dataset_config)

# Local mode (new):
dataset = load_dataset("json", data_files={"train": [...], "validation": [...]})
```

**Routing:** When `data_dir` is set, use local mode. When `data_dir` is None, use hub mode (existing behavior).

### New parameters

```python
class HFTextDataModule(LightningDataModule):
    def __init__(
        self,
        # --- Source: hub (existing) ---
        dataset_name: str | None = "wikitext",
        dataset_config: str | None = "wikitext-2-raw-v1",
        # --- Source: local (new) ---
        data_dir: str | None = None,
        train_files: str = "train/*.jsonl.zst",
        val_files: str = "val.jsonl.zst",
        text_column: str = "text",
        # --- Scale control (new) ---
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        num_shards: int | None = None,
        # --- Existing (unchanged) ---
        tokenizer_name: str = "gpt2",
        max_length: int = 128,
        batch_size: int = 8,
        num_workers: int = 0,
        probe_n_samples: int = 512,
        seed: int = 42,
    ):
```

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `data_dir` | Base directory for local shards. When set, switches to local mode. | `None` (hub mode) |
| `train_files` | Glob pattern relative to `data_dir` for training data | `"train/*.jsonl.zst"` |
| `val_files` | Glob pattern relative to `data_dir` for validation data | `"val.jsonl.zst"` |
| `text_column` | Column name containing raw text | `"text"` |
| `max_train_samples` | Cap on training examples (applied after loading) | `None` (use all) |
| `max_val_samples` | Cap on validation examples | `None` (use all) |
| `num_shards` | Load only first N train shards (for scale control) | `None` (use all) |

### Loading logic in `setup()`

```python
def setup(self, stage=None):
    self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
    if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

    if self.data_dir is not None:
        dataset = self._load_local()
    else:
        dataset = self._load_hub()

    # Apply sample caps
    if self.max_train_samples:
        dataset["train"] = dataset["train"].select(range(
            min(self.max_train_samples, len(dataset["train"]))
        ))
    if self.max_val_samples:
        dataset["validation"] = dataset["validation"].select(range(
            min(self.max_val_samples, len(dataset["validation"]))
        ))

    # Rest unchanged: tokenize, wrap in TokenizedDataset, create probe subset
```

```python
def _load_hub(self):
    """Load from HuggingFace Hub."""
    from datasets import load_dataset
    return load_dataset(self.dataset_name, self.dataset_config)

def _load_local(self):
    """Load from local jsonl/jsonl.zst shards."""
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

### `text_column` handling

The Pile and WikiText both use `"text"` as the column name, but other local datasets might not. `TokenizedDataset` currently hardcodes `example["text"]`. The fix is to pass `text_column` through:

```python
class TokenizedDataset(Dataset):
    def __init__(self, hf_dataset, tokenize_fn, max_length, text_column="text"):
        self.text_column = text_column
        # valid_indices filter uses self.text_column instead of hardcoded "text"
```

### Config files

**`data/wikitext.yaml`** — updated `_target_`, otherwise unchanged:
```yaml
_target_: manylatents.data.text.HFTextDataModule
dataset_name: "wikitext"
dataset_config: "wikitext-2-raw-v1"
tokenizer_name: "gpt2"
max_length: 128
batch_size: 8
num_workers: 0
probe_n_samples: 512
seed: ${seed}
```

**`data/pile.yaml`** — new:
```yaml
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

**`experiment/representation_probe_pythia.yaml`** — update data override:
```yaml
defaults:
  - override /data: pile    # was: wikitext
```
Remove `data.tokenizer_name` override (now in pile.yaml). Keep `data.max_length`, `data.batch_size`, `data.probe_n_samples` overrides if the experiment needs different values.

### Scale guidance

| Use case | `num_shards` | `max_train_samples` | ~Size |
|----------|-------------|-------------------|-------|
| Dev/smoke test | 1 | 1000 | <1 MB |
| Probing experiment | 1 | 50000 | ~25 MB |
| Fine-tune (small) | 3 | null | ~42 GB |
| Full training | null | null | ~427 GB |

### What does NOT change

- `TokenizedDataset` class — same interface (HF datasets from `load_dataset("json")` has identical row access)
- `_collate_fn` — unchanged
- `probe_dataloader()` — unchanged
- `prepare_data()` — only called in hub mode (local files already exist)
- No new base class, no subclass hierarchy — follows existing flat DataModule pattern

### Backward compatibility

- `TextDataModule` name becomes `HFTextDataModule`. Add a deprecation alias:
  ```python
  TextDataModule = HFTextDataModule  # backward compat
  ```
- `TextDataConfig` dataclass is unused externally (not referenced in configs) — rename to `HFTextDataConfig` or remove.
- The activation_tracker.py reference to "TextDataModule" in an error message gets updated.

## Testing

1. **Config unit test:** `HFTextDataModule` instantiable from both `wikitext.yaml` and `pile.yaml`
2. **Hub loading test:** existing wikitext tests still pass (regression)
3. **Local loading test:** mock local jsonl files, verify `_load_local()` returns correct HF Dataset structure
4. **`num_shards` test:** with 3 mock shard files, `num_shards=2` loads exactly 2
5. **`max_train_samples` test:** verify truncation
6. **`text_column` test:** dataset with non-"text" column name loads correctly
7. **Symlink resolution test:** verify `Path.resolve()` is called on local paths
8. **Pile smoke test (slow, cluster-only):** load 100 samples from real Pile val, tokenize, verify shapes

## Not in scope

- Streaming/IterableDataset support — `load_dataset("json")` uses memory-mapped Arrow, efficient enough for single-shard probing
- Filtering by `pile_set_name` — can be added later via a `filter_fn` param if needed
- New base class for DataModules — existing flat pattern works
