# manylatents/data/text.py
"""Text data module for HuggingFace language models."""
from dataclasses import dataclass
from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer


@dataclass
class HFTextDataConfig:
    """Configuration for text data module.

    Attributes:
        dataset_name: HuggingFace dataset name (e.g., "wikitext")
        dataset_config: Dataset config (e.g., "wikitext-2-raw-v1")
        tokenizer_name: Tokenizer to use (defaults to model name)
        max_length: Maximum sequence length
        batch_size: Training batch size
        num_workers: DataLoader workers
    """
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    tokenizer_name: Optional[str] = None
    max_length: int = 128
    batch_size: int = 8
    num_workers: int = 0


class HFTextDataModule(LightningDataModule):
    """Lightning DataModule for text data with HuggingFace models.

    Loads a HuggingFace dataset, tokenizes it, and provides DataLoaders
    for training, validation, and probing.
    """

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

    def prepare_data(self):
        """Download dataset and tokenizer (hub mode only)."""
        if self.data_dir is None:
            from datasets import load_dataset
            load_dataset(self.dataset_name, self.dataset_config)
        AutoTokenizer.from_pretrained(self.tokenizer_name)

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
            # Filter empty strings
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

        # Tokenize datasets
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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return self.val_dataloader()  # Use validation as test for simplicity

    def probe_dataloader(self):
        """Fixed subset for representation probing."""
        return DataLoader(
            self.probe_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """Collate tokenized examples into batch."""
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


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


# Backward compatibility aliases
TextDataModule = HFTextDataModule
TextDataConfig = HFTextDataConfig
