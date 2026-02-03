"""HuggingFace model wrapper for PyTorch Lightning."""
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import CausalLMOutput


@dataclass
class HFTrainerConfig:
    """Configuration for HFTrainerModule.

    Attributes:
        model_name_or_path: HuggingFace model identifier or local path
        learning_rate: Learning rate for AdamW
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps for scheduler
        adam_epsilon: Epsilon for AdamW
        torch_dtype: Optional dtype for model (e.g., torch.bfloat16)
        trust_remote_code: Whether to trust remote code for model loading
        attn_implementation: Attention implementation (e.g., "flash_attention_2")
    """
    model_name_or_path: str
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    adam_epsilon: float = 1e-8
    torch_dtype: Optional[torch.dtype] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None


class HFTrainerModule(LightningModule):
    """Lightning module wrapping HuggingFace causal LM.

    Features:
    - Lazy model initialization (for FSDP compatibility)
    - Exposes .network for activation extraction
    - Standard Lightning training interface
    """

    def __init__(self, config: HFTrainerConfig, datamodule=None):
        super().__init__()
        self.config = config
        self.network: Optional[AutoModelForCausalLM] = None
        self.tokenizer = None
        # datamodule passed by experiment.py but not used here

        self.save_hyperparameters({"config": config.__dict__})

    def configure_model(self) -> None:
        """Lazy model initialization for FSDP compatibility."""
        if self.network is not None:
            return

        model_kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": self.config.model_name_or_path,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.config.torch_dtype
        if self.config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.network = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )

    def forward(self, **inputs) -> CausalLMOutput:
        """Forward pass through the model."""
        return self.network(**inputs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Standard training step."""
        outputs: CausalLMOutput = self(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Standard validation step."""
        outputs: CausalLMOutput = self(**batch)
        loss = outputs.loss
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Standard test step."""
        outputs: CausalLMOutput = self(**batch)
        loss = outputs.loss
        self.log("test/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure AdamW with optional warmup."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

        if self.config.warmup_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer
