"""
PyTorch Lightning training modules.

Provides OLMoClassifier, OLMoRegressor, and OLMoPretrainer modules
with support for QLoRA and full finetuning.
"""

import math

import torch
import pytorch_lightning as pl
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchmetrics import Accuracy, AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .utils import get_device_map


class OLMoPretrainer(pl.LightningModule):
    """
    Lightning module for causal LM pretraining.

    Used for pretraining on SMILES (ZINC20, PubChem) or
    instruction tuning (USPTO).

    Args:
        model_name: HuggingFace model identifier or path to pretrained model
        use_qlora: Use 4-bit quantization with LoRA
        lr: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Fraction of steps for warmup
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        gradient_checkpointing: Enable gradient checkpointing
    """

    def __init__(
        self,
        model_name: str = "allenai/OLMo-7B-hf",
        use_qlora: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_ratio: float = 0.15,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = None
        self.tokenizer = None

    def configure_model(self):
        """Initialize model."""
        if self.model is not None:
            return

        hp = self.hparams

        self.tokenizer = AutoTokenizer.from_pretrained(hp.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if hp.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            hp.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )

        model.config.use_cache = False
        if hp.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if hp.use_qlora:
            model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            target_modules="all-linear",
            lora_dropout=hp.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(model, lora_cfg)

        if self.trainer.is_global_zero:
            self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # Perplexity
        perplexity = torch.exp(loss)

        # BPB: bits per byte
        num_tokens = (batch["labels"] != -100).sum()
        num_bytes = batch["num_bytes"].sum()
        bpb = (loss * num_tokens) / (num_bytes * math.log(2))

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", perplexity, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/bpb", bpb, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        hp = self.hparams

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(hp.warmup_ratio * total_steps)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps
                ),
                CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
            ],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
