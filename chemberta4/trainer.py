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

from .model import ClassificationHead, CausalLMClassificationHead, RegressionHead
from .utils import get_device_map


class OLMoRegressor(pl.LightningModule):
    """
    Lightning module for regression tasks.

    Uses RMSE loss and supports label normalization.
    Reports denormalized metrics for interpretability.

    Args:
        model_name: HuggingFace model identifier
        use_qlora: Use 4-bit quantization with LoRA
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        warmup_ratio: Fraction of steps for warmup
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout rate
        label_mean: Mean for label denormalization
        label_std: Std for label denormalization
    """

    def __init__(
        self,
        model_name: str = "allenai/OLMo-7B-hf",
        use_qlora: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        label_mean: float = 0.0,
        label_std: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = None
        self.tokenizer = None

    def configure_model(self):
        """Initialize model (called before training starts)."""
        if self.model is not None:
            return

        hp = self.hparams

        self.tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if hp.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        lora_cfg = LoraConfig(
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=hp.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        device_map = get_device_map(self.device)

        base = AutoModel.from_pretrained(
            hp.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
        )

        if hp.use_qlora:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        base = get_peft_model(base, lora_cfg)

        if self.global_rank == 0:
            base.print_trainable_parameters()

        self.model = RegressionHead(base)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask, labels)

    def _denormalize(self, values):
        """Convert normalized values back to original scale."""
        return values * self.hparams.label_std + self.hparams.label_mean

    def _shared_step(self, batch, stage: str):
        """Shared logic for train/val/test steps."""
        preds, loss = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        # Denormalize for metrics
        preds_denorm = self._denormalize(preds)
        labels_denorm = self._denormalize(batch["labels"])

        # Calculate denormalized metrics
        mse = torch.mean((preds_denorm - labels_denorm) ** 2)
        rmse = torch.sqrt(mse + 1e-6)
        mae = torch.mean(torch.abs(preds_denorm - labels_denorm))

        # Log
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/rmse", rmse, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/mae", mae, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        hp = self.hparams

        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "layer_norm" in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": hp.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=hp.lr,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * hp.warmup_ratio)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
                ),
                CosineAnnealingLR(
                    optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
                ),
            ],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
