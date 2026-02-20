"""
PyTorch Lightning training modules.

Provides OLMoClassifier, OLMoRegressor, and OLMoPretrainer modules
with support for QLoRA and full finetuning.
"""

import math
from typing import Any, Dict, Optional

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


class OLMoClassifier(pl.LightningModule):
    """Lightning module for classification tasks.

    Supports single-task and multi-task classification.
    Can use either a classification head or LM head (Yes/No prediction).
    Supports QLoRA (4-bit), LoRA, and full finetuning.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    num_tasks : int
        Number of classification tasks/labels.
    task_type : str
        One of 'single_task' or 'multi_task'.
    use_lm_head : bool
        If 'True', use Yes/No LM-head prediction instead of a classification head.
    finetune_strategy : str
        One of 'qlora' (4-bit + LoRA), 'lora' (LoRA only), or
        'full_finetune' (all parameters trainable).
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for AdamW.
    warmup_ratio : float
        Fraction of total steps used for linear warmup.
    lora_r : int
        LoRA rank.
    lora_alpha : int
        LoRA alpha (typically 2× rank).
    lora_dropout : float
        LoRA dropout rate.
    """

    def __init__(
        self,
        model_name: str = "allenai/OLMo-7B-hf",
        num_tasks: int = 1,
        task_type: str = "single_task",
        use_lm_head: bool = False,
        finetune_strategy: str = "qlora",
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = None
        self.tokenizer = None

        # Setup metrics based on task type
        if task_type == "single_task":
            metric_kwargs = {"task": "binary"}
        else:
            metric_kwargs = {
                "task": "multilabel",
                "num_labels": num_tasks,
                "average": "macro",
            }

        self.train_acc = Accuracy(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.val_auroc = AUROC(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)
        self.test_auroc = AUROC(**metric_kwargs)

    def configure_model(self) -> None:
        """Initialise the backbone model and optional LoRA adapters.

        Called by the trainer before training starts. Loads the base model,
        applies quantization (QLoRA) and LoRA adapters based on
        'finetune_strategy', then wraps with the appropriate head.
        """
        if self.model is not None:
            return

        hp = self.hparams

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config (only for qlora)
        bnb_config = None
        if hp.finetune_strategy == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        device_map = get_device_map(self.device)

        if hp.use_lm_head:
            # Use AutoModelForCausalLM with LM head
            base = AutoModelForCausalLM.from_pretrained(
                hp.model_name,
                quantization_config=bnb_config,
                device_map=device_map,
            )
            if hp.finetune_strategy == "qlora":
                base = prepare_model_for_kbit_training(
                    base, use_gradient_checkpointing=True
                )
            if hp.finetune_strategy != "full_finetune":
                lora_cfg = LoraConfig(
                    r=hp.lora_r,
                    lora_alpha=hp.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    lora_dropout=hp.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                base = get_peft_model(base, lora_cfg)

            if self.global_rank == 0:
                base.print_trainable_parameters()

            self.model = CausalLMClassificationHead(
                base, self.tokenizer, hp.num_tasks, hp.task_type
            )
        else:
            # Use AutoModel with classification head
            base = AutoModel.from_pretrained(
                hp.model_name,
                quantization_config=bnb_config,
                device_map=device_map,
            )
            if hp.finetune_strategy == "qlora":
                base = prepare_model_for_kbit_training(
                    base, use_gradient_checkpointing=True
                )
            if hp.finetune_strategy != "full_finetune":
                lora_cfg = LoraConfig(
                    r=hp.lora_r,
                    lora_alpha=hp.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    lora_dropout=hp.lora_dropout,
                    bias="none",
                    task_type="FEATURE_EXTRACTION",
                )
                base = get_peft_model(base, lora_cfg)

            if self.global_rank == 0:
                base.print_trainable_parameters()

            self.model = ClassificationHead(base, hp.num_tasks, hp.task_type)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """Run the forward pass through the classification model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape '(batch, seq_len)'.
        attention_mask : torch.Tensor
            Attention mask of shape '(batch, seq_len)'.
        labels : torch.Tensor, optional
            Ground-truth labels for loss computation.
        label_mask : torch.Tensor, optional
            Boolean mask for valid labels (multilabel/multitask tasks).

        Returns
        -------
        tuple
            '(logits, loss)' returned by the classification head.
        """
        return self.model(input_ids, attention_mask, labels, label_mask)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Compute loss and update metrics for a single batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dict with 'input_ids', 'attention_mask', 'labels', and
            optionally 'label_mask'.
        stage : str
            One of 'train', 'val', or 'test'.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        label_mask = batch.get("label_mask", None)
        logits, loss = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
            label_mask,
        )

        # Get metrics for this stage
        acc_metric = getattr(self, f"{stage}_acc")
        auroc_metric = getattr(self, f"{stage}_auroc", None)

        # Compute predictions and update metrics
        if self.hparams.task_type == "single_task":
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            acc_metric(preds, batch["labels"])
            if auroc_metric is not None:
                auroc_metric(probs, batch["labels"])
        else:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            if label_mask is not None:
                valid = label_mask.any(dim=1)
                if valid.any():
                    acc_metric(preds[valid], batch["labels"][valid].int())
                    if auroc_metric is not None:
                        auroc_metric(probs[valid], batch["labels"][valid].int())
            else:
                acc_metric(preds, batch["labels"].int())
                if auroc_metric is not None:
                    auroc_metric(probs, batch["labels"].int())

        # Log metrics
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/acc", acc_metric, on_epoch=True, sync_dist=True)
        if auroc_metric is not None:
            self.log(
                f"{stage}/roc_auc", auroc_metric, on_epoch=True, prog_bar=True, sync_dist=True
            )

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute a single training step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of tokenized samples from the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar training loss.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute a single validation step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of tokenized samples from the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar validation loss.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute a single test step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of tokenized samples from the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar test loss.
        """
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict:
        """Set up AdamW optimizer with warmup + cosine annealing scheduler.

        Returns
        -------
        Dict
            Dict with 'optimizer' and 'lr_scheduler' keys, as expected
            by PyTorch Lightning.
        """
        hp = self.hparams

        # Separate params for weight decay
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

        # Warmup + cosine annealing
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
