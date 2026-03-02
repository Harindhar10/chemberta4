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


class OLMoRegressor(pl.LightningModule):
    """This class implements a PyTorch Lightning module for molecular regression tasks.

    Supports two head types selected by 'use_lm_head':

    * 'False' (default): 'RegressionHead' — last-token pooling + linear layer,
      trained with RMSE loss on raw labels.
    * 'True': 'CausalLMRegressionHead' — teacher-forced causal LM with
      cross-entropy loss during training; generates text and parses a float
      with regex during validation/test for RMSE reporting.

    Examples
    --------
    >>> from chemberta4.trainer import OLMoRegressor
    >>> reg = OLMoRegressor()
    >>> reg.hparams.use_lm_head
    False
    """

    def __init__(
        self,
        model_name: str = "allenai/OLMo-7B-hf",
        finetune_strategy: str = "qlora",
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        use_lm_head: bool = False,
    ):
        """Initialise OLMoRegressor.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier.
        finetune_strategy : str
            One of 'qlora', 'lora', or 'full_finetune'.
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay for AdamW.
        warmup_ratio : float
            Fraction of total steps used for linear warmup.
        lora_r : int
            LoRA rank.
        lora_alpha : int
            LoRA alpha.
        lora_dropout : float
            LoRA dropout rate.
        use_lm_head : bool
            If 'True', use 'CausalLMRegressionHead' with cross-entropy training
            and text-generation evaluation. If 'False', use 'RegressionHead'
            with direct RMSE loss.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = None
        self.tokenizer = None

    def configure_model(self) -> None:
        """Initialise the backbone model and optional LoRA adapters.

        Called by the trainer before training starts. Applies quantization
        and LoRA based on 'finetune_strategy', then wraps with a regression head.
        """
        if self.model is not None:
            return

        hp = self.hparams

        self.tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if hp.finetune_strategy == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        device_map = "cpu" #get_device_map(self.device)

        if hp.use_lm_head:
            base = AutoModelForCausalLM.from_pretrained(
                hp.model_name,
                quantization_config=bnb_config,
                device_map=device_map,
            )
        else:
            base = AutoModel.from_pretrained(
                hp.model_name,
                quantization_config=bnb_config,
                device_map=device_map,
            )

        if hp.finetune_strategy == "qlora":
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        if hp.finetune_strategy != "full_finetune":
            lora_task_type = TaskType.CAUSAL_LM if hp.use_lm_head else "FEATURE_EXTRACTION"
            lora_cfg = LoraConfig(
                r=hp.lora_r,
                lora_alpha=hp.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=hp.lora_dropout,
                bias="none",
                task_type=lora_task_type,
            )
            base = get_peft_model(base, lora_cfg)

        if self.global_rank == 0:
            base.print_trainable_parameters()

        if hp.use_lm_head:
            self.model = CausalLMRegressionHead(base, self.tokenizer)
        else:
            self.model = RegressionHead(base)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """Run the forward pass through the regression model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape '(batch, seq_len)'.
        attention_mask : torch.Tensor
            Attention mask of shape '(batch, seq_len)'.
        labels : torch.Tensor, optional
            Ground-truth labels for loss computation.

        Returns
        -------
        tuple
            '(predictions, loss)' returned by the regression head.
        """
        return self.model(input_ids, attention_mask, labels)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Compute loss and log RMSE/MAE for a single batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dict with 'input_ids', 'attention_mask', and 'labels'.
        stage : str
            One of 'train', 'val', or 'test'.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        if self.hparams.use_lm_head:
            return self._clm_step(batch, stage)

        preds, loss = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        rmse = torch.sqrt(torch.nn.functional.mse_loss(preds, batch["labels"]) + 1e-6)
        mae = torch.mean(torch.abs(preds - batch["labels"]))

        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/rmse", rmse, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/mae", mae, on_epoch=True, sync_dist=True)

        return loss

    def _clm_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Handle a batch for the CausalLMRegressionHead.

        During training, computes and logs cross-entropy loss.
        During validation/test, generates text, parses floats with regex,
        and logs RMSE/MAE.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dict with 'input_ids', 'attention_mask', 'labels' (token IDs with
            -100 masking), and 'label_values' (raw float targets, eval only).
        stage : str
            One of 'train', 'val', or 'test'.

        Returns
        -------
        torch.Tensor
            Cross-entropy loss for training, RMSE for validation/test.
        """
        if stage == "train":
            _, loss = self(
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
            self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss

        # Evaluation: generate text and parse predicted numbers
        parsed_preds = self.model.generate_and_parse(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        true_vals = batch["label_values"].to(parsed_preds.device)

        valid = ~(torch.isnan(parsed_preds) | torch.isnan(true_vals))
        if valid.any():
            rmse = torch.sqrt(
                ((parsed_preds[valid] - true_vals[valid]) ** 2).mean() + 1e-6
            )
            mae = torch.abs(parsed_preds[valid] - true_vals[valid]).mean()
        else:
            rmse = torch.tensor(float("nan"), device=self.device)
            mae = torch.tensor(float("nan"), device=self.device)

        self.log(f"{stage}/rmse", rmse, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/mae", mae, on_epoch=True, sync_dist=True)

        return rmse

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
