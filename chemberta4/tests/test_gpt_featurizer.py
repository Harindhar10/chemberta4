import os
from typing import Optional, Tuple

import deepchem as dc
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepchem.data import _TorchIndexDiskDataset
from chemberta4.gpt_featurizer import GPTFeaturizer


class DummyMLPClassificationHead(nn.Module):
    """Lightweight MLP head for testing"""

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_dim: int = 256,
        num_tasks: int = 1,
        task_type: str = "single_task",
    ):
        super().__init__()
        self.task_type = task_type
        self.num_tasks = num_tasks
        output_dim = 2 if task_type == "single_task" else num_tasks

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embeds = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits = self.mlp(pooled)

        loss = None
        if labels is not None:
            if self.task_type == "single_task":
                loss = nn.CrossEntropyLoss()(logits, labels)
            else:
                loss = nn.BCEWithLogitsLoss()(logits, labels)
        return logits, loss


class DummyOLMoClassifier(nn.Module):
    """
    This class wraps ``DummyMLPClassificationHead`` and exposes the same
    ``configure_model`` / ``forward`` interface used by the real
    classifier, but without any PyTorch-Lightning or LoRA dependencies.
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_dim: int = 256,
        num_tasks: int = 1,
        task_type: str = "single_task",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.model = None

    def configure_model(self) -> None:
        if self.model is not None:
            return
        self.model = DummyMLPClassificationHead(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_tasks=self.num_tasks,
            task_type=self.task_type,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.model(input_ids, attention_mask, labels)


class TestGPTFeaturizer:
    """End-to-end tests for GPTFeaturizer.

    Each test mirrors the real training pipeline in
    ``train_classification.py``:  CSV → ``dc.data.CSVLoader`` with
    ``GPTFeaturizer`` → ``_TorchIndexDiskDataset`` → ``DataLoader`` →
    model forward pass.  This ensures the featurizer's output format
    (tensor shapes, dtypes, dict keys) is fully compatible with the
    downstream model's expectations.
    """

    def test_single_task_pipeline(self, load_tokenizer, tmp_path):
        """Test for the end-to-end single-task pipeline: CSV → CSVLoader + GPTFeaturizer →
        _TorchIndexDiskDataset → DataLoader → DummyOLMoClassifier forward.

        Mirrors the real flow in ``train_classification.py``.  Verifies that
        the featurizer produces ``(1, 128)`` tensors that, after DataLoader
        collation and ``squeeze(1)``, yield ``(B, 128)`` inputs with
        ``torch.long`` labels compatible with ``CrossEntropyLoss``.

        The dummy model must produce ``(B, 2)`` logits and a finite loss.
        """
        # Write a small CSV
        csv_path = tmp_path / "train.csv"
        pd.DataFrame({
            "smiles": ["CCO", "C1=CC=CC=C1", "CC(=O)O", "C"],
            "p_np": [1, 0, 1, 0],
        }).to_csv(csv_path, index=False)

        # Deepchem's CSVLoader + GPTFeaturizer
        featurizer = GPTFeaturizer(load_tokenizer, task_name="bbbp", task_type="single_task")
        loader = dc.data.CSVLoader(
            tasks=["p_np"],
            feature_field=["smiles", "p_np"],
            id_field="smiles",
            featurizer=featurizer,
        )
        disk_ds = loader.create_dataset(str(csv_path))

        # Wrap Deepchem's DiskDataset with _TorchIndexDiskDataset to make it a pytorch dataset
        torch_ds = dc.data._TorchIndexDiskDataset(disk_ds)
        assert len(torch_ds) == 4

        # DataLoader
        dl = DataLoader(torch_ds, batch_size=2, shuffle=False)
        batch = next(iter(dl))

        # Unpack 
        x_dict, _, _, _ = batch
        input_ids = x_dict["input_ids"].squeeze(1)          # (B,1,128) -> (B,128)
        attention_mask = x_dict["attention_mask"].squeeze(1)
        labels = x_dict["labels"]

        assert input_ids.shape == (2, 128)
        assert attention_mask.shape == (2, 128)
        assert labels.dtype == torch.long

        #Forward pass through dummy model
        model = DummyOLMoClassifier(vocab_size=load_tokenizer.vocab_size, hidden_dim=32, num_tasks=1)
        model.configure_model()
        logits, loss = model(input_ids, attention_mask, labels)

        assert logits.shape == (2, 2)
        assert loss is not None
        assert torch.isfinite(loss)

    def test_multi_task_pipeline(self, load_tokenizer, tmp_path):
        """End-to-end multi-task pipeline with a SIDER-style CSV (3 label
        columns) and ``task_type="multi_task"``.

        Multi-task labels must be ``torch.float32`` of shape
        ``(B, num_tasks)`` for ``BCEWithLogitsLoss``.  The dummy model
        must produce ``(B, num_tasks)`` logits and a finite loss.
        """

        csv_path = tmp_path / "train.csv"
        pd.DataFrame({
            "smiles": ["CCO", "C1=CC=CC=C1", "CC(=O)O", "C"],
            "task1": [1.0, 0.0, 1.0, 0.0],
            "task2": [0.0, 1.0, 0.0, 1.0],
            "task3": [1.0, 1.0, 0.0, 0.0],
        }).to_csv(csv_path, index=False)

        task_columns = ["task1", "task2", "task3"]
        featurizer = GPTFeaturizer(load_tokenizer, task_name="sider", task_type="multi_task")
        loader = dc.data.CSVLoader(
            tasks=task_columns,
            feature_field=["smiles"] + task_columns,
            id_field="smiles",
            featurizer=featurizer,
        )
        disk_ds = loader.create_dataset(str(csv_path))
        torch_ds = _TorchIndexDiskDataset(disk_ds)
        dl = DataLoader(torch_ds, batch_size=2, shuffle=False)
        batch = next(iter(dl))

        x_dict, _, _, _ = batch
        input_ids = x_dict["input_ids"].squeeze(1)
        attention_mask = x_dict["attention_mask"].squeeze(1)
        labels = x_dict["labels"]

        assert input_ids.shape == (2, 128)
        assert labels.dtype == torch.float32
        assert labels.shape == (2, 3)

        model = DummyOLMoClassifier(
            vocab_size=load_tokenizer.vocab_size,
            hidden_dim=32,
            num_tasks=3,
            task_type="multi_task",
        )
        model.configure_model()
        logits, loss = model(input_ids, attention_mask, labels)

        assert logits.shape == (2, 3)
        assert loss is not None
        assert torch.isfinite(loss)

    def test_formatting_prompts_func(self, load_tokenizer):
        """Verify that prompt templates correctly wrap SMILES strings.

        Each formatted text must contain the original SMILES, end with
        the tokenizer's EOS token, and include the task-specific prompt
        (e.g. "blood-brain barrier" for BBBP, "HIV" for HIV).

        A wrong template silently corrupts predictions with no runtime
        error, so this test catches the problem at the source.
        """
        featurizer_bbbp = GPTFeaturizer(load_tokenizer, task_name="bbbp")
        featurizer_hiv = GPTFeaturizer(load_tokenizer, task_name="hiv")

        texts_bbbp = featurizer_bbbp.formatting_prompts_func(["CCO"])
        texts_hiv = featurizer_hiv.formatting_prompts_func(["CCO"])

        # SMILES must appear in the formatted text
        assert "CCO" in texts_bbbp[0]
        assert "CCO" in texts_hiv[0]

        # Each text must end with the EOS token
        assert texts_bbbp[0].endswith(load_tokenizer.eos_token)
        assert texts_hiv[0].endswith(load_tokenizer.eos_token)

        # Task-specific content must be present
        assert "blood-brain barrier" in texts_bbbp[0]
        assert "HIV" in texts_hiv[0]

        # Different tasks must produce different prompts
        assert texts_bbbp[0] != texts_hiv[0]
