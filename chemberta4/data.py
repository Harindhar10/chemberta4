import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Any


class PretrainingDataset(Dataset):
    """
    Dataset for causal language modeling pretraining on SMILES.

    Simply formats SMILES strings for next-token prediction.

    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer,
        max_len: int = 256,
        prefix: str = "SMILES: ",
    ) -> None:
        """Initialise PretrainingDataset.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings.
        tokenizer : PreTrainedTokenizerBase
            HuggingFace tokenizer.
        max_len : int
            Maximum token sequence length for truncation/padding.
        prefix : str
            String prepended to each SMILES (default: 'SMILES: ').
        """

        texts = [f"{prefix}{s}" for s in smiles_list]

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

        # Labels for causal LM: same as input_ids, mask padding
        self.labels = self.encodings["input_ids"].clone()
        self.labels[self.labels == tokenizer.pad_token_id] = -100

        # Byte counts for BPB calculation
        self.num_bytes = torch.tensor(
            [len(t.encode("utf-8")) for t in texts], dtype=torch.long
        )

        self.num_samples = len(smiles_list)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single tokenized sample.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dict with 'input_ids', 'attention_mask', 'labels', and
            'num_bytes' (byte count used for BPB calculation).
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
            "num_bytes": self.num_bytes[idx],
        }


class InstructionDataset(Dataset):
    """
    Dataset for instruction tuning (USPTO-style).

    Formats instruction/input/output tuples for causal LM training.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_len: int = 512,
    ) -> None:
        """Initialise InstructionDataset.

        Parameters
        ----------
        data : List[Dict]
            List of dicts with 'instruction', 'input', and 'output' keys.
        tokenizer : PreTrainedTokenizerBase
            HuggingFace tokenizer.
        max_len : int
            Maximum token sequence length for truncation/padding.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Materialize if streaming
        self.data = list(data)

        # Precompute byte counts for BPB calculation
        self.num_bytes = [
            len(
                f"Instruction: {item['instruction']}\n"
                f"Input: {item['input']}\n"
                f"Output: {item['output']}".encode("utf-8")
            )
            for item in self.data
        ]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single tokenized instruction sample.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dict with 'input_ids', 'attention_mask', 'labels', and
            'num_bytes' (byte count used for BPB calculation).
        """
        item = self.data[idx]

        # Build instruction format
        prompt = (
            f"Instruction: {item['instruction']}\n"
            f"Input: {item['input']}\n"
            f"Output: {item['output']}"
        )

        encodings = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Labels: mask padding tokens
        labels = encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "num_bytes": torch.tensor(self.num_bytes[idx], dtype=torch.long),
        }

