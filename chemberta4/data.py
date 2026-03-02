import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Any



class PretrainingDataset(Dataset):
    """
    This class wraps SMILES strings as a PyTorch dataset for causal language model pretraining.

    It formats each SMILES string for next-token prediction. Each SMILES is
    prefixed with 'prefix' (default "SMILES: ") and tokenised to a fixed
    length with right-padding. The 'labels' tensor is identical to
    'input_ids' except that padding positions are set to '-100' so
    PyTorch's cross-entropy ignores them.

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> from chemberta4.data import PretrainingDataset
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> tokenizer.pad_token = tokenizer.eos_token
    >>> ds = PretrainingDataset(["CC", "CCO"], tokenizer, max_len=16)
    >>> sample = ds[0]
    >>> list(sample.keys())
    ['input_ids', 'attention_mask', 'labels']
    >>> sample["input_ids"].shape
    torch.Size([16])
    >>> (sample["labels"] == -100).any().item()
    True
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: PreTrainedTokenizerBase,
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
            Dict with 'input_ids', 'attention_mask', and 'labels'.

        Examples
        --------
        >>> from transformers import AutoTokenizer
        >>> from chemberta4.data import PretrainingDataset
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> ds = PretrainingDataset(["CC", "CCO"], tokenizer, max_len=16)
        >>> sample = ds[0]
        >>> list(sample.keys())
        ['input_ids', 'attention_mask', 'labels']
        >>> sample["input_ids"].shape
        torch.Size([16])
        >>> (sample["labels"] == -100).any().item()
        True
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class InstructionDataset(Dataset):
    """
    This class wraps instruction/input/output tuples as a PyTorch dataset for causal LM instruction tuning.

    It formats each sample as an instruction/input/output tuple for next-token prediction training. Each
    sample is formatted as '"Instruction: ...\nInput: ...\nOutput: ..."'
    and tokenised on-the-fly (lazy tokenisation). Padding tokens in 'labels'
    are masked to '-100'.

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> from chemberta4.data import InstructionDataset
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> tokenizer.pad_token = tokenizer.eos_token
    >>> data = [{"instruction": "Predict product.", "input": "CC + O", "output": "CCO"}]
    >>> ds = InstructionDataset(data, tokenizer, max_len=32)
    >>> sample = ds[0]
    >>> list(sample.keys())
    ['input_ids', 'attention_mask', 'labels']
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
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
            Dict with 'input_ids', 'attention_mask', and 'labels'.

        Examples
        --------
        >>> from transformers import AutoTokenizer
        >>> from chemberta4.data import InstructionDataset
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> data = [{"instruction": "Predict product.", "input": "CC + O", "output": "CCO"}]
        >>> ds = InstructionDataset(data, tokenizer, max_len=32)
        >>> sample = ds[0]
        >>> list(sample.keys())
        ['input_ids', 'attention_mask', 'labels']
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
        }

