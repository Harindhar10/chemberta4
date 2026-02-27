
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List
from transformers import PreTrainedTokenizerBase


class MoleculeNetDataset(Dataset):
    """
    A PyTorch dataset that wraps MoleculeNet CSV data for molecular property
    prediction. It supports classification (single-task and multi-task) and
    regression, with two prompt formats depending on whether a linear head or
    the language model head is used.

    Label processing depends on the combination of 'task_type' and
    'experiment_type':

    * **single_task classification** — rows with missing labels are dropped;
      labels are stored as integers for cross-entropy loss.
    * **multi_task classification** — all rows are kept; a boolean mask tracks
      which labels are present so that missing values (NaN) are excluded from
      the loss.
    * **Causal LM regression** ('use_lm_head=True', 'experiment_type="regression"')
      — the target number is embedded directly into the prompt text (e.g.
      ``"### Response:\\n3.14159"``). Tokenization happens per-sample in
      ``__getitem__`` and prompt tokens are masked with -100 so that only the
      answer portion contributes to the cross-entropy loss.
    * **Standard regression** — rows with missing labels are dropped; labels
      are stored as floats for RMSE loss.

    Examples
    --------
    >>> import pandas as pd
    >>> from transformers import AutoTokenizer
    >>> from chemberta4.data import MoleculeNetDataset
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> tokenizer.pad_token = tokenizer.eos_token
    >>> df = pd.DataFrame({"smiles": ["CC", "CCO"], "label": [0, 1]})
    >>> ds = MoleculeNetDataset(
    ...     df, tokenizer, ["label"], "Is it soluble?",
    ...     "single_task", "classification", max_len=32)
    >>> sample = ds[0]
    >>> list(sample.keys())
    ['input_ids', 'attention_mask', 'labels']
    >>> sample["labels"].item()
    0
    >>> sample["input_ids"].shape
    torch.Size([32])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        task_columns: List[str],
        prompt: str,
        task_type: str,
        experiment_type: str,
        max_len: int = 128,
        use_lm_head: bool = False,
        smiles_column: str = "smiles",
    ) -> None:
        """Initialise MoleculeNetDataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with SMILES and target columns.
        tokenizer : PreTrainedTokenizerBase
            HuggingFace tokenizer.
        task_columns : List[str]
            Column names containing the target labels.
        prompt : str
            Task-specific prompt text prepended to each molecule.
        task_type : str
            One of 'single_task' or 'multi_task' (for classification).
        experiment_type : str
            One of 'classification' or 'regression'.
        max_len : int
            Maximum token sequence length for truncation/padding.
        use_lm_head : bool
            If 'True', format prompts for Yes/No LM-head prediction
            (classification) or embed the answer in the text for teacher-forced
            causal LM regression.
        smiles_column : str
            Name of the column containing SMILES strings.
        """

        self.task_type = task_type
        self.experiment_type = experiment_type
        self.num_tasks = len(task_columns)
        self.use_lm_head = use_lm_head

        # Process labels based on task type
        if task_type == "single_task":
            df = df.dropna(subset=task_columns).copy()
            self.labels = torch.tensor(
                df[task_columns[0]].values.astype(int), dtype=torch.long
            )
            self.label_mask = None

        elif task_type == "multi_task":
            df = df.copy()
            labels_array = df[task_columns].values.astype(np.float32)
            # Mask for missing labels (NaN values)
            self.label_mask = torch.tensor(~np.isnan(labels_array), dtype=torch.bool)
            # Replace NaN with 0 for computation
            labels_array = np.nan_to_num(labels_array, nan=0.0)
            self.labels = torch.tensor(labels_array, dtype=torch.float32)

        elif use_lm_head and experiment_type == "regression":
            # CLM regression: embed the answer in the text; tokenize lazily
            # per sample in __getitem__ using the reference's split-separator
            # approach to avoid BPE tokenization boundary issues.
            _SEPARATOR = "### Response:\n"
            df = df.dropna(subset=task_columns).copy()
            labels = df[task_columns[0]].values.astype(np.float32)

            self._clm_texts = [
                f"Molecule: {s}\nQuestion: {prompt}\n{_SEPARATOR}{v:.5f}{tokenizer.eos_token}"
                for s, v in zip(df[smiles_column], labels)
            ]
            self._clm_separator = _SEPARATOR
            self._tokenizer = tokenizer
            self._max_len = max_len
            self.label_values = torch.tensor(labels, dtype=torch.float32)
            self.label_mask = None
            self.num_samples = len(df)
            return  # __getitem__ handles tokenization for CLM regression

        elif experiment_type == "regression":
            df = df.dropna(subset=task_columns).copy()
            labels = df[task_columns[0]].values.astype(np.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)
            self.label_mask = None

        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")

        # Build prompts
        if use_lm_head:
            texts = [
                f"Molecule: {s}\nQuestion: {prompt}\nAnswer:"
                for s in df[smiles_column]
            ]
        else:
            texts = [f"Molecule: {s}\n{prompt}" for s in df[smiles_column]]

        # Tokenize all at once
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.num_samples = len(df)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single dataset sample formatted for transformer training.

        This method supports two modes of operation:

        1. Standard encoder-style training (classification or regression)
        Returns pre-tokenized inputs stored in ``self.encodings`` along with
        their corresponding labels. Optionally includes a ``label_mask`` for
        multi-task setups with missing labels.

        2. Causal language modeling (CLM) regression mode
        Triggered when ``self.use_lm_head`` is True and
        ``self.experiment_type == "regression"``.
        In this case:
            - The full prompt + target text is tokenized.
            - ``labels`` are initialized as a clone of ``input_ids``.
            - Tokens corresponding to the prompt portion (before
            ``self._clm_separator``) are masked with -100 so that loss is
            computed only on the target portion.
            - Padding tokens are also masked with -100.
            - The original scalar regression value is returned separately as
            ``label_values``.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dict with 'input_ids', 'attention_mask', 'labels', and
            optionally 'label_mask' (for multi_task tasks).

        Examples
        --------
        >>> import pandas as pd
        >>> from transformers import AutoTokenizer
        >>> from chemberta4.data import MoleculeNetDataset
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> df = pd.DataFrame({"smiles": ["CC", "CCO"], "label": [0, 1]})
        >>> ds = MoleculeNetDataset(
        ...     df, tokenizer, ["label"], "Is it soluble?",
        ...     "single_task", "classification", max_len=32)
        >>> sample = ds[0]
        >>> list(sample.keys())
        ['input_ids', 'attention_mask', 'labels']
        >>> sample["labels"].item()
        0
        >>> sample["input_ids"].shape
        torch.Size([32])
        """
        if self.use_lm_head and self.experiment_type == "regression":
            text = self._clm_texts[idx]
            enc = self._tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self._max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            parts = text.split(self._clm_separator)
            if len(parts) >= 2:
                prompt_text = parts[0] + self._clm_separator
                prompt_enc = self._tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=self._max_len,
                    return_tensors="pt",
                )
                prompt_len = prompt_enc["input_ids"].shape[1]
                if prompt_len < len(labels):
                    labels[:prompt_len] = -100

            labels[labels == self._tokenizer.pad_token_id] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "label_values": self.label_values[idx],
            }

        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }
        if self.label_mask is not None:
            item["label_mask"] = self.label_mask[idx]
        return item

