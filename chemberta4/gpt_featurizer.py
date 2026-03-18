import pandas as pd
import torch
from deepchem.feat import Featurizer
from typing import Dict, List

from chemberta4.prompt_templates import PROMPT_TEMPLATES

try:
    from transformers import PreTrainedTokenizerBase
except ModuleNotFoundError:
    raise ImportError(
        "Transformers must be installed for GPTFeaturizer to be used!"
    )


class GPTFeaturizer(Featurizer):
    """DeepChem-compatible featurizer that wraps a HuggingFace tokenizer.

    Applies a task-specific prompt template to each SMILES string, tokenizes
    the result with padding/truncation, and attaches ground-truth labels.
    Designed for use with ``dc.data.CSVLoader`` so that data preparation
    stays inside the DeepChem ecosystem while leveraging HuggingFace
    tokenizers (e.g. ``GPTNeoXTokenizerFast``).

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer instance (e.g. from
        ``AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")``).
    task_name : str
        Key into ``PROMPT_TEMPLATES`` (e.g. ``"bbbp"``, ``"sider"``).
    task_type : str, optional
        ``"single_task"`` (default) for binary classification with
        ``torch.long`` labels, or ``"multi_task"`` for multi-label
        classification with ``torch.float32`` labels.

    Examples
    --------
    >>> import pandas as pd
    >>> from transformers import AutoTokenizer
    >>> from chemberta4.gpt_tokenizer import GPTFeaturizer
    >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")
    >>> tokenizer.pad_token = tokenizer.eos_token
    >>> featurizer = GPTFeaturizer(tokenizer, task_name="bbbp")
    >>> df = pd.DataFrame({"smiles": ["CCO", "C1=CC=CC=C1"], "p_np": [1, 0]})
    >>> encodings = featurizer.featurize(df)
    >>> encodings[0]["input_ids"].shape
    torch.Size([1, 128])
    >>> encodings[0]["labels"].dtype
    torch.int64
    """

    def __init__(self, tokenizer, task_name: str, task_type: str = "single_task"):
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.task_type = task_type

    def featurize(self, datapoints: pd.DataFrame, **kwargs) -> List[Dict[str, torch.Tensor]]:
        """Tokenize a DataFrame of SMILES strings and attach labels.

        Each SMILES is wrapped with the task prompt via
        ``formatting_prompts_func``, tokenized to fixed length 128, and
        paired with its ground-truth label(s).

        Parameters
        ----------
        datapoints : pd.DataFrame
            Must contain a ``"smiles"`` column and one or more label columns.
        **kwargs
            Passed through to the underlying tokenizer.

        Returns
        -------
        List[Dict[str, torch.Tensor]]
            One dict per row with keys ``"input_ids"``, ``"attention_mask"``,
            and ``"labels"``.  For ``single_task`` the label is a scalar
            ``torch.long``; for ``multi_task`` it is a 1-D ``torch.float32``
            tensor of length ``len(label_cols)``.
        """
        label_cols = datapoints.columns.drop('smiles')

        smiles = datapoints['smiles']
        smiles_with_prompts = self.formatting_prompts_func(smiles)

        encodings = []
        for i, text in enumerate(smiles_with_prompts):
            enc = self._featurize(text)
            if self.task_type == "multi_task":
                enc['labels'] = torch.tensor(
                    datapoints.iloc[i][label_cols].values.astype(float),
                    dtype=torch.float32,
                )
            else:
                enc['labels'] = torch.tensor(
                    datapoints.iloc[i][label_cols[0]], dtype=torch.long,
                )
            encodings.append(enc)
        return encodings

    def _featurize(self, datapoint: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize a single string.

        Parameters
        ----------
        datapoint : str
            Text to tokenize (typically a prompt-wrapped SMILES).
        **kwargs
            Forwarded to the HuggingFace tokenizer ``__call__``.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with at least ``"input_ids"`` and
            ``"attention_mask"`` tensors of shape ``(1, 128)``.
        """
        encoding = self.tokenizer(
            datapoint,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
            **kwargs,
        ).data
        return encoding

    def formatting_prompts_func(self, examples) -> List[str]:
        """Apply the task prompt template to each SMILES and append EOS.

        Parameters
        ----------
        examples : Iterable[str]
            SMILES strings to wrap with the prompt.

        Returns
        -------
        List[str]
            Prompt-formatted texts, each ending with the tokenizer's EOS
            token.

        Examples
        --------
        >>> # Assuming tokenizer and featurizer are already initialised:
        >>> texts = featurizer.formatting_prompts_func(["CCO"])
        >>> "CCO" in texts[0]
        True
        """
        prompt_template = PROMPT_TEMPLATES[self.task_name]
        EOS_TOKEN = self.tokenizer.eos_token
        texts = []
        for molecule in examples:
            text = prompt_template.format(smiles=molecule) + EOS_TOKEN
            texts.append(text)
        return texts
