import torch
import torch.nn as nn
from typing import Optional, Tuple



def last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """This function extracts the last non-padding token representation.

    For decoder-only models like OLMo, we use the last token's representation
    for classification/regression tasks. Decoder-only transformers process
    tokens left-to-right, so the final non-padding position has attended to
    the entire input sequence. This function uses
    'attention_mask.sum(dim=1) - 1' to locate that position for each item
    in the batch and extracts the corresponding hidden vector with
    'torch.gather', avoiding any loop over batch elements.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Hidden states of shape '[batch, seq_len, hidden_size]'.
    attention_mask : torch.Tensor
        Attention mask of shape '[batch, seq_len]'.

    Returns
    -------
    torch.Tensor
        Pooled output of shape '[batch, hidden_size]'.

    Examples
    --------
    >>> import torch
    >>> from chemberta4.model import last_token_pool
    >>> hidden = torch.zeros(2, 4, 8)
    >>> hidden[0, 2, :] = 1.0   # last real token at position 2 for sample 0
    >>> hidden[1, 3, :] = 1.0   # last real token at position 3 for sample 1
    >>> mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    >>> out = last_token_pool(hidden, mask)
    >>> out.shape
    torch.Size([2, 8])
    >>> out[0].sum().item()
    8.0
    >>> out[1].sum().item()
    8.0
    """
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]

    # Create indices for gathering
    indices = sequence_lengths.view(-1, 1, 1).expand(
        batch_size, 1, hidden_states.size(-1)
    )
    indices = indices.to(hidden_states.device)

    # Gather and squeeze
    return torch.gather(hidden_states, 1, indices).squeeze(1)



class RegressionHead(nn.Module):
    """This class implements a regression head with last-token pooling for scalar molecular property prediction.

    It uses RMSE loss by default.

    Takes a backbone 'nn.Module' and appends a single linear unit that maps
    the last-token hidden state to a scalar. Loss is the square-root of MSE
    (RMSE) with a small epsilon (1e-6) added for numerical stability.

    Examples
    --------
    >>> import torch, torch.nn as nn
    >>> from types import SimpleNamespace
    >>> from chemberta4.model import RegressionHead
    >>> class DummyBackbone(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.config = SimpleNamespace(hidden_size=16)
    ...         self.embed = nn.Embedding(100, 16)
    ...     def forward(self, input_ids, attention_mask):
    ...         h = self.embed(input_ids)
    ...         return SimpleNamespace(last_hidden_state=h)
    >>> head = RegressionHead(DummyBackbone())
    >>> input_ids = torch.zeros(2, 8, dtype=torch.long)
    >>> mask = torch.ones(2, 8, dtype=torch.long)
    >>> preds, loss = head(input_ids, mask)
    >>> preds.shape
    torch.Size([2])
    >>> loss is None
    True
    """

    def __init__(self, backbone: nn.Module):
        """Initialise RegressionHead.

        Parameters
        ----------
        backbone : nn.Module
            The base model (OLMo with LoRA).
        """
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Linear(backbone.config.hidden_size, 1)

        # Initialize with small weights
        nn.init.normal_(self.regressor.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.regressor.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the forward pass for regression.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape '[batch, seq_len]'.
        attention_mask : torch.Tensor
            Attention mask of shape '[batch, seq_len]'.
        labels : torch.Tensor, optional
            Normalized regression targets of shape '[batch]'.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Predicted values of shape '[batch]' and scalar RMSE loss if labels are provided, else None.

        Examples
        --------
        >>> import torch, torch.nn as nn
        >>> from types import SimpleNamespace
        >>> from chemberta4.model import RegressionHead
        >>> class DummyBackbone(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.config = SimpleNamespace(hidden_size=16)
        ...         self.embed = nn.Embedding(100, 16)
        ...     def forward(self, input_ids, attention_mask):
        ...         h = self.embed(input_ids)
        ...         return SimpleNamespace(last_hidden_state=h)
        >>> head = RegressionHead(DummyBackbone())
        >>> input_ids = torch.zeros(2, 8, dtype=torch.long)
        >>> mask = torch.ones(2, 8, dtype=torch.long)
        >>> preds, loss = head(input_ids, mask)
        >>> preds.shape
        torch.Size([2])
        >>> loss is None
        True
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_hidden_state = out.last_hidden_state
        pooled_output = last_token_pool(last_hidden_state, attention_mask)
        preds = self.regressor(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            # RMSE loss with epsilon for numerical stability
            loss = torch.sqrt(nn.functional.mse_loss(preds, labels) + 1e-6)

        return preds, loss


class CausalLMRegressionHead(nn.Module):
    """Regression head using the model's causal LM head for text generation.

    Trains with cross-entropy loss using teacher-forced generation: the answer
    number is embedded directly in the input text and prompt tokens are masked
    to -100 in the labels tensor. Only the response tokens contribute to the
    cross-entropy loss.

    At evaluation, generates text from the prompt-only portion and parses the
    first float from the decoded output using a regex pattern.

    Examples
    --------
    >>> import torch, torch.nn as nn
    >>> from types import SimpleNamespace
    >>> from chemberta4.model import CausalLMRegressionHead
    >>> class DummyCausalLM(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.embed = nn.Embedding(256, 16)
    ...         self.lm_head = nn.Linear(16, 256)
    ...     def forward(self, input_ids, attention_mask, labels=None):
    ...         h = self.embed(input_ids)
    ...         logits = self.lm_head(h)
    ...         loss = None
    ...         if labels is not None:
    ...             mask = labels != -100
    ...             loss = nn.CrossEntropyLoss()(logits[mask], labels[mask])
    ...         return SimpleNamespace(logits=logits, loss=loss)
    ...     def generate(self, input_ids, attention_mask, **kwargs):
    ...         return input_ids
    >>> class DummyTokenizer:
    ...     eos_token = "<eos>"
    ...     eos_token_id = 0
    ...     def decode(self, ids, skip_special_tokens=True):
    ...         return "3.14"
    >>> head = CausalLMRegressionHead(DummyCausalLM(), DummyTokenizer())
    >>> input_ids = torch.zeros(2, 8, dtype=torch.long)
    >>> mask = torch.ones(2, 8, dtype=torch.long)
    >>> labels = torch.full((2, 8), -100, dtype=torch.long)
    >>> labels[:, 6:] = 42
    >>> logits, loss = head(input_ids, mask, labels)
    >>> logits.shape
    torch.Size([2, 8, 256])
    >>> loss is not None
    True
    """

    def __init__(self, model: nn.Module, tokenizer) -> None:
        """Initialise CausalLMRegressionHead.

        Parameters
        ----------
        model : nn.Module
            A causal LM model with a language modelling head (e.g., OLMo loaded
            via AutoModelForCausalLM). Must accept a 'labels' argument and
            return an object with '.logits' and '.loss' attributes.
        tokenizer : PreTrainedTokenizerBase
            Tokenizer used to decode generated token IDs.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run teacher-forced forward pass for cross-entropy training.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape '[batch, seq_len]'. The full sequence including
            both the prompt and the answer number.
        attention_mask : torch.Tensor
            Attention mask of shape '[batch, seq_len]'.
        labels : torch.Tensor, optional
            Token IDs of shape '[batch, seq_len]' with prompt positions set to
            -100. HuggingFace computes cross-entropy only on non-masked positions.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Vocab logits of shape '[batch, seq_len, vocab_size]' and scalar
            cross-entropy loss if labels are provided, else None.

        Examples
        --------
        >>> import torch, torch.nn as nn
        >>> from types import SimpleNamespace
        >>> from chemberta4.model import CausalLMRegressionHead
        >>> class DummyCausalLM(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.embed = nn.Embedding(256, 16)
        ...         self.lm_head = nn.Linear(16, 256)
        ...     def forward(self, input_ids, attention_mask, labels=None):
        ...         h = self.embed(input_ids)
        ...         logits = self.lm_head(h)
        ...         loss = None
        ...         if labels is not None:
        ...             mask = labels != -100
        ...             loss = nn.CrossEntropyLoss()(logits[mask], labels[mask])
        ...         return SimpleNamespace(logits=logits, loss=loss)
        ...     def generate(self, input_ids, attention_mask, **kwargs):
        ...         return input_ids
        >>> class DummyTokenizer:
        ...     eos_token = "<eos>"
        ...     eos_token_id = 0
        ...     def decode(self, ids, skip_special_tokens=True):
        ...         return "3.14"
        >>> head = CausalLMRegressionHead(DummyCausalLM(), DummyTokenizer())
        >>> input_ids = torch.zeros(2, 8, dtype=torch.long)
        >>> mask = torch.ones(2, 8, dtype=torch.long)
        >>> labels = torch.full((2, 8), -100, dtype=torch.long)
        >>> labels[:, 6:] = 42
        >>> logits, loss = head(input_ids, mask, labels)
        >>> logits.shape
        torch.Size([2, 8, 256])
        >>> loss is not None
        True
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.logits, outputs.loss

    def generate_and_parse(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_ids: torch.Tensor,
        max_new_tokens: int = 15,
    ) -> torch.Tensor:
        """Generate text from the prompt and parse the first float from the output.

        Uses the label mask to determine where the prompt ends, then generates
        from that position and applies a regex to extract the predicted number.

        Parameters
        ----------
        input_ids : torch.Tensor
            Full token IDs of shape '[batch, seq_len]' (prompt + answer).
        attention_mask : torch.Tensor
            Attention mask of shape '[batch, seq_len]'.
        label_ids : torch.Tensor
            Token-level labels of shape '[batch, seq_len]' with prompt tokens
            set to -100. The first non-(-100) position marks the answer start.
        max_new_tokens : int
            Maximum number of tokens to generate for the answer.

        Returns
        -------
        torch.Tensor
            Parsed float predictions of shape '[batch]'. Unparseable outputs
            are set to 'float("nan")'.
        """
        preds = []

        for i in range(input_ids.shape[0]):
            # Find where the response starts (first token not masked to -100)
            response_mask = (label_ids[i] != -100)
            answer_start_idx = response_mask.int().argmax().item()

            if answer_start_idx == 0:
                # Entire sequence is prompt (no response tokens found)
                preds.append(float("nan"))
                continue

            prompt_ids = input_ids[i : i + 1, :answer_start_idx]
            prompt_mask = attention_mask[i : i + 1, :answer_start_idx]

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0, answer_start_idx:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            match = re.search(r"[-+]?\d*\.\d+|\d+", generated_text)
            preds.append(float(match.group()) if match else float("nan"))

        return torch.tensor(preds, dtype=torch.float32, device=input_ids.device)
