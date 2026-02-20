import torch
import torch.nn as nn
from typing import Optional, Tuple


def last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract the last non-padding token representation.

    For decoder-only models like OLMo, we use the last token's representation
    for classification/regression tasks.

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        attention_mask: [batch, seq_len]

    Returns:
        Pooled output: [batch, hidden_size]
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
    """Regression head with last-token pooling.

    Uses RMSE loss by default.
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
