"""Scoring heads and pairwise losses (torch).

A head maps the features of one edit to a score (and, for the uncertainty loss, a log variance). Training only
ever sees two edits of the same sample, A and B, and the probability that A is better:
- "bt": Bradley-Terry with hard labels, P(A > B) = sigmoid(s_A - s_B), target 1 if A is the final winner;
- "soft": the same model with the soft label p_a of the final ranking as target (the team's uncertainty);
- "uncertainty": as in HPSv3, each edit gets a Gaussian score N(mu, sigma^2), so
  P(A > B) = Phi((mu_A - mu_B) / sqrt(sigma_A^2 + sigma_B^2)), target p_a. Edits the model is unsure about
  (large sigma) push less on the loss.
At inference the score is s (or mu): one forward pass per edit, comparable across the edits of a request.
"""

import math

import torch
import torch.nn.functional as functional
from torch import nn

LOSSES = ("bt", "soft", "uncertainty")
HEADS = ("linear", "mlp")


class ScoreHead(nn.Module):
    """Linear: dropout on the inputs, then one linear layer. MLP: dropout on the inputs, one hidden layer (GELU,
    dropout). Input dropout is the main guard against overfitting: there are thousands of input features for
    about 20,000 training pairs."""

    def __init__(self, n_inputs: int, head: str = "mlp", hidden: int = 64, dropout: float = 0.3,
                 input_dropout: float = 0.1, uncertainty: bool = False):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
        outputs = 2 if uncertainty else 1
        self.uncertainty = uncertainty
        layers = [nn.Dropout(input_dropout)]
        if head == "linear":
            layers.append(nn.Linear(n_inputs, outputs))
        else:
            layers += [nn.Linear(n_inputs, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, outputs)]
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Score (mean) and log variance (None without uncertainty) of each row."""
        output = self.net(features)
        return output[:, 0], (output[:, 1] if self.uncertainty else None)


def pair_loss(mu_a, mu_b, logvar_a, logvar_b, target: torch.Tensor, loss: str) -> torch.Tensor:
    """Mean cross-entropy between P(A > B) and the target probability."""
    if loss in ("bt", "soft"):
        return functional.binary_cross_entropy_with_logits(mu_a - mu_b, target)
    if loss == "uncertainty":
        standardised = (mu_a - mu_b) / torch.sqrt(logvar_a.exp() + logvar_b.exp() + 1e-6)
        probability = (0.5 * (1 + torch.erf(standardised / math.sqrt(2)))).clamp(1e-6, 1 - 1e-6)
        return functional.binary_cross_entropy(probability, target)
    raise ValueError(f"unknown loss {loss!r}; expected one of {LOSSES}")
