"""
The Oracle — Macroscopic Goal State Predictor

Predicts ẑ_final (the conclusion of a proof) from z_0 (the premise).
This eliminates target leakage: at inference, the system only has the
premise, so it must autonomously predict where the proof should end.

Architecture: Deep residual MLP with skip connections.
Predicting the end state from the start is computationally irreducible —
a shallow MLP cannot learn this mapping. Each residual block provides
a non-linear "reasoning hop" that incrementally transforms the premise
toward the predicted goal.

Loss: L_oracle = ||ẑ_final - sg(z_final)||² trained alongside the JEPA.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Single residual block: Linear → GELU → LayerNorm → Linear → residual."""

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d_model),
        )
        # Small init so residual starts near identity
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Oracle(nn.Module):
    """
    Predicts ẑ_final from z_0 (premise only).

    The premise → conclusion leap is far harder than the step → step
    prediction (which the micro-Predictor handles). This requires a
    deeper network with skip connections at every layer.

    Args:
        d_model: Latent dimension (must match JEPA)
        n_layers: Number of residual blocks (default 4)
        expansion: FFN expansion factor per block (default 4)
    """

    def __init__(self, d_model: int, n_layers: int = 4, expansion: int = 4):
        super().__init__()
        self.d_model = d_model

        # Input projection (optional conditioning layer)
        self.input_norm = nn.LayerNorm(d_model)

        # Stack of residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(d_model, expansion) for _ in range(n_layers)]
        )

        # Output projection + normalization
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z_0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_0: [B, d] premise vector

        Returns:
            z_hat_final: [B, d] predicted goal state
        """
        x = self.input_norm(z_0)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)
