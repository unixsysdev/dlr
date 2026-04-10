"""
Energy Critic — Manifold Guardrail for Trajectories

Learns an energy function E(z) → scalar that distinguishes
in-manifold latent states from perturbed off-manifold states.

Architecture: MLP with spectral normalization for Lipschitz stability.
Spectral norm constrains the largest singular value of each weight
matrix to 1, preventing energy values from exploding and ensuring
smooth energy landscapes.

Training:
  Positive samples (low energy): Real waypoints from Z_true
  Negative samples (high energy): Gaussian-perturbed waypoints

The energy penalty is added to the Flow Expert's loss, forcing
the velocity field to curve around high-energy "mountains" of
latent drift rather than cutting through them.

Loss: Margin-based contrastive
  L = E[E(z_pos)] + E[max(0, margin - E(z_neg))]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyCritic(nn.Module):
    """
    Energy function E(z) → scalar.

    Low energy = in-manifold proof state.
    High energy = perturbed or off-manifold latent state.

    Architecture: MLP with spectral normalization on all linear layers
    for Lipschitz stability (prevents energy explosion).
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(d_model, hidden_dim)),
            nn.GELU(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, d] or [B, N, d] latent state(s)

        Returns:
            energy: [B] or [B, N] scalar energy values
        """
        shape = z.shape
        if z.dim() == 3:
            B, N, d = shape
            z_flat = z.reshape(B * N, d)
            energy = self.net(z_flat).squeeze(-1)  # [B*N]
            return energy.reshape(B, N)
        return self.net(z).squeeze(-1)  # [B]


def energy_contrastive_loss(
    critic: nn.Module,
    z_positive: torch.Tensor,
    noise_std: float = 0.5,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Train the Energy Critic with contrastive pairs.

    Args:
        critic: EnergyCritic module
        z_positive: [B, N, d] real trajectory waypoints (low energy target)
        noise_std: std of Gaussian perturbation for negatives
        margin: target energy gap between positives and negatives

    Returns:
        Scalar contrastive loss
    """
    # Positive: real waypoints should have LOW energy
    e_pos = critic(z_positive)  # [B, N]
    loss_pos = e_pos.mean()

    # Negative: perturbed waypoints should have HIGH energy (above margin)
    z_negative = z_positive + noise_std * torch.randn_like(z_positive)
    e_neg = critic(z_negative)  # [B, N]
    loss_neg = F.relu(margin - e_neg).mean()

    return loss_pos + loss_neg


def flow_energy_penalty(
    critic: nn.Module,
    x_t: torch.Tensor,
    v_pred: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Energy penalty that backpropagates into the Flow Expert.

    Evaluates the energy of the current predicted state and allows
    gradients to flow through v_pred into the Flow Expert's weights.

    The penalty is simply E(x_t + v_pred * dt) where dt is a small
    lookahead — the energy at a short extrapolation of the current
    velocity. Gradients flow through v_pred because we do NOT detach it.

    This replaces the previous gradient-ascent cosine approach which
    detached v_pred and contributed zero gradients to the Flow Expert.

    Args:
        critic: EnergyCritic module (weights frozen via optimizer separation)
        x_t: [B, N, d] current noisy trajectory (detached from flow graph)
        v_pred: [B, N, d] predicted velocity — MUST have grad enabled
        t: [B] current timestep

    Returns:
        Scalar energy penalty with gradients flowing through v_pred
    """
    # Short lookahead: where would the velocity take us in one small step?
    # Using a small fixed dt (not remaining_time) to avoid Euler assumption
    dt = 0.05  # Small fixed lookahead step
    x_ahead = x_t.detach() + v_pred * dt  # [B, N, d] — gradients flow through v_pred

    # Energy at the lookahead point — critic weights are frozen by
    # optimizer separation, but v_pred's graph is preserved
    energy = critic(x_ahead)  # [B, N]

    return energy.mean()
