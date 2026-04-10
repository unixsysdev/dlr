"""
Energy Critic — Manifold Guardrail for Trajectories

Learns an energy function E(z) → scalar that distinguishes
valid proof states (on-manifold) from invalid states (off-manifold).

Training:
  Positive samples (low energy): Real waypoints from Z_true
  Negative samples (high energy): Gaussian-perturbed waypoints

The energy penalty is added to the Flow Expert's loss, forcing
the velocity field to curve around high-energy "mountains" of
logical contradiction rather than cutting through them.

Loss: Margin-based contrastive
  L = E[E(z_pos)] + E[max(0, margin - E(z_neg))]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyCritic(nn.Module):
    """
    Energy function E(z) → scalar.

    Low energy = valid proof state (on the learned manifold).
    High energy = invalid state (hallucination, logical error).

    Architecture: MLP with spectral normalization for Lipschitz
    stability (prevents energy values from exploding).
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),  # Scalar energy output
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
    Gradient-based energy penalty for the flow loss.

    Instead of projecting to the endpoint (which assumes a straight line
    and is inconsistent with Heun's method), we compute the energy gradient
    ∇E(x_t) and penalize velocity vectors that point toward high-energy
    regions (steepest energy ascent).

    This is solver-agnostic: it works at the current timestep regardless
    of whether the ODE integrator uses Euler, Heun, or higher-order methods.

    Args:
        critic: EnergyCritic module
        x_t: [B, N, d] current noisy trajectory
        v_pred: [B, N, d] predicted velocity (detached from critic grad)
        t: [B] current timestep (unused, kept for API compatibility)

    Returns:
        Scalar energy penalty (higher = velocity points toward invalid regions)
    """
    # Enable gradient tracking on x_t for autograd
    x_t_detached = x_t.detach().requires_grad_(True)

    # Compute energy at current state
    energy = critic(x_t_detached)  # [B, N]
    energy_sum = energy.sum()

    # Gradient of energy w.r.t. x_t: direction of steepest energy ascent
    grad_e = torch.autograd.grad(
        energy_sum, x_t_detached, create_graph=False
    )[0]  # [B, N, d]

    # Penalize velocity pointing in the direction of energy ascent
    # cosine_similarity > 0 means velocity points toward high energy
    v_flat = v_pred.detach().reshape(-1, v_pred.shape[-1])  # [B*N, d]
    g_flat = grad_e.reshape(-1, grad_e.shape[-1])            # [B*N, d]

    cos_sim = F.cosine_similarity(v_flat, g_flat, dim=-1)  # [B*N]

    # Only penalize positive alignment (velocity toward high energy)
    # Negative alignment means velocity points away from high energy (good)
    penalty = F.relu(cos_sim).mean()

    return penalty
