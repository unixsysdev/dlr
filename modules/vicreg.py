"""
VICReg Loss — Variance-Invariance-Covariance Regularization

Replaces the simple MSE band-aid with a structural guarantee that
the latent space remains expanded, informative, and decorrelated.

Three components:
  Invariance: MSE between predicted and target embeddings (the sink)
  Variance:   Hinge penalty on per-dimension std (anti-collapse)
  Covariance: Off-diagonal penalty on the covariance matrix (decorrelation)

Reference: Bardes, Ponce & LeCun (2022) — VICReg
"""

import torch
import torch.nn.functional as F


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    Force per-dimension std to be at least gamma.

    v(Z) = (1/d) Σ_j max(0, γ - √(Var(z^j) + ε))

    Prevents ALL dimensions from collapsing, not just the average.

    Args:
        z: [B, d] batch of embeddings
        gamma: minimum target std per dimension
        eps: numerical stability

    Returns:
        Scalar variance loss
    """
    std = torch.sqrt(z.var(dim=0) + eps)  # [d]
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Penalize off-diagonal elements of the covariance matrix.

    c(Z) = (1/d) Σ_{i≠j} C(Z)_{i,j}²

    Forces each latent dimension to encode *different* information.
    Without this, the model can "cheat" by encoding the same signal
    across multiple dimensions.

    Args:
        z: [B, d] batch of embeddings

    Returns:
        Scalar covariance loss
    """
    B, d = z.shape
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / max(B - 1, 1)  # [d, d]

    # Zero out diagonal (we only penalize off-diagonal)
    cov.fill_diagonal_(0.0)

    # Sum of squared off-diagonal elements, normalized by d
    return (cov ** 2).sum() / d


def vicreg_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    lambda_inv: float = 25.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 1.0,
    gamma: float = 1.0,
):
    """
    Full VICReg loss.

    Args:
        z_pred: [B, d] predicted embeddings (from predictor)
        z_target: [B, d] target embeddings (from EMA encoder, detached)
        lambda_inv: weight for invariance (MSE) term
        lambda_var: weight for variance (anti-collapse) term
        lambda_cov: weight for covariance (decorrelation) term
        gamma: minimum target std per dimension

    Returns:
        total_loss: weighted sum of all three terms
        loss_dict: dict with individual losses for logging
    """
    # Invariance: pull predictions toward targets
    inv_loss = F.mse_loss(z_pred, z_target)

    # Variance: force spread in both predicted and target embeddings
    var_loss = variance_loss(z_pred, gamma) + variance_loss(z_target, gamma)

    # Covariance: decorrelate dimensions in both
    cov_loss = covariance_loss(z_pred) + covariance_loss(z_target)

    total = lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov_loss

    return total, {
        "inv": inv_loss.item(),
        "var": var_loss.item(),
        "cov": cov_loss.item(),
        "total": total.item(),
    }
