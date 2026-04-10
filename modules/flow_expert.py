"""
Module B: Flow Expert — The Latent Logic Engine

A Rectified Flow model (DiT architecture) that generates continuous
reasoning trajectories from noise → Z_true.

V3 Architecture (Unified Trajectory — No Double Dip):
  - AdaLN conditioning: z_0 (premise) + z_target (goal) + timestep
  - Self-attention ONLY on waypoints (no cross-attention to prompt_kv)
  - The premise is already baked into z_0 via cumulative encoding
  - Velocity Zero-Out: padded waypoints have zero velocity (masked loss)
"""

import math
import torch
import torch.nn as nn


class DiTBlock(nn.Module):
    """
    Transformer block with Adaptive Layer Normalization (AdaLN).

    V3: Self-attention + FFN only. No cross-attention.
    The premise context is injected via AdaLN conditioning (z_0),
    not via a separate KV cache. This eliminates the "double dip"
    and compresses the entire memory footprint into the conditioning vector.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN: condition → 4 params (gamma, beta for 2 sublayers)
        self.adaln_proj = nn.Linear(d_model, 4 * d_model)
        # Zero-init so AdaLN starts as identity
        nn.init.zeros_(self.adaln_proj.weight)
        nn.init.zeros_(self.adaln_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d] input sequence (noisy trajectory)
            cond: [B, d] conditioning vector (z_0 + z_target + t_embed)
        """
        # Compute AdaLN parameters
        params = self.adaln_proj(cond)  # [B, 4*d]
        g1, b1, g2, b2 = params.chunk(4, dim=-1)
        # Unsqueeze for sequence dim: [B, 1, d]
        g1, b1 = g1.unsqueeze(1), b1.unsqueeze(1)
        g2, b2 = g2.unsqueeze(1), b2.unsqueeze(1)

        # Self-attention with AdaLN modulation
        h = (1 + g1) * self.norm1(x) + b1
        x = x + self.self_attn(h, h, h)[0]

        # Feed-forward with AdaLN modulation
        h = (1 + g2) * self.norm2(x) + b2
        x = x + self.ffn(h)

        return x


class FlowExpert(nn.Module):
    """
    Rectified Flow model for trajectory generation.

    V3 Unified Trajectory Architecture:
    The Flow Expert does NOT query a text KV cache. It takes:
      - z_0: The compressed premise vector (boundary condition: start)
      - z_target: The compressed goal vector (boundary condition: end)
      - t: The current timestep
    All three are fused via AdaLN into a single conditioning signal.
    The model uses self-attention on the N waypoints to draw
    straight lines between them.

    This eliminates the "double dip" — the entire memory of the
    problem is compressed into the conditioning vector, not a
    separate KV cache.

    Rectified Flow formulation:
        x_t = t · Z_true + (1-t) · ε,  where ε ~ N(0, I)
        v_true = Z_true - ε
        L_flow = E[||v_θ(x_t, t, z_0, z_target) - v_true||²]
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_waypoints: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_waypoints = n_waypoints

        # Learned positional encoding for waypoints
        self.waypoint_pos = nn.Parameter(
            torch.randn(1, n_waypoints, d_model) * 0.02
        )

        # Timestep embedding: scalar t → d-dim vector
        self.t_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Combine z_0 + z_target + t_embed → conditioning vector
        # Three d-dim vectors → fused conditioning
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # DiT blocks (self-attention only, no cross-attention)
        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )

        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding: [B] → [B, d]"""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device).float()
            / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, d//2]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, d]

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_0: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: [B, N, d] noisy trajectory at time t
            t: [B] timestep in [0, 1]
            z_0: [B, d] premise vector (start boundary condition)
            z_target: [B, d] target goal vector (end boundary condition)

        Returns:
            v_theta: [B, N, d] predicted velocity field
        """
        B, N, d = x_t.shape

        # Add waypoint positional encoding
        x = x_t + self.waypoint_pos[:, :N]

        # Build conditioning: z_0 + z_target + timestep
        t_emb = self.t_embed(self._sinusoidal_embedding(t))  # [B, d]
        cond = self.cond_proj(
            torch.cat([z_0, z_target, t_emb], dim=-1)
        )  # [B, d]

        # Process through DiT blocks (self-attention only)
        for block in self.blocks:
            x = block(x, cond)

        # Output velocity
        v = self.out_proj(self.final_norm(x))
        return v

    @torch.no_grad()
    def generate(
        self,
        z_0: torch.Tensor,
        z_target: torch.Tensor,
        n_steps: int = 50,
        solver: str = "euler",
    ) -> torch.Tensor:
        """
        Generate a trajectory from noise using ODE integration.

        Args:
            z_0: [B, d] premise vector (start boundary)
            z_target: [B, d] goal vector (end boundary)
            n_steps: number of integration steps
            solver: 'euler' (1st order) or 'heun' (2nd order, corrects curvature)

        Returns:
            trajectory: [B, N, d] generated trajectory
        """
        B = z_target.shape[0]
        device = z_target.device

        # Start from Gaussian noise
        x = torch.randn(B, self.n_waypoints, self.d_model, device=device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_i = i * dt
            t_cur = torch.full((B,), t_i, device=device)
            v = self.forward(x, t_cur, z_0, z_target)

            if solver == "heun" and i < n_steps - 1:
                # Heun's method: predict-correct (2nd order Runge-Kutta)
                # 1. Euler predict
                x_euler = x + v * dt
                # 2. Evaluate velocity at predicted point
                t_next = torch.full((B,), t_i + dt, device=device)
                v_next = self.forward(x_euler, t_next, z_0, z_target)
                # 3. Average the two velocities (trapezoidal rule)
                x = x + (v + v_next) * 0.5 * dt
            else:
                # Standard Euler step
                x = x + v * dt

        return x


def masked_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    MSE loss with Velocity Zero-Out masking.

    Only computes loss on active waypoints (mask=1).
    Padded waypoints (copies of z_final) have zero velocity,
    and the loss ignores them so the flow learns to "park" there.

    Args:
        pred: [B, N, d] predicted velocity
        target: [B, N, d] true velocity (Z_true - noise)
        mask: [B, N] binary mask, 1=active, 0=padded

    Returns:
        Scalar masked MSE loss
    """
    diff = (pred - target) ** 2  # [B, N, d]
    diff = diff.mean(dim=-1)     # [B, N] — per-waypoint MSE
    diff = diff * mask           # Zero out padded waypoints
    return diff.sum() / mask.sum().clamp(min=1)
