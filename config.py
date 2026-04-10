"""
DLR — Centralized Configuration

Supports two profiles:
  - PoC:        d=128, 1K problems, ~7 hours on RTX 4090
  - Production: d=1024, 100K problems, ~14-18h on H200/B200
"""

import math
import torch
from dataclasses import dataclass


@dataclass
class DLRConfig:
    # ── Architecture ────────────────────────────────────────────────
    d_model: int = 128           # Latent dimension
    n_heads: int = 4             # Attention heads
    encoder_layers: int = 3      # JEPA encoder depth
    predictor_hidden: int = 64   # Predictor bottleneck dim
    flow_layers: int = 4         # Flow Expert (DiT) depth
    decoder_layers: int = 2      # Scribe decoder depth
    n_waypoints: int = 16        # Max trajectory waypoints N
    ff_mult: int = 4             # FFN expansion factor
    dropout: float = 0.1

    # ── Data ────────────────────────────────────────────────────────
    dataset_name: str = "AI-MO/NuminaMath-CoT"
    n_samples: int = 1000        # Problems to use
    max_seq_len: int = 256       # Tokenizer max length
    min_steps: int = 2           # Min proof steps to include
    max_steps: int = 20          # Max proof steps to include

    # ── Phase 1: JEPA Training ──────────────────────────────────────
    jepa_epochs: int = 15
    jepa_batch_size: int = 64
    jepa_lr: float = 3e-4
    jepa_weight_decay: float = 0.05
    ema_start: float = 0.996     # EMA momentum start
    ema_end: float = 1.0         # EMA momentum end (frozen)
    collapse_threshold: float = 1e-4  # z_var below this = collapse

    # ── Phase 2: Flow Expert Training ───────────────────────────────
    flow_epochs: int = 80
    flow_batch_size: int = 32
    flow_lr: float = 1e-4
    flow_weight_decay: float = 0.01

    # ── Phase 3: Decoder Training ───────────────────────────────────
    decoder_epochs: int = 25
    decoder_batch_size: int = 32
    decoder_lr: float = 3e-4
    decoder_weight_decay: float = 0.01
    decoder_max_seq_len: int = 256  # Decoder output max tokens
    decoder_window_half: int = 2    # Sliding window half-width
    use_gt_trajectories: bool = True  # False = generated trajectories only
    decoder_generated_mix_start: float = 0.0  # Fraction of generated trajs at epoch 0
    decoder_generated_mix_end: float = 0.5    # Fraction of generated trajs at final epoch

    # ── VICReg (Anti-Collapse) ─────────────────────────────────────
    vicreg_lambda_inv: float = 25.0   # Invariance (MSE) weight
    vicreg_lambda_var: float = 25.0   # Variance (anti-collapse) weight
    vicreg_lambda_cov: float = 1.0    # Covariance (decorrelation) weight
    vicreg_gamma: float = 1.0         # Variance hinge threshold (min std)

    # ── Oracle (Goal State Predictor) ─────────────────────────────
    oracle_layers: int = 4            # Depth of Oracle residual MLP
    oracle_expansion: int = 4         # FFN expansion factor
    oracle_loss_weight: float = 1.0   # Weight of Oracle loss in total
    oracle_exposure_rate: float = 0.2  # Mix Oracle goals into flow training

    # ── Energy Critic (Manifold Guardrail) ────────────────────────
    energy_hidden: int = 256          # Critic hidden dim
    energy_margin: float = 1.0        # Contrastive margin
    energy_noise_std: float = 0.5     # Std for negative samples
    energy_penalty_weight: float = 0.1  # α in flow loss
    energy_lr: float = 1e-4           # Critic learning rate
    stop_loss_weight: float = 0.2     # Weight for trajectory stop prediction

    # ── ODE Solver ──────────────────────────────────────────────────
    ode_steps: int = 50          # Euler integration steps
    ode_solver: str = "heun"     # 'euler' (fast) or 'heun' (accurate)
    eval_temperature: float = 0.0  # 0.0 = strict greedy decoding for eval

    # ── Compute Optimization ────────────────────────────────────────
    use_compile: bool = True     # torch.compile (graph compilation)
    use_bf16: bool = True        # BF16 mixed precision (H100/H200/B200)
    use_tf32: bool = True        # TF32 matmul precision (Ampere+)
    use_liger: bool = True       # Liger fused kernels (fused CE for decoder)
    num_workers: int = 4         # DataLoader workers
    pin_memory: bool = True      # Pin memory for GPU transfer
    persistent_workers: bool = True  # Keep workers alive between epochs

    # ── General ─────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    plot_dir: str = "plots"
    grad_clip: float = 1.0       # Gradient clipping norm

    def ema_schedule(self, step: int, total_steps: int) -> float:
        """Cosine EMA momentum schedule: 0.996 → 1.0"""
        progress = step / max(total_steps, 1)
        return self.ema_end - (self.ema_end - self.ema_start) * (
            math.cos(math.pi * progress) + 1
        ) / 2

    def apply_compute_optimizations(self):
        """Apply global compute settings. Call once at startup."""
        if self.use_tf32 and self.device == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  ⚡ TF32 matmul enabled")

    def maybe_compile(self, model):
        """Wrap model in torch.compile if enabled."""
        if self.use_compile and self.device == "cuda":
            print(f"  ⚡ Compiling {model.__class__.__name__}...")
            return torch.compile(model)
        return model

    @property
    def autocast_ctx(self):
        """Return the appropriate autocast context manager."""
        if self.use_bf16 and self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        # No-op context manager for CPU or FP32
        return torch.autocast(self.device, enabled=False)


def production_config() -> DLRConfig:
    """
    H200/B200 production config.
    Targets ~14-18 hours on a single H200 (141GB HBM3e).

    d=1024, 100K problems, 12-layer DiT, 4-layer decoder.
    """
    return DLRConfig(
        # Architecture — full scale
        d_model=1024,
        n_heads=16,
        encoder_layers=8,
        predictor_hidden=512,
        flow_layers=12,
        decoder_layers=4,
        n_waypoints=32,
        ff_mult=4,
        dropout=0.05,

        # Data — 100K problems
        n_samples=100_000,
        max_seq_len=1024,
        min_steps=2,
        max_steps=30,

        # Phase 1: JEPA
        jepa_epochs=8,
        jepa_batch_size=256,
        jepa_lr=2e-4,
        jepa_weight_decay=0.05,
        ema_start=0.996,

        # Phase 2: Flow Expert
        flow_epochs=40,
        flow_batch_size=128,
        flow_lr=5e-5,
        flow_weight_decay=0.01,

        # Phase 3: Decoder
        decoder_epochs=15,
        decoder_batch_size=128,
        decoder_lr=2e-4,
        decoder_weight_decay=0.01,
        decoder_max_seq_len=1024,
        decoder_window_half=3,
        decoder_generated_mix_start=0.1,
        decoder_generated_mix_end=0.75,

        # VICReg
        vicreg_lambda_inv=25.0,
        vicreg_lambda_var=25.0,
        vicreg_lambda_cov=1.0,
        vicreg_gamma=1.0,

        # Oracle
        oracle_layers=6,
        oracle_expansion=4,
        oracle_loss_weight=1.0,
        oracle_exposure_rate=0.2,

        # Energy Critic
        energy_hidden=512,
        energy_margin=1.0,
        energy_noise_std=0.5,
        energy_penalty_weight=0.1,
        energy_lr=5e-5,
        stop_loss_weight=0.2,

        # ODE
        ode_steps=100,
        ode_solver="heun",
        eval_temperature=0.0,

        # Compute — all optimizations ON
        use_compile=True,
        use_bf16=True,
        use_tf32=True,
        use_liger=True,
        num_workers=8,
        pin_memory=True,

        # General
        grad_clip=1.0,
    )
