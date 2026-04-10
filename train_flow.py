"""
Phase 2: Train the Flow Expert (Rectified Flow)

Trains the DiT-based Flow Expert to generate continuous
reasoning trajectories from noise → Z_true.

The JEPA is frozen. GPUs never run the JEPA during this phase
because trajectories are pre-extracted.

Rectified Flow:
  x_t = t · Z_true + (1-t) · ε,  ε ~ N(0,I)
  v_true = Z_true - ε
  L_flow = ||v_θ(x_t, t, z_target) - v_true||² (masked for active waypoints)
"""

import os
import json
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DLRConfig
from modules.flow_expert import FlowExpert, masked_mse_loss
from data_pipeline import TrajectoryDataset


def train_flow(config: DLRConfig) -> dict:
    """
    Train the Flow Expert and return training history.

    Returns:
        dict with model and history
    """
    print("=" * 60)
    print("PHASE 2: Training Flow Expert (Rectified Flow)")
    print("=" * 60)

    torch.manual_seed(config.seed)
    device = config.device
    config.apply_compute_optimizations()

    # ── Data ────────────────────────────────────────────────────
    print("\n[1/3] Loading pre-extracted trajectories...")
    trajectory_path = os.path.join(config.data_dir, "trajectories.pt")
    dataset = TrajectoryDataset(trajectory_path)
    dataloader = DataLoader(
        dataset,
        batch_size=config.flow_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        drop_last=False,
    )

    # ── Model ───────────────────────────────────────────────────
    print("\n[2/3] Building Flow Expert (DiT)...")
    model = FlowExpert(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.flow_layers,
        n_waypoints=config.n_waypoints,
        ff_mult=config.ff_mult,
        dropout=config.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.flow_lr,
        weight_decay=config.flow_weight_decay,
    )

    # ── Compute: torch.compile ──────────────────────────────────
    raw_model = model
    model = config.maybe_compile(model)

    # ── Training Loop ───────────────────────────────────────────
    print(f"\n[3/3] Training for {config.flow_epochs} epochs...")
    history = {"loss": [], "epoch_loss": []}
    t_start = time.time()

    for epoch in range(config.flow_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            dataloader,
            desc=f"  Epoch {epoch+1}/{config.flow_epochs}",
            leave=False,
        )
        for batch in pbar:
            Z_true = batch["Z_true"].to(device)            # [B, N, d]
            active_mask = batch["active_mask"].to(device)   # [B, N]
            z_target = batch["z_target"].to(device)         # [B, d]

            # V3: z_0 is the premise, already in the trajectory
            z_0 = Z_true[:, 0, :]                           # [B, d]

            B = Z_true.shape[0]

            # ── Rectified Flow ──────────────────────────────────
            # Sample timestep
            t = torch.rand(B, device=device)  # [B] ~ U(0,1)

            # Sample noise
            epsilon = torch.randn_like(Z_true)  # [B, N, d]

            # Noisy interpolation
            t_expand = t[:, None, None]  # [B, 1, 1]
            x_t = t_expand * Z_true + (1 - t_expand) * epsilon  # [B, N, d]

            # True velocity (straight-line target)
            v_true = Z_true - epsilon  # [B, N, d]

            # Forward (BF16 autocast)
            with config.autocast_ctx:
                # Predict velocity (V3: z_0 + z_target via AdaLN, no cross-attention)
                v_pred = model(x_t, t, z_0, z_target)  # [B, N, d]

                # Masked MSE loss (Velocity Zero-Out)
                loss = masked_mse_loss(v_pred, v_true, active_mask)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
            optimizer.step()

            history["loss"].append(loss.item())
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        history["epoch_loss"].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch+1}/{config.flow_epochs} | "
                f"Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s"
            )

    # Save (raw uncompiled model)
    checkpoint_path = os.path.join(config.checkpoint_dir, "flow_final.pt")
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "config": {
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "flow_layers": config.flow_layers,
                "n_waypoints": config.n_waypoints,
            },
        },
        checkpoint_path,
    )
    print(f"\n  ✓ Flow Expert saved to {checkpoint_path}")

    # Save history
    history_path = os.path.join(config.data_dir, "flow_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    total_time = time.time() - t_start
    print(f"  Phase 2 complete in {total_time/60:.1f} minutes")

    return {"model": model, "history": history}


if __name__ == "__main__":
    config = DLRConfig()
    train_flow(config)
