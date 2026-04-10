"""
Phase 2: Train the Flow Expert + Energy Critic

Trains the DiT-based Flow Expert to generate continuous
reasoning trajectories from noise → Z_true, with an Energy
Critic that penalizes off-manifold trajectories.

V5 Architecture:
  - Flow Expert uses scheduled Oracle replacement during training:
    most batches use ground-truth z_target, but a fraction use ẑ_final
    so the model learns to handle Oracle error at inference.
  - Energy Critic trains on contrastive pairs: real waypoints (low energy)
    vs. perturbed waypoints (high energy).
  - Energy penalty added to flow loss: α · E(predicted_endpoint)

Rectified Flow:
  x_t = t · Z_true + (1-t) · ε,  ε ~ N(0,I)
  v_true = Z_true - ε
  L_flow = ||v_θ(x_t, t, z_0, z_target) - v_true||² + α · E(endpoint)
"""

import os
import json
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DLRConfig
from modules.flow_expert import FlowExpert, masked_mse_loss
from modules.energy_critic import EnergyCritic, energy_contrastive_loss, flow_energy_penalty
from modules.text_jepa import TextJEPA
from data_pipeline import TrajectoryDataset


def train_flow(config: DLRConfig, jepa_model: TextJEPA = None) -> dict:
    """
    Train the Flow Expert + Energy Critic and return training history.

    Returns:
        dict with flow model, energy critic, and history
    """
    print("=" * 60)
    print("PHASE 2: Training Flow Expert + Energy Critic")
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

    # ── Models ──────────────────────────────────────────────────
    print("\n[2/4] Building Flow Expert + Energy Critic...")

    flow_model = FlowExpert(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.flow_layers,
        n_waypoints=config.n_waypoints,
        ff_mult=config.ff_mult,
        dropout=config.dropout,
    ).to(device)

    energy_critic = EnergyCritic(
        d_model=config.d_model,
        hidden_dim=config.energy_hidden,
    ).to(device)

    flow_params = sum(p.numel() for p in flow_model.parameters())
    critic_params = sum(p.numel() for p in energy_critic.parameters())
    print(f"  Flow Expert params: {flow_params:,}")
    print(f"  Energy Critic params: {critic_params:,}")

    if jepa_model is None:
        print("\n[3/4] Loading frozen JEPA Oracle...")
        jepa_ckpt = torch.load(
            os.path.join(config.checkpoint_dir, "jepa_final.pt"),
            map_location=device,
            weights_only=False,
        )
        jepa_model = TextJEPA(
            vocab_size=jepa_ckpt["vocab_size"],
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.encoder_layers,
            predictor_hidden=config.predictor_hidden,
            dropout=0.0,
            ff_mult=config.ff_mult,
            max_len=config.max_seq_len,
            oracle_layers=config.oracle_layers,
            oracle_expansion=config.oracle_expansion,
        ).to(device)
        jepa_model.load_state_dict(jepa_ckpt["model_state_dict"])
    else:
        print("\n[3/4] Using provided JEPA Oracle...")

    jepa_model.eval()
    for p in jepa_model.parameters():
        p.requires_grad = False

    # Separate optimizers (alternating updates)
    flow_optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=config.flow_lr,
        weight_decay=config.flow_weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        energy_critic.parameters(),
        lr=config.energy_lr,
    )

    # ── Compute: torch.compile ──────────────────────────────────
    raw_flow = flow_model
    raw_critic = energy_critic
    flow_model = config.maybe_compile(flow_model)
    # Don't compile the critic — it's small and alternates training

    # ── Training Loop ───────────────────────────────────────────
    print(f"\n[4/4] Training for {config.flow_epochs} epochs...")
    print(f"  Energy penalty weight α = {config.energy_penalty_weight}")
    print(f"  Noise std for negatives: {config.energy_noise_std}")
    print(f"  Oracle exposure rate = {config.oracle_exposure_rate:.0%}")

    history = {
        "loss": [], "epoch_loss": [],
        "flow_loss": [], "energy_penalty": [],
        "critic_loss": [],
        "oracle_exposure": [],
    }
    t_start = time.time()

    for epoch in range(config.flow_epochs):
        flow_model.train()
        energy_critic.train()
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

            # Scheduled Oracle replacement:
            # expose the flow model to predicted goals so evaluation-time
            # Oracle error is no longer out-of-distribution.
            use_oracle = torch.rand(B, device=device) < config.oracle_exposure_rate
            if use_oracle.any():
                with torch.no_grad():
                    z_hat = jepa_model.predict_goal(z_0[use_oracle])
                z_cond = z_target.clone()
                z_cond[use_oracle] = z_hat
            else:
                z_cond = z_target

            # ══════════════════════════════════════════════════
            # Step 1: Update Energy Critic (contrastive pairs)
            # ══════════════════════════════════════════════════
            critic_loss = energy_contrastive_loss(
                energy_critic, Z_true,
                noise_std=config.energy_noise_std,
                margin=config.energy_margin,
            )
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # ══════════════════════════════════════════════════
            # Step 2: Update Flow Expert (velocity + energy penalty)
            # ══════════════════════════════════════════════════

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
                v_pred = flow_model(x_t, t, z_0, z_cond)  # [B, N, d]

                # Masked MSE loss (Velocity Zero-Out)
                flow_mse = masked_mse_loss(v_pred, v_true, active_mask)

                # Energy penalty: penalize high-energy predicted endpoints
                e_penalty = flow_energy_penalty(
                    raw_critic, x_t, v_pred, t
                )

                loss = flow_mse + config.energy_penalty_weight * e_penalty

            # Backward
            flow_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_flow.parameters(), config.grad_clip)
            flow_optimizer.step()

            # Logging
            history["loss"].append(loss.item())
            history["flow_loss"].append(flow_mse.item())
            history["energy_penalty"].append(e_penalty.item())
            history["critic_loss"].append(critic_loss.item())
            history["oracle_exposure"].append(use_oracle.float().mean().item())
            epoch_loss += loss.item()
            n_batches += 1

            pbar.set_postfix(
                flow=f"{flow_mse.item():.4f}",
                e_pen=f"{e_penalty.item():.3f}",
                crit=f"{critic_loss.item():.3f}",
                ora=f"{use_oracle.float().mean().item():.2f}",
            )

        avg_loss = epoch_loss / max(n_batches, 1)
        history["epoch_loss"].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch+1}/{config.flow_epochs} | "
                f"Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s"
            )

    # Save (raw uncompiled models)
    flow_path = os.path.join(config.checkpoint_dir, "flow_final.pt")
    torch.save(
        {
            "model_state_dict": raw_flow.state_dict(),
            "config": {
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "flow_layers": config.flow_layers,
                "n_waypoints": config.n_waypoints,
            },
        },
        flow_path,
    )
    print(f"\n  ✓ Flow Expert saved to {flow_path}")

    critic_path = os.path.join(config.checkpoint_dir, "energy_critic_final.pt")
    torch.save(
        {"model_state_dict": raw_critic.state_dict()},
        critic_path,
    )
    print(f"  ✓ Energy Critic saved to {critic_path}")

    # Save history
    history_path = os.path.join(config.data_dir, "flow_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    total_time = time.time() - t_start
    print(f"  Phase 2 complete in {total_time/60:.1f} minutes")

    return {
        "flow_model": flow_model,
        "energy_critic": energy_critic,
        "history": history,
    }


if __name__ == "__main__":
    config = DLRConfig()
    train_flow(config)
