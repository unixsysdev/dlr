"""
Phase 1: Train the Text-JEPA + Oracle (VICReg)

Trains the JEPA to predict the next reasoning step's latent
representation, using VICReg loss for structural anti-collapse.
Simultaneously trains the Oracle to predict z_final from z_0.

Loss:
  L_total = L_vicreg(z_pred, z_target) + λ_oracle · L_oracle(ẑ_goal, z_goal)

Monitoring:
  - VICReg sub-losses: invariance, variance, covariance
  - Oracle loss (endpoint prediction quality)
  - z_var (legacy collapse monitor, now backed by VICReg guarantee)
  - EMA tau schedule
"""

import os
import json
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DLRConfig
from checkpointing import save_model_checkpoint
from modules.text_jepa import TextJEPA
from modules.vicreg import vicreg_loss
from data_pipeline import (
    load_dataset_split,
    parse_all_problems,
    prepare_tokenizer,
    JEPADataset,
)


def train_jepa(config: DLRConfig) -> dict:
    """
    Train the Text-JEPA + Oracle and return training history.

    Returns:
        dict with model, tokenizer, parsed_problems, and history
    """
    print("=" * 60)
    print("PHASE 1: Training Text-JEPA + Oracle (VICReg)")
    print("=" * 60)

    # ── Setup ───────────────────────────────────────────────────
    config.validate()
    torch.manual_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    config.apply_compute_optimizations()

    # Tokenizer
    print("\n[1/4] Preparing tokenizer...")
    tokenizer = prepare_tokenizer()
    vocab_size = len(tokenizer)
    print(f"  Vocab size (with special tokens): {vocab_size}")

    # Dataset
    print("\n[2/4] Loading and parsing dataset...")
    train_raw, test_raw = load_dataset_split(config.dataset_name, config.n_samples)
    parsed = parse_all_problems(train_raw, config.min_steps, config.max_steps)
    parsed_test = parse_all_problems(test_raw, config.min_steps, config.max_steps)
    print(f"  Parsed {len(parsed)} train / {len(parsed_test)} test problems")

    if len(parsed) == 0:
        raise RuntimeError("No valid problems found! Check min_steps/max_steps.")

    jepa_dataset = JEPADataset(parsed, tokenizer, config.max_seq_len)
    dataloader = DataLoader(
        jepa_dataset,
        batch_size=config.jepa_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        drop_last=False,
    )

    # Model
    print("\n[3/4] Building Text-JEPA + Oracle...")
    model = TextJEPA(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.encoder_layers,
        predictor_hidden=config.predictor_hidden,
        dropout=config.dropout,
        ff_mult=config.ff_mult,
        max_len=config.max_seq_len,
        oracle_layers=config.oracle_layers,
        oracle_expansion=config.oracle_expansion,
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    oracle_params = sum(p.numel() for p in model.oracle.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Oracle params: {oracle_params:,}")
    print(f"  Target encoder params (EMA, frozen): {total_params - trainable_params:,}")

    # Optimizer (context encoder + predictor + oracle)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=config.jepa_lr,
        weight_decay=config.jepa_weight_decay,
    )

    # ── Compute: torch.compile ──────────────────────────────────
    raw_model = model
    model = config.maybe_compile(model)

    total_steps = len(dataloader) * config.jepa_epochs

    # ── Training Loop ───────────────────────────────────────────
    print(f"\n[4/4] Training for {config.jepa_epochs} epochs "
          f"({total_steps} steps)...")
    print(f"  EMA schedule: {config.ema_start} → {config.ema_end} (cosine)")
    print(f"  VICReg: λ_inv={config.vicreg_lambda_inv}, "
          f"λ_var={config.vicreg_lambda_var}, "
          f"λ_cov={config.vicreg_lambda_cov}, "
          f"γ={config.vicreg_gamma}")
    print(f"  Oracle weight: {config.oracle_loss_weight}")

    history = {
        "loss": [], "z_var": [], "ema_tau": [], "epoch_loss": [],
        "inv_loss": [], "var_loss": [], "cov_loss": [],
        "oracle_loss": [],
    }
    global_step = 0
    t_start = time.time()

    for epoch in range(config.jepa_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_z_var = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{config.jepa_epochs}")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(config.device) for k, v in batch.items()}

            # Forward (BF16 autocast for speed)
            with config.autocast_ctx:
                result = model(
                    batch["context_ids"],
                    batch["context_mask"],
                    batch["target_ids"],
                    batch["target_mask"],
                    # Premise-only tokens: Oracle sees ONLY [PREMISE]...[/PREMISE]
                    # NO intermediate steps — matches inference behavior
                    premise_ids=batch["premise_ids"],
                    premise_mask=batch["premise_mask"],
                    # Goal-state tokens: the Oracle target now matches the
                    # exact full-proof endpoint used by trajectory extraction.
                    goal_ids=batch["goal_ids"],
                    goal_mask=batch["goal_mask"],
                )

                z_predicted = result["z_predicted"]
                z_target = result["z_target"]

                # VICReg loss (replaces simple MSE)
                vic_loss, vic_dict = vicreg_loss(
                    z_predicted,
                    z_target,
                    lambda_inv=config.vicreg_lambda_inv,
                    lambda_var=config.vicreg_lambda_var,
                    lambda_cov=config.vicreg_lambda_cov,
                    gamma=config.vicreg_gamma,
                )

                # Total loss: VICReg + Oracle
                oracle_loss = result["oracle_loss"]
                total_loss = vic_loss + config.oracle_loss_weight * oracle_loss

            z_var = result["z_var"]

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                raw_model.trainable_parameters(), config.grad_clip
            )
            optimizer.step()

            # EMA update (always FP32, on raw uncompiled model)
            tau = config.ema_schedule(global_step, total_steps)
            raw_model.ema_update(tau)

            # Logging
            history["loss"].append(total_loss.item())
            history["z_var"].append(z_var)
            history["ema_tau"].append(tau)
            history["inv_loss"].append(vic_dict["inv"])
            history["var_loss"].append(vic_dict["var"])
            history["cov_loss"].append(vic_dict["cov"])
            history["oracle_loss"].append(oracle_loss.item())

            epoch_loss += total_loss.item()
            epoch_z_var += z_var
            n_batches += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                inv=f"{vic_dict['inv']:.4f}",
                var=f"{vic_dict['var']:.3f}",
                ora=f"{oracle_loss.item():.4f}",
                z_v=f"{z_var:.4f}",
            )

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_z_var = epoch_z_var / max(n_batches, 1)
        history["epoch_loss"].append(avg_loss)

        elapsed = time.time() - t_start
        print(
            f"  Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
            f"z_var: {avg_z_var:.4f} | tau: {tau:.6f} | "
            f"Time: {elapsed:.0f}s"
        )

    # Save (always save raw uncompiled model)
    checkpoint_path = os.path.join(config.checkpoint_dir, "jepa_final.pt")
    save_model_checkpoint(
        checkpoint_path,
        raw_model.state_dict(),
        config,
        vocab_size=vocab_size,
    )
    print(f"\n  ✓ JEPA + Oracle saved to {checkpoint_path}")

    # Save tokenizer
    tokenizer_path = os.path.join(config.checkpoint_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"  ✓ Tokenizer saved to {tokenizer_path}")

    # Save history
    history_path = os.path.join(config.data_dir, "jepa_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    total_time = time.time() - t_start
    print(f"\n  Phase 1 complete in {total_time/60:.1f} minutes")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "parsed_problems": parsed,
        "parsed_problems_test": parsed_test,
        "history": history,
    }


if __name__ == "__main__":
    config = DLRConfig()
    train_jepa(config)
