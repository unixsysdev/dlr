"""
Phase 3: Train the Scribe Decoder

Trains the lightweight decoder to translate continuous
N×d trajectories into discrete text tokens.

Crucial: JEPA and Flow Expert are frozen during this phase.
The decoder receives ground-truth or generated trajectories
and learns the mapping via teacher forcing (cross-entropy loss).
"""

import os
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Liger fused cross-entropy: avoids materializing the full [B*L, V] logits
# tensor for large vocabularies. Falls back to F.cross_entropy if unavailable.
try:
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False

from config import DLRConfig
from modules.decoder import ScribeDecoder
from modules.flow_expert import FlowExpert
from modules.text_jepa import TextJEPA
from data_pipeline import (
    load_dataset_split,
    parse_all_problems,
    prepare_tokenizer,
    DecoderDataset,
    format_premise,
)


def _load_frozen_generation_models(
    config: DLRConfig,
    jepa_model=None,
    flow_model=None,
):
    """Load frozen JEPA + Flow models for generated-trajectory decoder training."""
    device = config.device

    if jepa_model is None:
        print("  Loading frozen JEPA for decoder trajectory generation...")
        ckpt = torch.load(
            os.path.join(config.checkpoint_dir, "jepa_final.pt"),
            map_location=device,
            weights_only=False,
        )
        jepa_model = TextJEPA(
            vocab_size=ckpt["vocab_size"],
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
        jepa_model.load_state_dict(ckpt["model_state_dict"])

    if flow_model is None:
        print("  Loading frozen Flow Expert for decoder trajectory generation...")
        ckpt = torch.load(
            os.path.join(config.checkpoint_dir, "flow_final.pt"),
            map_location=device,
            weights_only=False,
        )
        flow_model = FlowExpert(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.flow_layers,
            n_waypoints=config.n_waypoints,
            ff_mult=config.ff_mult,
            dropout=0.0,
        ).to(device)
        incompatible = flow_model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "  ⚠ Flow checkpoint missing keys for current architecture: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    jepa_model.eval()
    flow_model.eval()
    for p in jepa_model.parameters():
        p.requires_grad = False
    for p in flow_model.parameters():
        p.requires_grad = False

    return jepa_model, flow_model


def _generated_mix_rate(config: DLRConfig, epoch: int) -> float:
    """Linear schedule from mostly clean trajectories to more generated ones."""
    if not config.use_gt_trajectories:
        return 1.0
    if config.decoder_epochs <= 1:
        return config.decoder_generated_mix_end
    progress = epoch / max(config.decoder_epochs - 1, 1)
    mix_rate = (
        config.decoder_generated_mix_start
        + progress * (config.decoder_generated_mix_end - config.decoder_generated_mix_start)
    )
    return float(min(max(mix_rate, 0.0), 1.0))


def train_decoder(
    config: DLRConfig,
    parsed_problems: list = None,
    jepa_model=None,
    flow_model=None,
) -> dict:
    """
    Train the Scribe Decoder and return training history.

    Returns:
        dict with model, tokenizer, and history
    """
    print("=" * 60)
    print("PHASE 3: Training the Scribe Decoder")
    print("=" * 60)

    torch.manual_seed(config.seed)
    device = config.device
    config.apply_compute_optimizations()

    # ── Tokenizer ───────────────────────────────────────────────
    print("\n[1/4] Loading tokenizer...")
    tokenizer_path = os.path.join(config.checkpoint_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = prepare_tokenizer()
    vocab_size = len(tokenizer)

    # ── Data ────────────────────────────────────────────────────
    print("\n[2/4] Loading dataset...")
    if parsed_problems is None:
        raw_dataset, _ = load_dataset_split(config.dataset_name, config.n_samples)
        parsed_problems = parse_all_problems(
            raw_dataset, config.min_steps, config.max_steps
        )

    trajectory_path = os.path.join(config.data_dir, "trajectories.pt")
    dataset = DecoderDataset(
        trajectory_path, parsed_problems, tokenizer, config.decoder_max_seq_len
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.decoder_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        drop_last=False,
    )

    # ── Model ───────────────────────────────────────────────────
    print("\n[3/4] Building Scribe Decoder...")
    model = ScribeDecoder(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.decoder_layers,
        n_waypoints=config.n_waypoints,
        window_half=config.decoder_window_half,
        max_seq_len=config.decoder_max_seq_len,
        ff_mult=config.ff_mult,
        dropout=config.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.decoder_lr,
        weight_decay=config.decoder_weight_decay,
    )

    pad_id = tokenizer.pad_token_id

    needs_generated_trajectories = (
        (not config.use_gt_trajectories)
        or max(config.decoder_generated_mix_start, config.decoder_generated_mix_end) > 0.0
    )
    if needs_generated_trajectories:
        jepa_model, flow_model = _load_frozen_generation_models(
            config,
            jepa_model=jepa_model,
            flow_model=flow_model,
        )

    # Determine loss function
    use_liger = config.use_liger and HAS_LIGER and config.device == "cuda"
    if use_liger:
        print("  ⚡ Liger fused cross-entropy enabled")
    else:
        if config.use_liger and not HAS_LIGER:
            print("  ⚠ Liger not installed, falling back to F.cross_entropy")

    # ── Compute: torch.compile ──────────────────────────────────
    raw_model = model
    model = config.maybe_compile(model)

    # ── Training Loop ───────────────────────────────────────────
    print(f"\n[4/4] Training for {config.decoder_epochs} epochs...")
    history = {
        "loss": [],
        "epoch_loss": [],
        "perplexity": [],
        "generated_fraction": [],
    }
    t_start = time.time()

    for epoch in range(config.decoder_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        n_batches = 0
        mix_rate = _generated_mix_rate(config, epoch)
        print(f"  Epoch {epoch+1}: generated trajectory mix = {mix_rate:.0%}")

        pbar = tqdm(
            dataloader,
            desc=f"  Epoch {epoch+1}/{config.decoder_epochs}",
            leave=False,
        )
        for batch in pbar:
            trajectory = batch["trajectory"].to(device)   # [B, N, d]
            trajectory_mask = batch["active_mask"].to(device)  # [B, N]
            input_ids = batch["input_ids"].to(device)     # [B, L]
            target_ids = batch["target_ids"].to(device)   # [B, L]
            target_mask = batch["target_mask"].to(device)  # [B, L]
            problem_idx = batch["problem_idx"]

            if needs_generated_trajectories and mix_rate > 0.0:
                batch_size = trajectory.shape[0]
                use_generated = torch.rand(batch_size, device=device) < mix_rate
                if mix_rate >= 1.0 - 1e-6:
                    use_generated = torch.ones(batch_size, dtype=torch.bool, device=device)

                if use_generated.any():
                    if torch.is_tensor(problem_idx):
                        problem_idx_list = problem_idx.tolist()
                    else:
                        problem_idx_list = list(problem_idx)

                    selected = use_generated.nonzero(as_tuple=True)[0]
                    premise_texts = [
                        format_premise(parsed_problems[problem_idx_list[i]]["problem"])
                        for i in selected.tolist()
                    ]
                    premise_tokens = tokenizer(
                        premise_texts,
                        max_length=config.max_seq_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)

                    with torch.no_grad():
                        z_0, _ = jepa_model.encode(
                            premise_tokens["input_ids"],
                            premise_tokens["attention_mask"],
                        )
                        z_hat_final = jepa_model.predict_goal(z_0)
                        generated_trajectory, generated_mask = flow_model.generate(
                            z_0,
                            z_hat_final,
                            n_steps=config.ode_steps,
                            solver=config.ode_solver,
                            return_mask=True,
                        )

                    trajectory = trajectory.clone()
                    trajectory_mask = trajectory_mask.clone()
                    trajectory[selected] = generated_trajectory
                    trajectory_mask[selected] = generated_mask
            else:
                use_generated = torch.zeros(
                    trajectory.shape[0], dtype=torch.bool, device=device
                )

            # Forward (BF16 autocast)
            with config.autocast_ctx:
                logits = model(
                    input_ids,
                    trajectory,
                    trajectory_mask=trajectory_mask,
                )  # [B, L, V]

                # Cross-entropy loss (Liger fused or standard)
                if use_liger:
                    # Liger operates on 2D tensors directly
                    loss = LigerCrossEntropyFunction.apply(
                        logits.reshape(-1, vocab_size),
                        target_ids.reshape(-1),
                        pad_id,  # ignore_index
                    )
                else:
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        target_ids.reshape(-1),
                        ignore_index=pad_id,
                    )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
            optimizer.step()

            # Count non-padding tokens for perplexity
            n_tokens = target_mask.sum().item()

            history["loss"].append(loss.item())
            history["generated_fraction"].append(use_generated.float().mean().item())
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            n_batches += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                gen=f"{use_generated.float().mean().item():.2f}",
            )

        avg_loss = epoch_loss / max(epoch_tokens, 1)
        perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)
        history["epoch_loss"].append(avg_loss)
        history["perplexity"].append(perplexity)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch+1}/{config.decoder_epochs} | "
                f"Loss: {avg_loss:.4f} | PPL: {perplexity:.1f} | "
                f"Time: {elapsed:.0f}s"
            )

    # Save (raw uncompiled model)
    checkpoint_path = os.path.join(config.checkpoint_dir, "decoder_final.pt")
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "vocab_size": vocab_size,
            "config": {
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "decoder_layers": config.decoder_layers,
                "n_waypoints": config.n_waypoints,
                "decoder_window_half": config.decoder_window_half,
            },
        },
        checkpoint_path,
    )
    print(f"\n  ✓ Decoder saved to {checkpoint_path}")

    # Save history
    history_path = os.path.join(config.data_dir, "decoder_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    total_time = time.time() - t_start
    print(f"  Phase 3 complete in {total_time/60:.1f} minutes")

    return {"model": model, "tokenizer": tokenizer, "history": history}


if __name__ == "__main__":
    config = DLRConfig()
    train_decoder(config)
