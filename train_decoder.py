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
from data_pipeline import (
    load_dataset_split,
    parse_all_problems,
    prepare_tokenizer,
    DecoderDataset,
)


def train_decoder(config: DLRConfig, parsed_problems: list = None) -> dict:
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
        raw_dataset = load_dataset_split(config.dataset_name, config.n_samples)
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
    history = {"loss": [], "epoch_loss": [], "perplexity": []}
    t_start = time.time()

    for epoch in range(config.decoder_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        n_batches = 0

        pbar = tqdm(
            dataloader,
            desc=f"  Epoch {epoch+1}/{config.decoder_epochs}",
            leave=False,
        )
        for batch in pbar:
            trajectory = batch["trajectory"].to(device)   # [B, N, d]
            input_ids = batch["input_ids"].to(device)     # [B, L]
            target_ids = batch["target_ids"].to(device)   # [B, L]
            target_mask = batch["target_mask"].to(device)  # [B, L]

            # Forward (BF16 autocast)
            with config.autocast_ctx:
                logits = model(input_ids, trajectory)  # [B, L, V]

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
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

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
