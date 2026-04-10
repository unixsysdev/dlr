"""
Phase 1.5: Extract Z_true Trajectories

After the JEPA is trained, freeze it and extract the geometric
trajectory Z_true for every proof in the dataset.

V3 Unified Trajectory — No Double Dip:
  z_0     = E_y( [PREMISE] )                    ← IS the premise
  z_1     = E_y( [PREMISE] + [STEP 1] )          ← premise + step 1
  z_k     = E_y( [PREMISE] + [STEP 1] + ... + [STEP k] )
  z_final = E_y( Full Proof )                    ← IS the target

  No separate prompt_kv cache needed. z_0 contains the premise.

Variable-Length Handling (Velocity Zero-Out):
  If proof has K < N_max steps:
    - Indices 0..K-1 = actual encoded waypoints
    - Indices K..N_max-1 = copies of z_final
    - active_mask[0..K-1] = 1, active_mask[K..N_max-1] = 0
"""

import os
import torch
from tqdm import tqdm

from config import DLRConfig
from modules.text_jepa import TextJEPA
from data_pipeline import (
    load_dataset_split,
    parse_all_problems,
    prepare_tokenizer,
    format_context,
)


def extract_trajectories(
    config: DLRConfig,
    model: TextJEPA = None,
    tokenizer=None,
    parsed_problems: list = None,
) -> str:
    """
    Extract Z_true trajectories from the frozen JEPA.

    Can accept pre-loaded model/tokenizer/data (from run_poc.py)
    or load from checkpoints (standalone execution).

    Returns:
        Path to saved trajectories file.
    """
    print("=" * 60)
    print("PHASE 1.5: Extracting Z_true Trajectories")
    print("=" * 60)

    device = config.device
    os.makedirs(config.data_dir, exist_ok=True)

    # ── Load model if not provided ──────────────────────────────
    if model is None:
        print("\n[1/3] Loading frozen JEPA from checkpoint...")
        tokenizer = prepare_tokenizer()
        checkpoint = torch.load(
            os.path.join(config.checkpoint_dir, "jepa_final.pt"),
            map_location=device,
            weights_only=False,
        )
        vocab_size = checkpoint["vocab_size"]
        model = TextJEPA(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.encoder_layers,
            predictor_hidden=config.predictor_hidden,
            dropout=0.0,  # No dropout during extraction
            ff_mult=config.ff_mult,
            max_len=config.max_seq_len,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("\n[1/3] Using provided JEPA model...")

    # Freeze everything
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if parsed_problems is None:
        print("[2/3] Loading dataset...")
        raw_dataset, _ = load_dataset_split(config.dataset_name, config.n_samples)
        parsed_problems = parse_all_problems(
            raw_dataset, config.min_steps, config.max_steps
        )

    # ── Extract ─────────────────────────────────────────────────
    print(f"\n[3/3] Extracting trajectories for {len(parsed_problems)} problems...")
    print(f"  N_max = {config.n_waypoints}, d = {config.d_model}")

    all_Z_true = []
    all_active_masks = []
    all_z_targets = []
    problem_indices = []

    for prob_idx, item in enumerate(tqdm(parsed_problems, desc="  Extracting")):
        problem = item["problem"]
        steps = item["steps"]
        n_steps = len(steps)

        # ── Encode each cumulative prefix ───────────────────────
        waypoints = []

        for k in range(-1, n_steps):
            # k=-1: just the premise
            # k=0: premise + step 0
            # k=i: premise + steps 0..i
            if k == -1:
                text = f"[PREMISE] {problem} [/PREMISE]"
            else:
                text = format_context(problem, steps, k)

            tokens = tokenizer(
                text,
                max_length=config.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            z, seq = model.encode(tokens["input_ids"], tokens["attention_mask"])
            waypoints.append(z.squeeze(0))  # [d]
            # V3: No prompt_kv extraction — z_0 IS the premise

        # ── Assemble trajectory with Velocity Zero-Out ──────────
        n_actual = len(waypoints)  # n_steps + 1 (includes premise-only)
        N = config.n_waypoints
        d = config.d_model

        Z_true = torch.zeros(N, d, device=device)
        active_mask = torch.zeros(N, device=device)

        # Fill active waypoints
        n_fill = min(n_actual, N)
        for i in range(n_fill):
            Z_true[i] = waypoints[i]
            active_mask[i] = 1.0

        # Velocity Zero-Out: pad remaining with copies of z_final
        z_final = waypoints[min(n_actual - 1, N - 1)]
        for i in range(n_fill, N):
            Z_true[i] = z_final
            # active_mask[i] stays 0

        z_target = z_final  # [d]

        all_Z_true.append(Z_true.cpu())
        all_active_masks.append(active_mask.cpu())
        all_z_targets.append(z_target.cpu())
        problem_indices.append(prob_idx)

    # ── Stack and save ──────────────────────────────────────────
    # V3: No prompt_kv — the trajectory IS the unified cache
    trajectories = {
        "Z_true": torch.stack(all_Z_true),           # [P, N, d]
        "active_masks": torch.stack(all_active_masks),  # [P, N]
        "z_targets": torch.stack(all_z_targets),      # [P, d]
        "problem_indices": problem_indices,
    }

    save_path = os.path.join(config.data_dir, "trajectories.pt")
    torch.save(trajectories, save_path)

    print(f"\n  ✓ Saved {len(all_Z_true)} trajectories to {save_path}")
    print(f"    Z_true shape: {trajectories['Z_true'].shape}")
    print(f"    Active waypoints: "
          f"min={int(trajectories['active_masks'].sum(1).min())}, "
          f"max={int(trajectories['active_masks'].sum(1).max())}, "
          f"mean={trajectories['active_masks'].sum(1).mean():.1f}")

    return save_path


if __name__ == "__main__":
    config = DLRConfig()
    extract_trajectories(config)
