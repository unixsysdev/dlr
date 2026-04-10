#!/usr/bin/env python3
"""
DLR Overnight PoC — Master Orchestrator

V3 Unified Trajectory Architecture:
  - No "double dip" — z_0 IS the premise, baked into the trajectory
  - Flow Expert uses AdaLN conditioning (z_0 + z_target + t), no cross-attention
  - Decoder cross-attends ONLY to the trajectory (32×d dense cache)

Usage:
    python run_poc.py                          # PoC on GPU (d=128, ~7h)
    python run_poc.py --production             # H200/B200 run (d=1024, ~14-18h)
    python run_poc.py --quick-test --device cpu # Quick CPU sanity check
    python run_poc.py --skip-to flow           # Resume from Phase 2

Compute optimizations (enabled by default on CUDA):
    - torch.compile: Graph compilation, 1.3-2x speedup
    - BF16 autocast: Half-precision, 2x throughput
    - TF32 matmul:   Tensor core precision, free 2-3x on matmuls
"""

import os
import sys
import time
import argparse
import torch

from checkpointing import (
    build_decoder_from_checkpoint,
    build_flow_from_checkpoint,
    build_jepa_from_checkpoint,
)
from config import DLRConfig, production_config


def parse_args():
    parser = argparse.ArgumentParser(description="DLR Overnight PoC")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of problems to use")
    parser.add_argument("--d-model", type=int, default=None,
                        help="Latent dimension")
    parser.add_argument("--skip-to", type=str, default=None,
                        choices=["extract", "flow", "decoder", "eval"],
                        help="Skip to a specific phase")
    parser.add_argument("--production", action="store_true",
                        help="H200/B200 production config: d=1024, 100K problems")
    parser.add_argument("--quick-test", action="store_true",
                        help="Ultra-fast test: 50 samples, 2 epochs each")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable BF16 autocast")
    return parser.parse_args()


def main():
    args = parse_args()

    # Select base config
    if args.production:
        config = production_config()
        print("🚀 PRODUCTION MODE: H200/B200 config (d=1024, 100K problems)")
    else:
        config = DLRConfig()

    # Override config from args
    if args.device:
        config.device = args.device
    if args.n_samples:
        config.n_samples = args.n_samples
    if args.d_model:
        config.d_model = args.d_model
    if args.no_compile:
        config.use_compile = False
    if args.no_bf16:
        config.use_bf16 = False

    # Quick test mode: minimal everything
    if args.quick_test:
        config.n_samples = 50
        config.jepa_epochs = 2
        config.flow_epochs = 5
        config.decoder_epochs = 3
        config.jepa_batch_size = 16
        config.flow_batch_size = 8
        config.decoder_batch_size = 8
        config.use_compile = False  # Compilation overhead > training time
        config.use_bf16 = False
        config.use_liger = False
        config.full_pipeline_eval_samples = 10
        print("⚡ QUICK TEST MODE: minimal epochs and samples")

    config.validate()

    # Print config
    # Build compute flags string
    flags = []
    if config.use_compile:
        flags.append("compile")
    if config.use_bf16:
        flags.append("bf16")
    if config.use_tf32:
        flags.append("tf32")
    if config.use_liger:
        flags.append("liger")
    compute_str = " + ".join(flags) if flags else "none"

    print("╔" + "═" * 58 + "╗")
    print("║     DLR — Decoupled Latent Reasoner                      ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Device:       {config.device:<42}║")
    print(f"║  d_model:      {config.d_model:<42}║")
    print(f"║  Samples:      {config.n_samples:<42}║")
    print(f"║  JEPA epochs:  {config.jepa_epochs:<42}║")
    print(f"║  Flow epochs:  {config.flow_epochs:<42}║")
    print(f"║  Decoder epochs: {config.decoder_epochs:<40}║")
    print(f"║  Compute:      {compute_str:<42}║")
    print(f"║  Skip to:      {str(args.skip_to):<42}║")
    print("╚" + "═" * 58 + "╝")

    if config.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        config.device = "cpu"

    if config.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    t_start = time.time()

    # Shared state between phases
    jepa_model = None
    tokenizer = None
    parsed_problems = None       # Train split — for trajectory extraction
    parsed_problems_test = None  # Test split — for evaluation ONLY
    flow_model = None
    decoder_model = None

    # ── PHASE 1: Train JEPA ─────────────────────────────────────
    if args.skip_to is None:
        from train_jepa import train_jepa

        result = train_jepa(config)
        jepa_model = result["model"]
        tokenizer = result["tokenizer"]
        parsed_problems = result["parsed_problems"]
        parsed_problems_test = result["parsed_problems_test"]

    # ── PHASE 1.5: Extract Trajectories ─────────────────────────
    if args.skip_to in (None, "extract"):
        from extract_trajectories import extract_trajectories

        extract_trajectories(config, jepa_model, tokenizer, parsed_problems)

    # ── PHASE 2: Train Flow Expert ──────────────────────────────
    if args.skip_to in (None, "extract", "flow"):
        from train_flow import train_flow

        result = train_flow(config, jepa_model=jepa_model)
        flow_model = result["flow_model"]

    # ── PHASE 3: Train Decoder ──────────────────────────────────
    if args.skip_to in (None, "extract", "flow", "decoder"):
        from train_decoder import train_decoder

        result = train_decoder(config, parsed_problems)
        decoder_model = result["model"]
        tokenizer = result["tokenizer"]

    # ── EVALUATION ──────────────────────────────────────────────
    from visualize import generate_full_dashboard

    # Load models if we skipped phases
    if flow_model is None:
        flow_ckpt = os.path.join(config.checkpoint_dir, "flow_final.pt")
        if os.path.exists(flow_ckpt):
            ckpt = torch.load(flow_ckpt, map_location=config.device, weights_only=False)
            flow_model, _, incompatible = build_flow_from_checkpoint(ckpt, config)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(
                    "⚠ Flow checkpoint is not fully compatible with the current "
                    f"architecture: missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )

    if decoder_model is None:
        dec_ckpt = os.path.join(config.checkpoint_dir, "decoder_final.pt")
        if os.path.exists(dec_ckpt):
            from data_pipeline import prepare_tokenizer
            tokenizer = prepare_tokenizer()
            ckpt = torch.load(dec_ckpt, map_location=config.device, weights_only=False)
            decoder_model, _ = build_decoder_from_checkpoint(ckpt, config)

    if jepa_model is None:
        jepa_ckpt = os.path.join(config.checkpoint_dir, "jepa_final.pt")
        if os.path.exists(jepa_ckpt):
            from data_pipeline import prepare_tokenizer as _pt
            if tokenizer is None:
                tokenizer = _pt()
            ckpt = torch.load(jepa_ckpt, map_location=config.device, weights_only=False)
            jepa_model, _ = build_jepa_from_checkpoint(ckpt, config)

    if parsed_problems is None:
        from data_pipeline import load_dataset_split, parse_all_problems
        train_raw, test_raw = load_dataset_split(config.dataset_name, config.n_samples)
        parsed_problems = parse_all_problems(train_raw, config.min_steps, config.max_steps)
        parsed_problems_test = parse_all_problems(test_raw, config.min_steps, config.max_steps)

    generate_full_dashboard(
        config,
        jepa_model=jepa_model,
        flow_model=flow_model,
        decoder_model=decoder_model,
        parsed_problems_train=parsed_problems,
        parsed_problems_test=parsed_problems_test,
        tokenizer=tokenizer,
    )

    # ── Summary ─────────────────────────────────────────────────
    total_time = time.time() - t_start
    hours = total_time / 3600
    print(f"\n{'═' * 60}")
    print(f"  TOTAL TIME: {total_time/60:.1f} minutes ({hours:.2f} hours)")
    print(f"  Plots:       {config.plot_dir}/")
    print(f"  Checkpoints: {config.checkpoint_dir}/")
    print(f"  Data:        {config.data_dir}/")
    print(f"{'═' * 60}")
    print("\n  Check the plots/ directory for your morning dashboard. ☕")


if __name__ == "__main__":
    main()
