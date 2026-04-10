"""
DLR Overnight PoC — Visualization & Evaluation Dashboard

Implements the four evaluation metrics:
  A. JEPA Collapse Variance
  B. Cosine Monotonicity
  C. Flow Convergence (endpoint L2)
  D. Token Recovery Rate (number/operator exact match)

Plus training curve plots and trajectory visualizations.
"""

import os
import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("  ⚠ umap-learn not installed, falling back to t-SNE")

from sklearn.manifold import TSNE


def setup_plots(plot_dir: str):
    """Create plot directory and set matplotlib style."""
    os.makedirs(plot_dir, exist_ok=True)
    plt.style.use("dark_background")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12


# ─── Metric A: JEPA Collapse Variance ──────────────────────────


def plot_jepa_training(data_dir: str, plot_dir: str):
    """Plot JEPA loss curve, z_var, and EMA schedule."""
    history_path = os.path.join(data_dir, "jepa_history.json")
    if not os.path.exists(history_path):
        print("  ⚠ No JEPA history found, skipping")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    axes[0].plot(history["loss"], alpha=0.3, color="#00d4ff", linewidth=0.5)
    # Smoothed
    window = max(len(history["loss"]) // 50, 1)
    smoothed = np.convolve(history["loss"], np.ones(window) / window, mode="valid")
    axes[0].plot(smoothed, color="#00d4ff", linewidth=2)
    axes[0].set_title("JEPA Loss", fontweight="bold")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("L2 Loss")
    axes[0].set_yscale("log")

    # z_var (collapse detector)
    axes[1].plot(history["z_var"], color="#ff6b6b", linewidth=1)
    axes[1].axhline(y=1e-4, color="#ffcc00", linestyle="--", label="Collapse threshold")
    axes[1].set_title("Metric A: z_var (Collapse Detector)", fontweight="bold")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Var(z_target)")
    axes[1].set_yscale("log")
    axes[1].legend()

    # EMA tau
    axes[2].plot(history["ema_tau"], color="#a855f7", linewidth=2)
    axes[2].set_title("EMA Momentum (τ)", fontweight="bold")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("τ")

    fig.suptitle("Phase 1: Text-JEPA Training", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "01_jepa_training.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved 01_jepa_training.png")

    # Report
    final_z_var = history["z_var"][-1]
    collapsed = final_z_var < 1e-4
    print(f"    Final z_var: {final_z_var:.6f} {'⚠️ COLLAPSED!' if collapsed else '✓ Healthy'}")


# ─── Metric B: Cosine Monotonicity ─────────────────────────────


def evaluate_cosine_monotonicity(data_dir: str, plot_dir: str, n_examples: int = 20):
    """
    Check if cosine similarity to z_final increases monotonically
    along each trajectory.

    A proof trajectory [z_0, z_1, ..., z_final] should have:
      cos(z_0, z_final) < cos(z_1, z_final) < ... < cos(z_{n-1}, z_final) = 1.0
    """
    traj_path = os.path.join(data_dir, "trajectories.pt")
    if not os.path.exists(traj_path):
        print("  ⚠ No trajectories found, skipping")
        return

    data = torch.load(traj_path, weights_only=False)
    Z_true = data["Z_true"]            # [P, N, d]
    active_masks = data["active_masks"]  # [P, N]

    monotonic_count = 0
    total_count = 0
    all_cosines = []

    n_check = min(n_examples, len(Z_true))

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(n_check):
        z_traj = Z_true[i]          # [N, d]
        mask = active_masks[i]      # [N]
        n_active = int(mask.sum().item())

        if n_active < 2:
            continue

        z_active = z_traj[:n_active]  # [K, d]
        z_final = z_active[-1]        # [d]

        # Cosine similarities to z_final
        z_norm = torch.nn.functional.normalize(z_active, dim=-1)
        zf_norm = torch.nn.functional.normalize(z_final.unsqueeze(0), dim=-1)
        cosines = (z_norm * zf_norm).sum(dim=-1)  # [K]
        cosines_np = cosines.numpy()

        # Check monotonicity
        is_monotonic = all(
            cosines_np[j] <= cosines_np[j + 1] + 1e-6
            for j in range(len(cosines_np) - 1)
        )
        monotonic_count += int(is_monotonic)
        total_count += 1
        all_cosines.append(cosines_np)

        # Plot first 10
        if i < 10:
            color = "#00ff88" if is_monotonic else "#ff4444"
            axes[i].plot(cosines_np, marker="o", color=color, markersize=4)
            axes[i].set_title(f"Proof {i} ({'✓' if is_monotonic else '✗'})", fontsize=10)
            axes[i].set_xlabel("Step")
            axes[i].set_ylabel("cos(z_i, z_final)")
            axes[i].set_ylim(-0.2, 1.1)

    fig.suptitle(
        "Metric B: Cosine Monotonicity — cos(z_i, z_final) should increase",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "02_cosine_monotonicity.png"), dpi=150, bbox_inches="tight")
    plt.close()

    rate = monotonic_count / max(total_count, 1)
    print(f"  ✓ Saved 02_cosine_monotonicity.png")
    print(f"    Monotonic: {monotonic_count}/{total_count} ({rate:.0%})")


# ─── Trajectory Visualization (t-SNE / UMAP) ───────────────────


def visualize_trajectories(data_dir: str, plot_dir: str, n_trajectories: int = 10):
    """
    Visualize extracted trajectories in 2D using UMAP or t-SNE.
    Each trajectory is a colored line showing the path through latent space.
    """
    traj_path = os.path.join(data_dir, "trajectories.pt")
    if not os.path.exists(traj_path):
        print("  ⚠ No trajectories found, skipping")
        return

    data = torch.load(traj_path, weights_only=False)
    Z_true = data["Z_true"][:n_trajectories]       # [T, N, d]
    masks = data["active_masks"][:n_trajectories]   # [T, N]

    # Collect all active waypoints
    all_points = []
    traj_ids = []
    for i in range(len(Z_true)):
        n_active = int(masks[i].sum().item())
        points = Z_true[i, :n_active].numpy()  # [K, d]
        all_points.append(points)
        traj_ids.extend([i] * n_active)

    all_points = np.concatenate(all_points, axis=0)  # [total_pts, d]
    traj_ids = np.array(traj_ids)

    # Dimensionality reduction
    if HAS_UMAP and len(all_points) > 15:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_points) - 1))
        label = "UMAP"
    else:
        perp = min(30, max(2, len(all_points) - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perp)
        label = "t-SNE"

    embedded = reducer.fit_transform(all_points)  # [total_pts, 2]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.get_cmap("tab10")

    offset = 0
    for i in range(len(Z_true)):
        n_active = int(masks[i].sum().item())
        pts = embedded[offset : offset + n_active]
        color = cmap(i % 10)

        # Line
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=0.6, linewidth=1.5)
        # Start marker
        ax.scatter(pts[0, 0], pts[0, 1], color=color, marker="o", s=80, zorder=5)
        # End marker
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, marker="*", s=150, zorder=5)
        # Step indices
        for j in range(len(pts)):
            ax.annotate(str(j), (pts[j, 0], pts[j, 1]), fontsize=7, color=color, alpha=0.8)

        offset += n_active

    ax.set_title(
        f"Trajectory Visualization ({label}) — {n_trajectories} proofs\n"
        "○ = start, ★ = z_final",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(f"{label}-1")
    ax.set_ylabel(f"{label}-2")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "03_trajectory_visualization.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved 03_trajectory_visualization.png")


# ─── Metric C: Flow Convergence ─────────────────────────────────


def plot_flow_training(data_dir: str, plot_dir: str):
    """Plot Flow Expert loss curve."""
    history_path = os.path.join(data_dir, "flow_history.json")
    if not os.path.exists(history_path):
        print("  ⚠ No Flow history found, skipping")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-step loss
    axes[0].plot(history["loss"], alpha=0.2, color="#00d4ff", linewidth=0.5)
    window = max(len(history["loss"]) // 50, 1)
    smoothed = np.convolve(history["loss"], np.ones(window) / window, mode="valid")
    axes[0].plot(smoothed, color="#00d4ff", linewidth=2)
    axes[0].set_title("Flow Expert Loss (per step)", fontweight="bold")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Masked MSE")
    axes[0].set_yscale("log")

    # Per-epoch loss
    axes[1].plot(history["epoch_loss"], marker="o", color="#ff6b6b", linewidth=2, markersize=3)
    axes[1].set_title("Flow Expert Loss (per epoch)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Avg Masked MSE")
    axes[1].set_yscale("log")

    fig.suptitle("Phase 2: Rectified Flow Training", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "04_flow_training.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved 04_flow_training.png")

    print(f"    Final flow loss: {history['epoch_loss'][-1]:.6f}")


def evaluate_flow_endpoint(
    flow_model,
    data_dir: str,
    plot_dir: str,
    config,
    n_samples: int = 20,
):
    """
    Metric C: Generate trajectories from noise and measure
    endpoint distance to true z_final.
    """
    traj_path = os.path.join(data_dir, "trajectories.pt")
    data = torch.load(traj_path, weights_only=False)
    device = config.device

    Z_true = data["Z_true"][:n_samples].to(device)
    z_targets = data["z_targets"][:n_samples].to(device)
    active_masks = data["active_masks"][:n_samples]

    flow_model.eval()

    distances = []
    for i in range(n_samples):
        # V3: z_0 is the first waypoint of the trajectory
        z_0_i = Z_true[i, 0:1, :].squeeze(0).unsqueeze(0)  # [1, d]

        # Generate trajectory from noise
        gen = flow_model.generate(
            z_0_i,
            z_targets[i : i + 1],
            n_steps=config.ode_steps,
            solver=config.ode_solver,
        )  # [1, N, d]

        # Get active final waypoint index
        n_active = int(active_masks[i].sum().item())
        final_idx = max(n_active - 1, 0)

        # Endpoint distance
        gen_final = gen[0, final_idx]              # [d]
        true_final = Z_true[i, final_idx]          # [d]
        dist = torch.norm(gen_final - true_final).item()
        distances.append(dist)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(n_samples), distances, color="#a855f7", alpha=0.8)
    ax.axhline(y=np.mean(distances), color="#ffcc00", linestyle="--",
               label=f"Mean: {np.mean(distances):.4f}")
    ax.set_title("Metric C: Endpoint Distance (Generated vs True z_final)",
                 fontweight="bold")
    ax.set_xlabel("Problem")
    ax.set_ylabel("L2 Distance")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "05_flow_endpoint.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved 05_flow_endpoint.png")
    print(f"    Mean endpoint L2: {np.mean(distances):.4f}")
    print(f"    Min: {np.min(distances):.4f}, Max: {np.max(distances):.4f}")


# ─── Metric D: Token Recovery Rate ─────────────────────────────


def extract_math_tokens(text: str) -> set:
    """Extract numbers and math operators from text."""
    # Numbers (int, float, fractions)
    numbers = set(re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text))
    # Operators and keywords
    operators = set(re.findall(r"[+\-*/=<>≤≥±√∑∏∫]", text))
    # Common math words
    words = set(re.findall(r"\b(?:pi|sqrt|sin|cos|tan|log|ln|sum|prod|lim)\b", text, re.I))
    return numbers | operators | words


def evaluate_token_recovery(
    decoder_model,
    data_dir: str,
    plot_dir: str,
    parsed_problems: list,
    tokenizer,
    config,
    n_samples: int = 10,
):
    """
    Metric D: Generate text from trajectories and check if
    the correct numbers and operators are recovered.
    """
    traj_path = os.path.join(data_dir, "trajectories.pt")
    data = torch.load(traj_path, weights_only=False)
    device = config.device

    Z_true = data["Z_true"][:n_samples].to(device)
    indices = data["problem_indices"][:n_samples]

    decoder_model.eval()
    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.sep_token_id

    results = []
    for i in range(min(n_samples, len(Z_true))):
        # Generate text from trajectory
        generated_ids = decoder_model.generate(
            Z_true[i : i + 1],
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            max_length=config.decoder_max_seq_len,
            temperature=config.eval_temperature,  # Near-greedy for PoC clarity
        )

        gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Ground truth
        prob_idx = indices[i]
        if prob_idx < len(parsed_problems):
            gt_steps = parsed_problems[prob_idx]["steps"]
            gt_text = " ".join(gt_steps)
        else:
            gt_text = ""

        # Token recovery
        gen_tokens = extract_math_tokens(gen_text)
        gt_tokens = extract_math_tokens(gt_text)

        if gt_tokens:
            recovered = gen_tokens & gt_tokens
            recovery_rate = len(recovered) / len(gt_tokens)
        else:
            recovered = set()
            recovery_rate = 0.0

        results.append({
            "problem_idx": prob_idx,
            "generated": gen_text[:200],
            "ground_truth": gt_text[:200],
            "gen_math_tokens": list(gen_tokens),
            "gt_math_tokens": list(gt_tokens),
            "recovered": list(recovered),
            "recovery_rate": recovery_rate,
        })

    # Plot
    rates = [r["recovery_rate"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(rates)), rates, color="#00ff88", alpha=0.8)
    ax.axhline(y=np.mean(rates), color="#ffcc00", linestyle="--",
               label=f"Mean: {np.mean(rates):.2%}")
    ax.set_title("Metric D: Token Recovery Rate (Numbers & Operators)",
                 fontweight="bold")
    ax.set_xlabel("Problem")
    ax.set_ylabel("Recovery Rate")
    ax.set_ylim(0, 1.1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "06_token_recovery.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved 06_token_recovery.png")
    print(f"    Mean recovery rate: {np.mean(rates):.2%}")

    # Print samples
    print("\n  ── Sample Outputs ──")
    for r in results[:5]:
        print(f"\n    Problem {r['problem_idx']}:")
        print(f"    GT:  {r['ground_truth'][:120]}...")
        print(f"    Gen: {r['generated'][:120]}...")
        print(f"    Math tokens recovered: {r['recovered']} "
              f"({r['recovery_rate']:.0%})")

    # Save results
    results_path = os.path.join(data_dir, "token_recovery_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─── Decoder Training Curves ───────────────────────────────────


def plot_decoder_training(data_dir: str, plot_dir: str):
    """Plot decoder loss and perplexity curves."""
    history_path = os.path.join(data_dir, "decoder_history.json")
    if not os.path.exists(history_path):
        print("  ⚠ No decoder history found, skipping")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["loss"], alpha=0.2, color="#00d4ff", linewidth=0.5)
    window = max(len(history["loss"]) // 50, 1)
    smoothed = np.convolve(history["loss"], np.ones(window) / window, mode="valid")
    axes[0].plot(smoothed, color="#00d4ff", linewidth=2)
    axes[0].set_title("Decoder Cross-Entropy Loss", fontweight="bold")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("CE Loss")

    # Perplexity
    axes[1].plot(history["perplexity"], marker="o", color="#ff6b6b",
                 linewidth=2, markersize=4)
    axes[1].set_title("Decoder Perplexity", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_yscale("log")

    fig.suptitle("Phase 3: Scribe Decoder Training", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "07_decoder_training.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved 07_decoder_training.png")

    print(f"    Final perplexity: {history['perplexity'][-1]:.1f}")


# ─── Master Dashboard ──────────────────────────────────────────


def generate_full_dashboard(
    config,
    flow_model=None,
    decoder_model=None,
    parsed_problems=None,
    tokenizer=None,
):
    """Generate the complete evaluation dashboard."""
    print("\n" + "=" * 60)
    print("EVALUATION DASHBOARD")
    print("=" * 60)

    setup_plots(config.plot_dir)

    # Phase 1 plots
    print("\n── Phase 1: JEPA ──")
    plot_jepa_training(config.data_dir, config.plot_dir)

    # Trajectory analysis
    print("\n── Trajectory Analysis ──")
    evaluate_cosine_monotonicity(config.data_dir, config.plot_dir)
    visualize_trajectories(config.data_dir, config.plot_dir)

    # Phase 2 plots
    print("\n── Phase 2: Flow Expert ──")
    plot_flow_training(config.data_dir, config.plot_dir)
    if flow_model is not None:
        evaluate_flow_endpoint(flow_model, config.data_dir, config.plot_dir, config)

    # Phase 3 plots
    print("\n── Phase 3: Decoder ──")
    plot_decoder_training(config.data_dir, config.plot_dir)
    if decoder_model is not None and parsed_problems is not None and tokenizer is not None:
        evaluate_token_recovery(
            decoder_model, config.data_dir, config.plot_dir,
            parsed_problems, tokenizer, config,
        )

    print(f"\n  All plots saved to {config.plot_dir}/")
    print("=" * 60)
