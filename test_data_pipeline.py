#!/usr/bin/env python3
"""
DLR Data Pipeline Test

Downloads NuminaMath-CoT, parses solution steps, and validates
the structured text formatting WITHOUT any model training.

Usage:
    python test_data_pipeline.py
    python test_data_pipeline.py --n-samples 50 --verbose
"""

import argparse
import sys
from collections import Counter

from data_pipeline import (
    load_dataset_split,
    parse_all_problems,
    parse_solution_steps,
    format_context,
    format_target,
    format_full_solution,
    SPECIAL_TOKENS,
)


def main():
    parser = argparse.ArgumentParser(description="Test DLR data pipeline")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of problems to load")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed per-problem output")
    parser.add_argument("--show", type=int, default=5,
                        help="Number of full examples to display")
    args = parser.parse_args()

    print("=" * 60)
    print("DLR Data Pipeline Test")
    print("=" * 60)

    # ── 1. Load dataset ─────────────────────────────────────────
    print(f"\n[1/5] Loading {args.n_samples} problems from NuminaMath-CoT...")
    try:
        raw = load_dataset_split("AI-MO/NuminaMath-CoT", args.n_samples)
        print(f"  ✓ Loaded {len(raw)} problems")
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        sys.exit(1)

    # Show raw sample
    print(f"\n  ── Raw sample ──")
    sample = raw[0]
    print(f"  Problem: {sample['problem'][:150]}...")
    print(f"  Solution: {sample['solution'][:200]}...")

    # ── 2. Parse solution steps ─────────────────────────────────
    print(f"\n[2/5] Parsing solution steps...")
    parsed = parse_all_problems(raw, min_steps=2, max_steps=20)
    print(f"  ✓ {len(parsed)} problems passed step count filter")
    print(f"  ✗ {len(raw) - len(parsed)} rejected "
          f"(too few/many steps or empty)")

    if len(parsed) == 0:
        print("\n  ⚠ No problems survived parsing! Debugging...")
        for i in range(min(5, len(raw))):
            steps = parse_solution_steps(raw[i]["solution"])
            print(f"  Problem {i}: {len(steps)} steps parsed")
            if steps:
                print(f"    Step 0: {steps[0][:100]}...")
        sys.exit(1)

    # Step count distribution
    step_counts = [len(p["steps"]) for p in parsed]
    counter = Counter(step_counts)
    print(f"\n  Step count distribution:")
    for k in sorted(counter.keys()):
        bar = "█" * min(counter[k], 40)
        print(f"    {k:2d} steps: {counter[k]:4d} {bar}")
    print(f"  Mean: {sum(step_counts)/len(step_counts):.1f} steps/problem")
    print(f"  Min:  {min(step_counts)}, Max: {max(step_counts)}")

    # ── 3. Show formatted examples ─────────────────────────────
    print(f"\n[3/5] Formatted examples ({args.show} problems):")
    for i in range(min(args.show, len(parsed))):
        item = parsed[i]
        problem = item["problem"]
        steps = item["steps"]
        print(f"\n  {'─' * 56}")
        print(f"  Problem {i} ({len(steps)} steps):")
        print(f"  Raw problem: {problem[:120]}...")

        # Show each step
        for j, step in enumerate(steps):
            print(f"    Step {j}: {step[:100]}{'...' if len(step) > 100 else ''}")

        # Show formatted context→target pairs (JEPA training format)
        print(f"\n  JEPA training pairs:")
        for j in range(min(len(steps) - 1, 3)):
            ctx = format_context(problem, steps, j)
            tgt = format_target(steps[j + 1])
            print(f"    Pair {j}:")
            print(f"      Context: {ctx[:120]}...")
            print(f"      Target:  {tgt[:120]}...")

        # Show full solution format (decoder training)
        full = format_full_solution(steps)
        print(f"\n  Decoder format:")
        print(f"    {full[:200]}...")

    # ── 4. Count JEPA training pairs ───────────────────────────
    print(f"\n[4/5] JEPA training pair statistics:")
    total_pairs = sum(len(p["steps"]) - 1 for p in parsed)
    print(f"  Total context→target pairs: {total_pairs}")
    print(f"  Avg pairs per problem: {total_pairs / len(parsed):.1f}")

    # ── 5. Token length analysis ────────────────────────────────
    print(f"\n[5/5] Text length analysis:")
    context_lens = []
    target_lens = []
    full_lens = []

    for item in parsed:
        problem = item["problem"]
        steps = item["steps"]
        for j in range(len(steps) - 1):
            ctx = format_context(problem, steps, j)
            tgt = format_target(steps[j + 1])
            context_lens.append(len(ctx.split()))
            target_lens.append(len(tgt.split()))
        full = format_full_solution(steps)
        full_lens.append(len(full.split()))

    print(f"  Context lengths (words): "
          f"mean={sum(context_lens)/len(context_lens):.0f}, "
          f"max={max(context_lens)}, "
          f"p95={sorted(context_lens)[int(len(context_lens)*0.95)]}")
    print(f"  Target lengths (words):  "
          f"mean={sum(target_lens)/len(target_lens):.0f}, "
          f"max={max(target_lens)}, "
          f"p95={sorted(target_lens)[int(len(target_lens)*0.95)]}")
    print(f"  Full solution (words):   "
          f"mean={sum(full_lens)/len(full_lens):.0f}, "
          f"max={max(full_lens)}, "
          f"p95={sorted(full_lens)[int(len(full_lens)*0.95)]}")

    # Recommend max_seq_len
    p95_ctx = sorted(context_lens)[int(len(context_lens) * 0.95)]
    # Rough estimate: ~1.3 tokens per word for BPE
    recommended = min(512, max(128, int(p95_ctx * 1.3 / 64) * 64 + 64))
    print(f"\n  Recommended max_seq_len: {recommended} tokens")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Data pipeline validated: {len(parsed)} problems, "
          f"{total_pairs} training pairs")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
