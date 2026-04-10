"""
DLR Data Pipeline

Handles:
  1. Loading NuminaMath-CoT from HuggingFace
  2. Parsing CoT solutions into discrete reasoning steps
  3. Building structured [PREMISE]/[STEP] formatted text
  4. Creating PyTorch datasets for all three training phases
"""

import re
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

# Special tokens for structured CoT format
SPECIAL_TOKENS = ["[PREMISE]", "[/PREMISE]", "[STEP]", "[/STEP]", "[CONCLUSION]"]


def load_dataset_split(dataset_name: str, n_samples: int, test_ratio: float = 0.1):
    """
    Load NuminaMath-CoT with explicit train/test split.

    Returns:
        (train_dataset, test_dataset) — disjoint subsets
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    n_test = max(int(len(ds) * test_ratio), 1)
    n_train = len(ds) - n_test

    train_ds = ds.select(range(n_train))
    test_ds = ds.select(range(n_train, n_train + n_test))

    print(f"  Data split: {n_train} train / {n_test} test")
    return train_ds, test_ds


def parse_solution_steps(solution_text: str) -> List[str]:
    """
    Parse a Chain-of-Thought solution into discrete reasoning steps.

    Handles multiple formats:
      - "Step N: ..." explicit steps
      - "**Step N:**" markdown bold steps
      - Numbered lists "1. ...", "2. ..."
      - Double-newline separated paragraphs
      - Single-newline fallback
    """
    if not solution_text or not solution_text.strip():
        return []

    steps = []

    # Try 1: Explicit "Step N:" or "**Step N:**" markers
    parts = re.split(
        r"(?:^|\n)\s*(?:\*{0,2}Step\s+\d+[\.:]\*{0,2})\s*",
        solution_text,
        flags=re.IGNORECASE,
    )
    parts = [p.strip() for p in parts if p and len(p.strip()) > 10]
    if len(parts) >= 2:
        return parts

    # Try 2: Numbered lists "1. ...", "2) ..."
    parts = re.split(r"\n\s*\d+[\.\)]\s+", solution_text)
    parts = [p.strip() for p in parts if p and len(p.strip()) > 10]
    if len(parts) >= 2:
        return parts

    # Try 3: Double newline paragraphs
    parts = re.split(r"\n\s*\n", solution_text)
    parts = [p.strip() for p in parts if p and len(p.strip()) > 10]
    if len(parts) >= 2:
        return parts

    # Try 4: Single newline (most aggressive)
    parts = solution_text.split("\n")
    parts = [p.strip() for p in parts if p and len(p.strip()) > 10]
    if len(parts) >= 2:
        return parts

    # Last resort: treat entire solution as one step
    return [solution_text.strip()] if len(solution_text.strip()) > 10 else []


def format_context(problem: str, steps: List[str], up_to: int) -> str:
    """Format premise + steps[0..up_to] as structured text."""
    text = f"[PREMISE] {problem} [/PREMISE]"
    for i in range(up_to + 1):
        text += f" [STEP] {steps[i]} [/STEP]"
    return text


def format_premise(problem: str) -> str:
    """Format premise ONLY — no intermediate steps. Used for Oracle training."""
    return f"[PREMISE] {problem} [/PREMISE]"


def format_target(step: str) -> str:
    """Format a single step as structured text."""
    return f"[STEP] {step} [/STEP]"


def format_goal_state(problem: str, steps: List[str]) -> str:
    """
    Format the full cumulative proof state at the final prefix.

    This is the object the Flow Expert uses as z_target after trajectory
    extraction, so the Oracle must be trained against the same endpoint.
    """
    return format_context(problem, steps, len(steps) - 1)


def format_full_solution(steps: List[str]) -> str:
    """Format the complete solution for decoder training."""
    parts = [f"[STEP] {s} [/STEP]" for s in steps]
    return " ".join(parts) + " [CONCLUSION]"


def prepare_tokenizer(
    tokenizer_name: str = "bert-base-uncased",
    custom_path: str = "checkpoints/tokenizer",
):
    """
    Load tokenizer, preferring the custom math tokenizer if available.

    Priority:
      1. Custom math BPE from checkpoints/tokenizer/ (trained by train_tokenizer.py)
      2. Fallback to bert-base-uncased with DLR special tokens added
    """
    import os
    from transformers import AutoTokenizer

    if os.path.exists(custom_path):
        tokenizer = AutoTokenizer.from_pretrained(custom_path)
        print(f"  ✓ Loaded custom math tokenizer from {custom_path} "
              f"(vocab: {len(tokenizer)})")
        return tokenizer

    print(f"  ⚠ Custom tokenizer not found at {custom_path}, "
          f"falling back to {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tokenizer


def parse_all_problems(
    dataset, min_steps: int = 2, max_steps: int = 20
) -> List[Dict]:
    """
    Parse all problems into structured format.

    Returns list of dicts with:
      - problem: str (the problem text)
      - steps: List[str] (discrete reasoning steps)
      - solution: str (full solution text)
    """
    parsed = []
    for item in dataset:
        problem = item["problem"]
        solution = item["solution"]
        steps = parse_solution_steps(solution)

        # Filter by step count
        if len(steps) < min_steps or len(steps) > max_steps:
            continue

        parsed.append(
            {"problem": problem, "steps": steps, "solution": solution}
        )

    return parsed


class JEPADataset(Dataset):
    """
    Dataset for Phase 1: JEPA + Oracle training.

    Yields 5-tuples using a sliding window:
      Context   = [PREMISE] problem [/PREMISE] [STEP] s1 ... [STEP] si
      Target    = [STEP] s_{i+1} [/STEP]              (micro-predictor target)
      Premise   = [PREMISE] problem [/PREMISE]        (Oracle input — premise ONLY)
      GoalState = full cumulative proof prefix        (Oracle target / Flow endpoint)
    """

    def __init__(
        self,
        parsed_problems: List[Dict],
        tokenizer,
        max_seq_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pairs = []
        skipped = 0

        for item in parsed_problems:
            problem = item["problem"]
            steps = item["steps"]
            premise = format_premise(problem)
            goal_state = format_goal_state(problem, steps)

            # Sliding window: each pair is (prefix, next_step, premise, goal_state)
            for i in range(len(steps) - 1):
                context = format_context(problem, steps, i)
                target = format_target(steps[i + 1])

                # Guard: skip pairs where context would be truncated.
                # Truncation silently chops the premise or recent steps,
                # causing the JEPA to learn corrupted representations.
                ctx_len = len(tokenizer.encode(context))
                goal_len = len(tokenizer.encode(goal_state))
                if ctx_len > max_seq_len or goal_len > max_seq_len:
                    skipped += 1
                    continue

                self.pairs.append((context, target, premise, goal_state))

        print(f"  JEPADataset: {len(self.pairs)} context→target pairs"
              f" ({skipped} skipped due to truncation)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, target, premise, goal_state = self.pairs[idx]

        ctx = self.tokenizer(
            context,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            target,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        prem = self.tokenizer(
            premise,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        goal = self.tokenizer(
            goal_state,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "context_ids": ctx["input_ids"].squeeze(0),
            "context_mask": ctx["attention_mask"].squeeze(0),
            "target_ids": tgt["input_ids"].squeeze(0),
            "target_mask": tgt["attention_mask"].squeeze(0),
            "premise_ids": prem["input_ids"].squeeze(0),
            "premise_mask": prem["attention_mask"].squeeze(0),
            "goal_ids": goal["input_ids"].squeeze(0),
            "goal_mask": goal["attention_mask"].squeeze(0),
        }


class TrajectoryDataset(Dataset):
    """
    Dataset for Phase 2: Flow Expert training.

    Loads pre-extracted trajectories from extract_trajectories.py.
    V3: No prompt_kv — z_0 is the premise, accessed as Z_true[:, 0, :].
    Each item contains:
      - Z_true: [N, d] ground-truth trajectory (Velocity Zero-Out padded)
      - active_mask: [N] binary mask (1=active, 0=padded copy of z_final)
      - z_target: [d] the final waypoint
    """

    def __init__(self, trajectory_path: str):
        data = torch.load(trajectory_path, weights_only=False)
        self.Z_true = data["Z_true"]           # [num_problems, N, d]
        self.active_masks = data["active_masks"]  # [num_problems, N]
        self.z_targets = data["z_targets"]     # [num_problems, d]
        print(f"  TrajectoryDataset: {len(self.Z_true)} trajectories loaded")

    def __len__(self):
        return len(self.Z_true)

    def __getitem__(self, idx):
        return {
            "Z_true": self.Z_true[idx],
            "active_mask": self.active_masks[idx],
            "z_target": self.z_targets[idx],
        }


class DecoderDataset(Dataset):
    """
    Dataset for Phase 3: Decoder training.

    Each item contains:
      - trajectory: [N, d] the continuous trajectory
      - input_ids: [L] decoder input tokens (BOS + shifted solution)
      - target_ids: [L] decoder target tokens (solution)
      - target_mask: [L] attention mask for loss (1=real, 0=pad)
    """

    def __init__(
        self,
        trajectory_path: str,
        parsed_problems: List[Dict],
        tokenizer,
        max_seq_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.bos_id = tokenizer.cls_token_id
        self.pad_id = tokenizer.pad_token_id

        # Load trajectories
        data = torch.load(trajectory_path, weights_only=False)
        self.trajectories = data["Z_true"]         # [num_problems, N, d]
        self.active_masks = data["active_masks"]   # [num_problems, N]
        self.problem_indices = data["problem_indices"]  # maps trajectory → problem

        # Tokenize solutions
        self.items = []
        for i, prob_idx in enumerate(self.problem_indices):
            if prob_idx >= len(parsed_problems):
                continue
            steps = parsed_problems[prob_idx]["steps"]
            solution_text = format_full_solution(steps)

            tokens = tokenizer(
                solution_text,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            self.items.append(
                {
                    "trajectory_idx": i,
                    "problem_idx": prob_idx,
                    "token_ids": tokens["input_ids"].squeeze(0),
                    "token_mask": tokens["attention_mask"].squeeze(0),
                }
            )

        print(f"  DecoderDataset: {len(self.items)} trajectory→text pairs")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        trajectory = self.trajectories[item["trajectory_idx"]]
        active_mask = self.active_masks[item["trajectory_idx"]]
        token_ids = item["token_ids"]
        token_mask = item["token_mask"]

        # Teacher forcing: input = [BOS] + tokens[:-1], target = tokens
        input_ids = torch.cat(
            [torch.tensor([self.bos_id]), token_ids[:-1]]
        )

        return {
            "trajectory": trajectory,
            "active_mask": active_mask,
            "problem_idx": item["problem_idx"],
            "input_ids": input_ids,
            "target_ids": token_ids,
            "target_mask": token_mask,
        }
