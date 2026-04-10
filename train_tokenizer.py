#!/usr/bin/env python3
"""
DLR Custom Tokenizer — Math-Native BPE

Trains a BPE tokenizer directly on NuminaMath-CoT data so that:
  - Numbers stay as whole tokens (not split into sub-digits)
  - Math operators (+, -, ×, ÷, =, <, >, etc.) are single tokens
  - Our structural tokens ([PREMISE], [STEP], etc.) are native
  - Smaller vocab (~8K-16K vs BERT's 30K) = faster decoder softmax

Usage:
    python train_tokenizer.py                   # Default: 8192 vocab
    python train_tokenizer.py --vocab-size 16384  # Larger for production
    python train_tokenizer.py --n-samples 5000    # More training data
"""

import os
import argparse
from tokenizers import Tokenizer, pre_tokenizers, models, trainers, processors
from tokenizers.normalizers import NFKC
from transformers import PreTrainedTokenizerFast

from data_pipeline import (
    load_dataset_split,
    parse_all_problems,
    format_context,
    format_target,
    format_full_solution,
    SPECIAL_TOKENS,
)


# Math-specific pre-tokenization patterns
MATH_SPECIAL_CHARS = [
    "+", "-", "*", "/", "=", "<", ">", "≤", "≥", "≠",
    "(", ")", "[", "]", "{", "}",
    "^", "_", "√", "∑", "∏", "∫",
    "π", "∞", "±", "×", "÷",
    ".", ",", ":", ";", "!",
]


def build_training_corpus(parsed_problems: list) -> list:
    """
    Build the training corpus from parsed problems.
    Includes all formats the tokenizer will encounter:
      - Raw problems
      - Individual steps
      - Formatted context→target pairs
      - Full formatted solutions
    """
    corpus = []

    for item in parsed_problems:
        problem = item["problem"]
        steps = item["steps"]

        # Raw problem text
        corpus.append(problem)

        # Individual steps
        for step in steps:
            corpus.append(step)

        # Formatted context→target pairs (what JEPA sees)
        for j in range(len(steps) - 1):
            corpus.append(format_context(problem, steps, j))
            corpus.append(format_target(steps[j + 1]))

        # Full formatted solution (what decoder sees)
        corpus.append(format_full_solution(steps))

    return corpus


def train_tokenizer(
    corpus: list,
    vocab_size: int = 8192,
    min_frequency: int = 2,
) -> Tokenizer:
    """
    Train a BPE tokenizer on the math corpus.

    Design choices:
      - BPE (not WordPiece) for better handling of mathematical notation
      - Pre-split on whitespace + digits boundary to protect numbers
      - Special tokens for our DLR format
      - [PAD], [UNK], [CLS] (BOS), [SEP] (EOS) for compatibility
    """
    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Normalizer: Unicode normalization (handles ², ³, ½, etc.)
    tokenizer.normalizer = NFKC()

    # Pre-tokenizer: split on whitespace, then isolate digits
    # This prevents numbers from being merged into adjacent words
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Digits(individual_digits=False),  # Keep "123" together
    ])

    # Define special tokens
    special_tokens = [
        "[PAD]",    # 0 - padding
        "[UNK]",    # 1 - unknown
        "[CLS]",    # 2 - BOS (start of sequence)
        "[SEP]",    # 3 - EOS (end of sequence)
        "[MASK]",   # 4 - mask (for future use)
    ] + SPECIAL_TOKENS  # [PREMISE], [/PREMISE], [STEP], [/STEP], [CONCLUSION]

    # Train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # Post-processor: add [CLS] and [SEP] automatically
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_id),
            ("[SEP]", sep_id),
        ],
    )

    # Enable padding and truncation
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=512)

    return tokenizer


def wrap_as_hf_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    """Wrap the raw tokenizer as a HuggingFace PreTrainedTokenizerFast."""
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    return hf_tokenizer


def analyze_tokenizer(hf_tokenizer, parsed_problems: list, n_examples: int = 5):
    """Analyze tokenizer quality on math data."""
    print(f"\n{'─' * 56}")
    print(f"  Tokenizer Analysis")
    print(f"{'─' * 56}")
    print(f"  Vocab size: {len(hf_tokenizer)}")

    # Test on specific math expressions
    test_cases = [
        "2x + 3 = 15",
        "x = (15 - 3) / 2 = 6",
        "3.14159 * r^2",
        "√(a² + b²) = c",
        "∑_{i=1}^{n} i = n(n+1)/2",
        "[PREMISE] Solve 2x = 10 [/PREMISE]",
        "[STEP] Divide both sides by 2 [/STEP]",
        "[STEP] x = 5 [/STEP] [CONCLUSION]",
    ]

    print(f"\n  Math tokenization samples:")
    for text in test_cases:
        tokens = hf_tokenizer.tokenize(text)
        ids = hf_tokenizer.encode(text)
        print(f"    Input:  {text}")
        print(f"    Tokens: {tokens}")
        print(f"    IDs:    {ids}")
        print()

    # Token length stats on actual data
    all_lengths = []
    for item in parsed_problems[:100]:
        for step in item["steps"]:
            text = format_target(step)
            ids = hf_tokenizer.encode(text)
            all_lengths.append(len(ids))

    if all_lengths:
        all_lengths.sort()
        print(f"  Token lengths (per step):")
        print(f"    Mean: {sum(all_lengths)/len(all_lengths):.0f}")
        print(f"    P50:  {all_lengths[len(all_lengths)//2]}")
        print(f"    P95:  {all_lengths[int(len(all_lengths)*0.95)]}")
        print(f"    Max:  {max(all_lengths)}")

    # Number preservation check
    print(f"\n  Number preservation check:")
    numbers = ["0", "1", "42", "100", "3.14", "0.001", "-7", "1/2", "256"]
    for num in numbers:
        tokens = hf_tokenizer.tokenize(num)
        reconstructed = hf_tokenizer.decode(hf_tokenizer.encode(num), skip_special_tokens=True)
        intact = reconstructed.strip() == num
        status = "✓" if intact else "✗"
        print(f"    {status} {num:>8} → {tokens} → '{reconstructed.strip()}'")


def main():
    parser = argparse.ArgumentParser(description="Train DLR math tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8192,
                        help="Vocabulary size (default: 8192)")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of problems for tokenizer training")
    parser.add_argument("--output-dir", type=str, default="checkpoints/tokenizer",
                        help="Output directory for tokenizer")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Minimum token frequency")
    args = parser.parse_args()

    print("=" * 60)
    print("DLR Math Tokenizer Training")
    print("=" * 60)

    # ── Load data ───────────────────────────────────────────────
    print(f"\n[1/4] Loading {args.n_samples} problems...")
    raw, _ = load_dataset_split("AI-MO/NuminaMath-CoT", args.n_samples)
    parsed = parse_all_problems(raw, min_steps=2, max_steps=20)
    print(f"  ✓ {len(parsed)} problems parsed")

    # ── Build corpus ─────────────────────────────────────────────
    print(f"\n[2/4] Building training corpus...")
    corpus = build_training_corpus(parsed)
    total_chars = sum(len(t) for t in corpus)
    print(f"  ✓ {len(corpus)} text segments, {total_chars:,} total characters")

    # ── Train tokenizer ──────────────────────────────────────────
    print(f"\n[3/4] Training BPE tokenizer (vocab_size={args.vocab_size})...")
    raw_tokenizer = train_tokenizer(
        corpus,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # Wrap as HuggingFace tokenizer
    hf_tokenizer = wrap_as_hf_tokenizer(raw_tokenizer)

    # ── Save ─────────────────────────────────────────────────────
    print(f"\n[4/4] Saving tokenizer...")
    os.makedirs(args.output_dir, exist_ok=True)
    hf_tokenizer.save_pretrained(args.output_dir)
    print(f"  ✓ Saved to {args.output_dir}/")
    print(f"  ✓ Vocab size: {len(hf_tokenizer)}")

    # ── Analysis ─────────────────────────────────────────────────
    analyze_tokenizer(hf_tokenizer, parsed)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Math tokenizer ready: {len(hf_tokenizer)} tokens")
    print(f"  Use with: AutoTokenizer.from_pretrained('{args.output_dir}')")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
