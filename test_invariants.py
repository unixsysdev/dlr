#!/usr/bin/env python3
"""
Invariant tests for the DLR PoC.

These are lightweight regression tests for the silent failure modes the
reviewer called out: waypoint under-budgeting, decoder truncation, extraction
endpoint integrity, and checkpoint reconstruction.
"""

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from checkpointing import build_jepa_from_checkpoint, save_model_checkpoint
from config import DLRConfig
from data_pipeline import DecoderDataset
from extract_trajectories import extract_trajectories
from modules.text_jepa import TextJEPA


class DummyBatch(dict):
    """Small dict wrapper that mirrors HuggingFace BatchEncoding.to()."""

    def to(self, device):
        return DummyBatch({k: v.to(device) for k, v in self.items()})


class DummyTokenizer:
    """Whitespace tokenizer with deterministic ids for fast invariant tests."""

    cls_token_id = 101
    pad_token_id = 0

    def encode(self, text):
        pieces = text.split()
        return list(range(1, len(pieces) + 3))

    def __call__(
        self,
        text,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        batch_ids = []
        batch_masks = []
        for item in texts:
            token_ids = self.encode(item)[:max_length]
            mask = [1] * len(token_ids)
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
            mask = mask + [0] * (max_length - len(mask))
            batch_ids.append(token_ids)
            batch_masks.append(mask)

        return DummyBatch(
            {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
            }
        )


class DummyJEPA(torch.nn.Module):
    """Deterministic encoder that maps token sums into a simple latent vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = torch.nn.Linear(1, d_model, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.arange(1, d_model + 1).float().unsqueeze(1))

    def encode(self, input_ids, attention_mask):
        token_sum = (input_ids * attention_mask).sum(dim=1, keepdim=True).float()
        z = self.proj(token_sum)
        return z, z.unsqueeze(1)


class DLRInvariantTests(unittest.TestCase):
    def test_config_rejects_waypoint_underbudget(self):
        config = DLRConfig(n_waypoints=16, max_steps=20)
        with self.assertRaises(ValueError):
            config.validate()

    def test_decoder_dataset_skips_overlong_targets(self):
        tokenizer = DummyTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trajectories.pt")
            torch.save(
                {
                    "Z_true": torch.zeros(2, 5, 4),
                    "active_masks": torch.tensor(
                        [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float32
                    ),
                    "z_targets": torch.zeros(2, 4),
                    "problem_indices": [0, 1],
                },
                path,
            )

            parsed = [
                {
                    "problem": "p0",
                    "steps": ["too many words " * 4, "another very long step " * 4],
                    "solution": "",
                },
                {
                    "problem": "p1",
                    "steps": ["short one", "short two"],
                    "solution": "",
                },
            ]

            dataset = DecoderDataset(path, parsed, tokenizer, max_seq_len=12)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.skipped_due_to_truncation, 1)

    def test_extract_trajectories_keeps_true_endpoint_and_padded_tail(self):
        tokenizer = DummyTokenizer()
        model = DummyJEPA(d_model=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DLRConfig(
                d_model=4,
                n_waypoints=5,
                max_steps=4,
                max_seq_len=64,
                device="cpu",
                data_dir=tmpdir,
                checkpoint_dir=tmpdir,
                plot_dir=tmpdir,
            )
            parsed = [
                {
                    "problem": "solve x",
                    "steps": ["step one", "step two"],
                    "solution": "",
                }
            ]

            save_path = extract_trajectories(
                config,
                model=model,
                tokenizer=tokenizer,
                parsed_problems=parsed,
            )
            payload = torch.load(save_path, weights_only=False)
            traj = payload["Z_true"][0]
            active = payload["active_masks"][0]
            target = payload["z_targets"][0]

            self.assertEqual(int(active.sum().item()), 3)
            self.assertTrue(torch.allclose(target, traj[2]))
            self.assertTrue(torch.allclose(traj[3], target))
            self.assertTrue(torch.allclose(traj[4], target))

    def test_checkpoint_roundtrip_restores_saved_oracle_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DLRConfig(
                d_model=8,
                n_heads=2,
                encoder_layers=2,
                predictor_hidden=4,
                oracle_layers=3,
                oracle_expansion=2,
                max_steps=4,
                n_waypoints=5,
                device="cpu",
                checkpoint_dir=tmpdir,
                data_dir=tmpdir,
                plot_dir=tmpdir,
            )
            model = TextJEPA(
                vocab_size=32,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.encoder_layers,
                predictor_hidden=config.predictor_hidden,
                ff_mult=config.ff_mult,
                max_len=config.max_seq_len,
                oracle_layers=config.oracle_layers,
                oracle_expansion=config.oracle_expansion,
            )
            ckpt_path = os.path.join(tmpdir, "jepa.pt")
            save_model_checkpoint(
                ckpt_path,
                model.state_dict(),
                config,
                vocab_size=32,
            )

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            rebuilt, rebuilt_config = build_jepa_from_checkpoint(checkpoint, DLRConfig(device="cpu"))

            self.assertEqual(rebuilt_config.oracle_layers, 3)
            self.assertEqual(rebuilt_config.oracle_expansion, 2)
            self.assertEqual(len(rebuilt.oracle.blocks), 3)
            self.assertEqual(rebuilt.oracle.blocks[0].net[0].out_features, 16)


if __name__ == "__main__":
    unittest.main()
