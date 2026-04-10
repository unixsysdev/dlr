"""
Module C: The Scribe — Y-Decoder

Lightweight autoregressive causal decoder that translates
the N×d continuous trajectory into discrete text tokens.

Key design:
  - Cross-attention uses a sliding window: token i only attends
    to nearby trajectory waypoints, preventing attention smearing.
  - At PoC scale (d=128, 2 layers), this validates information flow
    from Module B → text, not mathematical accuracy.
"""

import torch
import torch.nn as nn
import math


class DecoderLayer(nn.Module):
    """Single decoder layer: causal self-attn → sliding-window cross-attn → FFN."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        # Causal self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention to trajectory waypoints
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        trajectory: torch.Tensor,
        causal_mask: torch.Tensor,
        sw_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, d] token embeddings
            trajectory: [B, N, d] waypoint trajectory
            causal_mask: [L, L] float mask (-inf=blocked, 0=allowed)
            sw_mask: [L, N] float sliding window mask
        """
        # Self-attention (causal)
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, attn_mask=causal_mask, is_causal=False)[0]

        # Cross-attention (sliding window) to trajectory
        h = self.norm2(x)
        x = x + self.cross_attn(h, trajectory, trajectory, attn_mask=sw_mask)[0]

        # FFN
        h = self.norm3(x)
        x = x + self.ffn(h)

        return x


class ScribeDecoder(nn.Module):
    """
    Autoregressive decoder that reads continuous trajectories
    and outputs discrete tokens.

    The sliding window constraint ensures token i only attends
    to trajectory waypoints in [k-w, k+w], where
    k = floor(i · N / L) maps token position to trajectory position.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_waypoints: int = 16,
        window_half: int = 2,
        max_seq_len: int = 256,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_waypoints = n_waypoints
        self.window_half = window_half
        self.max_seq_len = max_seq_len

        # Token embedding + positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (embedding ↔ lm_head)
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask: 0=attend, -inf=block."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def _make_sliding_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Sliding window cross-attention mask.
        Token i attends to trajectory waypoints [k-w, k+w]
        where k = floor(i · N / L).

        Returns:
            [L, N] float mask: 0=attend, -inf=block
        """
        N = self.n_waypoints
        w = self.window_half
        mask = torch.full((seq_len, N), float("-inf"), device=device)
        for i in range(seq_len):
            k = int(i * N / max(seq_len, 1))
            start = max(0, k - w)
            end = min(N, k + w + 1)
            mask[i, start:end] = 0.0
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Args:
            input_ids: [B, L] input token IDs (BOS + shifted target)
            trajectory: [B, N, d] waypoint trajectory

        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Token + positional embeddings
        positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # Build masks
        causal_mask = self._make_causal_mask(L, device)    # [L, L]
        sw_mask = self._make_sliding_window_mask(L, device)  # [L, N]

        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, trajectory, causal_mask, sw_mask)

        # Output logits
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        trajectory: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 256,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        """
        Autoregressive generation from a trajectory.

        Args:
            trajectory: [B, N, d] waypoint trajectory
            bos_token_id: beginning-of-sequence token ID
            eos_token_id: end-of-sequence token ID
            max_length: maximum generation length
            temperature: sampling temperature

        Returns:
            generated_ids: [B, L] generated token IDs
        """
        B = trajectory.shape[0]
        device = trajectory.device

        # Start with BOS
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            logits = self.forward(generated, trajectory)  # [B, L, V]
            next_logits = logits[:, -1, :] / temperature   # [B, V]
            next_token = torch.multinomial(
                torch.softmax(next_logits, dim=-1), num_samples=1
            )  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have produced EOS
            if (next_token == eos_token_id).all():
                break

        return generated
