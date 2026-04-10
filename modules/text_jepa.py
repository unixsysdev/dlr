"""
Module A: Text-JEPA — The Semantic Anchor

A 1D-sequence adaptation of I-JEPA for structured mathematical reasoning.
Learns the latent geometry of logical transitions by predicting the future
state of a proof in continuous space.

Architecture:
  - Context Encoder (E_x): Trainable Transformer, processes premise + prev steps
  - Target Encoder  (E_y): EMA-updated copy of E_x (collapse prevention)
  - Predictor       (P):   Narrow MLP that predicts z_target from z_context
"""

import math
import torch
import torch.nn as nn

from modules.oracle import Oracle


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Text encoder for the JEPA.
    Maps token IDs → d-dimensional latent representations.
    Provides both full sequence output (for prompt_kv) and
    mean-pooled output (for z vectors).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Init weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, S] token IDs
            attention_mask: [B, S] where 1=real, 0=pad

        Returns:
            [B, S, d] full sequence output
        """
        x = self.embedding(input_ids)
        x = self.pos_enc(x)

        # Convert mask: 1=real,0=pad → True=pad for src_key_padding_mask
        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Mean pool over non-padding tokens → [B, d]."""
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
            return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return x.mean(dim=1)


class Predictor(nn.Module):
    """
    Narrow MLP that predicts z_target from z_context.
    Includes a learned "next step" embedding that signals
    the prediction task.
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.next_step_embed = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        """[B, d] → [B, d]"""
        return self.net(z_context + self.next_step_embed)


class TextJEPA(nn.Module):
    """
    Full Text-JEPA module.

    Training:
        loss_dict = jepa(ctx_ids, ctx_mask, tgt_ids, tgt_mask, premise_ids, premise_mask)
        loss_dict['total_loss'].backward()
        optimizer.step()
        jepa.ema_update(tau)  # Update target_encoder

    Extraction (after training):
        z, seq = jepa.encode(input_ids, attention_mask)
        # z: [B, d] mean-pooled vector (for trajectory)
        # seq: [B, S, d] full sequence (for prompt_kv)

    Inference:
        z_hat_final = jepa.predict_goal(z_0)
        # Use z_hat_final as the goal for the Flow Expert
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        predictor_hidden: int = 64,
        dropout: float = 0.1,
        ff_mult: int = 4,
        max_len: int = 512,
        oracle_layers: int = 4,
        oracle_expansion: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        # Context encoder (trainable)
        self.context_encoder = TransformerEncoder(
            vocab_size, d_model, n_heads, n_layers, dropout, ff_mult, max_len
        )

        # Target encoder (EMA updated, no gradients)
        self.target_encoder = TransformerEncoder(
            vocab_size, d_model, n_heads, n_layers, dropout, ff_mult, max_len
        )

        # Initialize target as exact copy of context
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor (trainable) — micro: predicts next step
        self.predictor = Predictor(d_model, predictor_hidden)

        # Oracle (trainable) — macro: predicts final proof state from premise
        self.oracle = Oracle(
            d_model=d_model,
            n_layers=oracle_layers,
            expansion=oracle_expansion,
        )

    @torch.no_grad()
    def ema_update(self, tau: float):
        """
        Update target encoder with EMA of context encoder.
        target = tau * target + (1 - tau) * context

        Uses torch.lerp_: a.lerp_(b, weight) = a + weight*(b-a)
        So lerp_(context, 1-tau) = tau*target + (1-tau)*context ✓
        """
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_tgt.data.lerp_(p_ctx.data, 1.0 - tau)

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        premise_ids: torch.Tensor = None,
        premise_mask: torch.Tensor = None,
    ):
        """
        Forward pass for JEPA + Oracle training.

        Args:
            context_ids: [B, S] context (premise + steps 0..i)
            context_mask: [B, S]
            target_ids: [B, S] next step (i+1)
            target_mask: [B, S]
            premise_ids: [B, S] premise only (for Oracle training)
            premise_mask: [B, S]

        Returns:
            dict with:
                jepa_loss: L2 prediction loss
                oracle_loss: Oracle MSE loss (0 if no premise provided)
                z_var: variance of z_target (collapse detector)
                z_target: [B, d] target representations (detached)
        """
        # Context path (with gradients)
        ctx_seq = self.context_encoder(context_ids, context_mask)
        z_context = self.context_encoder.pool(ctx_seq, context_mask)  # [B, d]

        # Target path (no gradients — EMA updated)
        with torch.no_grad():
            tgt_seq = self.target_encoder(target_ids, target_mask)
            z_target = self.target_encoder.pool(tgt_seq, target_mask)  # [B, d]

        # Predict target from context (micro predictor)
        z_predicted = self.predictor(z_context)  # [B, d]

        # JEPA L2 loss (invariance component — will be wrapped in VICReg)
        jepa_loss = torch.nn.functional.mse_loss(z_predicted, z_target)

        # Collapse detection: per-dim variance averaged across batch
        z_var = z_target.var(dim=0).mean().item()

        # Oracle loss (if premise provided)
        oracle_loss = torch.tensor(0.0, device=context_ids.device)
        if premise_ids is not None:
            with torch.no_grad():
                prem_seq = self.target_encoder(premise_ids, premise_mask)
                z_0 = self.target_encoder.pool(prem_seq, premise_mask)  # [B, d]

            # Oracle predicts z_final from z_0
            z_hat_final = self.oracle(z_0)  # [B, d]

            # z_final = the full-context encoding (last step)
            # Use z_target as proxy for z_final in the current batch
            oracle_loss = torch.nn.functional.mse_loss(z_hat_final, z_target.detach())

        return {
            "jepa_loss": jepa_loss,
            "oracle_loss": oracle_loss,
            "z_var": z_var,
            "z_target": z_target.detach(),
            "z_predicted": z_predicted,
        }

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Encode text using the stable target encoder (for extraction).

        Returns:
            z: [B, d] mean-pooled representation (for trajectory waypoints)
            seq: [B, S, d] full sequence output (retained for extensibility)
        """
        seq = self.target_encoder(input_ids, attention_mask)
        z = self.target_encoder.pool(seq, attention_mask)
        return z, seq

    @torch.no_grad()
    def predict_goal(self, z_0: torch.Tensor) -> torch.Tensor:
        """
        Predict the final proof state from the premise.
        Used at inference when no ground-truth z_target is available.

        Args:
            z_0: [B, d] premise vector

        Returns:
            z_hat_final: [B, d] predicted goal state
        """
        return self.oracle(z_0)

    def trainable_parameters(self):
        """Returns only the parameters that should be optimized."""
        return (
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.oracle.parameters())
        )
