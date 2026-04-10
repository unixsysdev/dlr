"""
Checkpoint helpers for saving full configs and reconstructing models safely.
"""

import torch

from config import DLRConfig
from modules.decoder import ScribeDecoder
from modules.flow_expert import FlowExpert
from modules.text_jepa import TextJEPA


def config_from_checkpoint(checkpoint: dict, fallback: DLRConfig) -> DLRConfig:
    """
    Reconstruct the full config saved with a checkpoint.

    Runtime paths and device come from the caller's current config so resumed runs
    still write into the active workspace.
    """
    raw = checkpoint.get("full_config") or checkpoint.get("config") or {}
    loaded = DLRConfig.from_dict(raw, base=fallback)
    loaded.device = fallback.device
    loaded.checkpoint_dir = fallback.checkpoint_dir
    loaded.data_dir = fallback.data_dir
    loaded.plot_dir = fallback.plot_dir
    return loaded


def save_model_checkpoint(
    path: str,
    model_state_dict: dict,
    config: DLRConfig,
    **extra,
):
    """Save a model checkpoint together with the full runtime config."""
    payload = {
        "model_state_dict": model_state_dict,
        "full_config": config.to_dict(),
    }
    payload.update(extra)
    torch.save(payload, path)


def build_jepa_from_checkpoint(
    checkpoint: dict,
    fallback_config: DLRConfig,
    vocab_size: int = None,
):
    """Instantiate TextJEPA using the architecture saved in its checkpoint."""
    config = config_from_checkpoint(checkpoint, fallback_config)
    model = TextJEPA(
        vocab_size=vocab_size or checkpoint["vocab_size"],
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.encoder_layers,
        predictor_hidden=config.predictor_hidden,
        dropout=0.0,
        ff_mult=config.ff_mult,
        max_len=config.max_seq_len,
        oracle_layers=config.oracle_layers,
        oracle_expansion=config.oracle_expansion,
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def build_flow_from_checkpoint(
    checkpoint: dict,
    fallback_config: DLRConfig,
):
    """Instantiate FlowExpert using the architecture saved in its checkpoint."""
    config = config_from_checkpoint(checkpoint, fallback_config)
    model = FlowExpert(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.flow_layers,
        n_waypoints=config.n_waypoints,
        ff_mult=config.ff_mult,
        dropout=0.0,
    ).to(config.device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model, config, incompatible


def build_decoder_from_checkpoint(
    checkpoint: dict,
    fallback_config: DLRConfig,
):
    """Instantiate ScribeDecoder using the architecture saved in its checkpoint."""
    config = config_from_checkpoint(checkpoint, fallback_config)
    model = ScribeDecoder(
        vocab_size=checkpoint["vocab_size"],
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.decoder_layers,
        n_waypoints=config.n_waypoints,
        window_half=config.decoder_window_half,
        use_sliding_window=config.use_sliding_window,
        max_seq_len=config.decoder_max_seq_len,
        ff_mult=config.ff_mult,
        dropout=0.0,
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config
