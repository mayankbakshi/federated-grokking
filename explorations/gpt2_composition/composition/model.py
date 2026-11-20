"""Fresh GPT-2-style decoder with a task-specific vocabulary."""

from contextlib import nullcontext
import torch
from transformers import GPT2Config, GPT2LMHeadModel


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    return torch.device(spec)


def build_model(vocab_size: int, cfg, seed: int) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    gcfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=cfg.n_positions,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        resid_pdrop=cfg.dropout,
        embd_pdrop=cfg.dropout,
        attn_pdrop=cfg.dropout,
        bos_token_id=None,
        eos_token_id=None,
    )
    return GPT2LMHeadModel(gcfg)


def precision_context(precision: str, device: torch.device):
    if precision == "fp16_autocast":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if precision != "fp32":
        raise ValueError(f"unknown precision {precision!r}")
    return nullcontext()
