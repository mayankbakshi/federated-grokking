"""Run either composition objective with the archived world or a small CPU check."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from composition.config import ModelConfig, TrainConfig, WorldConfig
from composition.data.composition import CompositionAdapter
from composition.train import train_run

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--objective", choices=["full_sequence", "answer_only"], required=True
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.steps is not None and args.steps < 1:
        ap.error("--steps must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        ap.error("Output directory is not empty; use a new directory or --resume")
    if args.resume and not (args.output / "latest.pt").exists():
        ap.error("--resume requires a checkpoint produced by this runner")
    args.output.mkdir(parents=True, exist_ok=True)
    saved_run = (
        "model-v0-n_entities_200"
        if args.objective == "full_sequence"
        else "model-v1-diag-phi18"
    )
    cfg = json.loads((ROOT / "data" / saved_run / "config.json").read_text())
    mcfg = ModelConfig(**cfg["model"])
    cfg["train"]["betas"] = tuple(cfg["train"]["betas"])
    tcfg = TrainConfig(**cfg["train"])
    adapter = CompositionAdapter()
    if args.smoke:
        wcfg = WorldConfig(n_entities=50, n_relations=50, out_degree=10)
        world = adapter.build_world(wcfg, root=args.output / "worlds")
        mcfg = ModelConfig(n_layer=2, n_embd=32, n_head=2)
        tcfg = replace(
            tcfg,
            phi=3.6,
            batch_size=16,
            warmup_steps=2,
            dense_every=2,
            dense_until=100,
            sparse_every=100,
            eval_cap=8,
            num_threads=2,
            num_workers=0,
        )
        default_steps = 6
    else:
        world = ROOT / "data/world"
        # These are the last saved observations, not the original configured caps.
        default_steps = 150000 if args.objective == "full_sequence" else 8000
    tcfg = replace(
        tcfg,
        max_steps=args.steps or default_steps,
        train_seed=args.seed,
        device=args.device,
        loss_mode=args.objective,
    )
    torch.set_num_threads(tcfg.num_threads)
    data = adapter.load(world, tcfg.phi, tcfg.eval_cap)
    train_run(
        f"composition-{args.objective}",
        data,
        mcfg,
        tcfg,
        resume=args.resume,
        out_dir=args.output,
    )
    print(f"Completed through step {tcfg.max_steps}: {args.output}")


if __name__ == "__main__":
    main()
