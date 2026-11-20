"""Seeded minibatch training with checkpoints and answer-level evaluation."""

import json
import os
import random
from dataclasses import asdict
from itertools import count
from pathlib import Path
import numpy as np
import torch
from . import paths
from .data.adapter import Example, TaskData
from .evaluate import eval_accuracy
from .model import build_model, precision_context, resolve_device


def checkpoint_schedule(t: int, cfg) -> bool:
    return (
        t <= cfg.dense_until
        and t % cfg.dense_every == 0
        or t % cfg.sparse_every == 0
        or t == cfg.max_steps
    )


def collate(batch: list[Example], pad_id: int, loss_mode: str) -> dict:
    width = max((len(ex.full_ids) for ex in batch))
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for row, ex in enumerate(batch):
        n = len(ex.full_ids)
        ids[row, :n] = torch.tensor(ex.full_ids)
        mask[row, :n] = 1
        if loss_mode == "full_sequence":
            labels[row, :n] = ids[row, :n]
        elif loss_mode == "answer_only":
            a = len(ex.input_ids)
            y = len(ex.answer_ids)
            assert n == a + y + 1
            assert ex.full_ids[a : a + y] == ex.answer_ids
            labels[row, a : a + y] = torch.tensor(ex.answer_ids)
        else:
            raise ValueError(f"unknown loss_mode {loss_mode!r}")
    if loss_mode == "answer_only":
        expected = sum((len(ex.answer_ids) for ex in batch))
        assert int((labels != -100).sum().item()) == expected
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def _epoch_seed(train_seed: int, epoch: int) -> int:
    return (train_seed * 1000003 + epoch) % 2**63


def _rng_states(device: torch.device) -> dict:
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "mps":
        states["torch_mps"] = torch.mps.get_rng_state()
    return states


def _restore_rng(states: dict, device: torch.device) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if device.type == "mps" and "torch_mps" in states:
        torch.mps.set_rng_state(states["torch_mps"])


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_run(
    run_name: str,
    data: TaskData,
    mcfg,
    tcfg,
    resume: bool = False,
    out_dir: "Path | None" = None,
    logger=None,
) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else paths.run_dir(run_name)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(tcfg.device)
    latest = out_dir / "latest.pt"
    metrics_path = out_dir / "metrics.jsonl"
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "task": data.metadata.get("task"),
                "world_dir": data.metadata.get("world_dir"),
                "phi": data.metadata.get("phi"),
                "vocab_size": len(data.vocab),
                "model": asdict(mcfg),
                "train": asdict(tcfg),
            },
            indent=2,
        )
    )
    random.seed(tcfg.train_seed)
    np.random.seed(tcfg.train_seed % 2**32)
    model = build_model(len(data.vocab), mcfg, tcfg.train_seed).to(device)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.lr,
        betas=tcfg.betas,
        eps=tcfg.eps,
        weight_decay=tcfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / tcfg.warmup_steps) if tcfg.warmup_steps > 0 else 1.0
    )
    step, start_epoch, start_batch = (0, 0, 0)
    if resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, start_epoch, start_batch = (ck["step"], ck["epoch"], ck["batch_idx"])
        _restore_rng(ck["rng"], device)
    if step >= tcfg.max_steps:
        return out_dir
    n = len(data.train)
    pad_id = data.pad_id
    done = False
    with metrics_path.open("a") as metrics_f:
        for epoch in count(start_epoch):
            g = torch.Generator().manual_seed(_epoch_seed(tcfg.train_seed, epoch))
            order = torch.randperm(n, generator=g)
            batches = [
                order[i : i + tcfg.batch_size] for i in range(0, n, tcfg.batch_size)
            ]
            first_batch = start_batch if epoch == start_epoch else 0
            if first_batch >= len(batches):
                continue
            for bidx in range(first_batch, len(batches)):
                step += 1
                if step > tcfg.max_steps:
                    done = True
                    break
                batch = [data.train[i] for i in batches[bidx].tolist()]
                t = collate(batch, pad_id, tcfg.loss_mode)
                with precision_context(tcfg.precision, device):
                    outp = model(
                        input_ids=t["input_ids"].to(device),
                        attention_mask=t["attention_mask"].to(device),
                        labels=t["labels"].to(device),
                    )
                    loss = outp.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if checkpoint_schedule(step, tcfg):
                    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    torch.save(sd, ckpt_dir / f"step{step}.pt")
                    model.eval()
                    accs = {
                        name: eval_accuracy(
                            model, split, pad_id, tcfg.batch_size, device
                        )
                        for name, split in data.eval_splits.items()
                    }
                    model.train()
                    rec = {
                        "step": step,
                        "loss": float(loss.detach()),
                        "lr": sched.get_last_lr()[0],
                        "acc": accs,
                    }
                    metrics_f.write(json.dumps(rec) + "\n")
                    metrics_f.flush()
                    if logger is not None:
                        logger.log(rec, step=step)
                if step % tcfg.dense_every == 0 or step == tcfg.max_steps:
                    _atomic_save(
                        {
                            "step": step,
                            "epoch": epoch,
                            "batch_idx": bidx + 1,
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "sched": sched.state_dict(),
                            "rng": _rng_states(device),
                            "run_name": run_name,
                        },
                        latest,
                    )
                if logger is not None:
                    logger.heartbeat(done=step, total=tcfg.max_steps)
            if done:
                break
    return out_dir
