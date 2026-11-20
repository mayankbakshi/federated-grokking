"""Bounded greedy decoding and saved-checkpoint rescoring."""

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import torch
from .data.adapter import Example, TaskData
from .model import build_model


def greedy_continuations(
    model, examples: list[Example], pad_id: int, batch_size: int, device
) -> list[list[int]]:
    end_id = examples[0].full_ids[-1]
    max_len = model.config.n_positions
    out: list = [None] * len(examples)
    by_len: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        by_len.setdefault(len(ex.input_ids), []).append(i)
    with torch.no_grad():
        for length, idxs in sorted(by_len.items()):
            for start in range(0, len(idxs), batch_size):
                chunk = idxs[start : start + batch_size]
                ids = torch.tensor(
                    [examples[i].input_ids for i in chunk], device=device
                )
                conts = [[] for _ in chunk]
                finished = [False] * len(chunk)
                max_answer = max((len(examples[i].answer_ids) for i in chunk))
                while ids.shape[1] < max_len and (not all(finished)):
                    logits = model(input_ids=ids).logits[:, -1, :]
                    nxt = logits.argmax(dim=-1)
                    for j, tok in enumerate(nxt.tolist()):
                        if finished[j]:
                            continue
                        if tok == end_id:
                            finished[j] = True
                        else:
                            conts[j].append(tok)
                            ex = examples[chunk[j]]
                            if len(conts[j]) == len(ex.answer_ids):
                                finished[j] = True
                    ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
                    if max((len(c) for c in conts)) >= max_answer:
                        for j, i in enumerate(chunk):
                            if len(conts[j]) >= len(examples[i].answer_ids):
                                finished[j] = True
                for j, i in enumerate(chunk):
                    out[i] = conts[j]
    return out


def eval_accuracy(
    model, split: list[Example], pad_id: int, batch_size: int, device
) -> float:
    if not split:
        return float("nan")
    conts = greedy_continuations(model, split, pad_id, batch_size, device)
    hits = sum((c == ex.answer_ids for c, ex in zip(conts, split)))
    return hits / len(split)


def _checkpoint_steps(run_dir: Path) -> list[int]:
    ckpt_dir = run_dir / "checkpoints"
    return sorted(
        (
            int(m.group(1))
            for p in ckpt_dir.glob("step*.pt")
            if (m := re.fullmatch("step(\\d+)\\.pt", p.name))
        )
    )


def rescore_checkpoints(
    run_dir: Path,
    data: TaskData,
    mcfg,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
    logger=None,
) -> Path:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    run_name = cfg.get("run_name", run_dir.name)
    source_loss_mode = cfg.get("train", {}).get("loss_mode")
    train_seed = cfg.get("train", {}).get("train_seed", 0)
    steps = _checkpoint_steps(run_dir)
    if not steps:
        raise ValueError(
            f"no step*.pt checkpoints found under {run_dir / 'checkpoints'}"
        )
    rescore_path = out_dir / "rescore.jsonl"
    with rescore_path.open("w") as f:
        for done, step in enumerate(steps, start=1):
            model = build_model(len(data.vocab), mcfg, train_seed)
            sd = torch.load(
                run_dir / "checkpoints" / f"step{step}.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(sd)
            model.to(device).eval()
            accs = {
                name: eval_accuracy(model, split, data.pad_id, batch_size, device)
                for name, split in data.eval_splits.items()
            }
            rec = {
                "step": step,
                "acc": accs,
                "eval_rule": "bounded_decode",
                "source_run": run_name,
                "source_loss_mode": source_loss_mode,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if logger is not None:
                logger.log(rec, step=step)
                logger.heartbeat(done=done, total=len(steps))
    config_payload = {
        "source_run": run_name,
        "source_run_dir": str(run_dir.resolve()),
        "source_loss_mode": source_loss_mode,
        "eval_rule": "bounded_decode",
        "steps": steps,
        "eval_cap": data.metadata.get("eval_cap"),
        "device": str(device),
        "model": asdict(mcfg) if hasattr(mcfg, "__dataclass_fields__") else str(mcfg),
        "rescored_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "rescore_config.json").write_text(json.dumps(config_payload, indent=2))
    return rescore_path
