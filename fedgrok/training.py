"""Train the modular-division regimes and measure the fixed-probe diagnostic."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .config import CONFIG
from .data import (
    generate_all_pairs,
    partition_type_B,
    split_local_train_test,
    to_tensors,
)
from .model import ModularTransformer
from .kernels import (
    backbone_readout_masks,
    randproj_gram,
    label_gram,
    centered_alignment,
    block_alignment,
)


def evaluate(model, x, y, z, batch_size: int = 1024):
    """Batched held-out accuracy and mean cross-entropy."""
    model.eval()
    total_correct, total_loss, N = (0, 0.0, len(x))
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb, yb, zb = (
                x[i : i + batch_size],
                y[i : i + batch_size],
                z[i : i + batch_size],
            )
            logits = model(xb, yb)
            loss = F.cross_entropy(logits, zb, reduction="sum")
            pred = logits.argmax(dim=-1)
            total_correct += (pred == zb).sum().item()
            total_loss += loss.item()
    model.train()
    return (total_correct / N, total_loss / N)


def make_eval_schedule(n_steps, E=None, log_per_decade=None, dense_bands=None):
    """Use the study evaluation grid: logarithmic sampling plus dense transition bands; snap federated/local evaluation to averaging steps."""
    import math

    cfg = CONFIG
    if log_per_decade is None:
        log_per_decade = cfg.eval_log_per_decade
    if dense_bands is None:
        dense_bands = cfg.eval_dense_bands
    steps = {0, int(n_steps)}
    n_log = max(2, int(log_per_decade * math.log10(max(n_steps, 10))))
    for s in np.geomspace(1, n_steps, n_log):
        steps.add(int(round(s)))
    for lo, hi, st in dense_bands:
        for s in range(int(lo), min(int(hi), n_steps) + 1, int(st)):
            steps.add(s)
    if E is not None:
        steps = {min(int(round(s / E)) * E, n_steps) for s in steps}
        steps.add(0)
        if n_steps % E == 0:
            steps.add(n_steps)
    steps = {s for s in steps if 0 <= s <= n_steps}
    return frozenset(sorted(steps))


def build_model():
    c = CONFIG
    return ModularTransformer(c.p, c.d_model, c.n_heads, c.n_layers, c.d_ff)


def build_client_data(seed):
    bands, _ = partition_type_B(generate_all_pairs(CONFIG.p), CONFIG.V, CONFIG.p)
    clients, held, pooled = [], [], []
    for c, band in enumerate(bands):
        train, test = split_local_train_test(band, CONFIG.alpha, seed * 100 + c)
        clients.append(
            {"train": to_tensors(train, "cpu"), "test": to_tensors(test, "cpu")}
        )
        held.append(test)
        pooled.append(train)
    return (
        clients,
        to_tensors(np.concatenate(held), "cpu"),
        to_tensors(np.concatenate(pooled), "cpu"),
    )


def build_probe(seed, per_band=64):
    bands, _ = partition_type_B(generate_all_pairs(CONFIG.p), CONFIG.V, CONFIG.p)
    rng = np.random.RandomState(10000 + seed)
    selected = np.concatenate(
        [band[rng.choice(len(band), size=per_band, replace=False)] for band in bands]
    )
    x, y, z = to_tensors(selected, "cpu")
    return {
        "x": x,
        "y": y,
        "z": z.numpy(),
        "bands": np.repeat(np.arange(CONFIG.V), per_band),
    }


def average(states):
    return {
        key: torch.stack([state[key].float() for state in states])
        .mean(0)
        .to(states[0][key].dtype)
        for key in states[0]
    }


def measure(scratch, state, probe, projections=8):
    mask = backbone_readout_masks(scratch)["readout"]
    kernel = randproj_gram(
        scratch,
        state,
        probe["x"],
        probe["y"],
        mask,
        R=projections,
        proj_seed=0,
        method="vmap",
        row_chunk=64,
    )["backbone"]
    labels = label_gram(probe["z"], CONFIG.p)
    return {
        "A_backbone": centered_alignment(kernel, labels),
        "A_in_backbone": block_alignment(kernel, labels, probe["bands"], "in"),
        "A_out_backbone": block_alignment(kernel, labels, probe["bands"], "out"),
    }


def run_regime(
    seed,
    mode,
    n_steps=10000,
    E=10,
    round_every=100,
    per_band=64,
    projections=8,
    diagnostics=True,
    record_individual=False,
):
    if mode not in {"central", "local", "fedavg"}:
        raise ValueError(mode)
    if n_steps < 1 or E < 1 or round_every < 1 or round_every % E:
        raise ValueError(
            "Positive steps/intervals required; round_every must be divisible by E."
        )
    clients, held, pooled = build_client_data(seed)
    probe = build_probe(seed, per_band)
    if mode == "local":
        models = []
        for client in range(CONFIG.V):
            torch.manual_seed(seed * 100 + client)
            models.append(build_model())
    else:
        torch.manual_seed(seed)
        initial = build_model()
        if mode == "central":
            models = [initial]
        else:
            models = []
            for _ in range(CONFIG.V):
                model = build_model()
                model.load_state_dict(initial.state_dict())
                models.append(model)
    opts = [
        torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG.lr,
            weight_decay=CONFIG.weight_decay,
            betas=(CONFIG.beta1, CONFIG.beta2),
        )
        for model in models
    ]
    evaluator = build_model()
    scratch = build_model()
    schedule = make_eval_schedule(n_steps, E=None if mode == "central" else E)
    records, accuracy = [], []
    individual = []
    individual_schedule = make_eval_schedule(n_steps) if record_individual else set()
    for model in models:
        model.train()
    # Preserve the recorded indexing convention: step 0 is after the first optimizer update.
    for step in range(n_steps + 1):
        for client, (model, opt) in enumerate(zip(models, opts)):
            x, y, z = pooled if mode == "central" else clients[client]["train"]
            loss = F.cross_entropy(model(x, y), z)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if mode == "fedavg" and step % E == 0:
            state = average([model.state_dict() for model in models])
            for model in models:
                model.load_state_dict(state)
        if (diagnostics and step % round_every == 0) or step in schedule:
            state = (
                models[0].state_dict()
                if mode == "central"
                else average([m.state_dict() for m in models])
            )
            if diagnostics and step % round_every == 0:
                records.append(
                    {"step": step, **measure(scratch, state, probe, projections)}
                )
            if step in schedule:
                evaluator.load_state_dict(state)
                acc, _ = evaluate(evaluator, *held)
                accuracy.append((step, acc))
        if step in individual_schedule:
            own = [
                evaluate(model, *clients[c]["test"])[0]
                for c, model in enumerate(models)
            ]
            global_test = [evaluate(model, *held)[0] for model in models]
            individual.append((step, own, global_test))
    arrays = (
        {key: np.array([row[key] for row in records]) for key in records[0]}
        if records
        else {"step": np.array([], dtype=int)}
    )
    if record_individual:
        arrays.update(
            individual_step=np.array([s for s, _, _ in individual], dtype=int),
            individual_local=np.array([a for _, a, _ in individual]),
            individual_global=np.array([a for _, _, a in individual]),
        )
    arrays.update(
        acc_step=np.array([s for s, _ in accuracy], dtype=int),
        acc=np.array([a for _, a in accuracy]),
        seed=seed,
        mode=mode,
        E=E,
        V=CONFIG.V,
        n_steps=n_steps,
        grok_step=-1,
    )
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument(
        "--mode", choices=["central", "local", "fedavg", "all"], default="all"
    )
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--E", type=int, default=10)
    parser.add_argument("--round-every", type=int, default=100)
    parser.add_argument("--per-band", type=int, default=64)
    parser.add_argument("--projections", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("output/training"))
    args = parser.parse_args()
    if (
        args.output.resolve()
        == (Path(__file__).resolve().parents[1] / "data/battery").resolve()
    ):
        parser.error(
            "Choose a new output directory; bundled reference data is immutable."
        )
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        **vars(args),
        "output": str(args.output),
        "model": CONFIG.as_dict(),
        "torch": torch.__version__,
    }
    (args.output / "configuration.json").write_text(json.dumps(settings, indent=2))
    modes = ["central", "local", "fedavg"] if args.mode == "all" else [args.mode]
    for seed in args.seeds:
        for mode in modes:
            path = args.output / f"seed{seed}_{mode}.npz"
            if path.exists():
                raise FileExistsError(path)
            arrays = run_regime(
                seed,
                mode,
                args.n_steps,
                args.E,
                args.round_every,
                args.per_band,
                args.projections,
            )
            np.savez(path, **arrays)
            print(path, flush=True)


if __name__ == "__main__":
    main()
