"""Verify the saved evidence and regenerate the composition learning curves."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FULL = "model-v0-n_entities_200"
ANSWER = "model-v1-diag-phi18"


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    steps = [row["step"] for row in rows]
    if steps != sorted(set(steps)):
        raise ValueError(f"Expected unique, increasing steps: {path}")
    for row in rows:
        if not all(0 <= value <= 1 for value in row["acc"].values()):
            raise ValueError(f"Invalid accuracy: {path}")
    return rows


def load_evidence(root=ROOT):
    manifest = json.loads((root / "data/SHA256.json").read_text())
    for name, expected in manifest.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Evidence checksum mismatch: {name}")
    full = read_rows(root / "data" / FULL / "metrics.jsonl")
    answer = read_rows(root / "data" / ANSWER / "metrics.jsonl")
    rescored = read_rows(root / "data" / FULL / "rescore.jsonl")
    original = {row["step"]: row for row in full}
    for row in rescored:
        if row["acc"] != original[row["step"]]["acc"]:
            raise ValueError(f"Rescore differs at step {row['step']}")
    return full, answer, rescored


def summarize(full, answer, rescored):
    def describe(rows):
        keys = list(rows[0]["acc"])
        return {
            "observations": len(rows),
            "last_step": rows[-1]["step"],
            "final_accuracy": rows[-1]["acc"],
            "first_observed_at_or_above_99_percent": {
                key: next(
                    (row["step"] for row in rows if row["acc"][key] >= 0.99), None
                )
                for key in keys
            },
        }

    return {
        "full_sequence": describe(full),
        "answer_only": describe(answer),
        "rescored_checkpoints": len(rescored),
        "rescore_matches_original_accuracy_exactly": True,
        "full_sequence_rescored_endpoint": rescored[-1],
        "interpretation": "A short observed training/generalization lag in the full-sequence run; no pronounced delayed transition established in either run.",
    }


def plot(rescored, answer, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), sharex=True, sharey=True)
    composition = [
        ("train_inferred", "Training compositions (sampled)", "#243b53", "-"),
        ("test_inferred_id", "Held-out ID compositions", "#008a91", "-"),
        ("inferred_ood", "Held-out OOD compositions", "#b46432", "--"),
    ]
    atomic = [
        ("id_atomic", "ID atomic facts (sampled)", "#6e4b9e", "-"),
        ("ood_atomic", "OOD-designated atomic facts", "#708b26", "--"),
    ]
    for col, (rows, title) in enumerate(
        [(rescored, "Full-sequence objective"), (answer, "Answer-only objective")]
    ):
        for row_index, series in enumerate([composition, atomic]):
            ax = axes[row_index, col]
            for key, label, color, style in series:
                ax.plot(
                    [r["step"] for r in rows],
                    [r["acc"][key] for r in rows],
                    label=label,
                    color=color,
                    linestyle=style,
                    marker="o",
                    markersize=3,
                    linewidth=1.8,
                )
            ax.set_xlim(0, 12000)
            ax.set_ylim(-0.035, 1.055)
            ax.set_xticks([0, 4000, 8000, 12000], ["0", "4k", "8k", "12k"])
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
            ax.grid(alpha=0.17)
            if col == 0:
                ax.set_ylabel(
                    "Composition accuracy" if row_index == 0 else "Atomic-fact accuracy"
                )
            if row_index == 0:
                ax.set_title(title, fontsize=12, loc="left", pad=10)
            else:
                ax.set_xlabel("Optimizer steps")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, top=0.87, bottom=0.12, hspace=0.22, wspace=0.13
    )
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.svg")
    svg = output / "learning_curves.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "output/report")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    full, answer, rescored = load_evidence()
    summary = summarize(full, answer, rescored)
    reference = ROOT / "reference/summary.json"
    if reference.exists() and summary != json.loads(reference.read_text()):
        raise ValueError("Computed summary differs from the released reference")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "accuracy.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "evaluation", "step", "split", "accuracy"])
        for run, evaluation, rows in [
            (FULL, "original", full),
            (FULL, "bounded_decode_rescore", rescored),
            (ANSWER, "bounded_decode", answer),
        ]:
            for row in rows:
                for split, accuracy in row["acc"].items():
                    writer.writerow([run, evaluation, row["step"], split, accuracy])
    if not args.no_plot:
        plot(rescored, answer, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
