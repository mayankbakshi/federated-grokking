"""Reproduce the three-panel figure from the saved 60-run battery."""

import argparse
import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .analysis import analyze
from .comparison import load_curves, summary as comparison_summary

ROOT = Path(__file__).resolve().parents[1]


def build(output, data=None, comparison_data=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    DATA = Path(data) if data is not None else ROOT / "data/battery"
    comparison = load_curves(comparison_data)
    (output / "comparison.json").write_text(
        json.dumps(comparison_summary(comparison, DATA), indent=2) + "\n"
    )
    result = analyze(DATA, list(range(20)))
    summary = {
        k: result[k]
        for k in (
            "n",
            "npos",
            "mean_lead",
            "ci_lo",
            "ci_hi",
            "p_sign",
            "leak_loc_n",
            "leak_cen_n",
            "c1",
            "c2",
            "c3",
            "passed",
            "rows",
        )
    }

    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    (output / "statistics.json").write_text(
        json.dumps(summary, indent=2, default=encode)
    )
    with (output / "per_seed.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])

    def load(seed, mode):
        with np.load(DATA / f"seed{seed}_{mode}.npz", allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def stack(mode, value_key, time_key):
        runs = [load(s, mode) for s in range(20)]
        x = runs[0][time_key]
        # Require the same checkpoint grid across seeds.
        assert all(np.array_equal(x, r[time_key]) for r in runs)
        return x, np.stack([r[value_key] for r in runs])

    TEAL = "#087f83"
    INK = "#182b38"
    GRAY = "#78818a"
    ORANGE = "#c26a35"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#b5bec5",
            "xtick.color": "#475560",
            "ytick.color": "#475560",
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(8.25, 4.3), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.12, 1],
        height_ratios=[1, 1],
        left=0.083,
        right=0.98,
        top=0.89,
        bottom=0.14,
        wspace=0.32,
        hspace=0.70,
    )
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_marker = fig.add_subplot(gs[1, 0])
    ax_lead = fig.add_subplot(gs[:, 1])

    def ribbon(ax, mode, key, time_key, color, label):
        t, a = stack(mode, key, time_key)
        q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(t, q1, q3, color=color, alpha=0.12, linewidth=0)
        ax.plot(t, med, color=color, lw=1.8, label=label)

    for key, color, label, linestyle in [
        ("fedavg", TEAL, "Averaging: global", "-"),
        ("local_own", ORANGE, "Local: own band", "-"),
        ("local_global", "#7462a8", "Local: global", "--"),
    ]:
        t, a = comparison[key]
        q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75], axis=0)
        shown = t >= 100
        ax_acc.fill_between(
            t[shown], q1[shown], q3[shown], color=color, alpha=0.12, linewidth=0
        )
        ax_acc.plot(
            t[shown], med[shown], color=color, lw=1.5, ls=linestyle, label=label
        )
    ax_acc.axhline(0.9, color=GRAY, lw=0.8, ls="--", label="90% threshold")
    ax_acc.set(
        title="A  Averaging vs local-only training",
        ylabel="Test accuracy",
        ylim=(-0.04, 1.07),
        xscale="log",
        xlim=(100, 100000),
        xlabel="Training step (log scale)",
    )
    ax_acc.xaxis.label.set_fontsize(8)
    ax_acc.set_xticks([100, 1000, 10000, 100000], ["100", "1k", "10k", "100k"])
    ax_acc.set_yticks([0, 0.5, 1], ["0", "0.5", "1.0"])
    ax_acc.legend(
        loc="upper left",
        fontsize=6.7,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.95,
        handlelength=1.4,
        labelspacing=0.2,
        borderaxespad=0.3,
    )

    for mode, c, label in [
        ("fedavg", TEAL, "Model averaging"),
        ("central", GRAY, "Pooled"),
        ("local", ORANGE, "No-share mean"),
    ]:
        ribbon(ax_marker, mode, "A_out_backbone", "step", c, label)
    ax_marker.axhline(0.02, color="#65727b", lw=0.8, ls="--", label="Threshold 0.02")
    ax_marker.set(
        title="B  Sketched cross-client marker",
        ylabel="Alignment",
        xlabel="Training step",
        xlim=(0, 10000),
        ylim=(-0.01, 0.17),
    )
    ax_marker.set_xticks(
        [0, 2500, 5000, 7500, 10000], ["0", "2.5k", "5k", "7.5k", "10k"]
    )
    ax_marker.set_yticks([0, 0.05, 0.1, 0.15], ["0", ".05", ".10", ".15"])
    ax_marker.legend(
        loc="upper left",
        fontsize=7.1,
        frameon=False,
        ncol=1,
        handlelength=1.4,
        labelspacing=0.25,
        borderaxespad=0.3,
    )

    ordered = sorted(result["rows"], key=lambda r: r["lead"])
    leads = np.array([r["lead"] for r in ordered])
    ax_lead.bar(
        np.arange(20),
        leads,
        color=[TEAL if v > 0 else ORANGE for v in leads],
        width=0.78,
    )
    ax_lead.axhline(0, color=GRAY, lw=0.8)
    ax_lead.axhline(result["mean_lead"], color=INK, ls="--", lw=1)
    ax_lead.set(
        title="C  Lead to sustained 90% accuracy",
        ylabel="90% accuracy onset - marker onset (steps)",
        xlabel="20 runs, ordered by lead",
        ylim=(-150, 1700),
        xlim=(-0.8, 19.8),
    )
    ax_lead.set_xticks([])
    # Label the two negative leads, which are small at this scale.
    for i, value in enumerate(leads):
        if value <= 0:
            ax_lead.scatter(
                [i],
                [value],
                s=16,
                facecolors="white",
                edgecolors=ORANGE,
                linewidths=1,
                zorder=4,
            )
            ax_lead.annotate(
                str(int(value)),
                xy=(i, value),
                xytext=(1.6 if i == 0 else 3.7, 160 if i == 0 else 70),
                textcoords="data",
                fontsize=7.4,
                color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.7),
            )
    ax_lead.set_yticks([0, 500, 1000, 1500])
    ax_lead.text(
        0.03,
        0.96,
        f"{result['npos']}/{result['n']} positive leads\nMean {result['mean_lead']:.0f} steps\n95% bootstrap CI [{result['ci_lo']:.0f}, {result['ci_hi']:.0f}]",
        transform=ax_lead.transAxes,
        va="top",
        fontsize=9,
        linespacing=1.4,
    )
    for ax in [ax_acc, ax_marker, ax_lead]:
        ax.grid(axis="y", alpha=0.13)
        ax.set_axisbelow(True)
    fig.savefig(output / "figure.png", dpi=300)
    fig.savefig(output / "figure.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--comparison-data", type=Path, default=None)
    args = parser.parse_args()
    build(args.output, args.data, args.comparison_data)
    print(f"Wrote figure and statistics to {args.output}")


if __name__ == "__main__":
    main()
