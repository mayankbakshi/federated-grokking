"""Check the first full-size diagnostic and early accuracy against saved seed-0 runs."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .training import run_regime

ROOT = Path(__file__).resolve().parents[1]


def check(output):
    torch.set_num_threads(1)
    report = []
    for mode in ["central", "local", "fedavg"]:
        actual = run_regime(
            0, mode, n_steps=20, E=10, round_every=100, per_band=64, projections=8
        )
        with np.load(
            ROOT / "data/battery" / f"seed0_{mode}.npz", allow_pickle=False
        ) as saved:
            metric_differences = {}
            for key in ["A_backbone", "A_in_backbone", "A_out_backbone"]:
                reference = saved[key][: len(actual[key])]
                metric_differences[key] = float(np.max(np.abs(actual[key] - reference)))
                np.testing.assert_array_equal(actual[key], reference)
            shared = sorted(set(actual["acc_step"]).intersection(saved["acc_step"]))
            new_acc, old_acc = (
                dict(zip(actual["acc_step"], actual["acc"])),
                dict(zip(saved["acc_step"], saved["acc"])),
            )
            differences = [abs(float(new_acc[s]) - float(old_acc[s])) for s in shared]
            assert max(differences) == 0, f"{mode}: early accuracy differs"
            report.append(
                {
                    "regime": mode,
                    "seed": 0,
                    "recorded_steps_checked": [0, 20],
                    "kernel_checkpoints_checked": [0],
                    "probe_size": 256,
                    "projections": 8,
                    "kernel_max_abs_differences": metric_differences,
                    "shared_accuracy_steps": [int(s) for s in shared],
                    "accuracy_max_abs_difference": max(differences),
                }
            )
        print(f"{mode}: first diagnostic and early accuracy match exactly", flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("output/training-check.json")
    )
    check(parser.parse_args().output)


if __name__ == "__main__":
    main()
