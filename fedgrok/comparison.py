"""Individual local-only learning and FedAvg accuracy on explicit evaluation domains."""

import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_curves(directory=None):
    directory = (
        Path(directory) if directory is not None else ROOT / "data/local_comparison"
    )
    fed, own, global_test = [], [], []
    fed_step = local_step = None
    for seed in range(20):
        folder = directory / f"seed{seed}"
        f = np.array(
            json.loads((folder / "fedavg_wd1.0_E10.json").read_text())["global"]
        )
        clients = [
            np.array(
                json.loads((folder / f"local_wd1.0_c{c}.json").read_text())["series"]
            )
            for c in range(4)
        ]
        if fed_step is None:
            fed_step, local_step = f[:, 0], clients[0][:, 0]
        np.testing.assert_array_equal(fed_step, f[:, 0])
        for client in clients:
            np.testing.assert_array_equal(local_step, client[:, 0])
        fed.append(f[:, 1])
        own.append(np.mean([c[:, 1] for c in clients], axis=0))
        global_test.append(np.mean([c[:, 2] for c in clients], axis=0))
    return {
        "fedavg": (fed_step, np.array(fed)),
        "local_own": (local_step, np.array(own)),
        "local_global": (local_step, np.array(global_test)),
    }


def check_overlap(curves, battery=None):
    battery = Path(battery) if battery is not None else ROOT / "data/battery"
    step, fed = curves["fedavg"]
    shared_counts = []
    for seed in range(20):
        lookup = dict(zip(step, fed[seed]))
        with np.load(battery / f"seed{seed}_fedavg.npz", allow_pickle=False) as run:
            shared = [
                (s, a) for s, a in zip(run["acc_step"], run["acc"]) if s in lookup
            ]
            assert shared, "No overlapping FedAvg observations"
            assert all(lookup[s] == a for s, a in shared), (
                f"FedAvg mismatch at seed {seed}"
            )
            shared_counts.append(len(shared))
    return shared_counts


def summary(curves, battery=None):
    result = {
        "seeds": 20,
        "clients_per_seed": 4,
        "local_aggregation": "equal-weight mean of individual client accuracies within seed, then median/IQR across seeds",
        "horizon": 100000,
        "display_min_step": 100,
        "shared_fedavg_observations_per_seed": check_overlap(curves, battery),
    }
    for key, (step, values) in curves.items():
        median = np.median(values, axis=0)
        result[key] = {
            "final_median_accuracy": float(median[-1]),
            "maximum_median_accuracy": float(median.max()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Retrain the local-only/FedAvg accuracy comparison without kernel measurements."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--n-steps", type=int, default=100000)
    parser.add_argument(
        "--output", type=Path, default=Path("output/comparison-training")
    )
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "data/local_comparison").resolve():
        parser.error(
            "Choose a new output directory; bundled reference data is immutable."
        )
    import torch
    from .training import run_regime

    torch.set_num_threads(1)
    for seed in args.seeds:
        folder = args.output / f"seed{seed}"
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir(parents=True)
        for mode in ["local", "fedavg"]:
            run = run_regime(
                seed,
                mode,
                n_steps=args.n_steps,
                diagnostics=False,
                record_individual=(mode == "local"),
            )
            if mode == "fedavg":
                record = {
                    "seed": seed,
                    "component": mode,
                    "wd": 1.0,
                    "E": 10,
                    "n_steps": args.n_steps,
                    "global": np.column_stack([run["acc_step"], run["acc"]]).tolist(),
                }
                (folder / "fedavg_wd1.0_E10.json").write_text(json.dumps(record))
            else:
                for client in range(4):
                    record = {
                        "seed": seed,
                        "component": mode,
                        "wd": 1.0,
                        "client": client,
                        "n_steps": args.n_steps,
                        "series": np.column_stack(
                            [
                                run["individual_step"],
                                run["individual_local"][:, client],
                                run["individual_global"][:, client],
                            ]
                        ).tolist(),
                    }
                    (folder / f"local_wd1.0_c{client}.json").write_text(
                        json.dumps(record)
                    )
            print(f"seed {seed}: {mode} complete", flush=True)


if __name__ == "__main__":
    main()
