"""Checks of the experimental definitions and kernel calculations."""

from pathlib import Path
import json
import numpy as np
import pytest
import torch
from fedgrok.analysis import analyze, persist_cross
from fedgrok.data import generate_all_pairs
from fedgrok.kernels import (
    backbone_readout_masks,
    block_alignment,
    full_jacobian,
    randproj_gram,
    trace_gram,
)
from fedgrok.model import ModularTransformer
from fedgrok.training import (
    build_client_data,
    build_probe,
    make_eval_schedule,
    run_regime,
)
from fedgrok.comparison import load_curves, check_overlap

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_collective_train_and_test_are_disjoint_and_complete():
    clients, held, pooled = build_client_data(7)

    def triples(tensors):
        return set(zip(*(x.tolist() for x in tensors)))

    train, test = triples(pooled), triples(held)
    assert len(train) == len(test) == 4656
    assert not train.intersection(test)
    assert train.union(test) == set(map(tuple, generate_all_pairs()))
    assert set.union(*(triples(client["train"]) for client in clients)) == train


def test_probe_is_fixed_balanced_and_independent_of_global_rng():
    torch.manual_seed(71)
    before = torch.get_rng_state().clone()
    first = build_probe(7)
    second = build_probe(7)
    assert torch.equal(before, torch.get_rng_state())
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(np.bincount(first["bands"]), [64] * 4)
    _, held, pooled = build_client_data(7)
    probe = set(zip(first["x"].tolist(), first["y"].tolist()))
    assert probe.intersection(zip(pooled[0].tolist(), pooled[1].tolist()))
    assert probe.intersection(zip(held[0].tolist(), held[1].tolist()))


def test_persistence_rejects_a_dip_and_an_incomplete_tail():
    step = [0, 100, 200, 300, 400, 500]
    assert persist_cross(step, [0, 1, 1, 0, 1, 1], 0.9) is None
    assert persist_cross(step, [0, 1, 1, 1, 1, 0], 0.9) == 100
    assert persist_cross(step, [1, 1, float("nan"), 1, 1, 1], 0.9) is None


def test_accuracy_grids_equal_the_saved_experiment_grids():
    for mode in ["central", "local", "fedavg"]:
        expected = np.array(
            sorted(make_eval_schedule(10000, E=None if mode == "central" else 10))
        )
        with np.load(
            ROOT / "data/battery" / f"seed0_{mode}.npz", allow_pickle=False
        ) as run:
            np.testing.assert_array_equal(run["acc_step"], expected)
            np.testing.assert_array_equal(run["step"], np.arange(0, 10001, 100))


def test_exact_trace_equals_full_jacobian_contraction():
    torch.manual_seed(4)
    model = ModularTransformer(p=5, d_model=8, n_heads=2, n_layers=1, d_ff=16)
    state = model.state_dict()
    x, y = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
    jac = full_jacobian(model, state, x, y)
    direct = np.einsum("ncd,mcd->nm", jac, jac)
    exact = trace_gram(model, state, x, y, method="loop")
    np.testing.assert_allclose(exact, direct, rtol=2e-5, atol=2e-5)


def test_kernel_backend_agreement_and_parameter_partition():
    torch.manual_seed(5)
    model = ModularTransformer(p=5, d_model=8, n_heads=2, n_layers=1, d_ff=16)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    masks = backbone_readout_masks(model)
    assert not np.any(masks["backbone"] & masks["readout"])
    assert np.all(masks["backbone"] | masks["readout"])
    x, y = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
    before = torch.get_rng_state().clone()
    batched = randproj_gram(model, state, x, y, masks["readout"], R=3, method="vmap")
    loop = randproj_gram(model, state, x, y, masks["readout"], R=3, method="loop")
    assert torch.equal(before, torch.get_rng_state())
    for key in batched:
        np.testing.assert_allclose(batched[key], loop[key], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(batched["all"], batched["backbone"] + batched["readout"])
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])


def test_alignment_handles_degenerate_cross_band_sets():
    kernel = np.eye(4)
    labels = np.eye(4)
    assert np.isnan(block_alignment(kernel, labels, np.zeros(4), "out"))
    assert np.isclose(block_alignment(kernel, labels, [0, 0, 1, 1], "out"), 1.0)


def test_saved_result_retains_failed_control_and_negative_runs():
    result = analyze(ROOT / "data/battery", list(range(20)))
    assert result["n"] == 20 and result["npos"] == 18
    assert result["leak_cen_n"] == 17 and result["leak_loc_n"] == 0
    assert result["mean_lead"] == 529.5 and not result["passed"]
    assert sorted(row["lead"] for row in result["rows"] if row["lead"] <= 0) == [
        -100.0,
        -10.0,
    ]


def test_three_training_regimes_execute_and_are_repeatable():
    for mode in ["central", "local", "fedavg"]:
        first = run_regime(
            0, mode, n_steps=10, round_every=10, per_band=1, projections=1
        )
        second = run_regime(
            0, mode, n_steps=10, round_every=10, per_band=1, projections=1
        )
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        assert np.all(np.isfinite(first["A_out_backbone"]))
        assert np.all((first["acc"] >= 0) & (first["acc"] <= 1))


def test_comparison_averages_individual_clients_and_matches_diagnostic_runs():
    curves = load_curves()
    assert len(check_overlap(curves)) == 20
    clients = [
        np.array(
            json.loads(
                (
                    ROOT / "data/local_comparison/seed0" / f"local_wd1.0_c{c}.json"
                ).read_text()
            )["series"]
        )
        for c in range(4)
    ]
    np.testing.assert_array_equal(
        curves["local_own"][1][0], np.mean([c[:, 1] for c in clients], axis=0)
    )
    np.testing.assert_array_equal(
        curves["local_global"][1][0], np.mean([c[:, 2] for c in clients], axis=0)
    )
    assert curves["local_own"][0][-1] == 100000


def test_individual_training_matches_saved_early_accuracy():
    actual = run_regime(
        0, "local", n_steps=10, diagnostics=False, record_individual=True
    )
    for client in range(4):
        saved = json.loads(
            (
                ROOT / "data/local_comparison/seed0" / f"local_wd1.0_c{client}.json"
            ).read_text()
        )["series"]
        lookup = {row[0]: row[1:] for row in saved}
        for i, step in enumerate(actual["individual_step"]):
            np.testing.assert_array_equal(
                [
                    actual["individual_local"][i, client],
                    actual["individual_global"][i, client],
                ],
                lookup[step],
            )


def test_lead_summary_rejects_censored_runs(monkeypatch):
    from fedgrok import analysis

    steps = np.array([0, 100, 200, 300])
    for missing_onset in ["accuracy", "marker"]:

        def load_run(directory, seed, mode):
            return {
                "step": steps,
                "acc_step": steps,
                "acc": np.full(4, 0.5 if missing_onset == "accuracy" else 0.95),
                "A_out_backbone": np.full(
                    4, 0.0 if mode == "local" or missing_onset == "marker" else 0.03
                ),
                "grok_step": -1,
            }

        monkeypatch.setattr(analysis, "_load", load_run)
        with pytest.raises(ValueError, match="censored seeds: "):
            analysis.analyze(Path("unused"), [0])
