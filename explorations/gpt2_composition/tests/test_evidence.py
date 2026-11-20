import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from report import load_evidence, summarize
from composition.config import ModelConfig, WorldConfig
from composition.data.composition import CompositionAdapter, _content_sha256
from composition.model import build_model
from composition.train import collate


def test_saved_evidence_and_rescore_agree():
    rows = load_evidence()
    observed = summarize(*rows)
    assert observed == json.loads((ROOT / "reference/summary.json").read_text())
    assert observed["rescored_checkpoints"] == 21


def test_world_regeneration_matches_archived_arrays(tmp_path):
    meta = json.loads((ROOT / "data/world/meta.json").read_text())
    generated = CompositionAdapter().build_world(
        WorldConfig(**meta["config"]), root=tmp_path
    )
    with (
        np.load(ROOT / "data/world/world.npz") as archived,
        np.load(generated / "world.npz") as fresh,
    ):
        assert set(archived.files) == set(fresh.files)
        for key in archived.files:
            np.testing.assert_array_equal(archived[key], fresh[key])
        assert _content_sha256(dict(archived)) == meta["sha256"]


@pytest.fixture(scope="module")
def data():
    return CompositionAdapter().load(ROOT / "data/world", phi=18.0, eval_cap=3000)


def test_composition_splits_and_atomic_exposure(data):
    training = {ex.fact for ex in data.train if ex.kind == "inferred"}
    held_out = {ex.fact for ex in data.eval_splits["test_inferred_id"]}
    ood = {ex.fact for ex in data.eval_splits["inferred_ood"]}
    assert training.isdisjoint(held_out)
    assert training.isdisjoint(ood)
    assert len(data.train) == 181000
    assert len(data.eval_splits["test_inferred_id"]) == 932
    assert len(data.eval_splits["inferred_ood"]) == 498
    atomic = {ex.fact for ex in data.train if ex.kind == "atomic"}
    assert all(ex.fact in atomic for ex in data.eval_splits["ood_atomic"])


def test_answer_objective_equals_answer_token_log_loss(data):
    batch = [data.train[0], data.train[-1]]
    packed = collate(batch, data.pad_id, "answer_only")
    model = build_model(
        len(data.vocab), ModelConfig(n_layer=1, n_embd=16, n_head=2), seed=42
    ).eval()
    with torch.no_grad():
        prediction = model(**packed)
        expected = torch.stack(
            [
                -torch.log_softmax(prediction.logits[i, len(ex.input_ids) - 1], dim=-1)[
                    ex.answer_ids[0]
                ]
                for i, ex in enumerate(batch)
            ]
        ).mean()
    assert prediction.loss.item() == pytest.approx(expected.item(), abs=1e-5)
    assert (packed["labels"] != -100).sum().item() == len(batch)


def test_full_sequence_includes_prefix_targets(data):
    batch = [data.train[0], data.train[-1]]
    full = collate(batch, data.pad_id, "full_sequence")["labels"]
    answer = collate(batch, data.pad_id, "answer_only")["labels"]
    assert (full != -100).sum().item() > (answer != -100).sum().item()
    for i, ex in enumerate(batch):
        assert full[i, : len(ex.full_ids)].tolist() == ex.full_ids
