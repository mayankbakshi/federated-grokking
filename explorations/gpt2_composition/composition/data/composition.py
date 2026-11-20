"""Seeded two-hop relational composition, following Wang et al. (2024)."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .. import paths
from ..config import WorldConfig
from .adapter import Example, TaskData

STRUCTURAL = ("<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>")
_SPLIT_SEEDS = {
    "id_atomic": 1,
    "ood_atomic": 2,
    "train_inferred": 3,
    "test_inferred_id": 4,
    "inferred_ood": 5,
}


def _phi_key(phi: float) -> int:
    return int(round(phi * 1000000))


def _content_sha256(arrays: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _world_id(cfg: WorldConfig) -> str:
    return f"E{cfg.n_entities}_R{cfg.n_relations}_deg{cfg.out_degree}_idf{cfg.atomic_id_frac:g}_res{cfg.id_test_reserve_p:g}_seed{cfg.data_seed}"


class CompositionAdapter:
    name = "composition"

    def build_world(self, cfg: WorldConfig, root: "Path | None" = None) -> Path:
        root = Path(root) if root is not None else paths.worlds_dir(self.name)
        world_dir = root / _world_id(cfg)
        meta_path = world_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["config"] != asdict(cfg):
                raise ValueError(f"{world_dir} exists with a different config")
            return world_dir
        E, R, deg = (cfg.n_entities, cfg.n_relations, cfg.out_degree)
        rng = np.random.default_rng(cfg.data_seed)
        rels = np.stack([rng.choice(R, size=deg, replace=False) for _ in range(E)])
        heads = np.repeat(np.arange(E), deg).astype(np.int64)
        rels = rels.reshape(-1).astype(np.int64)
        tails = rng.integers(0, E, size=E * deg).astype(np.int64)
        n = E * deg
        n_ood = int(round((1 - cfg.atomic_id_frac) * n))
        ood_mask = np.zeros(n, dtype=bool)
        ood_mask[rng.choice(n, size=n_ood, replace=False)] = True
        first = np.repeat(np.arange(n, dtype=np.int64), deg)
        second = tails.repeat(deg) * deg + np.tile(np.arange(deg, dtype=np.int64), n)
        both_id = ~ood_mask[first] & ~ood_mask[second]
        both_ood = ood_mask[first] & ood_mask[second]
        u = rng.random(len(first))
        te = both_id & (u < cfg.id_test_reserve_p)
        cand = both_id & ~te
        arrays = {
            "heads": heads,
            "rels": rels,
            "tails": tails,
            "ood_mask": ood_mask,
            "inferred_cand": np.stack([first[cand], second[cand]], axis=1),
            "inferred_te": np.stack([first[te], second[te]], axis=1),
            "inferred_ood": np.stack([first[both_ood], second[both_ood]], axis=1),
        }
        meta = {
            "task": self.name,
            "world_id": _world_id(cfg),
            "config": asdict(cfg),
            "counts": {
                "n_atomic": int(n),
                "n_atomic_id": int(n - n_ood),
                "n_atomic_ood": int(n_ood),
                "n_paths": int(len(first)),
                "n_inferred_cand": int(cand.sum()),
                "n_inferred_te": int(te.sum()),
                "n_inferred_ood": int(both_ood.sum()),
            },
            "vocab_size": E + R + len(STRUCTURAL),
            "sha256": _content_sha256(arrays),
        }
        world_dir.mkdir(parents=True, exist_ok=True)
        np.savez(world_dir / "world.npz", **arrays)
        meta_path.write_text(json.dumps(meta, indent=2))
        return world_dir

    def _read(self, world_dir: Path):
        world = np.load(Path(world_dir) / "world.npz")
        meta = json.loads((Path(world_dir) / "meta.json").read_text())
        return (world, meta)

    def _vocab(self, E: int, R: int) -> dict[str, int]:
        vocab = {f"e_{i}": i for i in range(E)}
        vocab.update({f"r_{j}": E + j for j in range(R)})
        vocab.update({tok: E + R + k for k, tok in enumerate(STRUCTURAL)})
        return vocab

    def _train_inferred_pairs(self, world, meta, phi: float) -> np.ndarray:
        cand = world["inferred_cand"]
        n_id = meta["counts"]["n_atomic_id"]
        n_target = int(round(phi * n_id))
        if n_target > len(cand):
            raise ValueError(
                f"phi={phi} needs {n_target} inferred training facts but only {len(cand)} candidates exist in {meta['world_id']}"
            )
        rng = np.random.default_rng([meta["config"]["data_seed"], _phi_key(phi)])
        sel = np.sort(rng.choice(len(cand), size=n_target, replace=False))
        return cand[sel]

    def _atomic_example(self, world, E: int, end: int, i: int) -> Example:
        h, r, t = (
            int(world["heads"][i]),
            int(world["rels"][i]),
            int(world["tails"][i]),
        )
        return Example(
            input_ids=[h, E + r],
            full_ids=[h, E + r, t, end],
            answer_ids=[t],
            kind="atomic",
            fact=(h, r, t),
        )

    def _inferred_example(self, world, E: int, end: int, pair) -> Example:
        i, j = (int(pair[0]), int(pair[1]))
        h, r1 = (int(world["heads"][i]), int(world["rels"][i]))
        r2, t = (int(world["rels"][j]), int(world["tails"][j]))
        return Example(
            input_ids=[h, E + r1, E + r2],
            full_ids=[h, E + r1, E + r2, t, end],
            answer_ids=[t],
            kind="inferred",
            fact=(h, r1, r2, t),
        )

    def load(self, world_dir: Path, phi: float, eval_cap: int) -> TaskData:
        world, meta = self._read(world_dir)
        cfgd = meta["config"]
        E, R = (cfgd["n_entities"], cfgd["n_relations"])
        vocab = self._vocab(E, R)
        end = vocab["</a>"]
        ood = world["ood_mask"]
        atomic = [self._atomic_example(world, E, end, i) for i in range(len(ood))]
        tr_pairs = self._train_inferred_pairs(world, meta, phi)
        train_inferred = [self._inferred_example(world, E, end, p) for p in tr_pairs]
        test_inferred = [
            self._inferred_example(world, E, end, p) for p in world["inferred_te"]
        ]
        inferred_ood = [
            self._inferred_example(world, E, end, p) for p in world["inferred_ood"]
        ]
        train = atomic + train_inferred

        def cap(examples: list[Example], split: str) -> list[Example]:
            k = min(eval_cap, len(examples))
            rng = np.random.default_rng(
                [cfgd["data_seed"], _phi_key(phi), _SPLIT_SEEDS[split]]
            )
            idx = np.sort(rng.choice(len(examples), size=k, replace=False))
            return [examples[i] for i in idx]

        eval_splits = {
            "id_atomic": cap(
                [ex for i, ex in enumerate(atomic) if not ood[i]], "id_atomic"
            ),
            "ood_atomic": cap(
                [ex for i, ex in enumerate(atomic) if ood[i]], "ood_atomic"
            ),
            "train_inferred": cap(train_inferred, "train_inferred"),
            "test_inferred_id": cap(test_inferred, "test_inferred_id"),
            "inferred_ood": cap(inferred_ood, "inferred_ood"),
        }
        probe_splits = {
            "train_inferred": train_inferred,
            "test_inferred_id": test_inferred,
            "inferred_ood": inferred_ood,
        }
        metadata = dict(meta)
        metadata.update(
            {
                "phi": phi,
                "n_train_inferred": len(train_inferred),
                "world_dir": str(Path(world_dir).resolve()),
                "eval_cap": eval_cap,
            }
        )
        return TaskData(
            vocab=vocab,
            pad_id=vocab["<sep>"],
            train=train,
            eval_splits=eval_splits,
            probe_splits=probe_splits,
            metadata=metadata,
        )
