"""Serialized examples and named evaluation splits."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    input_ids: list[int]
    full_ids: list[int]
    answer_ids: list[int]
    kind: str
    fact: tuple


@dataclass
class TaskData:
    vocab: dict[str, int]
    pad_id: int
    train: list[Example]
    eval_splits: dict[str, list[Example]]
    probe_splits: dict[str, list[Example]]
    metadata: dict
