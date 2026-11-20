"""Configurations for the synthetic composition experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    task: str = "composition"
    n_entities: int = 500
    n_relations: int = 200
    out_degree: int = 20
    atomic_id_frac: float = 0.95
    id_test_reserve_p: float = 0.005
    data_seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 8
    n_embd: int = 768
    n_head: int = 12
    n_positions: int = 10
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    phi: float
    train_seed: int = 42
    batch_size: int = 512
    lr: float = 0.0001
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-08
    warmup_steps: int = 2000
    max_steps: int = 200000
    max_grad_norm: float = 1.0
    loss_mode: str = "answer_only"
    precision: str = "fp32"
    device: str = "auto"
    num_workers: int = 2
    num_threads: int = 4
    eval_cap: int = 3000
    dense_every: int = 2000
    dense_until: int = 40000
    sparse_every: int = 50000
