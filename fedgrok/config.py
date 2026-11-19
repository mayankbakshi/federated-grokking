from dataclasses import dataclass, asdict


@dataclass
class Config:
    p: int = 97
    V: int = 4
    alpha: float = 0.5
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    lr: float = 0.001
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    eval_log_per_decade: int = 40
    eval_dense_bands: tuple = ((1000, 6000, 100), (10000, 28000, 200))

    def as_dict(self):
        return asdict(self)


CONFIG = Config()
