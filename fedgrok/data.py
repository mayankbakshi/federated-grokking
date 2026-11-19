import numpy as np
import torch


def generate_all_pairs(p: int = 97) -> np.ndarray:
    """All modular-division triples (x, y, x/y mod p), with y nonzero."""
    inv = [pow(y, -1, p) for y in range(1, p)]
    rows = []
    for x in range(p):
        for y_idx, y in enumerate(range(1, p)):
            z = x * inv[y_idx] % p
            rows.append((x, y, z))
    return np.array(rows, dtype=np.int64)


def partition_type_B(pairs: np.ndarray, V: int = 4, p: int = 97):
    """Split x into contiguous bands; allocate the remainder to the last bands."""
    band_size = p // V
    remainder = p % V
    bands = []
    start = 0
    for v in range(V):
        end = start + band_size + (1 if v >= V - remainder else 0)
        bands.append((start, end))
        start = end
    assert bands[-1][1] == p, f"bands do not cover [0, p): {bands}"
    client_pairs = []
    for lo, hi in bands:
        mask = (pairs[:, 0] >= lo) & (pairs[:, 0] < hi)
        client_pairs.append(pairs[mask])
    return (client_pairs, bands)


def split_local_train_test(client_pairs: np.ndarray, alpha: float = 0.5, seed: int = 0):
    """Use a dedicated random stream for the per-client split."""
    rng = np.random.RandomState(seed)
    n = len(client_pairs)
    perm = rng.permutation(n)
    n_train = int(alpha * n)
    train = client_pairs[perm[:n_train]]
    local_test = client_pairs[perm[n_train:]]
    return (train, local_test)


def to_tensors(pairs: np.ndarray, device: str = "mps"):
    x = torch.tensor(pairs[:, 0], dtype=torch.long, device=device)
    y = torch.tensor(pairs[:, 1], dtype=torch.long, device=device)
    z = torch.tensor(pairs[:, 2], dtype=torch.long, device=device)
    return (x, y, z)
