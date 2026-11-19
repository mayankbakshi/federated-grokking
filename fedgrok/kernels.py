"""Randomized trace kernels and within-/cross-band label alignment."""

import numpy as np
import torch
from torch.func import functional_call, grad, jacrev, vmap


def param_names(model):
    return [name for name, _ in model.named_parameters()]


@torch.no_grad()
def _load(model, state):
    model.load_state_dict(state)


def backbone_readout_masks(model):
    head = np.concatenate(
        [
            np.full(p.numel(), name.startswith("head"), dtype=bool)
            for name, p in model.named_parameters()
        ]
    )
    return {"all": np.ones_like(head), "backbone": ~head, "readout": head}


def _params_dict(model):
    return {n: p.detach() for n, p in model.named_parameters()}


def _jac_vmap(model, x, y, W, row_chunk=64):
    """Compute per-example projected-output gradients in fixed-size chunks."""
    names = param_names(model)
    params = _params_dict(model)

    def f_weighted(pd, xi, yi, wi):
        logits = functional_call(model, pd, (xi.unsqueeze(0), yi.unsqueeze(0))).squeeze(
            0
        )
        return (logits * wi).sum()

    per_ex = vmap(grad(f_weighted, argnums=0), in_dims=(None, 0, 0, 0))
    n = x.shape[0]
    out = []
    for i in range(0, n, row_chunk):
        sl = slice(i, min(i + row_chunk, n))
        g = per_ex(params, x[sl], y[sl], W[sl])
        nc = x[sl].shape[0]
        out.append(torch.cat([g[nm].reshape(nc, -1) for nm in names], dim=1))
    return torch.cat(out, dim=0).detach().cpu().double().numpy()


def _jac_loop(model, x, y, W):
    """Compute the same gradients one example at a time for numerical cross-checks."""
    params = [p for _, p in model.named_parameters()]
    n = x.shape[0]
    rows = []
    for i in range(n):
        logits = model(x[i : i + 1], y[i : i + 1]).squeeze(0)
        s = (logits * W[i]).sum()
        g = torch.autograd.grad(s, params, retain_graph=False, create_graph=False)
        rows.append(torch.cat([gi.reshape(-1) for gi in g]))
    return torch.stack(rows, dim=0).detach().cpu().double().numpy()


def pntk_jacobian(model, x, y, W, method="vmap", row_chunk=64):
    """Return an n-by-d float64 Jacobian on a scratch model."""
    model.eval()
    if method == "vmap":
        return _jac_vmap(model, x, y, W, row_chunk=row_chunk)
    if method == "loop":
        return _jac_loop(model, x, y, W)
    raise ValueError(f"unknown method {method!r}")


def trace_gram(model, state, x, y, readout_mask=None, method="vmap", row_chunk=64):
    """Sum the classwise Jacobian Gram matrices to obtain the exact trace kernel."""
    _load(model, state)
    model.eval()
    P = int(model.head.out_features)
    n = x.shape[0]
    K = np.zeros((n, n), dtype=np.float64)
    rmask = np.asarray(readout_mask, dtype=bool) if readout_mask is not None else None
    K_read = np.zeros((n, n), dtype=np.float64) if rmask is not None else None
    for c in range(P):
        W = torch.zeros(n, P, device=x.device)
        W[:, c] = 1.0
        Jc = pntk_jacobian(model, x, y, W, method=method, row_chunk=row_chunk)
        K += Jc @ Jc.T
        if rmask is not None:
            Jr = Jc[:, rmask]
            K_read += Jr @ Jr.T
    if rmask is None:
        return K
    return {"all": K, "backbone": K - K_read, "readout": K_read}


def randproj_gram(
    model, state, x, y, readout_mask, R=8, proj_seed=0, method="vmap", row_chunk=64
):
    """Average R Rademacher output-projection Grams. This estimates the trace kernel without bias; normalized alignment is not guaranteed unbiased. Projection vectors use a separate, fixed random stream."""
    _load(model, state)
    model.eval()
    P = int(model.head.out_features)
    n = x.shape[0]
    rmask = np.asarray(readout_mask, dtype=bool)
    g = torch.Generator().manual_seed(int(proj_seed))
    U = torch.randint(0, 2, (R, P), generator=g).float() * 2.0 - 1.0
    K_all = np.zeros((n, n), dtype=np.float64)
    K_read = np.zeros((n, n), dtype=np.float64)
    for r in range(R):
        W = U[r].unsqueeze(0).expand(n, P).to(x.device)
        J = pntk_jacobian(model, x, y, W, method=method, row_chunk=row_chunk)
        K_all += J @ J.T
        Jr = J[:, rmask]
        K_read += Jr @ Jr.T
    K_all /= R
    K_read /= R
    return {"all": K_all, "backbone": K_all - K_read, "readout": K_read}


def full_jacobian(model, state, x, y):
    """Full output Jacobian for small-model reference calculations."""
    _load(model, state)
    model.eval()
    names = param_names(model)
    params = _params_dict(model)

    def f_logits(pd, xi, yi):
        return functional_call(model, pd, (xi.unsqueeze(0), yi.unsqueeze(0))).squeeze(0)

    jac = vmap(jacrev(f_logits, argnums=0), in_dims=(None, 0, 0))(params, x, y)
    n, P = (x.shape[0], int(model.head.out_features))
    return (
        torch.cat([jac[nm].reshape(n, P, -1) for nm in names], dim=2)
        .detach()
        .cpu()
        .double()
        .numpy()
    )


def _centering(n):
    return np.eye(n) - np.ones((n, n)) / n


def label_gram(z, P):
    """The (i, j) entry is 1 when labels i and j agree, and 0 otherwise."""
    z = np.asarray(z.cpu() if torch.is_tensor(z) else z)
    Y = np.eye(P)[z]
    return Y @ Y.T


def centered_alignment(K, L):
    """Frobenius cosine of the globally centered kernel and label Gram; NaN for degenerate centered energy."""
    n = K.shape[0]
    H = _centering(n)
    HKH = H @ K @ H
    HLH = H @ L @ H
    kE = float((HKH * HKH).sum())
    lE = float((HLH * HLH).sum())
    if kE <= 1e-10 * float((K * K).sum()) or lE <= 1e-10 * float((L * L).sum()):
        return np.nan
    return float((HKH * HLH).sum() / (np.sqrt(kE) * np.sqrt(lE)))


def block_alignment(K, L, bands, which):
    """Globally center both Grams, then take a separately normalized cosine on within-band or cross-band entries. The two normalized values need not add to the global alignment."""
    n = K.shape[0]
    H = _centering(n)
    HKH = H @ K @ H
    HLH = H @ L @ H
    bands = np.asarray(bands)
    same = bands[:, None] == bands[None, :]
    M = same if which == "in" else ~same
    kE = float((HKH * HKH * M).sum())
    lE = float((HLH * HLH * M).sum())
    if kE <= 1e-10 * float((K * K).sum()) or lE <= 1e-10 * float((L * L).sum()):
        return np.nan
    return float((HKH * HLH * M).sum() / (np.sqrt(kE) * np.sqrt(lE)))
