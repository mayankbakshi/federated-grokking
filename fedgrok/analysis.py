"""Frozen analysis definitions for the saved 20-seed study."""

import math
from pathlib import Path
import numpy as np

W = 300
TAU = 0.90
FLOOR_MIN = 0.02
KSIG = 3
ALPHA = 0.05
NBOOT = 10000
BOOT_SEED = 0


def _load(directory, seed, mode):
    path = Path(directory) / f"seed{seed}_{mode}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def persist_cross(step, val, thresh, w=W):
    """First threshold crossing sustained at all sampled checkpoints in [s, s+w]; date the event to s. Incomplete windows are excluded."""
    step = np.asarray(step, float)
    val = np.asarray(val, float)
    last = step.max() if step.size else -np.inf
    for i in range(step.size):
        s0 = step[i]
        if s0 + w > last:
            break
        m = (step >= s0) & (step <= s0 + w)
        v = val[m]
        if v.size and np.all(np.isfinite(v)) and np.all(v >= thresh):
            return float(s0)
    return None


def seed_floor(local):
    """Compute the threshold from the full no-sharing trajectory."""
    if local is None:
        return (FLOOR_MIN, "FLOOR_MIN (local missing)")
    a = np.asarray(local["A_out_backbone"], float)
    a = a[np.isfinite(a)]
    phi = max(FLOOR_MIN, float(a.mean() + KSIG * a.std()))
    src = "FLOOR_MIN" if phi == FLOOR_MIN else f"mean+{KSIG}std"
    return (phi, src)


def binom_ge(k, n, p=0.5):
    """One-sided exact binomial tail probability."""
    return sum((math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def analyze(d, seeds, e50_dir=None):
    """Summarize retrospective onset leads and control crossings for the requested seeds."""
    rng = np.random.RandomState(BOOT_SEED)
    rows = []
    for s in seeds:
        fed = _load(d, s, "fedavg")
        loc = _load(d, s, "local")
        cen = _load(d, s, "central")
        if fed is None:
            rows.append(dict(seed=s, missing=True))
            continue
        phi, phi_src = seed_floor(loc)
        t_g = persist_cross(fed["acc_step"], fed["acc"], TAU)
        t_a = persist_cross(fed["step"], fed["A_out_backbone"], phi)
        censored = t_g is None or t_a is None
        lead = None if censored else t_g - t_a
        leak_loc = (
            loc is not None
            and persist_cross(loc["step"], loc["A_out_backbone"], phi) is not None
        )
        leak_cen = (
            cen is not None
            and persist_cross(cen["step"], cen["A_out_backbone"], phi) is not None
        )
        lead50 = None
        if e50_dir is not None:
            fed50 = _load(e50_dir, s, "fedavg")
            if fed50 is not None:
                tg50 = persist_cross(fed50["acc_step"], fed50["acc"], TAU)
                ta50 = persist_cross(fed50["step"], fed50["A_out_backbone"], phi)
                lead50 = None if tg50 is None or ta50 is None else tg50 - ta50
        rows.append(
            dict(
                seed=s,
                missing=False,
                phi=phi,
                phi_src=phi_src,
                t_g=t_g,
                t_a=t_a,
                censored=censored,
                lead=lead,
                leak_local=leak_loc,
                leak_central=leak_cen,
                lead50=lead50,
                grok_ref=int(fed["grok_step"]),
                first_cross=persist_cross(fed["step"], fed["A_out_backbone"], phi, w=0),
            )
        )
    present = [r for r in rows if not r.get("missing")]
    n = len(present)
    censored_seeds = [row["seed"] for row in present if row["censored"]]
    if not n or censored_seeds:
        raise ValueError(
            "Lead summaries require both onsets for every run; "
            f"censored seeds: {censored_seeds}. Inspect the trajectories or use "
            "an analysis that explicitly handles censoring."
        )
    npos = sum((1 for r in present if r["lead"] is not None and r["lead"] > 0))
    p_sign = binom_ge(npos, n)
    boot_vals = np.array([r["lead"] for r in present], float)
    means = np.array(
        [rng.choice(boot_vals, size=n, replace=True).mean() for _ in range(NBOOT)]
    )
    ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
    mean_lead = boot_vals.mean()
    unc = np.array([r["lead"] for r in present if r["lead"] is not None], float)
    se = unc.std(ddof=1) / math.sqrt(len(unc)) if len(unc) > 1 else float("nan")
    leak_loc_n = sum((1 for r in present if r["leak_local"]))
    leak_cen_n = sum((1 for r in present if r["leak_central"]))
    c1 = p_sign <= ALPHA
    c2 = mean_lead > 0 and ci_lo > 0
    c3 = leak_loc_n <= 2 and leak_cen_n <= 2
    return dict(
        rows=rows,
        present=present,
        n=n,
        npos=npos,
        p_sign=p_sign,
        mean_lead=mean_lead,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        unc_mean=float(unc.mean()) if len(unc) else float("nan"),
        unc_se=se,
        leak_loc_n=leak_loc_n,
        leak_cen_n=leak_cen_n,
        c1=c1,
        c2=c2,
        c3=c3,
        passed=c1 and c2 and c3,
    )
