"""
Drill-target detection, depth estimation & probabilistic ranking. Pure NumPy.

Conventions: grid [row=North, col=East]; component order [East, North, Down].
"""
from __future__ import annotations
import numpy as np
from . import processing as pf


def detect_peaks(asig, E, N, dx, top_k=8, min_sep_m=700.0, percentile=90, border=6):
    """Local maxima of the analytic signal, with non-maximum suppression + border exclusion."""
    dil = asig.copy()
    for sy in (-1, 0, 1):
        for sx in (-1, 0, 1):
            if sx or sy:
                dil = np.maximum(dil, np.roll(np.roll(asig, sy, 0), sx, 1))
    thr = np.percentile(asig, percentile)
    ny, nx = asig.shape
    ys, xs = np.where((asig >= dil) & (asig >= thr))
    cand = sorted(([asig[y, x], y, x] for y, x in zip(ys, xs)), reverse=True)
    kept = []
    for amp, y, x in cand:
        if y < border or y >= ny - border or x < border or x >= nx - border:
            continue
        ex, nn = float(E[y, x]), float(N[y, x])
        if all(np.hypot(ex - k["east"], nn - k["north"]) >= min_sep_m for k in kept):
            kept.append({"iy": y, "ix": x, "east": ex, "north": nn, "amp": float(amp)})
        if len(kept) >= top_k:
            break
    return kept


def euler_depth_multi(tmi, dx, E, N, iy, ix, SI=3.0, wins=range(9, 16)):
    ds = [euler_depth(tmi, dx, E, N, iy, ix, win=w, SI=SI) for w in wins]
    return float(np.median(ds))


# depth from analytic-signal half-width (robust for compact sources).
# DEPTH_K calibrated against the physics forward model (depth ≈ K x half-width); see notebook.
DEPTH_K = 1.75


def depth_halfwidth(asig, E, N, iy, ix, dx, k=DEPTH_K, rmax_m=800.0):
    peak = asig[iy, ix]
    if peak <= 0:
        return np.nan
    rr = int(rmax_m / dx)
    ny, nx = asig.shape
    sl = (slice(max(iy - rr, 0), min(iy + rr + 1, ny)),
          slice(max(ix - rr, 0), min(ix + rr + 1, nx)))
    r = np.sqrt((E[sl] - E[iy, ix])**2 + (N[sl] - N[iy, ix])**2).ravel()
    a = asig[sl].ravel()
    order = np.argsort(r); r, a = r[order], a[order]
    below = np.where(a < peak / 2)[0]
    if below.size == 0:
        return np.nan
    return float(k * r[below[0]])


def euler_depth(tmi, dx, E, N, iy, ix, win=12, SI=3.0):
    gN, gE = np.gradient(tmi, dx)
    Tz = pf.vertical_derivative(tmi, dx, 1)
    ny, nx = tmi.shape
    sl = (slice(max(iy - win, 0), min(iy + win + 1, ny)),
          slice(max(ix - win, 0), min(ix + win + 1, nx)))
    e, n = E[sl].ravel(), N[sl].ravel()
    ge, gn, gz, t = gE[sl].ravel(), gN[sl].ravel(), Tz[sl].ravel(), tmi[sl].ravel()
    A = np.column_stack([ge, gn, gz, SI * np.ones_like(ge)])
    b = e * ge + n * gn + SI * t
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return abs(float(sol[2]))


def monte_carlo_rank(tmi, dx, E, N, cfg, n_runs=150, seed=1, top_k=8):
    """Detect on the detrended analytic signal (source-centred, reduction-independent); depth by
    multi-window Euler. Monte-Carlo over noise -> detection probability + depth median/IQR."""
    rng = np.random.default_rng(seed)
    detr, _ = pf.polynomial_detrend(tmi, E, N, order=1)
    base_asig = pf.analytic_signal(pf.upward_continue(detr, dx, 100.0), dx)
    base = detect_peaks(base_asig, E, N, dx, top_k=top_k, percentile=85)
    tol = 400.0
    hits = np.zeros(len(base)); amps = np.zeros(len(base)); depths = [[] for _ in base]
    for _ in range(n_runs):
        g = tmi + rng.normal(0, cfg.noise_nT, tmi.shape)
        d, _ = pf.polynomial_detrend(g, E, N, order=1)
        asig = pf.analytic_signal(pf.upward_continue(d, dx, 100.0), dx)  # mild denoise
        peaks = detect_peaks(asig, E, N, dx, top_k=top_k + 3, percentile=85)
        for i, bc in enumerate(base):
            near = [p for p in peaks if np.hypot(p["east"] - bc["east"], p["north"] - bc["north"]) <= tol]
            if near:
                p = max(near, key=lambda z: z["amp"])
                hits[i] += 1; amps[i] += p["amp"]
                dh = depth_halfwidth(asig, E, N, p["iy"], p["ix"], dx)
                if not np.isnan(dh):
                    depths[i].append(dh)
    amax = amps.max() or 1.0
    out = []
    for i, bc in enumerate(base):
        p = hits[i] / n_runs
        dz = np.array(depths[i]) if depths[i] else np.array([np.nan])
        out.append({"east": round(bc["east"]), "north": round(bc["north"]),
                    "detection_prob": round(float(p), 3),
                    "depth_median_m": round(float(np.nanmedian(dz))),
                    "depth_iqr_m": round(float(np.nanpercentile(dz, 75) - np.nanpercentile(dz, 25))) if depths[i] else np.nan,
                    "amp_norm": round(float(amps[i] / amax), 3),
                    "score": round(float(p * amps[i] / amax), 3)})
    out.sort(key=lambda z: -z["score"])
    for k, z in enumerate(out, 1):
        z["rank"] = k
    return out, detr
