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


# Depth from the azimuthally averaged analytic-signal profile.
#
# The previous estimator sorted the pixels of a square window by radius and returned the first
# radius at which *any single pixel* fell below half amplitude. That is a noise-driven extreme
# value, not a half-width: at the repo's own 2 nT noise it returned errors of +91 m at 300 m,
# -248 m at 950 m and -772 m at 1200 m, with IQRs of 0-49 m. Averaging the amplitude over
# annuli first suppresses the noise, and the crossing is then interpolated on the mean profile.
# Calibration: depth = DEPTH_A + DEPTH_B * half-width.
# Fitted by src/calibrate_depth.py on single-body synthetic scenes at 250-700 m depth, five
# noise seeds each (20 scenes), using the same preprocessing as the pipeline. Residual sd 5 m,
# max |error| 13 m over the fit range; held out at 900-1100 m the error stays within ~50 m.
# The negative intercept absorbs the 100 m upward continuation applied before the analytic
# signal, which adds a constant smoothing offset to the half-width.
DEPTH_A = -168.0
DEPTH_B = 1.904
DEPTH_K = DEPTH_B       # kept for callers that still reference the old name


def radial_profile(asig, E, N, iy, ix, dx, rmax_m=2000.0, bin_m=None):
    """Azimuthally averaged amplitude against distance from a peak.

    Returns ``(r_centres, amplitude)``; empty annuli are skipped, so both arrays are the same
    length and monotonically increasing in radius.
    """
    rmax = float(rmax_m)
    width = float(bin_m if bin_m is not None else dx)
    if rmax <= 0 or width <= 0:
        raise ValueError("rmax_m and bin_m must be positive")

    r = np.hypot(E - E[iy, ix], N - N[iy, ix]).ravel()
    a = np.asarray(asig, dtype=float).ravel()
    keep = np.isfinite(a) & (r <= rmax)
    r, a = r[keep], a[keep]
    if r.size == 0:
        return np.empty(0), np.empty(0)

    nbins = max(int(np.ceil(rmax / width)), 4)
    edges = np.linspace(0.0, rmax, nbins + 1)
    which = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)

    centres, means = [], []
    for k in range(nbins):
        sel = which == k
        if np.any(sel):
            centres.append(0.5 * (edges[k] + edges[k + 1]))
            means.append(float(a[sel].mean()))
    return np.asarray(centres), np.asarray(means)


def halfwidth_from_profile(r, a, peak=None):
    """Radius at which the descending profile crosses half amplitude, by interpolation."""
    if r.size < 3 or a.size != r.size:
        return np.nan
    peak = float(np.max(a)) if peak is None else float(peak)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    half = peak / 2.0
    i_peak = int(np.argmax(a))
    for i in range(i_peak + 1, r.size):
        if a[i] <= half:
            r0, r1 = float(r[i - 1]), float(r[i])
            a0, a1 = float(a[i - 1]), float(a[i])
            if a0 == a1:
                return r1
            return r0 + (a0 - half) * (r1 - r0) / (a0 - a1)
    return np.nan


def depth_halfwidth(asig, E, N, iy, ix, dx, a=DEPTH_A, b=DEPTH_B, rmax_m=2000.0):
    """Depth from the half-amplitude radius of the azimuthally averaged profile.

    ``depth = a + b * half-width``, with (a, b) fitted by src/calibrate_depth.py. The offset
    matters because the profile is computed after 100 m of upward continuation, which widens
    every anomaly by a roughly constant amount.
    """
    r, amp = radial_profile(asig, E, N, iy, ix, dx, rmax_m=rmax_m)
    hw = halfwidth_from_profile(r, amp)
    if not np.isfinite(hw) or hw <= 0:
        return np.nan
    return float(max(a + b * hw, 0.0))


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
    if len(base) == 0:
        return [], detr
    # amps[i] accumulates the amplitude on every hit, so divide by the hit count before
    # normalising: score must be p x (mean amplitude), not p x (sum) = p^2 x mean.
    mean_amps = np.divide(amps, hits, out=np.zeros_like(amps), where=hits > 0)
    amax = mean_amps.max() or 1.0
    out = []
    for i, bc in enumerate(base):
        p = hits[i] / n_runs
        if depths[i]:
            dz = np.asarray(depths[i], dtype=float)
            med = float(np.nanmedian(dz))
            iqr = float(np.nanpercentile(dz, 75) - np.nanpercentile(dz, 25))
            depth_median = round(med) if np.isfinite(med) else None
            depth_iqr = round(iqr) if np.isfinite(iqr) else None
        else:
            depth_median, depth_iqr = None, None
        out.append({"east": round(bc["east"]), "north": round(bc["north"]),
                    "detection_prob": round(float(p), 3),
                    "depth_median_m": depth_median,
                    "depth_iqr_m": depth_iqr,
                    "amp_norm": round(float(mean_amps[i] / amax), 3),
                    "score": round(float(p * mean_amps[i] / amax), 3)})
    out.sort(key=lambda z: -z["score"])
    for k, z in enumerate(out, 1):
        z["rank"] = k
    return out, detr
