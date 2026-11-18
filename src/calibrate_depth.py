"""Calibrate the depth estimator on single-body synthetic scenes.

The estimator returns a half-amplitude radius; this script turns that radius into metres and
makes the conversion auditable. It replaces the earlier claim that DEPTH_K was "calibrated
against the forward model, see notebook" — the notebook contained no calibration.

Method
------
For each depth and noise seed, build a scene with one body at the grid centre, apply exactly
the pipeline preprocessing (linear detrend, 100 m upward continuation, analytic signal), take
the azimuthally averaged radial profile around the peak, and record its half-amplitude radius.
A straight line ``depth = a + b * halfwidth`` is then fitted by least squares on the scenes
whose peak amplitude is above the noise floor (0.2 nT), because below that the peak is not a
detection at all.

Outputs
-------
results/depth_calibration.csv   every scene, its half-width, estimate and error
results/depth_calibration.png   half-width vs true depth with the fitted line

Run:  python -m src.calibrate_depth
"""
from __future__ import annotations

import csv
import os

import numpy as np

from . import forward_model as fm
from . import processing as pf
from . import targeting as tg

DEPTHS = (250, 400, 550, 700, 900, 1100, 1400)
SEEDS = (1, 2, 3, 4, 5)
BODY = dict(susceptibility=0.2, volume=4.0e7)   # strong body, so depth is the only variable
AMPLITUDE_FLOOR_NT = 0.2                        # ~10% of the 2 nT observational noise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def measure(depth: float, seed: int) -> dict:
    cfg = fm.SurveyConfig(seed=seed)
    g = fm.tmi_grid(cfg, [fm.Body(0.0, 0.0, depth, **BODY)])
    detr, _ = pf.polynomial_detrend(g["tmi"], g["E"], g["N"], order=1)
    asig = pf.analytic_signal(pf.upward_continue(detr, g["dx"], 100.0), g["dx"])

    near = np.hypot(g["E"], g["N"]) <= 300.0          # the body sits at the centre
    iy, ix = np.unravel_index(np.argmax(np.where(near, asig, -np.inf)), asig.shape)
    r, a = tg.radial_profile(asig, g["E"], g["N"], iy, ix, g["dx"])
    hw = tg.halfwidth_from_profile(r, a)
    return {"depth_m": depth, "seed": seed, "halfwidth_m": hw,
            "amplitude_nt": float(asig[iy, ix])}


def fit(rows: list[dict]) -> dict:
    usable = [r for r in rows if np.isfinite(r["halfwidth_m"])
              and r["amplitude_nt"] > AMPLITUDE_FLOOR_NT]
    d = np.array([r["depth_m"] for r in usable], dtype=float)
    hw = np.array([r["halfwidth_m"] for r in usable], dtype=float)
    A = np.column_stack([np.ones_like(hw), hw])
    (a, b), *_ = np.linalg.lstsq(A, d, rcond=None)
    pred = a + b * hw
    resid = pred - d
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((d - d.mean()) ** 2))
    return {"a": float(a), "b": float(b), "n": len(usable),
            "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
            "resid_sd_m": float(np.std(resid)), "max_abs_error_m": float(np.max(np.abs(resid))),
            "depth_min": float(d.min()), "depth_max": float(d.max())}


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    rows = [measure(d, s) for d in DEPTHS for s in SEEDS]
    f = fit(rows)

    path = os.path.join(RESULTS, "depth_calibration.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["depth_m", "seed", "halfwidth_m", "amplitude_nt",
                                           "depth_est_m", "error_m"])
        w.writeheader()
        for r in rows:
            est = f["a"] + f["b"] * r["halfwidth_m"] if np.isfinite(r["halfwidth_m"]) else ""
            err = est - r["depth_m"] if est != "" else ""
            w.writerow({**{k: r[k] for k in ("depth_m", "seed", "halfwidth_m", "amplitude_nt")},
                        "depth_est_m": round(est, 1) if est != "" else "",
                        "error_m": round(err, 1) if err != "" else ""})

    print(f"fit: depth = {f['a']:.1f} + {f['b']:.3f} x halfwidth")
    print(f"     n={f['n']} scenes, R2={f['r2']:.3f}, residual sd={f['resid_sd_m']:.0f} m, "
          f"max |error|={f['max_abs_error_m']:.0f} m over {f['depth_min']:.0f}-{f['depth_max']:.0f} m")
    print(f"wrote {path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        d = np.array([r["depth_m"] for r in rows], dtype=float)
        hw = np.array([r["halfwidth_m"] for r in rows], dtype=float)
        amp = np.array([r["amplitude_nt"] for r in rows], dtype=float)
        good = np.isfinite(hw) & (amp > AMPLITUDE_FLOOR_NT)
        ax.scatter(hw[good], d[good], s=26, color="#1f3864", label="above noise floor")
        ax.scatter(hw[~good], d[~good], s=26, facecolors="none", edgecolors="#999999",
                   label="below noise floor (not fitted)")
        xs = np.linspace(0.0, float(np.nanmax(hw[good])) * 1.15, 50)
        ax.plot(xs, f["a"] + f["b"] * xs, color="#C0504D",
                label=f"depth = {f['a']:.0f} + {f['b']:.2f}·hw  (R²={f['r2']:.2f})")
        ax.set_xlabel("Half-amplitude radius (m)")
        ax.set_ylabel("True depth (m)")
        ax.set_title("Depth calibration — azimuthally averaged analytic signal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "depth_calibration.png"), dpi=140)
        print("wrote", os.path.join(RESULTS, "depth_calibration.png"))
    except Exception as exc:  # pragma: no cover - plotting is optional
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
