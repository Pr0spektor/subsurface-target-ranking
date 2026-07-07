"""
End-to-end drill-target ranking pipeline.

Forward model -> polynomial detrend -> RTP + analytic signal -> probabilistic target ranking
(depth + uncertainty) -> validation vs known ground truth -> figures + CSVs.

Run:  python -m src.pipeline
Real data: replace forward_model.tmi_grid(...) with real_data.load_grid(...) (see src/real_data.py).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import forward_model as fm, processing as pf, targeting as tg

RESULTS = Path(__file__).resolve().parent.parent / "results"; RESULTS.mkdir(exist_ok=True)
MATCH_TOL = 450.0


def match_truth(targets, truth):
    rows = []
    for t in truth:
        te, tn, td = t[0], t[1], t[2]
        best, bd = None, 1e9
        for z in targets:
            d = np.hypot(z["east"] - te, z["north"] - tn)
            if d < bd:
                bd, best = d, z
        ok = best is not None and bd <= MATCH_TOL
        rows.append({"true_east": te, "true_north": tn, "true_depth_m": td, "matched": ok,
                     "found_rank": best["rank"] if ok else None,
                     "detection_prob": best["detection_prob"] if ok else 0.0,
                     "est_depth_m": best["depth_median_m"] if ok else None,
                     "depth_err_m": (best["depth_median_m"] - td) if ok else None,
                     "horiz_err_m": round(bd) if ok else None})
    return pd.DataFrame(rows)


def main():
    cfg = fm.SurveyConfig(); g = fm.tmi_grid(cfg, fm.default_scene())
    E, N, dx = g["E"], g["N"], g["dx"]
    targets, detr = tg.monte_carlo_rank(g["tmi"], dx, E, N, cfg, n_runs=150)
    rtp = pf.reduction_to_pole(detr, dx, cfg.inc_deg, cfg.dec_deg)
    asig = pf.analytic_signal(detr, dx)

    tdf = pd.DataFrame(targets)[["rank", "east", "north", "detection_prob",
                                 "depth_median_m", "depth_iqr_m", "amp_norm", "score"]]
    tdf.to_csv(RESULTS / "ranked_targets.csv", index=False)
    rec = match_truth(targets, g["truth"]); rec.to_csv(RESULTS / "recovery.csv", index=False)

    print("Ranked drill targets (probabilistic; depth + uncertainty):\n")
    print(tdf.to_string(index=False))
    nf = int(rec["matched"].sum())
    mde = rec.loc[rec["matched"], "depth_err_m"].abs().mean()
    mhe = rec.loc[rec["matched"], "horiz_err_m"].abs().mean()
    print(f"\nGround-truth recovery: {nf}/{len(g['truth'])} matched (<= {MATCH_TOL:.0f} m); "
          f"mean |horiz err| = {mhe:.0f} m; mean |depth err| = {mde:.0f} m.")
    print(rec.to_string(index=False))

    ext = [E.min(), E.max(), N.min(), N.max()]
    truth = g["truth"]
    def stars(ax): ax.scatter(truth[:, 0], truth[:, 1], marker="*", s=150, edgecolor="k",
                              facecolor="none", linewidths=1.4, label="true body")

    fig, ax = plt.subplots(figsize=(6.6, 5.6)); im = ax.imshow(g["tmi"], extent=ext, origin="lower", cmap="RdBu_r")
    stars(ax); ax.set_title("Observed TMI (nT)"); ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.legend(fontsize=8); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(RESULTS / "tmi_observed.png", dpi=140)

    fig, ax = plt.subplots(figsize=(6.6, 5.6)); im = ax.imshow(rtp, extent=ext, origin="lower", cmap="RdBu_r")
    stars(ax); ax.set_title("Reduction-to-Pole (detrended)"); ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout(); fig.savefig(RESULTS / "rtp.png", dpi=140)

    fig, ax = plt.subplots(figsize=(7.2, 5.8)); im = ax.imshow(asig, extent=ext, origin="lower", cmap="magma")
    for t in targets:
        ax.scatter(t["east"], t["north"], s=40 + 420 * t["score"], facecolor="none", edgecolor="cyan", linewidths=1.6)
        ax.text(t["east"] + 55, t["north"] + 55, f"#{t['rank']} {t['detection_prob']:.0%}", color="cyan", fontsize=7)
    stars(ax); ax.set_title("Analytic signal + probabilistic drill targets\n(circle size = score; ★ = true body)")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.legend(fontsize=8, loc="lower left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout(); fig.savefig(RESULTS / "targets.png", dpi=140)

    m = rec[rec["matched"]].copy()
    if len(m):
        iqr = [float(tdf.set_index("rank").loc[r, "depth_iqr_m"]) for r in m["found_rank"]]
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.errorbar(m["true_depth_m"], m["est_depth_m"], yerr=np.array(iqr) / 2, fmt="o", color="#1f3864", capsize=4)
        lim = [0, max(m["true_depth_m"].max(), m["est_depth_m"].max()) * 1.2]
        ax.plot(lim, lim, "--", color="grey", label="1:1")
        for _, r in m.iterrows():
            ax.annotate(f"{r['detection_prob']:.0%}", (r["true_depth_m"], r["est_depth_m"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_xlabel("True depth (m)"); ax.set_ylabel("Estimated depth ± IQR/2 (m)")
        ax.set_title("Depth recovery vs ground truth (analytic-signal half-width)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(RESULTS / "depth_recovery.png", dpi=140)
    print(f"\nSaved figures + CSVs to {RESULTS}/")


if __name__ == "__main__":
    main()
