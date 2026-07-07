"""
Test suite — validates the physics and the recovery pipeline.

Runs with pytest (`pytest -q`) or standalone (`python -m tests.test_core`). Pure NumPy.
"""
from __future__ import annotations
import numpy as np
from src import forward_model as fm, processing as pf, targeting as tg


def test_forward_localises_source():
    """At the pole (inc=90) the analytic-signal peak of a central body sits at the origin."""
    cfg = fm.SurveyConfig(noise_nT=0.0, regional_amp_nT=0.0, inc_deg=90.0, dec_deg=0.0)
    g = fm.tmi_grid(cfg, [fm.Body(0, 0, 400, 0.1, 2e7)])
    a = pf.analytic_signal(g["tmi"], g["dx"])
    iy, ix = np.unravel_index(np.argmax(a), a.shape)
    assert abs(g["E"][iy, ix]) <= g["dx"] and abs(g["N"][iy, ix]) <= g["dx"]


def test_rtp_identity_at_pole():
    """At the magnetic pole (inc=90) reduction-to-pole is (near) identity."""
    cfg = fm.SurveyConfig(noise_nT=0.0, regional_amp_nT=0.0, inc_deg=90.0, dec_deg=0.0)
    g = fm.tmi_grid(cfg, [fm.Body(0, 0, 400, 0.1, 2e7)])
    rtp = pf.reduction_to_pole(g["tmi"], g["dx"], 90.0, 0.0)
    a, b = g["tmi"] - g["tmi"].mean(), rtp - rtp.mean()
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert corr > 0.98


def test_vertical_derivative_of_constant_is_zero():
    fld = np.ones((64, 64)) * 5.0
    assert np.allclose(pf.vertical_derivative(fld, 50.0, 1), 0.0, atol=1e-6)


def test_depth_halfwidth_monotonic():
    """Deeper bodies must yield larger half-width depth estimates."""
    def est(depth):
        cfg = fm.SurveyConfig(noise_nT=0.0, regional_amp_nT=0.0)
        g = fm.tmi_grid(cfg, [fm.Body(0, 0, depth, 0.1, 2e7)])
        a = pf.analytic_signal(g["tmi"], g["dx"])
        iy, ix = np.unravel_index(np.argmax(a), a.shape)
        return tg.depth_halfwidth(a, g["E"], g["N"], iy, ix, g["dx"])
    d1, d2, d3 = est(300), est(600), est(900)
    assert d1 < d2 < d3


def test_pipeline_recovers_truth():
    """The MC pipeline recovers the strong bodies and rejects the weak decoy."""
    cfg = fm.SurveyConfig()
    g = fm.tmi_grid(cfg, fm.default_scene())
    targets, _ = tg.monte_carlo_rank(g["tmi"], g["dx"], g["E"], g["N"], cfg, n_runs=40)
    tol = 450.0
    matched, depth_errs = 0, []
    for t in g["truth"]:
        near = [z for z in targets
                if np.hypot(z["east"] - t[0], z["north"] - t[1]) <= tol]
        if near:
            matched += 1
            best = min(near, key=lambda z: np.hypot(z["east"] - t[0], z["north"] - t[1]))
            depth_errs.append(abs(best["depth_median_m"] - t[2]))
    assert matched >= 3, f"expected >=3 recovered, got {matched}"
    assert np.mean(depth_errs) < 200, f"mean depth error too high: {np.mean(depth_errs):.0f} m"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
