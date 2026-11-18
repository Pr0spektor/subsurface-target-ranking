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


def test_coarse_grid_does_not_crash():
    """The real-data path uses ~2 km spacing; rr = int(800/dx) used to become 0,
    collapsing the search window to one pixel and raising ValueError on round(nan)."""
    cfg = fm.SurveyConfig(n=11, extent=20000.0)          # dx = 2000 m
    g = fm.tmi_grid(cfg, fm.default_scene())
    assert g["dx"] == 2000.0
    targets, _ = tg.monte_carlo_rank(g["tmi"], g["dx"], g["E"], g["N"], cfg, n_runs=3)
    assert isinstance(targets, list)                      # no ValueError, empty is acceptable
    for t in targets:
        assert t["depth_median_m"] is None or isinstance(t["depth_median_m"], int)


def test_score_does_not_double_count_probability():
    """score must be p x mean amplitude, not p x sum = p^2 x amplitude."""
    cfg = fm.SurveyConfig()
    g = fm.tmi_grid(cfg, fm.default_scene())
    targets, _ = tg.monte_carlo_rank(g["tmi"], g["dx"], g["E"], g["N"], cfg, n_runs=40)
    for t in targets:
        assert abs(t["score"] - t["detection_prob"] * t["amp_norm"]) < 0.002


def test_depth_estimator_tracks_depth_under_noise():
    """The azimuthally averaged half-width must recover depth across the usable range.

    The old single-pixel estimator returned +91 m at 300 m and -772 m at 1200 m at the same
    noise level; this pins the replacement to a much tighter band.
    """
    errors = []
    for depth in (300.0, 600.0, 900.0):
        for seed in (1, 2, 3):
            cfg = fm.SurveyConfig(seed=seed)
            g = fm.tmi_grid(cfg, [fm.Body(0.0, 0.0, depth, 0.2, 4.0e7)])
            detr, _ = pf.polynomial_detrend(g["tmi"], g["E"], g["N"], order=1)
            asig = pf.analytic_signal(pf.upward_continue(detr, g["dx"], 100.0), g["dx"])
            iy, ix = np.unravel_index(np.argmax(asig), asig.shape)
            est = tg.depth_halfwidth(asig, g["E"], g["N"], iy, ix, g["dx"])
            assert np.isfinite(est)
            errors.append(est - depth)
    assert abs(float(np.mean(errors))) < 40.0, errors
    assert max(abs(e) for e in errors) < 80.0, errors


def test_clutter_population_is_bounded_and_varied():
    from src import forward_model as fm
    import numpy as np
    cfg = fm.SurveyConfig()
    rng = np.random.default_rng(0)
    bodies = fm.clutter_bodies(cfg, rng)
    assert len(bodies) == cfg.clutter_count
    assert all(abs(b.east) <= cfg.extent / 2 for b in bodies)
    assert len({round(b.depth) for b in bodies}) > 1


def test_deeper_and_weaker_bodies_produce_smaller_anomalies():
    from src import forward_model as fm
    import numpy as np
    cfg = fm.SurveyConfig()
    ax = np.linspace(-cfg.extent / 2, cfg.extent / 2, cfg.n)
    E, N = np.meshgrid(ax, ax)
    shallow = fm.body_tmi(cfg, fm.Body(0, 0, 200, 0.15, 2.0e7), E, N)
    deep = fm.body_tmi(cfg, fm.Body(0, 0, 1200, 0.05, 5.0e6), E, N)
    assert np.abs(shallow).max() > 10 * np.abs(deep).max()


def test_detection_probability_is_not_trivially_one():
    from src import forward_model as fm
    from src import targeting as tg
    cfg = fm.SurveyConfig()
    g = fm.tmi_grid(cfg, fm.default_scene())
    out, _ = tg.monte_carlo_rank(g["tmi"], g["dx"], g["E"], g["N"], cfg, n_runs=40, seed=3)
    probs = [t["detection_prob"] for t in out]
    assert probs and max(probs) <= 1.0
    assert min(probs) < 0.95, "every target detected on every run: the noise model is too clean"
