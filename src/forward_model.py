"""
Magnetic forward model (physics-based synthetic survey)
=======================================================

TMI anomaly over buried, induced-magnetised compact bodies (point-dipole approximation) in a chosen
ambient geomagnetic field, on a regular grid, plus a smooth regional field and observational noise.

A forward model is the "modelled" half of modelled-vs-observed, and — because the true source
locations/depths are known — it lets us *score target recovery against ground truth*. The identical
downstream code runs on real survey grids (see src/real_data.py).

Coordinate convention throughout the repo: component/vector order = [East, North, Down];
grids indexed [row=North (axis0), col=East (axis1)]. Pure NumPy.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MU0 = 4e-7 * np.pi


def field_unit_vector(inc_deg, dec_deg):
    i, d = np.radians(inc_deg), np.radians(dec_deg)
    return np.array([np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), np.sin(i)])  # [E, N, Down]


@dataclass
class Body:
    east: float; north: float; depth: float
    susceptibility: float = 0.05
    volume: float = 3.0e5


@dataclass
class SurveyConfig:
    n: int = 121
    extent: float = 4000.0
    height: float = 60.0
    inc_deg: float = 60.0
    dec_deg: float = 5.0
    field_strength: float = 50000.0
    noise_nT: float = 2.0
    regional_amp_nT: float = 60.0
    seed: int = 11


def default_scene():
    """Realistic mineralised bodies (magnetite-bearing): tens–hundreds of nT anomalies."""
    return [
        Body(-900, -1100, 300, 0.15, 2.0e7),    # strong, shallow (~400 nT)
        Body(600, 700, 550, 0.10, 2.5e7),       # strong, mid (~75 nT)
        Body(-1000, 1200, 800, 0.20, 4.0e7),    # deep but strong (~90 nT)
        Body(1100, -800, 450, 0.08, 1.5e7),     # moderate (~60 nT)
        Body(-200, 200, 950, 0.03, 5.0e6),      # weak/deep decoy (~1 nT, near noise)
    ]


def tmi_grid(cfg: SurveyConfig, bodies):
    rng = np.random.default_rng(cfg.seed)
    ax = np.linspace(-cfg.extent / 2, cfg.extent / 2, cfg.n)
    E, N = np.meshgrid(ax, ax)                 # E along columns, N along rows
    Fhat = field_unit_vector(cfg.inc_deg, cfg.dec_deg)
    B0 = cfg.field_strength
    tmi = np.zeros_like(E)
    for b in bodies:
        m = (b.susceptibility / MU0) * (B0 * 1e-9) * Fhat * b.volume    # [E,N,Down]
        rE, rN, rD = E - b.east, N - b.north, (cfg.height + b.depth)
        r = np.sqrt(rE**2 + rN**2 + rD**2) + 1e-6
        rhat = np.stack([rE, rN, rD * np.ones_like(rE)]) / r
        mdotr = m[0] * rhat[0] + m[1] * rhat[1] + m[2] * rhat[2]
        Bvec = (MU0 / (4 * np.pi)) * (3 * mdotr * rhat - m[:, None, None]) / r**3
        tmi += (Bvec[0] * Fhat[0] + Bvec[1] * Fhat[1] + Bvec[2] * Fhat[2]) * 1e9
    reg = cfg.regional_amp_nT * (0.6 * E / cfg.extent + 0.4 * N / cfg.extent
                                 + 0.3 * E * N / cfg.extent**2)
    noise = rng.normal(0, cfg.noise_nT, E.shape)
    truth = np.array([[b.east, b.north, b.depth, b.susceptibility] for b in bodies])
    return {"ax": ax, "E": E, "N": N, "tmi": tmi + reg + noise, "regional": reg,
            "dx": ax[1] - ax[0], "truth": truth, "cfg": cfg}


if __name__ == "__main__":
    g = tmi_grid(SurveyConfig(), default_scene())
    print("grid", g["E"].shape, "dx", round(g["dx"], 1),
          "TMI nT range", round(g["tmi"].min(), 1), round(g["tmi"].max(), 1))
