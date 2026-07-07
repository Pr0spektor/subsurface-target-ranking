"""
Real-data loader — apply the identical pipeline to a real aeromagnetic survey.
=============================================================================

The processing/targeting code is grid-agnostic: give it a gridded TMI array with a cell size and a
field inclination/declination and it runs unchanged. This module fetches a **real, open** magnetic
survey via `ensaio` (Fatiando's curated real datasets) and grids it with `verde`.

These packages need internet + a scientific-Python environment, so run this in Colab or locally
(`pip install ensaio harmonica verde pooch`), not in a restricted sandbox.

Example
-------
    from src import real_data, targeting as tg, processing as pf
    grid = real_data.load_lightning_creek()          # real Queensland aeromagnetic survey
    targets, detr = tg.monte_carlo_rank(grid["tmi"], grid["dx"], grid["E"], grid["N"], grid["cfg"])
"""
from __future__ import annotations
import numpy as np


def load_lightning_creek(spacing=200.0):
    """Real Lightning Creek aeromagnetic survey (Mount Isa, Queensland) via ensaio + verde.

    Returns a dict with the same keys the pipeline expects: E, N (grids), tmi, dx, cfg.
    """
    import ensaio, pandas as pd, verde as vd
    from types import SimpleNamespace

    fname = ensaio.fetch_lightning_creek_magnetic(version=1)
    df = pd.read_csv(fname)
    # project lon/lat to metres and grid the total-field anomaly
    proj = vd.projection.pyproj_wrap if hasattr(vd, "projection") else None
    coords = (df["easting_m"].values, df["northing_m"].values) if "easting_m" in df else \
             (df["longitude"].values, df["latitude"].values)
    field = df["total_field_anomaly_nt"].values if "total_field_anomaly_nt" in df else \
            df.filter(like="anomaly").iloc[:, 0].values
    grd = vd.grid_coordinates(vd.get_region(coords), spacing=spacing)
    tmi = vd.KNeighbors().fit(coords, field).grid(coordinates=grd).scalars.values
    E, N = grd
    cfg = SimpleNamespace(inc_deg=-52.0, dec_deg=6.0, noise_nT=float(np.nanstd(field) * 0.02))
    return {"E": E, "N": N, "tmi": np.nan_to_num(tmi, nan=np.nanmean(tmi)),
            "dx": spacing, "cfg": cfg, "truth": np.empty((0, 4))}


if __name__ == "__main__":
    g = load_lightning_creek()
    print("real grid", g["E"].shape, "dx", g["dx"], "TMI nT", round(float(np.nanmin(g["tmi"])), 1),
          round(float(np.nanmax(g["tmi"])), 1))
