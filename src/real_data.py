"""
Real-data loader — apply the identical pipeline to a real aeromagnetic survey.
=============================================================================

The processing/targeting code is grid-agnostic: give it a gridded TMI array, a cell size and the
field inclination/declination and it runs unchanged. This module loads a **real, open** aeromagnetic
survey (Great Britain, British Geological Survey) via `ensaio`, projects and grids it with `verde`,
and returns it in the exact structure the pipeline expects.

Requires a scientific-Python environment with internet (Colab or local):
    pip install ensaio verde pyproj pooch

Example
-------
    from src import real_data, targeting as tg
    g = real_data.load_britain_magnetic(spacing=2000.0)
    targets, _ = tg.monte_carlo_rank(g["tmi"], g["dx"], g["E"], g["N"], g["cfg"], n_runs=60)
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np


def load_britain_magnetic(spacing: float = 2000.0):
    """Great Britain aeromagnetic anomaly (BGS), projected and gridded to `spacing` metres."""
    import ensaio
    import pandas as pd
    import pyproj
    import verde as vd

    fname = ensaio.fetch_britain_magnetic(version=1)
    data = pd.read_csv(fname)
    field = data["total_field_anomaly_nt"].to_numpy()

    proj = pyproj.Proj(proj="merc", lat_ts=float(data["latitude"].mean()))
    easting, northing = proj(data["longitude"].to_numpy(), data["latitude"].to_numpy())
    coords = (easting, northing)

    # de-spike / decimate to the target resolution, then grid with nearest-neighbour
    block_coords, block_field = vd.BlockReduce(np.median, spacing=spacing).filter(coords, field)
    grd = vd.KNeighbors().fit(block_coords, block_field)
    ds = grd.grid(spacing=spacing, data_names="tmi")

    E, N = np.meshgrid(ds.easting.to_numpy(), ds.northing.to_numpy())
    tmi = np.nan_to_num(ds.tmi.to_numpy(), nan=float(np.nanmedian(ds.tmi.to_numpy())))
    # Britain reference field (approx): inclination ≈66 deg, declination ≈ -1 deg
    cfg = SimpleNamespace(inc_deg=66.0, dec_deg=-1.0, noise_nT=float(np.nanstd(field) * 0.02))
    return {"E": E, "N": N, "tmi": tmi, "dx": spacing, "cfg": cfg,
            "truth": np.empty((0, 4))}


if __name__ == "__main__":
    g = load_britain_magnetic()
    print("real grid", g["E"].shape, "dx", g["dx"],
          "TMI nT", round(float(np.nanmin(g["tmi"])), 1), round(float(np.nanmax(g["tmi"])), 1))
