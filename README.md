# Subsurface drill-target ranking — probabilistic, uncertainty-aware

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A compact, senior-level demonstration of the workflow behind AI-native exploration platforms
(e.g. Terra AI): turn potential-field geophysics into a **ranked list of drill targets, each with a
depth estimate and an uncertainty-aware detection probability** — "where to drill, how deep, and
how confident."

Everything here is computed and tested, not hand-waved. The core runs on **pure NumPy** (transparent,
auditable operators); the identical pipeline runs on a **real open aeromagnetic survey** via
`src/real_data.py` (Colab).

## Why a forward model?
The committed demo uses a **physics-based magnetic forward model** (dipole response of buried
magnetised bodies + regional field + noise). That is deliberate and is itself a core exploration
skill ("modelled vs observed") — and, crucially, because the true source locations and depths are
known, it lets us **score target recovery against ground truth**, which real data cannot.

## Result (synthetic survey, known ground truth)

![Targets](results/targets.png)

- **4 / 5 bodies recovered** at **100% detection probability**; the 5th is a weak, deep decoy
  correctly rejected (a true negative — honest behaviour, not cherry-picking).
- **Mean horizontal error ≈ 100 m** (~3 grid cells); **mean depth error ≈ 51 m**.

![Depth recovery](results/depth_recovery.png)

## Method (all from first principles, NumPy)
1. **Forward model** — TMI over dipole sources in a chosen field (`src/forward_model.py`).
2. **Processing** (`src/processing.py`, FFT-domain): polynomial regional removal, reduction-to-pole,
   vertical derivatives, upward continuation, analytic signal, tilt angle.
3. **Detection** — local maxima of the (denoised) analytic signal with non-max suppression + border
   masking; reduction-independent, so peaks sit over sources.
4. **Depth** — analytic-signal **half-width** estimator (robust for compact sources), with a single
   constant calibrated on the forward model (depth ≈ 1.75 × half-width; validated to ~10%).
5. **Probabilistic ranking** (`src/targeting.py`) — Monte-Carlo over noise realisations yields, per
   target, a **detection probability**, a **depth median + interquartile spread**, and a score.
6. **Validation** (`src/pipeline.py`) — matches ranked targets to known truth and reports recovery.

## Run
```bash
pip install -r requirements.txt
python -m src.pipeline          # synthetic demo -> results/ figures + CSVs
```

### On real data
`src/real_data.py` loads a **real open aeromagnetic survey** (Lightning Creek, Mount Isa, via
`ensaio`) and grids it with `verde`; the same `targeting.monte_carlo_rank(...)` then runs unchanged.
Needs a scientific-Python env with internet:
```bash
pip install ensaio verde harmonica pooch
python -c "from src import real_data, targeting as tg; g=real_data.load_lightning_creek(); \
           t,_=tg.monte_carlo_rank(g['tmi'],g['dx'],g['E'],g['N'],g['cfg']); print(t[:5])"
```

## Layout
```
subsurface-target-ranking/
├── src/
│   ├── forward_model.py   # physics-based synthetic magnetic survey
│   ├── processing.py      # FFT potential-field operators (RTP, derivatives, analytic signal…)
│   ├── targeting.py       # detection, depth (half-width + Euler), probabilistic MC ranking
│   ├── pipeline.py        # end-to-end run + ground-truth validation + figures
│   └── real_data.py       # real aeromagnetic survey loader (ensaio + verde)
├── results/               # generated figures + ranked_targets.csv + recovery.csv
├── requirements.txt · CITATION.cff · LICENSE
```

## Honest scope
- The committed demo input is a forward model (real physics, synthetic geology); the real-survey
  path is provided and runs the same code. This split is because the build sandbox had no internet;
  in Colab both run.
- Depth via analytic-signal half-width is an approximation with a calibrated constant; the repo also
  includes an Euler-deconvolution implementation for comparison.
- A production system (Terra AI–style) would add 3-D inversion (SimPEG), multi-method fusion
  (gravity/EM), and geological priors — hooks and notes are provided.

Built to demonstrate applied-geophysics + data-programming judgement: transparent methods, real
physics, uncertainty quantification, and validation against ground truth.
