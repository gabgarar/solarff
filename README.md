# solarff

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24133574.svg)](https://doi.org/10.5281/zenodo.24133574)
[![Release](https://img.shields.io/github/v/release/gabgarar/solarff)](https://github.com/gabgarar/solarff/releases)
[![tests](https://github.com/gabgarar/solarff/actions/workflows/tests.yml/badge.svg)](https://github.com/gabgarar/solarff/actions/workflows/tests.yml)
[![docs](https://github.com/gabgarar/solarff/actions/workflows/docs.yml/badge.svg)](https://gabgarar.github.io/solarff/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

*Extrapolation of the solar coronal magnetic field from SDO/HMI vector magnetograms, with a
calibrated confidence for every field line.*

**Documentation: <https://gabgarar.github.io/solarff/>**

A nonlinear force-free (NLFFF) extrapolation gives a full 3D field, but not all of it deserves the
same trust. `solarff` computes the extrapolation and then tells you, line by line, how likely each
field line is to reproduce the true geometry.

## What it does

| Task | Functions |
|---|---|
| Potential, linear and nonlinear force-free extrapolation | `potential_field`, `lfff`, `nlfff_lbfgs`, `nlfff` |
| Preprocessing of the vector magnetogram | `preprocess` |
| Field-line tracing | `trace_field_lines`, `seeds_from_magnetogram` |
| **Per-line confidence** | `assess_lines`, `confidence_map`, `reliable_volume` |
| Ranking of the lines fitted on MHD simulations (experimental) | `potential_departure`, `rank_lines` |
| Free energy and relative helicity with their uncertainties | `helicity_report`, `monte_carlo` |
| Twist, decay index and squashing factor Q | `twist_map`, `decay_index_profile`, `squashing_q` |
| SHARP download and reading, sunpy interoperability | `download_sharp`, `load_sharp_cea`, `from_sunpy`, `to_sunpy` |
| Public MHD simulation cubes as a truth | `solarff.simulations.reduced_field` |
| PDF report of a region | `run_study`, `generate_report` |

## Installation

```bash
git clone https://github.com/gabgarar/solarff.git
cd solarff
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

`requirements-lock.txt` pins the exact versions used for the published results.

## Quick start

```python
import solarff as sff

mag = sff.load_sharp_cea("data/harp377", rebin=4).balance_flux()   # SHARP CEA: Br, Bp, Bt
nl = sff.nlfff_lbfgs(sff.preprocess(mag), nz=48)

seeds = sff.seeds_from_magnetogram(mag, n=300)
lines = sff.trace_field_lines(nl, seeds)

a = sff.assess_lines(nl, lines, mag=mag)
a["confidence"]      # one value in [0, 1] per line
a["reliable"]        # confidence >= 0.8
```

To check the installation without data, `examples/01_lowlou_benchmark.py` runs the three models
against the analytical solution of Low & Lou (1990).

## The per-line confidence in brief

- A logistic regression on three indicators of each line: the variation of α along it, the fraction
  inside the boundary layer of the extrapolation, and its length, measured against the horizontal
  size and against the height of the box.
- Calibrated on 162 extrapolations of exact force-free fields, in boxes of four proportions
  (48 062 lines with known true error), and validated by leaving out one whole family of fields at
  a time.
- With the default threshold, 51 % of the lines are accepted and 4.4 % of them are wrong, against
  29 % of all lines.
- On reference fields built on observed regions the accepted lines are wrong about twice as often
  (median 9 %), and the extrapolations of real regions are not converged with resolution.
- **Against a radiative MHD simulation of an active region, taken as the truth, it does not hold:**
  62 % of the lines are wrong and so are 51 % of the accepted ones. A treatment of the base that
  trusts it less lowers the wrong lines from 64 to 48 % on snapshots never used before. See
  [Tests against simulations](https://gabgarar.github.io/solarff/realistic/).
- The probability holds within the calibration domain. Read the
  [limits](https://gabgarar.github.io/solarff/confidence/#known-limits) before applying it.

## Where to find the rest

| Topic | Page |
|---|---|
| Physical context, equations, inputs and outputs of each step | [Method and equations](https://gabgarar.github.io/solarff/method/) |
| Installation and recipes for each use case, with every parameter explained | [Getting started](https://gabgarar.github.io/solarff/getting-started/) |
| How to apply the confidence, and its limits | [Per-line confidence](https://gabgarar.github.io/solarff/confidence/) |
| The confidence against MHD simulations, and the recommended treatment of a photospheric base | [Tests against simulations](https://gabgarar.github.io/solarff/realistic/) |
| Energy, helicity, twist, decay index, QSLs, Monte Carlo, figures, limitations | [Diagnostics](https://gabgarar.github.io/solarff/diagnostics/) |
| Every figure and number from the scripts | [Reproducing the results](https://gabgarar.github.io/solarff/reproduce/) |
| All functions and classes | [API reference](https://gabgarar.github.io/solarff/api/core/) |

## Repository layout

| Path | Content |
|---|---|
| `solarff/` | the library |
| `examples/` | numbered scripts that produce every result |
| `tests/` | unit and integration tests (`pytest`) |
| `tools/` | reproduction script, interactive 3D view of the field lines (`lines3d.py`) and documentation helpers |
| `docs/resultados/`, `docs/revision/resultados/` | result files behind the published numbers, including the per-line table of the calibration (`docs/resultados/line_table.md`) |
| `docs/web/` | source of the documentation site |

## Development note

AI-assisted tools provided support during software development, technical documentation, and language
editing. All scientific and methodological decisions, final implementation, testing, validation, and
maintenance remain the responsibility of the author.

## License and citation

`solarff` is distributed under the [BSD 3-Clause](LICENSE) license. If you use it in academic work,
please cite the software (`CITATION.cff`; GitHub shows it under "Cite this repository").

> García, G. (2026). *solarff: NLFFF extrapolations with a calibrated per-line confidence* (v1.3.0).
> Zenodo. <https://doi.org/10.5281/zenodo.24378815>

The DOI [10.5281/zenodo.24133574](https://doi.org/10.5281/zenodo.24133574) always resolves to the latest version.
