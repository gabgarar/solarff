# Getting started

## Installation

`solarff` needs Python 3.10 or later.

```bash
git clone https://github.com/gabgarar/solarff.git
cd solarff
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `jsoc` | `drms` | downloading SHARP data from JSOC |
| `lbfgs` | `torch` | the recommended NLFFF solver (exact gradient, L-BFGS) |
| `aia` | `sunpy`, `drms` | sunpy interoperability and the comparison with AIA loops |
| `calibration` | `scikit-learn` | comparing classifiers when recalibrating |
| `dev` | `pytest` | running the tests |

`requirements-lock.txt` pins the exact versions used for the published results.

## First run: an exact solution

Before using real data, check the installation against the analytical field of Low & Lou (1990).
This takes about a minute.

```python
import solarff as sff

mag, exact = sff.low_lou_magnetogram(nx=32, ny=32, nz=24, noise=10.0)

pot = sff.potential_field(mag, nz=24)
nl = sff.nlfff_lbfgs(sff.preprocess(mag), nz=24)

seeds = sff.seeds_from_magnetogram(mag, n=200)
models = {"Exact": exact, "Potential": pot, "NLFFF": nl}
sff.print_table(sff.compare_models(models, reference="Exact", seeds=seeds))
sff.plot_comparison(models, seeds, save="lowlou_lines.png")
```

## Real data: an HMI active region

The input is the SHARP product in cylindrical equal-area projection, `hmi.sharp_cea_720s`, which is
already a vector magnetogram on a local Cartesian grid with the 180° ambiguity resolved.

| Segment | Meaning | In the library |
|---|---|---|
| `Br` | radial component | `Bz` |
| `Bp` | φ component (westward) | `Bx = Bp` |
| `Bt` | θ component (southward) | `By = -Bt` |

Downloading needs an email address [registered at JSOC](http://jsoc.stanford.edu/ajax/register_email.html).
It is only passed to JSOC and never stored.

```python
import solarff as sff

sff.search_sharp("2011-02-15T00:00:00", noaa=11158)          # find the HARP number
folder = sff.download_sharp(377, "2011-02-15T00:00:00", email="you@example.org")

mag = sff.load_sharp_cea(folder, rebin=4).balance_flux()
study = sff.run_study(mag, nz=48)      # potential + LFFF + preprocessing + NLFFF
```

Practical advice:

- Use regions within 30–45° of the central meridian. Farther out the transverse noise grows quickly.
- Reduce the resolution with `rebin=2..4`. A full-resolution SHARP is not tractable.
- The geometry is Cartesian, valid for regions up to about 200–300 Mm.

## Working with sunpy

```python
import sunpy.map
import solarff as sff

mag = sff.from_sunpy(sunpy.map.Map("Br.fits"), sunpy.map.Map("Bp.fits"), sunpy.map.Map("Bt.fits"), rebin=4)
maps = sff.to_sunpy(mag)               # {"Br": Map, "Bp": Map, "Bt": Map}
```

## A PDF report of a region

```python
study = sff.run_study(mag, nz=48)
sff.generate_report(study, "docs/informes/my_region")     # report in English and Spanish
```

The report needs LaTeX (MiKTeX or TeX Live). Without it, the `.tex` files are left ready to compile
elsewhere.

!!! note "Console messages"
    Progress messages printed by the solvers and the command-line help of the example scripts are
    still in Spanish. The API, the docstrings and this documentation are in English.
