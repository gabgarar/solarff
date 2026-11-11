"""Validation with loops observed by SDO/AIA: do the reliable lines align better?

For HARP 377 (2011-02-15 00:00 TAI) the NLFFF lines (with their confidence) and
the same seeds traced in the potential field and in the LFFF are projected onto
the closest AIA 171 Å and 193 Å images, and the misalignment angle with the loops
is measured (:func:`solarff.aia.aia_validation`):

* reliable versus unreliable lines, also **at equal height** (terciles and
  regression with height), because the reliable ones tend to be lower;
* NLFFF versus potential and LFFF with the same seeds, by group;
* sensitivity to the scale and to the loop detection threshold.

The email is used only to download from JSOC (it is not stored). Saves
``data/aia_validation_harp377.pkl`` (used by the HARP 377 report) and, with ``--json``,
a copy of the summary in JSON.

    python examples/08_aia_validation.py --email you@example.com
"""
import argparse
import json
import pickle
import warnings

import numpy as np

from solarff.aia import aia_validation, download_aia
from solarff.report.analysis import analyze
from solarff.tracing import trace_field_lines
from solarff.utils import code_version

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--email", help="email registrado en JSOC (sólo si hay que descargar)")
ap.add_argument("--cache", default="data/harp377_study_mc10.pkl")
ap.add_argument("--sharp", default="data/harp377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits")
ap.add_argument("--time-utc", default="2011-02-14T23:59:19.75Z", help="T_OBS de HMI en UTC (TAI − 34 s)")
ap.add_argument("--out", default="data/aia_validation_harp377.pkl")
ap.add_argument("--region", default="HARP 377 / NOAA 11158")
ap.add_argument("--json", default=None, help="resumen JSON adicional (opcional; el resumen ya va en --out)")
a = ap.parse_args()
if a.time_utc == "auto":  # T_OBS of the SHARP (TAI) -> UTC
    from astropy.io import fits
    from astropy.time import Time
    t_obs = fits.getheader(a.sharp, 1 if len(fits.open(a.sharp)) > 1 else 0)["T_OBS"].replace("_TAI", "").replace("_", "T").replace(".", "-", 2)
    a.time_utc = Time(t_obs, scale="tai").utc.isot + "Z"
    print("T_OBS en UTC:", a.time_utc)

study, mc = pickle.load(open(a.cache, "rb"))
print("Analizando el modelo (confianza por línea)...", flush=True)
A = analyze(study, monte_carlo=mc)
cm = A["_maps"]["confidence"]
lines = cm["lines"]
seeds = np.array([l.seed for l in lines])
other = {k: trace_field_lines(f, seeds) for k, f in (("potential", study.potential), ("lfff", study.lfff))
         if f is not None}
files = download_aia(a.time_utc, (171, 193), email=a.email)
res = aia_validation(lines, cm["assess"]["confidence"], cm["assess"]["reliable"], files, a.sharp, other)
res["summary"].update(code_version=code_version()["label"], calibration=cm["assess"]["calibration"],
                      region=a.region)
with open(a.out, "wb") as fh:
    pickle.dump(res, fh)
if a.json:
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res["summary"], fh, indent=1, ensure_ascii=False)
for w, r in res["summary"]["by_wavelength"].items():
    print(f"AIA {w} Å: {r['n_measured']} líneas; fiables {r['median_reliable']:.1f}°, no fiables "
          f"{r['median_unreliable']:.1f}° (p = {r['p_mannwhitney']:.2g}); con la altura: {r['ols_reliable']:+.1f}° "
          f"{np.round(r['ols_reliable_ci'], 1).tolist()}")
    for k, v in r["models"].items():
        print(f"   NLFFF − {k}: fiables {v['reliable']['median']:+.1f}° (p={v['reliable']['p']:.2g}), "
              f"no fiables {v['unreliable']['median']:+.1f}° (p={v['unreliable']['p']:.2g})")
    t = r["terciles"][1]
    print(f"   tercil medio: NLFFF − potencial fiables {t['nlfff_minus_potential_reliable']}, "
          f"no fiables {t['nlfff_minus_potential_unreliable']}")
print("guardado en", a.out)
