"""How much the preprocessing changes the magnetogram: real regions and an exact base.

For each region of the paper (observed and preprocessed magnetograms saved in the study) the
relative change in norm of B_z, of B_t and of the full vector, and the Aly criteria (net force
and torque) before and after are measured; the observed one is also preprocessed again with ``fix_bz=True`` (only the
transverse field changes) to give the alternative that was not used. On the exact base of Low & Lou I (already
force-free) at 32², 48² and 64² the same is measured with the default values (μ₃ = μ₄ = 0.01) and without
smoothing (μ₄ = 0): the effect of the smoothing depends on the resolution, and this is the check cited in Sect. 3.2.

Writes ``docs/resultados/preprocess_change.json``.

    python examples/19_preprocess_change.py        # ~2 min
"""
import json
import os
import pickle
import warnings

import numpy as np

from solarff.analytic import CALIBRATION_FIELDS, field_magnetogram
from solarff.calibration import _nz
from solarff.magnetogram import aly_criteria
from solarff.preprocess import preprocess
from solarff.utils import code_version

warnings.filterwarnings("ignore")
OUT = os.path.join("docs", "resultados", "preprocess_change.json")
REGIONS = {"11158": "data/harp377_study_mc10.pkl", "11429": "data/harp1449_20120306_2336_r7_nz32_mc10.pkl",
           "12673": "data/harp7115_20170906_1136_r4_nz56_mc10.pkl"}


def rel(a, b):
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)) / np.linalg.norm(np.asarray(a)))


def change(mag, pp):
    bt0, bt1 = np.hypot(mag.bx, mag.by), np.hypot(pp.bx, pp.by)
    v0, v1 = np.stack([mag.bx, mag.by, mag.bz]), np.stack([pp.bx, pp.by, pp.bz])
    return {"bz": rel(mag.bz, pp.bz), "bt": rel(bt0, bt1), "vector": rel(v0, v1),
            "aly_before": aly_criteria(mag), "aly_after": aly_criteria(pp)}


out = {"code_version": code_version()["label"], "regions": {}, "exact": {}}
for noaa, p in REGIONS.items():
    if not os.path.exists(p):
        continue
    st, _ = pickle.load(open(p, "rb"))
    pp = st.mag_pp if st.mag_pp is not None else preprocess(st.mag, verbose=False)
    r = {"default": change(st.mag, pp), "fix_bz": change(st.mag, preprocess(st.mag, fix_bz=True, verbose=False))}
    out["regions"][noaa] = r
    print(f"NOAA {noaa}: por defecto B_z {r['default']['bz']:.1%}, B_t {r['default']['bt']:.1%}, vector {r['default']['vector']:.1%} "
          f"(eps_force {r['default']['aly_before']['eps_force']:.3f} → {r['default']['aly_after']['eps_force']:.3f}); "
          f"fix_bz: B_t {r['fix_bz']['bt']:.1%} (eps_force → {r['fix_bz']['aly_after']['eps_force']:.3f})")

for n in (32, 48, 64):  # the smoothing depends on the resolution: given for the three of the suite
    exact = CALIBRATION_FIELDS["LL-I"](n, n, _nz(n))
    mag = field_magnetogram(exact, noise=0.0, noise_t=0.0, rng=0)
    out["exact"][str(n)] = {}
    for name, kw in (("default", {}), ("no_smoothing", {"mu4": 0.0})):
        out["exact"][str(n)][name] = change(mag, preprocess(mag, verbose=False, **kw))
        e = out["exact"][str(n)][name]
        print(f"LL-I {n}² sin ruido, {name:12s}: B_z {e['bz']:.1%}, B_t {e['bt']:.1%}, vector {e['vector']:.1%}")
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("guardado", OUT)
