"""With which divergence weight is the energy most accurate? Test with exact fields.

For each field family of the calibration (and the relaxed rope), with uniform
10 G noise and HMI-like noise, extrapolates the same preprocessed magnetogram
with several w_d and compares with the true field: total energy, free energy
with respect to the volume potential, relative helicity, E_div/E and gauge error.
Saves ``data/divergence_weight_exact.json``; with ``--package``, also the rows and their summary
(mean |error| without the rope; rope error apart) in ``solarff/divergence_weight.json``,
used by the report.

    python examples/11_divergence_weight_exact.py --weights 1 4 --package
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

from solarff.analytic import CALIBRATION_FIELDS, EXTERNAL_TEST_FIELDS, field_magnetogram
from solarff.calibration import NOISE_MODELS, _nz
from solarff.helicity import helicity_report
from solarff.nlfff_torch import nlfff_lbfgs
from solarff.preprocess import preprocess

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--weights", type=float, nargs="+", default=[1.0, 4.0])
ap.add_argument("--size", type=int, default=48)
ap.add_argument("--noises", nargs="+", default=["uniforme-10G", "HMI (10/60 G)"])
ap.add_argument("--out", default="data/divergence_weight_exact.json")
ap.add_argument("--package", action="store_true", help="escribir filas y resumen en solarff/divergence_weight.json")
a = ap.parse_args()

known = {**CALIBRATION_FIELDS, **EXTERNAL_TEST_FIELDS}
fields = list(CALIBRATION_FIELDS) + ["TD-relax"]
rows = []
for fname in fields:
    exact = known[fname](a.size, a.size, _nz(a.size))
    hx = helicity_report(exact)
    ex = {"E": hx["energy"]["E"], "E_free": hx["energy"]["E_free"], "H_V": hx["helicity"]["H_V"]}
    for j, nm in enumerate(a.noises):
        sz, st = NOISE_MODELS[nm]
        mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=1000 + j)
        pp = preprocess(mag, verbose=False)
        for wd in a.weights:
            t0 = time.time()
            f = nlfff_lbfgs(pp, _nz(a.size), verbose=False, div_weight=wd)
            h = helicity_report(f)
            e = h["energy"]
            r = {"field": fname, "noise": nm, "div_weight": wd,
                 "err_E": e["E"] / ex["E"] - 1, "err_E_free": e["E_free"] / ex["E_free"] - 1,
                 "err_H_V": h["helicity"]["H_V"] / ex["H_V"] - 1 if ex["H_V"] else None,
                 "E_div_frac": e["E_div_frac"], "gauge": h["gauge_diff"], "seconds": time.time() - t0}
            rows.append(r)
            print(f"{fname:9s} {nm:14s} w_d={wd:g}: E {r['err_E']:+.1%}  E_free {r['err_E_free']:+.1%}  "
                  f"H_V {r['err_H_V'] if r['err_H_V'] is None else format(r['err_H_V'], '+.1%')}  "
                  f"E_div/E {r['E_div_frac']:.3f}  gauge {r['gauge']:.1%} ({r['seconds']:.0f}s)", flush=True)
            json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("\nResumen (media de |error| sobre campos y ruidos):")
for wd in a.weights:
    rr = [r for r in rows if r["div_weight"] == wd]
    m = lambda k: np.mean([abs(r[k]) for r in rr if r[k] is not None])
    print(f"w_d={wd:g}: |err E| {m('err_E'):.1%}, |err E_free| {m('err_E_free'):.1%}, |err H_V| {m('err_H_V'):.1%}, "
          f"E_div/E {np.mean([r['E_div_frac'] for r in rr]):.3f}, gauge {np.mean([r['gauge'] for r in rr]):.1%}")

if a.package:
    exact = {}
    for wd in a.weights:
        rr = [r for r in rows if r["div_weight"] == wd and r["field"] != "TD-relax"]
        rope = [r for r in rows if r["div_weight"] == wd and r["field"] == "TD-relax"]
        m = lambda k: float(np.mean([abs(r[k]) for r in rr if r[k] is not None]))  # noqa: E731
        exact[str(float(wd))] = {"err_E_free": m("err_E_free"), "err_H_V": m("err_H_V"), "err_E": m("err_E"),
                                 "gauge": float(np.mean([r["gauge"] for r in rr])),
                                 "E_div_frac": float(np.mean([r["E_div_frac"] for r in rr])),
                                 "rope_err_E_free": float(np.mean([r["err_E_free"] for r in rope])) if rope else None,
                                 "n": len(rr)}
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "solarff", "divergence_weight.json")
    D = json.load(open(pkg, encoding="utf-8"))
    D["exact"], D["exact_rows"] = exact, rows
    json.dump(D, open(pkg, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("resumen escrito en", pkg)
