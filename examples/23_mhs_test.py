"""Test of the per-line confidence with a base that is not force-free, without recalibration.

The published model (``DEFAULT_CALIBRATION``) is applied as is to two families of exact magnetohydrostatic
fields (Low 1991; ``solarff.analytic.mhs_multipole_field``): the regions and alpha of ``LFFF-bip`` and
``LFFF-quad`` with a current perpendicular to the field in the lowest layers, so that the synthetic
magnetogram violates the force-free assumption as a photospheric one does. The reference lines are those of
the exact magnetohydrostatic field. Same suite as the calibration: 3 noise models x 3 resolutions x 2
realizations per family. Criteria registered before running in the plan of the paper (12-Nov-2026).

Saves ``data/mhs_families.pkl`` and the summary in ``docs/resultados/mhs_families.json``.

    python examples/23_mhs_test.py                    # ~20 min
    python examples/23_mhs_test.py --fields MHS-bip
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.analytic import EXTERNAL_TEST_FIELDS
from solarff.calibration import NOISE_MODELS, _nz, bootstrap_ci, calibration_suite, ece
from solarff.confidence import CALIBRATION_V1, DEFAULT_CALIBRATION, auc, confidence
from solarff.magnetogram import Magnetogram, aly_criteria
from solarff.metrics import force_free_metrics
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--fields", nargs="*", default=["MHS-bip", "MHS-quad"])
ap.add_argument("--sizes", type=int, nargs="+", default=[32, 48, 64])
ap.add_argument("--out", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "mhs_families.json"))
a = ap.parse_args()


def top_fraction(d, q, cal=DEFAULT_CALIBRATION):
    """Wrong lines among the fraction ``q`` with the highest confidence of each extrapolation."""
    wrong = n = 0
    for x in d:
        c = confidence(x["ind"], cal)
        ok = np.isfinite(c)
        order = np.argsort(-c[ok])
        k = max(1, int(round(q * ok.sum())))
        wrong += int(x["bad"][ok][order[:k]].sum())
        n += k
    return wrong / n


def evaluate(d, cal=DEFAULT_CALIBRATION):
    """Metrics of the published model on the cases ``d`` (intervals by bootstrap grouped by extrapolation)."""
    c = np.concatenate([confidence(x["ind"], cal) for x in d])
    y = np.concatenate([x["bad"] for x in d]).astype(int)
    g = np.concatenate([np.full(len(x["bad"]), i) for i, x in enumerate(d)])
    ok = np.isfinite(c)
    c, y, g = c[ok], y[ok], g[ok]
    p = 1 - c
    out = {"n_cases": int(len(d)), "n_lines": int(y.size), "bad_rate": float(y.mean()), "auc": auc(p, y),
           "auc_ci": list(bootstrap_ci(p, y, auc, groups=g, n_boot=500)), "ece": ece(p, y), "mean_predicted_bad": float(p.mean()),
           "top20_bad": top_fraction(d, 0.2, cal)}
    for t in (0.8, 0.9, 0.95):
        acc = c >= t
        out[f"accepted_{t}"] = float(acc.mean())
        out[f"bad_accepted_{t}"] = float(y[acc].mean()) if acc.any() else None
    acc = c >= 0.8
    risk = lambda pp, yy: float(yy[pp <= 0.2].mean()) if (pp <= 0.2).any() else float("nan")  # noqa: E731
    out["bad_accepted_0.8_ci"] = list(bootstrap_ci(p, y, risk, groups=g, n_boot=500)) if acc.any() else None
    return out


def describe(name, sizes):
    """How far from force-free the exact field is: net force of its base and current-weighted sine of the J-B angle."""
    rows = {}
    for n in sizes:
        f = EXTERNAL_TEST_FIELDS[name](n, n, _nz(n))
        base = Magnetogram(bz=f.bz[:, :, 0], bx=f.bx[:, :, 0], by=f.by[:, :, 0], dx=f.dx)
        rows[str(n)] = {"dx_Mm": float(f.dx), "eps_force": aly_criteria(base)["eps_force"], "cwsin": float(force_free_metrics(f)["CWsin"])}
        meta = {k: f.meta[k] for k in ("alpha_frac", "a", "scale_mm")}
    return {"params": meta, "exact": rows}


out = {"code_version": code_version()["label"], "calibration": "DEFAULT_CALIBRATION (three indicators in five terms, six families in boxes of four proportions), not refitted",
       "sizes": list(a.sizes), "noise_models": list(NOISE_MODELS), "realizations": 2, "families": {}}
data = {}
with keep_awake():
    for name in a.fields:
        t0 = time.time()
        d = calibration_suite(fields=[name], sizes=tuple(a.sizes), realizations=2,
                              checkpoint=os.path.join(a.out, f"mhs_{name}.ckpt"))
        data[name] = d
        s = {**describe(name, a.sizes), "all": evaluate(d), "all_v1": evaluate(d, CALIBRATION_V1), "by_size": {str(n): evaluate([x for x in d if x["size"] == n]) for n in a.sizes},
             "by_noise": {nm: evaluate([x for x in d if x["noise_model"] == nm]) for nm in NOISE_MODELS}, "seconds": time.time() - t0}
        out["families"][name] = s
        r = s["all"]
        print(f"{name:9s} wrong {r['bad_rate']:.0%} | AUC {r['auc']:.3f} ECE {r['ece']:.3f} | accepted {r['accepted_0.8']:.0%}, wrong among them "
              f"{r['bad_accepted_0.8']:.3f} (0.9: {r['bad_accepted_0.9']}, top 20%: {r['top20_bad']:.3f}) | version 1.1: AUC "
              f"{s['all_v1']['auc']:.3f}, wrong among the accepted {s['all_v1']['bad_accepted_0.8']:.3f}", flush=True)
        pickle.dump(data, open(os.path.join(a.out, "mhs_families.pkl"), "wb"))
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("written", a.json)
