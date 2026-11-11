"""External test of the per-line confidence on new families, without recalibration.

The published model (``DEFAULT_CALIBRATION``: three indicators, six families) is applied as is to
exact fields that took part neither in the calibration nor in the design: ``LL-IV`` (Low & Lou with
another depth and inclination), ``LFFF-delta`` (exact linear field, compact configuration with high α) and
``TD3-relax`` (third relaxed flux rope, thick and shallow). Same suite as the
calibration: 3 noise models × 3 resolutions × 2 realizations per family. For comparison, the
external rope ``TD2-relax`` of ``data/revision/suite_r0.pkl`` is evaluated in the same way if present.

Saves ``data/external_families.pkl`` and the summary in ``docs/resultados/external_families.json``.

    python examples/17_external_families.py                     # ~1 h
    python examples/17_external_families.py --fields LL-IV      # one family
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
from solarff.confidence import DEFAULT_CALIBRATION, auc, confidence
from solarff.metrics import force_free_metrics
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--fields", nargs="*", default=["LL-IV", "LFFF-delta", "TD3-relax"])
ap.add_argument("--sizes", type=int, nargs="+", default=[32, 48, 64])
ap.add_argument("--out", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "external_families.json"))
a = ap.parse_args()


def evaluate(d, cal=DEFAULT_CALIBRATION):
    """Metrics of the published model on the cases ``d`` (CI by bootstrap grouped by case)."""
    c = np.concatenate([confidence(x["ind"], cal) for x in d])
    y = np.concatenate([x["bad"] for x in d]).astype(int)
    g = np.concatenate([np.full(len(x["bad"]), i) for i, x in enumerate(d)])
    ok = np.isfinite(c)
    c, y, g = c[ok], y[ok], g[ok]
    p = 1 - c
    acc = c >= cal["threshold"]
    return {"n_cases": int(len(d)), "n_lines": int(y.size), "bad_rate": float(y.mean()), "auc": auc(p, y),
            "auc_ci": list(bootstrap_ci(p, y, auc, groups=g, n_boot=500)),
            "ece": ece(p, y), "ece_ci": list(bootstrap_ci(p, y, ece, groups=g, n_boot=500)),
            "accepted": float(acc.mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None,
            "mean_predicted_bad": float(p.mean())}


def summarize(name, d):
    s = {"all": evaluate(d), "by_noise": {}, "by_size": {}, "exact_cwsin": {}}
    for nm in NOISE_MODELS:
        s["by_noise"][nm] = evaluate([x for x in d if x["noise_model"] == nm])
    for n in sorted({x["size"] for x in d}):
        s["by_size"][str(n)] = evaluate([x for x in d if x["size"] == n])
        f = EXTERNAL_TEST_FIELDS[name](n, n, _nz(n))
        s["exact_cwsin"][str(n)] = float(force_free_metrics(f)["CWsin"])
        s["params"] = {k: v for k, v in f.meta.items()
                       if k in ("n", "m", "l", "phi", "alpha_frac", "spots", "R", "a", "d", "L", "n_turns", "handedness")}
    return s


out = {"code_version": code_version()["label"], "calibration": "DEFAULT_CALIBRATION (3 indicadores, 6 familias)",
       "sizes": list(a.sizes), "noise_models": list(NOISE_MODELS), "realizations": 2, "families": {}}
data = {}
with keep_awake():
    for name in a.fields:
        t0 = time.time()
        try:
            d = calibration_suite(fields=[name], sizes=tuple(a.sizes), realizations=2,
                                  checkpoint=os.path.join(a.out, f"external_{name}.ckpt"))
        except Exception as e:  # one failing family must not bring down the others
            print(f"{name}: ERROR {type(e).__name__}: {e}")
            out["families"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        data[name] = d
        out["families"][name] = {**summarize(name, d), "seconds": time.time() - t0}
        r = out["families"][name]["all"]
        print(f"{name:11s} malas {r['bad_rate']:.0%} | AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}] "
              f"ECE {r['ece']:.3f} | ≥ 0.8: {r['accepted']:.0%} aceptadas, {r['bad_accepted']:.1%} malas "
              f"({out['families'][name]['seconds'] / 60:.0f} min)")
        pickle.dump(data, open(os.path.join(a.out, "external_families.pkl"), "wb"))
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# Reference: the external rope of the paper, evaluated in the same way
r0 = os.path.join(a.out, "revision", "suite_r0.pkl")
if os.path.exists(r0):
    td2 = [x for x in pickle.load(open(r0, "rb")) if x["field"] == "TD2-relax"]
    if td2:
        out["families"]["TD2-relax"] = {**summarize("TD2-relax", td2), "source": r0}
        r = out["families"]["TD2-relax"]["all"]
        print(f"{'TD2-relax':11s} malas {r['bad_rate']:.0%} | AUC {r['auc']:.3f} ECE {r['ece']:.3f} | "
              f"≥ 0.8: {r['accepted']:.0%} aceptadas, {r['bad_accepted']:.1%} malas (referencia, suite_r0)")
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("guardado", a.json)
