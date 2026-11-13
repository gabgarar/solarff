"""Revision R8: why the model uses 3 of the 9 indicators, and what the fifth term gives.

For each indicator: individual AUC, sign of its coefficient when added to the model
(positive = a larger value indicates a worse line) and leave-one-family-out AUC/ECE when
added to the model (or, for the 3 of the model, when removed; the length is removed with its three terms).
The model has five terms: the length enters relative to the horizontal diagonal of the box, squared, and
relative to its height. ``fifth_term`` compares, by box proportion, the model with and without the last one.

    python examples/revision/r8_indicators.py
"""
import json
import os
import pickle
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np  # noqa: E402

from solarff.calibration import FEATURE_SIGN, USED_FEATURES, V1_FEATURES, _fit_logistic, box_name, cross_validate, ece, stack  # noqa: E402
from solarff.confidence import auc  # noqa: E402

warnings.filterwarnings("ignore")
S = pickle.load(open("data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax_boxes.pkl", "rb"))
data, single = S["data"], S["single"]
base = list(USED_FEATURES)
LENGTH_TERMS = ("length", "length_sq", "length_h")


def cv(feats):
    c = cross_validate(data, features=feats, models=["logistic"], group="field", verbose=False)
    p, y = c["models"]["logistic"]["p"], c["y"].astype(float)
    return float(auc(p, y)), float(ece(p, y))


a0, e0 = cv(base)
R = {"base": {"features": base, "auc": a0, "ece": e0}, "indicators": {}}
for f in ("sin_jb", "alpha_spread", "div", "buffer", "open", "foot_snr", "base_change", "apex", "length"):
    if f in base:
        feats = [x for x in base if x != f and not (f == "length" and x in LENGTH_TERMS)]
        mode = "quitar"
    else:
        feats = base + [f]
        mode = "añadir"
    a, e = cv(feats)
    X, y, _ = stack(data, base + ([f] if f not in base else []))
    cal = _fit_logistic(X, y.astype(float))
    coef = cal["coef"][(base + [f]).index(f) if f not in base else base.index(f)]
    R["indicators"][f] = {"single_auc": single.get(f), "mode": mode, "auc": a, "ece": e, "d_auc": a - a0,
                          "coef": float(coef), "expected_sign": FEATURE_SIGN.get(f)}
    print(f"{f:13s} AUC sola {single.get(f, float('nan')):.3f} | {mode} -> AUC {a:.3f} ({a - a0:+.3f}), ECE {e:.3f} | "
          f"coef {coef:+.2f} (signo esperado {FEATURE_SIGN.get(f):+d})")


def by_box(feats):
    """Leave-one-family-out figures of a list of terms, in all the boxes and in those of each proportion."""
    c = cross_validate(data, features=feats, models=["logistic"], group="field", verbose=False)
    p, y = c["models"]["logistic"]["p"], c["y"].astype(float)
    box = np.array([box_name(d) for d in data])[np.asarray(c["case"])]
    out = {}
    for g in ("all", "standard", "flat", "very flat", "tall"):
        m = np.ones(len(y), bool) if g == "all" else box == g
        acc = (1 - p[m]) >= 0.8
        out[g] = {"auc": float(auc(p[m], y[m])), "ece": float(ece(p[m], y[m])), "accepted": float(acc.mean()),
                  "bad_accepted": float(y[m][acc].mean()), "n_lines": int(m.sum())}
    return out


R["fifth_term"] = {"with": by_box(base), "without": by_box(list(V1_FEATURES))}
for g in R["fifth_term"]["with"]:
    w, wo = R["fifth_term"]["with"][g], R["fifth_term"]["without"][g]
    print(f"cajas {g:10s}: malas entre aceptadas sin el quinto término {wo['bad_accepted']:.1%} -> con él {w['bad_accepted']:.1%} | "
          f"AUC {wo['auc']:.3f} -> {w['auc']:.3f} | aceptadas {wo['accepted']:.1%} -> {w['accepted']:.1%}")
json.dump(R, open("docs/revision/resultados/r8_indicators.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"modelo: AUC {a0:.3f}, ECE {e0:.3f}")
