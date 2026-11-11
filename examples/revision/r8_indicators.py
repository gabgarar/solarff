"""Revision R8: why the model uses 3 of the 9 indicators.

For each indicator: individual AUC, sign of its coefficient when added to the 3-indicator model
(positive = a larger value indicates a worse line) and leave-one-family-out AUC/ECE when
added to the 3-indicator model (or, for the 3 of the model, when removed).

    python examples/revision/r8_indicators.py
"""
import json
import os
import pickle
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.calibration import FEATURE_SIGN, USED_FEATURES, _fit_logistic, cross_validate, ece, stack  # noqa: E402
from solarff.confidence import auc  # noqa: E402

warnings.filterwarnings("ignore")
S = pickle.load(open("data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl", "rb"))
data, single = S["data"], S["single"]
base = list(USED_FEATURES)


def cv(feats):
    c = cross_validate(data, features=feats, models=["logistic"], group="field", verbose=False)
    p, y = c["models"]["logistic"]["p"], c["y"].astype(float)
    return float(auc(p, y)), float(ece(p, y))


a0, e0 = cv(base)
R = {"base": {"features": base, "auc": a0, "ece": e0}, "indicators": {}}
for f in ("sin_jb", "alpha_spread", "div", "buffer", "open", "foot_snr", "base_change", "apex", "length"):
    if f in base:
        feats = [x for x in base if x != f and not (f == "length" and x == "length_sq")]
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
json.dump(R, open("docs/revision/resultados/r8_indicators.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"modelo de 3: AUC {a0:.3f}, ECE {e0:.3f}")
