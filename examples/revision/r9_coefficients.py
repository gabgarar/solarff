"""Revision R9: coefficients of the logistic model with their uncertainty.

Fits the default confidence (three indicators in five terms: the length enters relative to the horizontal
diagonal of the box, squared, and relative to its height) with the 162 extrapolations and estimates the error
of each coefficient with a bootstrap grouped by extrapolation (the lines of the same case are
not independent). Also gives the mean and standard deviation with which each indicator is
standardised (its typical range) and the odds ratio per standard deviation.

    python examples/revision/r9_coefficients.py
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.calibration import USED_FEATURES, _fit_logistic, stack  # noqa: E402

warnings.filterwarnings("ignore")
data = pickle.load(open("data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax_boxes.pkl", "rb"))["data"]
feats = list(USED_FEATURES)
X, y, _ = stack(data, feats)
# case index of each valid line, in the same order as stack()
from solarff.confidence import _design  # noqa: E402
ids = []
for i, d in enumerate(data):
    Xi = _design(d["ind"], feats)
    ids.append(np.full(int(np.all(np.isfinite(Xi), axis=1).sum()), i))
case = np.concatenate(ids)
assert len(case) == len(y)
base = _fit_logistic(X, y.astype(float))
rng = np.random.default_rng(0)
idx = {c: np.where(case == c)[0] for c in np.unique(case)}
B = []
for _ in range(300):
    sel = np.concatenate([idx[c] for c in rng.choice(list(idx), len(idx), replace=True)])
    f = _fit_logistic(X[sel], y[sel].astype(float))
    B.append([f["intercept"]] + list(f["coef"]))
B = np.array(B)
se = B.std(axis=0, ddof=1)
R = {"features": feats, "n_cases": len(data), "n_lines": int(len(y)), "intercept": base["intercept"], "intercept_se": float(se[0]),
     "intercept_ci95": [float(np.percentile(B[:, 0], 2.5)), float(np.percentile(B[:, 0], 97.5))], "rows": []}
# the two length scales are correlated: how much, and what it does to the uncertainty of their coefficients
Z = (X - np.array(base["mu"])) / np.array(base["sd"])
C = np.corrcoef(Z.T)
R["correlation"] = {f"{feats[i]}|{feats[j]}": float(C[i, j]) for i in range(len(feats)) for j in range(i + 1, len(feats))}
R["vif"] = {f: float(v) for f, v in zip(feats, np.diag(np.linalg.inv(C)))}
R["coef_correlation_bootstrap"] = {f"{feats[i]}|{feats[j]}": float(np.corrcoef(B[:, i + 1], B[:, j + 1])[0, 1])
                                   for i in range(len(feats)) for j in range(i + 1, len(feats))}
print("correlación longitud / longitud en alturas:", round(R["correlation"]["length|length_h"], 3), "| VIF:", {k: round(v, 2) for k, v in R["vif"].items()})
print(f"intercepto {base['intercept']:+.3f} ± {se[0]:.3f}")
for k, f in enumerate(feats):
    c = base["coef"][k]
    row = {"feature": f, "mu": base["mu"][k], "sd": base["sd"][k], "coef": c, "se": float(se[k + 1]),
           "odds_ratio_per_sd": float(np.exp(c)), "ci95": [float(np.percentile(B[:, k + 1], 2.5)), float(np.percentile(B[:, k + 1], 97.5))]}
    R["rows"].append(row)
    print(f"{f:12s} μ {row['mu']:+.3f} σ {row['sd']:.3f} | coef {c:+.3f} ± {row['se']:.3f} [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}] | OR {row['odds_ratio_per_sd']:.2f}")
json.dump(R, open("docs/revision/resultados/r9_coefficients.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
