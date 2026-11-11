"""Revision R7: external rope (C3), definition of "wrong line" (C4) and reduced model.

Uses the suite ``data/revision/suite_r0.pkl`` (6 training families + ``TD2-relax``,
which was not used to design anything).

* C4 — does the U shape of the length come from the relative metric? Fraction of wrong lines
  by length bins with the relative label (paper), the absolute one (Mm) and the mixed one, and
  gain of the quadratic term in leave-one-family-out validation with each label.
* C3 — external test on TD2-relax: models fitted with the 6 families (5 and 3 indicators) and
  with 5 families without ropes (linear and quadratic).
* Reduced model — the rule registered in ``docs/revision/fase1_auditoria_conceptual.md`` is applied.

    python examples/revision/r7_models.py
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.calibration import FULL5_FEATURES, LINEAR5_FEATURES as LINEAR_FEATURES, _Logistic, cross_validate, ece, stack  # noqa: E402
from solarff.confidence import auc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = "docs/revision/resultados"
D = pickle.load(open("data/revision/suite_r0.pkl", "rb"))
TRAIN6 = ["LL-I", "LL-II", "LL-III", "LFFF-bip", "LFFF-quad", "TD-relax"]
REDUCED = ["alpha_spread", "buffer", "length", "length_sq"]
FULL = list(FULL5_FEATURES)


def label(d, kind, mm=2.0):
    e = dict(d)
    err = d["err"]
    topo = err["topology"] > 0
    rel = err["shape"] > 0.1
    ab = np.nan_to_num(err["shape_abs"], nan=np.inf) > mm
    e["bad"] = {"rel": rel | topo, "abs": ab | topo, "mixed": (rel & ab) | topo}[kind]
    return e


def cv(data, feats):
    c = cross_validate(data, features=feats, models=["logistic"], group="field", verbose=False)
    p, y, g = c["models"]["logistic"]["p"], c["y"].astype(float), np.asarray(c["groups"])
    fam = {str(f): float(auc(p[g == f], y[g == f])) for f in np.unique(g)}
    return {"auc": float(auc(p, y)), "ece": float(ece(p, y)), "family_auc": fam, "worst": min(fam.values()),
            "accepted": float((1 - p >= 0.8).mean()), "bad_accepted": float(y[1 - p >= 0.8].mean())}


def external(train, test, feats):
    X, y, _ = stack(train, feats)
    m = _Logistic().fit(X, y)
    Xt, yt, _ = stack(test, feats)
    p = m.predict_proba(Xt)[:, 1]
    acc = 1 - p >= 0.8
    return {"auc": float(auc(p, yt)), "ece": float(ece(p, yt)), "bad_rate": float(yt.mean()),
            "accepted": float(acc.mean()), "bad_accepted": float(yt[acc].mean()) if acc.any() else None}


R = {}
# --- C4: U shape according to the label
L = np.concatenate([np.asarray(d["ind"]["length"], float) for d in D if d["field"] in TRAIN6])
bins = np.quantile(L[np.isfinite(L)], np.linspace(0, 1, 9))
R["C4_bad_rate_by_length_octile"] = {}
for kind in ("rel", "abs", "mixed"):
    y = np.concatenate([label(d, kind)["bad"] for d in D if d["field"] in TRAIN6]).astype(float)
    k = np.clip(np.digitize(L, bins[1:-1]), 0, 7)
    R["C4_bad_rate_by_length_octile"][kind] = [float(y[k == i].mean()) for i in range(8)]
    print(f"C4 {kind:5s} malas por octil de longitud:", np.round(R["C4_bad_rate_by_length_octile"][kind], 2).tolist())
R["C4_quadratic_gain"] = {}
for kind in ("rel", "abs", "mixed"):
    data = [label(d, kind) for d in D if d["field"] in TRAIN6]
    lin = cv(data, list(LINEAR_FEATURES))
    quad = cv(data, FULL)
    R["C4_quadratic_gain"][kind] = {"linear": lin, "quadratic": quad}
    print(f"C4 {kind:5s}: AUC lineal {lin['auc']:.3f} -> cuadrático {quad['auc']:.3f} (peor familia {lin['worst']:.3f} -> {quad['worst']:.3f})")

# --- Reduced model (label of the paper) and registered rule
data6 = [label(d, "rel") for d in D if d["field"] in TRAIN6]
full, red = cv(data6, FULL), cv(data6, REDUCED)
R["reduced_vs_full_cv"] = {"full": full, "reduced": red}
# --- C3: external rope
td2 = [label(d, "rel") for d in D if d["field"] == "TD2-relax"]
data5 = [d for d in data6 if d["field"] != "TD-relax"]
R["C3_external_TD2"] = {
    "full_6fam": external(data6, td2, FULL),
    "reduced_6fam": external(data6, td2, REDUCED),
    "linear_5fam_no_rope": external(data5, td2, list(LINEAR_FEATURES)),
    "quadratic_5fam_no_rope": external(data5, td2, FULL),
    "reduced_5fam_no_rope": external(data5, td2, REDUCED),
}
for k, v in R["C3_external_TD2"].items():
    print(f"C3 TD2 {k:24s}: AUC {v['auc']:.3f} ECE {v['ece']:.3f} malas {v['bad_rate']:.0%} aceptadas {v['accepted']:.0%} malas entre fiables {v['bad_accepted']:.1%}")
rule = {"auc": red["auc"] >= full["auc"] - 0.005, "ece": red["ece"] <= full["ece"] + 0.01,
        "worst_family": red["worst"] >= full["worst"] - 0.01,
        "external_bad_accepted": R["C3_external_TD2"]["reduced_6fam"]["bad_accepted"] <= R["C3_external_TD2"]["full_6fam"]["bad_accepted"]}
R["rule"] = {**rule, "adopt_reduced": all(rule.values())}
print(f"Modelo completo: AUC {full['auc']:.3f} ECE {full['ece']:.3f} peor {full['worst']:.3f} | reducido: AUC {red['auc']:.3f} ECE {red['ece']:.3f} peor {red['worst']:.3f}")
print("Regla:", R["rule"])
# --- C3 with the absolute label (in case C4 favours it)
td2a = [label(d, "abs") for d in D if d["field"] == "TD2-relax"]
d6a = [label(d, "abs") for d in D if d["field"] in TRAIN6]
R["C3_external_TD2_abs_label"] = {"full_6fam": external(d6a, td2a, FULL), "reduced_6fam": external(d6a, td2a, REDUCED)}
json.dump(R, open(os.path.join(OUT, "r7_models.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
