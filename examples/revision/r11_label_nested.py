"""Revision R11: (1) sensitivity to the definition of "wrong line"; (2) nested cross-validation.

(1) With the suite ``data/revision/suite_r0.pkl`` (6 training families) several definitions
of wrong line are tried: relative shape error (5, 10, 20 %), absolute (> 2 Mm), mixed, and
connectivity criteria with the endpoint distance (> 2 and > 5 Mm). For each one: fraction
of wrong lines, AUC/ECE leaving out each family with the refitted 3-indicator model, and AUC of
the published model (fitted with the label of the paper) against that other label.

(2) Nested cross-validation: in each outer fold (one family out) the indicators are
chosen only with the 5 training families, by backward elimination with the same rule as the
paper (an indicator is removed if, leaving out each inner family, the AUC does not drop by more than
0.005); then the model is fitted and evaluated on the outer family. The resulting AUC is not affected
by the indicator selection.

    python examples/revision/r11_label_nested.py
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.calibration import USED_FEATURES, _Logistic, ece  # noqa: E402
from solarff.confidence import DEFAULT_CALIBRATION, _design, auc, confidence  # noqa: E402

warnings.filterwarnings("ignore")
D = [d for d in pickle.load(open("data/revision/suite_r0.pkl", "rb")) if d["field"] != "TD2-relax"]
ALL = ["sin_jb", "alpha_spread", "div", "buffer", "open", "foot_snr", "base_change", "apex", "length", "length_sq"]
fam = np.concatenate([[d["field"]] * len(d["bad"]) for d in D])
Xall = np.vstack([_design(d["ind"], ALL) for d in D])
ok = np.all(np.isfinite(Xall), axis=1)


def labels(kind):
    out = []
    for d in D:
        e = d["err"]
        topo = e["topology"] > 0
        rel = lambda t: np.nan_to_num(e["shape"], nan=np.inf) > t
        ab = np.nan_to_num(e["shape_abs"], nan=np.inf) > 2.0
        endp = lambda t: np.nan_to_num(e["endpoint"], nan=np.inf) > t
        out.append({"rel5": rel(0.05) | topo, "rel10": rel(0.10) | topo, "rel20": rel(0.20) | topo,
                    "abs2mm": ab | topo, "mixed": (rel(0.10) & ab) | topo,
                    "rel10_or_end2mm": rel(0.10) | endp(2.0) | topo,
                    "rel10_or_end5mm": rel(0.10) | endp(5.0) | topo}[kind])
    return np.concatenate(out).astype(float)


def lofo(X, y, g, feats_idx):
    p = np.full(len(y), np.nan)
    for f in np.unique(g):
        te = g == f
        m = _Logistic().fit(X[~te][:, feats_idx], y[~te])
        p[te] = m.predict_proba(X[te][:, feats_idx])[:, 1]
    return p


X, F = Xall[ok], fam[ok]
base_idx = [ALL.index(f) for f in USED_FEATURES]
R = {"labels": {}}
c_pub = np.concatenate([confidence(d["ind"], DEFAULT_CALIBRATION) for d in D])[ok]
for kind in ("rel5", "rel10", "rel20", "abs2mm", "mixed", "rel10_or_end2mm", "rel10_or_end5mm"):
    y = labels(kind)[ok]
    p = lofo(X, y, F, base_idx)
    acc = 1 - p >= 0.8
    r = {"bad_rate": float(y.mean()), "auc_refit": float(auc(p, y)), "ece_refit": float(ece(p, y)),
         "accepted": float(acc.mean()), "bad_accepted": float(y[acc].mean()),
         "auc_published_model": float(auc(1 - c_pub, y))}
    R["labels"][kind] = r
    print(f"{kind:16s} malas {r['bad_rate']:.1%} | reajustado: AUC {r['auc_refit']:.3f} ECE {r['ece_refit']:.3f}, "
          f"≥0.8: {r['accepted']:.0%} aceptadas, {r['bad_accepted']:.1%} malas | modelo publicado: AUC {r['auc_published_model']:.3f}", flush=True)

# (2) nested cross-validation
y = labels("rel10")[ok]


def inner_auc(Xi, yi, gi, idx):
    return auc(lofo(Xi, yi, gi, idx), yi)


p_nested = np.full(len(y), np.nan)
chosen = {}
for f in np.unique(F):
    tr = F != f
    Xi, yi, gi = X[tr], y[tr], F[tr]
    cur = list(range(len(ALL)))
    a_cur = inner_auc(Xi, yi, gi, cur)
    while len(cur) > 1:
        trials = []
        for k in cur:
            if ALL[k] == "length" and ALL.index("length_sq") in cur:
                continue  # the square is not used without the length
            idx = [j for j in cur if j != k]
            trials.append((inner_auc(Xi, yi, gi, idx), k))
        best_auc, k = max(trials)
        if best_auc >= a_cur - 0.005:  # removing it does not lower the AUC by more than 0.005
            cur.remove(k)
            a_cur = max(a_cur, best_auc) if best_auc > a_cur else a_cur
        else:
            break
    chosen[str(f)] = [ALL[k] for k in cur]
    m = _Logistic().fit(Xi[:, cur], yi)
    p_nested[~tr] = m.predict_proba(X[~tr][:, cur])[:, 1]
    print(f"fuera {f:9s}: indicadores elegidos {chosen[str(f)]}", flush=True)
R["nested"] = {"auc": float(auc(p_nested, y)), "ece": float(ece(p_nested, y)), "chosen": chosen,
               "auc_non_nested_3": float(auc(lofo(X, y, F, base_idx), y))}
print(f"anidada: AUC {R['nested']['auc']:.3f}, ECE {R['nested']['ece']:.3f} | sin anidar (3 indicadores): AUC {R['nested']['auc_non_nested_3']:.3f}")
json.dump(R, open("docs/revision/resultados/r11_label_nested.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
