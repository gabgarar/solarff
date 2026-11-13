"""Revision R11: (1) sensitivity to the definition of "wrong line"; (2) and (3) nested cross-validation.

(1) With the rerun of the suite ``data/revision/suite_r0.pkl`` (6 training families) and the cases in boxes of
other proportions (162 extrapolations) several definitions of wrong line are tried: relative shape error (5, 10, 20 %), absolute (> 2 Mm), mixed, and
connectivity criteria with the endpoint distance (> 2 and > 5 Mm). For each one: fraction
of wrong lines, AUC/ECE leaving out each family with the refitted model, and AUC of
the published model (fitted with the label of the paper) against that other label.

(2) Nested cross-validation. In each outer fold (one family out) the whole selection is repeated with the 5 training
families only, in the two stages in which the model was built. First the indicators are chosen in the boxes of
standard proportion, among the 9 indicators and the square of the length, by backward elimination with the same
rule as the paper (a term is removed if, leaving out each inner family, the AUC does not drop by more than 0.005).
Then the fifth term (the length in box heights), which was not chosen by AUC but because it evens out the wrong
lines accepted in boxes of different proportions, is kept only if it lowers the largest fraction of wrong lines
accepted among the box proportions (inner families left out in turn). The model is then fitted with all the
training boxes and evaluated on the outer family. The figures are not affected by the selection.

Two variants are also written. ``nested_one_stage``: backward elimination over the eleven terms with the boxes of
all the proportions at once, a selection that was not the one used to build the model. ``nested_v1_design``: the
design of version 1.1 (standard boxes only, ten terms), which reproduces its figure.

(3) ``nested_fifth_term``: the three indicators fixed and only the choice of the fifth term nested.

    python examples/revision/r11_label_nested.py
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.calibration import USED_FEATURES, V1_FEATURES, _Logistic, box_name, ece, with_proportions  # noqa: E402
from solarff.confidence import DEFAULT_CALIBRATION, _design, auc, confidence  # noqa: E402

warnings.filterwarnings("ignore")
# the rerun of the suite (other random seeds, with absolute errors) plus the cases in boxes of other proportions
D = with_proportions([d for d in pickle.load(open("data/revision/suite_r0.pkl", "rb")) if d["field"] != "TD2-relax"])
ALL = ["sin_jb", "alpha_spread", "div", "buffer", "open", "foot_snr", "base_change", "apex", "length", "length_sq", "length_h"]
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
BOX = np.concatenate([[box_name(d)] * len(d["bad"]) for d in D])[ok]     # proportion of the box of each line
base_idx = [ALL.index(f) for f in USED_FEATURES]
R = {"labels": {}}
c_pub = np.concatenate([confidence(d["ind"], DEFAULT_CALIBRATION) for d in D])[ok]
for kind in ("rel5", "rel10", "rel20", "abs2mm", "mixed", "rel10_or_end2mm", "rel10_or_end5mm"):
    y = labels(kind)[ok]
    p = lofo(X, y, F, base_idx)
    acc = 1 - p >= 0.8
    r = {"bad_rate": float(y.mean()), "auc_refit": float(auc(p, y)), "ece_refit": float(ece(p, y)),
         "accepted": float(acc.mean()), "bad_accepted": float(y[acc].mean()),
         "auc_published_model": float(auc(1 - c_pub, y)), "accepted_published_model": float((c_pub >= 0.8).mean()),
         "bad_accepted_published_model": float(y[c_pub >= 0.8].mean())}
    R["labels"][kind] = r
    print(f"{kind:16s} malas {r['bad_rate']:.1%} | reajustado: AUC {r['auc_refit']:.3f} ECE {r['ece_refit']:.3f}, "
          f"≥0.8: {r['accepted']:.0%} aceptadas, {r['bad_accepted']:.1%} malas | modelo publicado: AUC {r['auc_published_model']:.3f}", flush=True)

# (2) nested cross-validation
y = labels("rel10")[ok]
STD = BOX == "standard"
H = ALL.index("length_h")
C10 = [k for k in range(len(ALL)) if k != H]
V1_IDX = [ALL.index(f) for f in V1_FEATURES]


def select(Xi, yi, gi, cand):
    """Backward elimination: a term is removed if the inner AUC (each inner family left out) does not fall by more than 0.005."""
    cur = list(cand)
    a_cur = auc(lofo(Xi, yi, gi, cur), yi)
    while len(cur) > 1:
        trials = []
        for k in cur:
            if ALL[k] == "length" and (ALL.index("length_sq") in cur or H in cur):
                continue  # the other terms of the length are not used without the length
            trials.append((auc(lofo(Xi, yi, gi, [j for j in cur if j != k]), yi), k))
        best_auc, k = max(trials)
        if best_auc >= a_cur - 0.005:
            cur.remove(k)
            a_cur = max(a_cur, best_auc)
        else:
            break
    return cur


def worst_box(p, yy, bb):
    """Largest fraction of wrong lines among the accepted ones (confidence >= 0.8) over the box proportions."""
    return float(max(yy[(bb == g) & (1 - p >= 0.8)].mean() for g in np.unique(bb)))


def nested(rows, cand, select_on_standard=False, fifth=False, fixed=None):
    """Out-of-sample probabilities with the selection repeated in each outer fold, and what was chosen."""
    Xr, yr, Fr, Br, Sr = X[rows], y[rows], F[rows], BOX[rows], STD[rows]
    p, chosen, kept = np.full(len(yr), np.nan), {}, {}
    for f in np.unique(Fr):
        tr = Fr != f
        sel = tr & Sr if select_on_standard else tr
        cur = list(fixed) if fixed is not None else select(Xr[sel], yr[sel], Fr[sel], cand)
        if fifth:
            w5 = worst_box(lofo(Xr[tr], yr[tr], Fr[tr], cur + [H]), yr[tr], Br[tr])
            w4 = worst_box(lofo(Xr[tr], yr[tr], Fr[tr], cur), yr[tr], Br[tr])
            kept[str(f)] = {"worst_box_with": w5, "worst_box_without": w4, "kept": bool(w5 < w4)}
            if w5 < w4:
                cur = cur + [H]
        chosen[str(f)] = [ALL[k] for k in cur]
        p[~tr] = _Logistic().fit(Xr[tr][:, cur], yr[tr]).predict_proba(Xr[~tr][:, cur])[:, 1]
    acc = 1 - p >= 0.8
    out = {"auc": float(auc(p, yr)), "ece": float(ece(p, yr)), "accepted": float(acc.mean()), "bad_accepted": float(yr[acc].mean()),
           "n_lines": int(len(yr)), "chosen": chosen,
           "auc_by_family": {str(f): float(auc(p[Fr == f], yr[Fr == f])) for f in np.unique(Fr)},
           "bad_accepted_by_family": {str(f): float(yr[(Fr == f) & acc].mean()) for f in np.unique(Fr)},
           "bad_accepted_by_box": {str(g): float(yr[(Br == g) & acc].mean()) for g in np.unique(Br)}}
    if fifth:
        out["fifth_term"] = kept
    return out


def show(name, r):
    print(f"{name}: AUC {r['auc']:.3f}, ECE {r['ece']:.3f}, {r['accepted']:.1%} aceptadas, {r['bad_accepted']:.1%} malas entre ellas | por caja "
          + ", ".join(f"{g} {v:.1%}" for g, v in r["bad_accepted_by_box"].items()) + " | por familia "
          + ", ".join(f"{g} {v:.1%}" for g, v in r["bad_accepted_by_family"].items()), flush=True)
    for f, c in r["chosen"].items():
        print(f"   fuera {f:9s}: {c}", flush=True)


every = np.ones(len(y), bool)
R["nested"] = nested(every, C10, select_on_standard=True, fifth=True)
R["nested"]["auc_non_nested_3"] = float(auc(lofo(X, y, F, base_idx), y))
show("anidada en dos etapas", R["nested"])
print(f"   sin anidar: AUC {R['nested']['auc_non_nested_3']:.3f}")
R["nested_one_stage"] = nested(every, list(range(len(ALL))))
show("anidada en una etapa, once términos, todas las cajas", R["nested_one_stage"])
R["nested_v1_design"] = nested(STD, C10)
show("diseño de la versión 1.1: cajas estándar, diez términos", R["nested_v1_design"])
R["nested_fifth_term"] = nested(every, None, fifth=True, fixed=V1_IDX)
show("tres indicadores fijos y quinto término anidado", R["nested_fifth_term"])
json.dump(R, open("docs/revision/resultados/r11_label_nested.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
