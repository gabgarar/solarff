"""External review 2 (10-Nov): how much does the physics add over the geometry, and is the fit balanced?

With the rerun of the six-family suite (``data/revision/suite_r0.pkl`` without TD2) plus the cases in boxes of other
proportions (162 extrapolations) and the same validation scheme as the paper (leave out each family, L2 = 0.01):

(1) Baselines: length only (its three terms: relative to the diagonal, squared, and relative to the height of the
    box), apex height only, length + buffer, length + buffer +
    α variation (published model), and each indicator alone. For each: AUC, Brier, ECE with 10 and 15
    bins, calibration slope and intercept, risk (wrong among the accepted) at fixed coverage
    of 25, 50 and 75 % and coverage at a fixed risk of 5 %, all with out-of-sample predictions.
(2) Stratification: in terciles of length × terciles of apex height, AUC of the α variation alone
    (direct ranking, no fit) and change of the AUC of the length + buffer model when α is added, with the
    same out-of-sample predictions restricted to each cell.
(3) Folds: AUC of the published model and standardised coefficients in each fold (family out),
    and indicators chosen in the nested selection (from r11).
(4) Balance of the fit: valid lines per family, per box proportion and per extrapolation (minimum, median, maximum).
(5) Weighting: the published model fitted with equal weight per line, per extrapolation, per family or per box
    proportion.

Writes ``docs/revision/resultados/r14_baselines.json``.

    python examples/revision/r14_baselines.py      # ~1 min
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.calibration import _Logistic, box_name, brier, ece, with_proportions  # noqa: E402
from solarff.confidence import _design, auc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join("docs", "revision", "resultados", "r14_baselines.json")
D = with_proportions([d for d in pickle.load(open("data/revision/suite_r0.pkl", "rb")) if d["field"] != "TD2-relax"])
ALL = ["sin_jb", "alpha_spread", "div", "buffer", "open", "foot_snr", "base_change", "apex", "length", "length_sq", "length_h"]
LEN = ["length", "length_sq", "length_h"]      # the length: relative to the diagonal, squared, and relative to the height of the box
X = np.vstack([_design(d["ind"], ALL) for d in D])
y = np.concatenate([d["bad"] for d in D]).astype(float)
fam = np.concatenate([[d["field"]] * len(d["bad"]) for d in D])
case = np.concatenate([[i] * len(d["bad"]) for i, d in enumerate(D)])
box = np.concatenate([[box_name(d)] * len(d["bad"]) for d in D])
ok = np.all(np.isfinite(X), axis=1)
X, y, fam, case, box = X[ok], y[ok], fam[ok], case[ok], box[ok]
col = {k: i for i, k in enumerate(ALL)}
FAMS = sorted(set(fam))


def cal_slope_intercept(p, yy):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    b = np.zeros(2)
    M = np.column_stack([np.ones_like(z), z])
    for _ in range(50):
        q = 1 / (1 + np.exp(-M @ b))
        W = q * (1 - q) + 1e-9
        b = b + np.linalg.solve(M.T @ (M * W[:, None]), M.T @ (yy - q))
    return float(b[1]), float(b[0])


def lofo(feats):
    idx = [col[f] for f in feats]
    p = np.full(len(y), np.nan)
    coefs = {}
    for f in FAMS:
        te = fam == f
        m = _Logistic().fit(X[~te][:, idx], y[~te])
        p[te] = m.predict_proba(X[te][:, idx])[:, 1]
        coefs[f] = {k: float(c) for k, c in zip(feats, m.cal["coef"])}
    return p, coefs


def risk_coverage(p, yy):
    c = 1 - p
    order = np.argsort(-c)
    cum_bad = np.cumsum(yy[order]) / np.arange(1, len(yy) + 1)
    out = {}
    for cov in (0.25, 0.5, 0.75):
        k = int(round(cov * len(yy))) - 1
        out[f"risk_at_coverage_{int(100 * cov)}"] = float(cum_bad[k])
    below = np.where(cum_bad <= 0.05)[0]
    out["coverage_at_risk_5"] = float((below[-1] + 1) / len(yy)) if len(below) else 0.0
    acc = c >= 0.8
    out["accepted_at_0.8"] = float(acc.mean())
    out["risk_at_0.8"] = float(yy[acc].mean()) if acc.any() else None
    return out


def metrics(p, yy):
    s, i = cal_slope_intercept(p, yy)
    worst = min(auc(p[fam == f], yy[fam == f]) for f in FAMS) if len(yy) == len(y) else None
    return {"auc": auc(p, yy), "brier": brier(p, yy), "ece_10": ece(p, yy, 10), "ece_15": ece(p, yy, 15),
            "cal_slope": s, "cal_intercept": i, "auc_worst_family": worst, **risk_coverage(p, yy)}


BASELINES = {"length": LEN, "apex": ["apex"], "alpha_spread": ["alpha_spread"], "buffer": ["buffer"],
             "length+buffer": ["buffer"] + LEN, "length+apex": ["apex"] + LEN,
             "length+buffer+apex": ["buffer", "apex"] + LEN,
             "published (alpha+buffer+length)": ["alpha_spread", "buffer"] + LEN}
res = {"n_lines": int(len(y)), "n_cases": len(D), "bad_rate": float(y.mean()), "families": FAMS, "baselines": {}, "strata": {}, "folds": {}, "balance": {}}
P = {}
for name, feats in BASELINES.items():
    p, coefs = lofo(feats)
    P[name] = p
    res["baselines"][name] = {"features": feats, **metrics(p, y)}
    if name.startswith("published"):
        res["folds"]["coefficients_by_left_out_family"] = coefs
    m = res["baselines"][name]
    print(f"{name:32s} AUC {m['auc']:.3f} (peor familia {m['auc_worst_family']:.3f}) Brier {m['brier']:.3f} ECE10 {m['ece_10']:.3f} "
          f"pend {m['cal_slope']:.2f} int {m['cal_intercept']:+.2f} | riesgo a 25/50/75 %: {m['risk_at_coverage_25']:.1%} "
          f"{m['risk_at_coverage_50']:.1%} {m['risk_at_coverage_75']:.1%} | cobertura a riesgo 5 %: {m['coverage_at_risk_5']:.0%}")

# (2) stratification: terciles of log-length × terciles of log-height
lg, ap = X[:, col["length"]], np.log10(np.maximum(X[:, col["apex"]], 1e-3))
ql, qa = np.quantile(lg, [1 / 3, 2 / 3]), np.quantile(ap, [1 / 3, 2 / 3])
bl, ba = np.digitize(lg, ql), np.digitize(ap, qa)
pA, pB = P["length+buffer"], P["published (alpha+buffer+length)"]
cells = []
for i in range(3):
    for j in range(3):
        m = (bl == i) & (ba == j)
        if m.sum() < 200 or y[m].mean() in (0.0, 1.0):
            continue
        a_alpha = auc(X[m, col["alpha_spread"]], y[m])
        cells.append({"length_tercile": i, "apex_tercile": j, "n": int(m.sum()), "bad_rate": float(y[m].mean()),
                      "auc_alpha_alone": a_alpha, "auc_length_buffer": auc(pA[m], y[m]), "auc_published": auc(pB[m], y[m]),
                      "delta_auc_adding_alpha": auc(pB[m], y[m]) - auc(pA[m], y[m])})
w = np.array([c["n"] for c in cells], float)
res["strata"] = {"cells": cells, "length_tercile_edges_log10": [float(v) for v in ql], "apex_tercile_edges_log10_mm": [float(v) for v in qa],
                 "weighted_mean_auc_alpha_alone": float(np.average([c["auc_alpha_alone"] for c in cells], weights=w)),
                 "weighted_mean_delta_auc_adding_alpha": float(np.average([c["delta_auc_adding_alpha"] for c in cells], weights=w)),
                 "min_auc_alpha_alone": float(min(c["auc_alpha_alone"] for c in cells)),
                 "min_delta_auc_adding_alpha": float(min(c["delta_auc_adding_alpha"] for c in cells)),
                 "n_cells": len(cells)}
print(f"estratos ({len(cells)} celdas): AUC de α sola dentro de celda {res['strata']['min_auc_alpha_alone']:.3f}–"
      f"{max(c['auc_alpha_alone'] for c in cells):.3f} (media {res['strata']['weighted_mean_auc_alpha_alone']:.3f}); "
      f"ΔAUC al añadir α a longitud+capa {res['strata']['min_delta_auc_adding_alpha']:+.3f}–"
      f"{max(c['delta_auc_adding_alpha'] for c in cells):+.3f} (media {res['strata']['weighted_mean_delta_auc_adding_alpha']:+.3f})")

# (3) folds: AUC per left-out family and indicators chosen by the nested selection
pub = P["published (alpha+buffer+length)"]
res["folds"]["auc_by_left_out_family"] = {f: auc(pub[fam == f], y[fam == f]) for f in FAMS}
r11 = os.path.join("docs", "revision", "resultados", "r11_label_nested.json")
if os.path.exists(r11):
    res["folds"]["nested_chosen"] = json.load(open(r11, encoding="utf-8"))["nested"]["chosen"]
co = res["folds"]["coefficients_by_left_out_family"]
res["folds"]["coefficient_range"] = {k: [min(v[k] for v in co.values()), max(v[k] for v in co.values())] for k in co[FAMS[0]]}
print("coeficientes estandarizados entre particiones:", {k: f"{a:.2f}–{b:.2f}" for k, (a, b) in res["folds"]["coefficient_range"].items()})

# (5) weighting: fit with equal weight per line, per extrapolation or per family (same L2), family out
from scipy.optimize import minimize  # noqa: E402


def fit_weighted(Xf, yf, w, l2=1e-2):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-12
    Z = (Xf - mu) / sd
    w = w / w.sum()

    def f(beta):
        z = Z @ beta[1:] + beta[0]
        p = 1 / (1 + np.exp(-z))
        eps = 1e-12
        nll = -np.sum(w * (yf * np.log(p + eps) + (1 - yf) * np.log(1 - p + eps)))
        g = Z.T @ (w * (p - yf))
        return nll + l2 * (beta[1:] ** 2).sum(), np.r_[np.sum(w * (p - yf)), g + 2 * l2 * beta[1:]]

    r = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B")
    return mu, sd, r.x


idx = [col[k] for k in BASELINES["published (alpha+buffer+length)"]]
res["weighting"] = {}
for scheme in ("line", "extrapolation", "family", "box"):
    p = np.full(len(y), np.nan)
    for f in FAMS:
        te = fam == f
        if scheme == "line":
            w = np.ones((~te).sum())
        elif scheme == "extrapolation":
            cnt = {c: int((case[~te] == c).sum()) for c in np.unique(case[~te])}
            w = np.array([1.0 / cnt[c] for c in case[~te]])
        elif scheme == "family":
            cnt = {g: int((fam[~te] == g).sum()) for g in np.unique(fam[~te])}
            w = np.array([1.0 / cnt[g] for g in fam[~te]])
        else:
            cnt = {g: int((box[~te] == g).sum()) for g in np.unique(box[~te])}
            w = np.array([1.0 / cnt[g] for g in box[~te]])
        mu, sd, b = fit_weighted(X[~te][:, idx], y[~te], w)
        p[te] = 1 / (1 + np.exp(-(((X[te][:, idx] - mu) / sd) @ b[1:] + b[0])))
    rc = risk_coverage(p, y)
    res["weighting"][scheme] = {"auc": auc(p, y), "accepted_at_0.8": rc["accepted_at_0.8"], "risk_at_0.8": rc["risk_at_0.8"],
                               "risk_at_coverage_50": rc["risk_at_coverage_50"],
                               "risk_at_0.8_by_box": {g: float(y[(box == g) & (1 - p >= 0.8)].mean()) for g in np.unique(box)}}
    m = res["weighting"][scheme]
    print(f"ponderación {scheme:13s}: AUC {m['auc']:.4f} aceptadas {m['accepted_at_0.8']:.1%} malas aceptadas {m['risk_at_0.8']:.2%} malas al 50 % {m['risk_at_coverage_50']:.2%}")
wl = res["weighting"]["line"]
OTHER = ("extrapolation", "family", "box")
res["weighting"]["max_change"] = {"auc": max(abs(res["weighting"][s]["auc"] - wl["auc"]) for s in OTHER),
                                  "risk_at_0.8_points": max(abs(res["weighting"][s]["risk_at_0.8"] - wl["risk_at_0.8"]) for s in OTHER) * 100,
                                  "accepted_at_0.8_points": max(abs(res["weighting"][s]["accepted_at_0.8"] - wl["accepted_at_0.8"]) for s in OTHER) * 100}
print("malas aceptadas por caja según la ponderación:", {s: {g: round(100 * v, 1) for g, v in res["weighting"][s]["risk_at_0.8_by_box"].items()} for s in ("line",) + OTHER})
print("cambio máximo por ponderar:", {k: round(v, 3) for k, v in res["weighting"]["max_change"].items()})

# (4) balance of the design
per_fam = {f: int((fam == f).sum()) for f in FAMS}
per_case = np.bincount(case)[np.unique(case)]
res["balance"] = {"lines_by_family": per_fam, "lines_by_box": {g: int((box == g).sum()) for g in np.unique(box)}, "lines_by_case": {"min": int(per_case.min()), "median": float(np.median(per_case)), "max": int(per_case.max())},
                  "cases_by_family": {f: int(len({c for c in case[fam == f]})) for f in FAMS}}
print("líneas por familia", per_fam, "| por extrapolación", res["balance"]["lines_by_case"])
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("guardado", OUT)
