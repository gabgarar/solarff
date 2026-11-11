"""External review 7 (11-Nov): checks asked for by a referee-style report, on data already computed.

(a) Probability model, calibration and threshold kept apart. Three variants under leave-one-family-out
    validation (on the original suite, which the paper quotes, and on its rerun of r14 and r15): the published penalised fit
    (L2 = 0.01), the unpenalised fit, and the unpenalised fit recalibrated with a logistic slope and
    intercept that are fitted only inside each training set (inner leave-one-family-out predictions of
    the five training families). For each: AUC, ECE, calibration slope and intercept, coverage and risk
    at confidence 0.8, risk at 50 % coverage and coverage at 5 % risk.
(b) Grouped-bootstrap intervals of the risk (wrong lines among the accepted) and of the coverage at 0.8:
    the calibration suite (out-of-sample predictions; by extrapolation and by family) and the four
    external families (published model; by extrapolation).
(c) What the wrong lines are made of: shape error only, open/closed change only, both; AUC of the
    out-of-sample probability against each label separately.
(d) Spearman correlations between the indicators and with the apex height.
(e) AIA: lines per region, lines with a measured angle, and sign of the accepted − rejected difference
    for the stored combinations of smoothing scale and ridge percentile (171 Å).

The predictions were registered before running. Writes ``docs/revision/resultados/r16_decision_checks.json``.

    python examples/revision/r16_decision_checks.py      # ~2 min
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.calibration import bootstrap_ci, ece  # noqa: E402
from solarff.confidence import DEFAULT_CALIBRATION, _design, auc, confidence  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join("docs", "revision", "resultados", "r16_decision_checks.json")
FEATS = ["alpha_spread", "buffer", "length", "length_sq"]
T = 0.8
N_BOOT = 2000


def stack(cases):
    """Design matrix, label, family and case index of the lines with finite indicators."""
    X = np.vstack([_design(d["ind"], FEATS) for d in cases])
    y = np.concatenate([d["bad"] for d in cases]).astype(float)
    fam = np.concatenate([[d["field"]] * len(d["bad"]) for d in cases])
    case = np.concatenate([[i] * len(d["bad"]) for i, d in enumerate(cases)])
    ok = np.all(np.isfinite(X), axis=1)
    return X[ok], y[ok], fam[ok], case[ok], ok


def fit(Xf, yf, l2):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-12
    Z = (Xf - mu) / sd

    def f(beta):
        p = 1 / (1 + np.exp(-(Z @ beta[1:] + beta[0])))
        nll = -np.mean(yf * np.log(p + 1e-12) + (1 - yf) * np.log(1 - p + 1e-12))
        return nll + l2 * (beta[1:] ** 2).sum(), np.r_[np.mean(p - yf), Z.T @ (p - yf) / len(yf) + 2 * l2 * beta[1:]]

    r = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-10})
    return mu, sd, r.x


def predict(model, Xf):
    mu, sd, b = model
    return 1 / (1 + np.exp(-(((Xf - mu) / sd) @ b[1:] + b[0])))


def logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def slope_intercept(p, yy):
    """Logistic recalibration logit P(y = 1) = a + b logit p (Newton); returns (b, a)."""
    z = logit(p)
    M = np.column_stack([np.ones_like(z), z])
    b = np.zeros(2)
    for _ in range(50):
        q = 1 / (1 + np.exp(-M @ b))
        b = b + np.linalg.solve(M.T @ (M * (q * (1 - q) + 1e-9)[:, None]), M.T @ (yy - q))
    return float(b[1]), float(b[0])


def lofo(X, y, fam, l2, recalibrate=False):
    p = np.full(len(y), np.nan)
    for f in sorted(set(fam)):
        te = fam == f
        Xt, yt, ft = X[~te], y[~te], fam[~te]
        p_out = predict(fit(Xt, yt, l2), X[te])
        if recalibrate:  # slope and intercept from predictions that are out of sample inside the training set
            p_in = np.full(len(yt), np.nan)
            for g in sorted(set(ft)):
                ti = ft == g
                p_in[ti] = predict(fit(Xt[~ti], yt[~ti], l2), Xt[ti])
            b, a = slope_intercept(p_in, yt)
            p_out = 1 / (1 + np.exp(-(a + b * logit(p_out))))
        p[te] = p_out
    return p


def risk(p, yy):
    acc = (1 - p) >= T
    return float(yy[acc].mean()) if acc.any() else float("nan")


def coverage(p, yy):
    return float(((1 - p) >= T).mean())


def summary(p, yy):
    order = np.argsort(p)
    cum = np.cumsum(yy[order]) / np.arange(1, len(yy) + 1)
    below = np.where(cum <= 0.05)[0]
    b, a = slope_intercept(p, yy)
    return {"auc": auc(p, yy), "ece_10": ece(p, yy, 10), "cal_slope": b, "cal_intercept": a,
            "accepted_at_0.8": coverage(p, yy), "risk_at_0.8": risk(p, yy),
            "risk_at_coverage_50": float(cum[int(round(0.5 * len(yy))) - 1]),
            "coverage_at_risk_5": float((below[-1] + 1) / len(yy)) if len(below) else 0.0}


def ci(p, yy, metric, groups):
    return list(bootstrap_ci(p, yy, metric, groups=groups, n_boot=N_BOOT))


res = {"threshold": T, "n_boot": N_BOOT}

# ------------------------------------------------------------------ (a) model, calibration, threshold
SUITES = {"original": pickle.load(open("data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl", "rb"))["data"],
          "rerun": [d for d in pickle.load(open("data/revision/suite_r0.pkl", "rb")) if d["field"] != "TD2-relax"]}
res["variants"] = {}
for sname, cases in SUITES.items():   # the paper quotes the original suite; the rerun is the one of r14 and r15
    X, y, fam, case, _ = stack(cases)
    V = {"penalised": lofo(X, y, fam, 1e-2), "unpenalised": lofo(X, y, fam, 0.0),
         "unpenalised_recalibrated": lofo(X, y, fam, 0.0, recalibrate=True)}
    out = {"n_lines": int(len(y)), **{k: summary(p, y) for k, p in V.items()}}
    r50 = [out[k]["risk_at_coverage_50"] for k in V]
    aucs = [out[k]["auc"] for k in V]
    out["spread"] = {"risk_at_coverage_50_points": 100 * (max(r50) - min(r50)), "auc": max(aucs) - min(aucs)}
    res["variants"][sname] = out
    for k in V:
        m = out[k]
        print(f"{sname:8s} {k:26s} AUC {m['auc']:.3f} ECE {m['ece_10']:.3f} pend {m['cal_slope']:.2f} int {m['cal_intercept']:+.2f} | 0.8: "
              f"{m['accepted_at_0.8']:.1%} aceptadas, {m['risk_at_0.8']:.1%} malas | al 50 %: {m['risk_at_coverage_50']:.1%} | a riesgo 5 %: {m['coverage_at_risk_5']:.0%}")

# ------------------------------------------------------------------ (b) intervals of risk and coverage
S = pickle.load(open("data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl", "rb"))["data"]
Xs, ys, fs, cs, oks = stack(S)
ps = lofo(Xs, ys, fs, 1e-2)
res["suite"] = {"n_lines": int(len(ys)), "n_extrapolations": int(len(set(cs))), "n_families": int(len(set(fs))),
                "auc": auc(ps, ys), "risk": risk(ps, ys), "coverage": coverage(ps, ys),
                "risk_ci_extrapolation": ci(ps, ys, risk, cs), "risk_ci_family": ci(ps, ys, risk, fs),
                "coverage_ci_extrapolation": ci(ps, ys, coverage, cs), "coverage_ci_family": ci(ps, ys, coverage, fs)}
m = res["suite"]
print(f"batería: riesgo {m['risk']:.1%} IC extrap {m['risk_ci_extrapolation'][0]:.1%}-{m['risk_ci_extrapolation'][1]:.1%}, "
      f"familia {m['risk_ci_family'][0]:.1%}-{m['risk_ci_family'][1]:.1%} | cobertura {m['coverage']:.1%} "
      f"IC extrap {m['coverage_ci_extrapolation'][0]:.1%}-{m['coverage_ci_extrapolation'][1]:.1%}, familia {m['coverage_ci_family'][0]:.1%}-{m['coverage_ci_family'][1]:.1%}")
EXT = dict(pickle.load(open("data/external_families.pkl", "rb")))
EXT = {"TD2-relax": [d for d in pickle.load(open("data/revision/suite_r0.pkl", "rb")) if d["field"] == "TD2-relax"], **EXT}
res["external"] = {}
for name, cases in EXT.items():
    c = np.concatenate([confidence(d["ind"], DEFAULT_CALIBRATION) for d in cases])
    yy = np.concatenate([d["bad"] for d in cases]).astype(float)
    g = np.concatenate([[i] * len(d["bad"]) for i, d in enumerate(cases)])
    ok = np.isfinite(c)
    p, yy, g = 1 - c[ok], yy[ok], g[ok]
    res["external"][name] = {"n_extrapolations": int(len(cases)), "n_lines": int(len(yy)), "risk": risk(p, yy), "coverage": coverage(p, yy),
                             "risk_ci": ci(p, yy, risk, g), "coverage_ci": ci(p, yy, coverage, g)}
    m = res["external"][name]
    print(f"{name:11s} riesgo {m['risk']:.1%} IC {m['risk_ci'][0]:.1%}-{m['risk_ci'][1]:.1%} | cobertura {m['coverage']:.0%} IC {m['coverage_ci'][0]:.0%}-{m['coverage_ci'][1]:.0%}")

# ------------------------------------------------------------------ (c) what the wrong lines are made of
shape = np.concatenate([d["err"]["shape"] for d in S])[oks]
topo = np.concatenate([d["err"]["topology"] for d in S])[oks] > 0
sh = np.nan_to_num(shape, nan=0.0) > S[0]["bad_threshold"]
assert np.array_equal(sh | topo, ys.astype(bool)), "the label is not shape | topology"
nb = ys.sum()
res["label"] = {"wrong": float(ys.mean()), "shape_only_of_wrong": float((sh & ~topo).sum() / nb), "topology_only_of_wrong": float((topo & ~sh).sum() / nb),
                "both_of_wrong": float((sh & topo).sum() / nb), "topology_change_of_all": float(topo.mean()),
                "auc_shape_without_topology_change": auc(ps[~topo], sh[~topo].astype(float)),
                "auc_topology_change": auc(ps, topo.astype(float)), "n_without_topology_change": int((~topo).sum())}
m = res["label"]
print(f"etiqueta: sólo forma {m['shape_only_of_wrong']:.1%}, sólo abierta/cerrada {m['topology_only_of_wrong']:.1%}, ambas {m['both_of_wrong']:.1%} "
      f"de las malas | AUC forma (sin cambio de topología) {m['auc_shape_without_topology_change']:.3f}, AUC cambio de topología {m['auc_topology_change']:.3f}")

# ------------------------------------------------------------------ (d) correlations between indicators
ind = {k: np.concatenate([np.asarray(d["ind"][k], float) for d in S])[oks] for k in ("alpha_spread", "buffer", "length", "apex")}
res["spearman"] = {}
keys = list(ind)
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        res["spearman"][f"{a}~{b}"] = float(spearmanr(ind[a], ind[b]).correlation)
print("Spearman:", {k: round(v, 2) for k, v in res["spearman"].items()})

# ------------------------------------------------------------------ (e) AIA: lines and stored sensitivity
AIA = {"11158": "data/aia_validation_harp377.pkl", "11429": "data/harp1449_20120306_2336_aia.pkl", "12673": "data/harp7115_20170906_1136_aia.pkl"}
if all(os.path.exists(f) for f in AIA.values()):
    res["aia"] = {}
    for noaa, f in AIA.items():
        sm = pickle.load(open(f, "rb"))["summary"]
        w = sm["by_wavelength"]["171"] if "171" in sm["by_wavelength"] else sm["by_wavelength"][171]
        sens = w["sensitivity"]
        res["aia"][noaa] = {"n_lines": int(sm["n_lines"]), "n_measured_171": int(w["n_measured"]), "sigma_px": float(sm["sigma_px"]),
                            "ridge_pct": float(sm["ridge_pct"]), "sigmas": sorted({float(s["sigma"]) for s in sens}),
                            "percentiles": sorted({float(s["pct"]) for s in sens}), "n_combinations": len(sens),
                            "n_accepted_better": int(sum(s["median_reliable"] < s["median_unreliable"] for s in sens))}
    print("AIA:", {k: (v["n_lines"], v["n_measured_171"], f"{v['n_accepted_better']}/{v['n_combinations']}") for k, v in res["aia"].items()})

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print("guardado", OUT)
