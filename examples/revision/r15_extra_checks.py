"""External review 4 (10-Nov): three inexpensive extra checks of the published confidence model.

With the rerun of the six-family suite (``data/revision/suite_r0.pkl`` without TD2) plus the cases in boxes of
other proportions (162 extrapolations) and the five terms of the published model:

(1) Unpenalised alternative: coefficients of the fit with L2 = 0 on the 162 extrapolations and, from its
    leave-one-family-out predictions, the confidence threshold that gives the same risk (wrong lines among
    the accepted) as the published model at 0.8, with its coverage.
(2) Observed versus predicted fraction of wrong lines of the published model (L2 = 0.01, leave one family
    out) in the range used for acceptance: p < 0.05, 0.05-0.1 and 0.1-0.2.
(3) Validation leaving out one noise model, one resolution or one box proportion instead of one family (same
    model), and the model fitted without any flat box evaluated on the flat ones, with five and with four terms.

Writes ``docs/revision/resultados/r15_extra_checks.json``.

    python examples/revision/r15_extra_checks.py      # ~1 min
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.calibration import box_name, ece, with_proportions  # noqa: E402
from solarff.confidence import _design, auc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join("docs", "revision", "resultados", "r15_extra_checks.json")
FEATS = ["alpha_spread", "buffer", "length", "length_sq", "length_h"]
# the rerun of the suite (other random seeds, with absolute errors) plus the cases in boxes of other proportions
D = with_proportions([d for d in pickle.load(open("data/revision/suite_r0.pkl", "rb")) if d["field"] != "TD2-relax"])
X = np.vstack([_design(d["ind"], FEATS) for d in D])
y = np.concatenate([d["bad"] for d in D]).astype(float)
grp = {k: np.concatenate([[str(d[k])] * len(d["bad"]) for d in D]) for k in ("field", "noise_model", "size")}
grp["box"] = np.concatenate([[box_name(d)] * len(d["bad"]) for d in D])
ok = np.all(np.isfinite(X), axis=1)
X, y = X[ok], y[ok]
grp = {k: v[ok] for k, v in grp.items()}


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


def leave_out(group, l2):
    p = np.full(len(y), np.nan)
    for g in sorted(set(grp[group])):
        te = grp[group] == g
        p[te] = predict(fit(X[~te], y[~te], l2), X[te])
    return p


def at_threshold(p, t):
    acc = (1 - p) >= t
    return float(acc.mean()), float(y[acc].mean())


res = {"n_lines": int(len(y)), "n_cases": len(D)}
p_pen, p_unp = leave_out("field", 1e-2), leave_out("field", 0.0)
cov_pen, risk_pen = at_threshold(p_pen, 0.8)
cov_unp, risk_unp = at_threshold(p_unp, 0.8)

# (1) unpenalised alternative
mu, sd, b = fit(X, y, 0.0)
ts = np.round(np.arange(0.80, 0.951, 0.001), 3)
t_star = next(float(t) for t in ts if at_threshold(p_unp, t)[1] <= risk_pen)
cov_star, risk_star = at_threshold(p_unp, t_star)
res["unpenalised"] = {"features": FEATS, "mean": mu.tolist(), "std": sd.tolist(), "intercept": float(b[0]), "coef": b[1:].tolist(),
                      "auc": auc(p_unp, y), "ece_10": ece(p_unp, y, 10), "accepted_at_0.8": cov_unp, "risk_at_0.8": risk_unp,
                      "threshold_same_risk": t_star, "accepted_at_threshold": cov_star, "risk_at_threshold": risk_star}
res["published_check"] = {"auc": auc(p_pen, y), "ece_10": ece(p_pen, y, 10), "accepted_at_0.8": cov_pen, "risk_at_0.8": risk_pen}

# (2) observed versus predicted in the acceptance range
res["acceptance_range"] = []
for lo, hi in ((0.0, 0.05), (0.05, 0.1), (0.1, 0.2)):
    m = (p_pen >= lo) & (p_pen < hi if hi < 0.2 else p_pen <= hi)
    res["acceptance_range"].append({"p_lo": lo, "p_hi": hi, "n": int(m.sum()), "predicted": float(p_pen[m].mean()), "observed": float(y[m].mean())})

# (3) leaving out a noise model or a resolution
res["leave_out"] = {}
for group in ("noise_model", "size", "box"):
    p = leave_out(group, 1e-2)
    cov, risk = at_threshold(p, 0.8)
    per = {g: auc(p[grp[group] == g], y[grp[group] == g]) for g in sorted(set(grp[group]))}
    rk = {g: float(y[(grp[group] == g) & (1 - p >= 0.8)].mean()) for g in sorted(set(grp[group]))}
    res["leave_out"][group] = {"auc": auc(p, y), "ece_10": ece(p, y, 10), "accepted_at_0.8": cov, "risk_at_0.8": risk, "auc_by_group": per,
                               "risk_by_group": rk}

# the model fitted without any flat box (standard and tall ones only), evaluated on the flat ones
flat = np.isin(grp["box"], ["flat", "very flat"])
res["trained_without_flat_boxes"] = {}
for name, cols in (("five_terms", slice(None)), ("four_terms", slice(0, 4))):
    p = predict(fit(X[~flat][:, cols], y[~flat], 1e-2), X[flat][:, cols])
    res["trained_without_flat_boxes"][name] = {
        g: {"auc": auc(p[grp["box"][flat] == g], y[flat][grp["box"][flat] == g]),
            "accepted_at_0.8": float((1 - p[grp["box"][flat] == g] >= 0.8).mean()),
            "risk_at_0.8": float(y[flat][(grp["box"][flat] == g) & (1 - p >= 0.8)].mean())} for g in ("flat", "very flat")}

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
