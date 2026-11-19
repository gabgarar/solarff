"""Realistic truths: the level of the error, and a choice between the NLFFF line and the potential line.

A model fitted on simulations ranks the lines of another simulation (``examples/35_realistic_truths.py`` and
``36_mixed_model.py``), but it does not know how many of them are wrong, and in the quiet-Sun runs the potential
field is better than the NLFFF with nothing to warn of it. Two questions, with the per-line records already saved:

1. the level: does a model with one more indicator, the signal to noise of the transverse field at the feet of the
   line (``foot_snr``), predict the fraction of wrong lines of a simulation it has not seen?
2. the choice: for every seed there are two lines, that of the NLFFF and that of the potential field. A model
   predicts which one is closer to the truth, and the hybrid set takes the NLFFF line where it is favoured.

Every simulation is predicted by a fit made with the other two, with the same weight for each. The indicator, the
models and the criteria were written down in the plan of the paper before running this (19-Nov-2026).

Saves ``docs/resultados/realistic_choice.json``.

    python examples/37_realistic_choice.py
"""
import argparse
import json
import os
import pickle

import numpy as np
from scipy.optimize import minimize

from solarff.confidence import DEFAULT_CALIBRATION, _design, auc
from solarff.simulations import SIMULATIONS
from solarff.utils import code_version

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "realistic_choice.json"))
a = ap.parse_args()

F5 = list(DEFAULT_CALIBRATION["features"])
EXTRA = ["sin_jb", "div", "base_change", "apex", "open"]
MODELS = {"six": F5 + ["departure"], "seven": F5 + ["departure", "foot_snr"], "all": F5 + ["departure", "foot_snr"] + EXTRA}


def departure(ind):
    key = "departure" if "departure" in ind else "ens_pot"
    s = np.asarray(ind[key], float)
    return np.where(np.asarray(ind[key + "_topo"]) > 0, np.maximum(np.nan_to_num(s, nan=0.5), 0.5), s)


def columns(ind, feats):
    return np.column_stack([np.log10(np.maximum(departure(ind), 1e-3)) if f == "departure" else _design(ind, [f])[:, 0] for f in feats])


def table(cases):
    cat = lambda f: np.concatenate([np.asarray(f(x), float) for x in cases])  # noqa: E731
    sh, tp = cat(lambda x: x["err"]["shape"]), cat(lambda x: x["err"]["topology"]) > 0
    psh, ptp = cat(lambda x: x["pot"]["pp"]["shape"]), cat(lambda x: x["pot"]["pp"]["topology"]) > 0
    eff_n, eff_p = np.where(tp, np.inf, sh), np.where(ptp, np.inf, psh)          # a line with the wrong topology loses
    t = {"X": np.vstack([columns(x["ind"], MODELS["all"]) for x in cases]), "better": (eff_n < eff_p).astype(float)}      # a tie goes to the potential line
    for lab, thr in (("10", 0.10), ("20", 0.20)):
        t["y" + lab], t["p" + lab] = ((sh > thr) | tp).astype(float), ((psh > thr) | ptp).astype(float)
    ok = np.isfinite(t["X"]).all(axis=1) & np.isfinite(sh) & np.isfinite(psh)
    return {k: v[ok] for k, v in t.items()}


def fit(X, y, w, l2=1e-2):
    w = w / w.mean()
    mu = np.average(X, 0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, 0, weights=w)) + 1e-12
    Z = (X - mu) / sd

    def f(beta):
        p = 1 / (1 + np.exp(-(Z @ beta[1:] + beta[0])))
        nll = -np.mean(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
        return nll + l2 * (beta[1:] ** 2).sum(), np.r_[np.mean(w * (p - y)), Z.T @ (w * (p - y)) / len(y) + 2 * l2 * beta[1:]]

    r = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B")
    return {"intercept": float(r.x[0]), "coef": r.x[1:].tolist(), "mu": mu.tolist(), "sd": sd.tolist()}


def predict(m, X):
    return 1 / (1 + np.exp(-((X - np.array(m["mu"])) / np.array(m["sd"]) @ np.array(m["coef"]) + m["intercept"])))


tabs = {}
for sim in SIMULATIONS:
    path = os.path.join(a.data, f"realistic_{sim}.pkl")
    if os.path.exists(path):
        tabs[sim] = table(list(pickle.load(open(path, "rb"))["cases"].values()))
out = {"code_version": code_version()["label"], "models": {k: v for k, v in MODELS.items()}, "level": {}, "choice": {}}
for name, feats in MODELS.items():
    cols = slice(0, len(feats))
    out["level"][name], out["choice"][name] = {}, {}
    for sim, t in tabs.items():
        others = [s for s in tabs if s != sim]
        X = np.vstack([tabs[s]["X"][:, cols] for s in others])
        w = np.concatenate([np.full(len(tabs[s]["y10"]), 1.0 / len(tabs[s]["y10"])) for s in others])
        # 1. the level of the error
        lev = {}
        for lab in ("y10", "y20"):
            p = predict(fit(X, np.concatenate([tabs[s][lab] for s in others]), w), t["X"][:, cols])
            lev[lab] = {"auc": auc(p, t[lab] > 0), "predicted_bad": float(p.mean()), "bad_rate": float(t[lab].mean()),
                        "level_error": float(abs(p.mean() - t[lab].mean()))}
        out["level"][name][sim] = lev
        # 2. the choice between the two lines
        m = fit(X, np.concatenate([tabs[s]["better"] for s in others]), w)
        q = predict(m, t["X"][:, cols])
        use = q > 0.5
        ch = {"nlfff_better": float(t["better"].mean()), "chosen_nlfff": float(use.mean()), "auc_choice": auc(q, t["better"] > 0),
              "coef": dict(zip(feats, m["coef"]))}
        for lab in ("10", "20"):
            hyb, yn, yp = np.where(use, t["y" + lab], t["p" + lab]), t["y" + lab], t["p" + lab]
            ch["bad_" + lab] = {"nlfff": float(yn.mean()), "potential": float(yp.mean()), "hybrid": float(hyb.mean()), "oracle": float(np.minimum(yn, yp).mean()),
                                "hybrid_minus_best": float(hyb.mean() - min(yn.mean(), yp.mean()))}
        out["choice"][name][sim] = ch
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

for name in MODELS:
    print(f"{name} terms: every simulation predicted by a fit with the other two")
    for sim in tabs:
        lv, ch = out["level"][name][sim], out["choice"][name][sim]
        print(f"  {sim:16s} level: " + " | ".join(f"{lab}: AUC {v['auc']:.3f}, predicted {v['predicted_bad']:.2f} of {v['bad_rate']:.2f}" for lab, v in lv.items()))
        print(f"  {'':16s} choice: NLFFF closer in {ch['nlfff_better']:.2f}, chosen in {ch['chosen_nlfff']:.2f}, AUC {ch['auc_choice']:.3f} | " + " | ".join(
            f"wrong at {lab} %: NLFFF {v['nlfff']:.3f}, potential {v['potential']:.3f}, hybrid {v['hybrid']:.3f}, the better of the two per line {v['oracle']:.3f}"
            for lab, v in (("10", ch["bad_10"]), ("20", ch["bad_20"]))))
print("written", a.json)
