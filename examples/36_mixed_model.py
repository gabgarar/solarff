"""A confidence fitted on analytical and realistic truths together.

The published confidence is fitted on exact force-free fields and does not rank the lines against radiative MHD
simulations (``examples/33_mhd_truth.py``). There are too few public simulations to fit a model on them alone. This
script tries both kinds of truth together, each analytical family and each simulation being one group:

- the analytical cases are computed again with the potential-field line of every seed, so that they have the
  indicator ``departure`` (how far the NLFFF line is from the potential line): the ten families of the paper, three
  noise models, grids of 32 and 48 points, one realisation;
- the realistic cases are those of ``examples/35_realistic_truths.py``;
- logistic models with the five published terms, and with those five and the departure, are fitted with the same
  weight for every group and evaluated on the group left out.

The models, the validations and the criteria were written down in the plan of the paper before running the analysis
(19-Nov-2026).

Saves ``data/mixed_analytic.pkl`` (checkpoint, with the per-line records) and ``docs/resultados/mixed_model.json``.

    python -u examples/36_mixed_model.py            # about half an hour the first time
    python examples/36_mixed_model.py --analyze-only
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np
from scipy.optimize import minimize

from solarff.analytic import CALIBRATION_FIELDS, EXTERNAL_TEST_FIELDS, field_magnetogram
from solarff.calibration import NOISE_MODELS, SIX_FAMILIES, _nz, ece
from solarff.confidence import DEFAULT_CALIBRATION, _design, auc, calibration_case, confidence
from solarff.simulations import SIMULATIONS
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "mixed_model.json"))
ap.add_argument("--sizes", nargs="*", type=int, default=[32, 48])
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

EXTERNAL = ("TD2-relax", "LL-IV", "LFFF-delta", "TD3-relax")
FAMILIES = list(SIX_FAMILIES) + list(EXTERNAL)
KNOWN = {**CALIBRATION_FIELDS, **EXTERNAL_TEST_FIELDS}
T = DEFAULT_CALIBRATION["threshold"]
F5 = list(DEFAULT_CALIBRATION["features"])
CKPT = os.path.join(a.data, "mixed_analytic.pkl")

# ------------------------------------------------------------------ analytical cases with the potential lines
state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {"cases": {}}
if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        for size in a.sizes:
            for fi, fam in enumerate(FAMILIES):
                exact = None
                for j, (nm, (sz, st)) in enumerate(NOISE_MODELS.items()):
                    key = f"{fam} | {size} | {nm}"
                    if key in state["cases"]:
                        continue
                    exact = KNOWN[fam](size, size, _nz(size)) if exact is None else exact
                    tc = time.time()
                    k = 500009 + 1000 * fi + 10 * size + j
                    mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=k)
                    rec = calibration_case(exact, mag, _nz(size), n_seeds=300, seed=k + 7, noise_bt=st, potential=True)
                    rec.update(group=fam, size=int(size), noise_model=nm, seconds=time.time() - tc)
                    state["cases"][key] = rec
                    pickle.dump(state, open(CKPT, "wb"))
                    bp = (np.asarray(rec["pot"]["pp"]["shape"], float) > 0.1) | (np.asarray(rec["pot"]["pp"]["topology"]) > 0)
                    print(f"  {key}: {np.mean(rec['bad']):.0%} wrong lines; potential field {bp.mean():.0%} ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)",
                          flush=True)


# ------------------------------------------------------------------ all the groups
def departure(ind):
    """Departure of the NLFFF line from the potential line; a change between open and closed counts as at least half the length."""
    key = "departure" if "departure" in ind else "ens_pot"            # the records of examples/34 and 35 use the second name
    s = np.asarray(ind[key], float)
    return np.where(np.asarray(ind[key + "_topo"]) > 0, np.maximum(np.nan_to_num(s, nan=0.5), 0.5), s)


def rows(d, group):
    sh = np.asarray(d["err"]["shape"], float)
    tp = np.asarray(d["err"]["topology"]) > 0
    psh, ptp = np.asarray(d["pot"]["pp"]["shape"], float), np.asarray(d["pot"]["pp"]["topology"]) > 0
    X = np.column_stack([_design(d["ind"], F5), np.log10(np.maximum(departure(d["ind"]), 1e-3))])
    return {"X": X, "y10": ((sh > 0.10) | tp).astype(float), "y20": ((sh > 0.20) | tp).astype(float),
            "pot10": ((psh > 0.10) | ptp).astype(float), "conf": confidence(d["ind"], DEFAULT_CALIBRATION), "group": np.full(len(sh), group, dtype=object)}


parts = [rows(x, x["group"]) for x in state["cases"].values()]
for sim in SIMULATIONS:
    path = os.path.join(a.data, f"realistic_{sim}.pkl")
    if os.path.exists(path):
        parts += [rows(x, sim) for x in pickle.load(open(path, "rb"))["cases"].values()]
out = {"code_version": code_version()["label"], "features": F5 + ["departure"], "sizes": a.sizes, "groups": {}, "models": {}}
if parts:
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ok = np.isfinite(D["X"]).all(axis=1) & np.isfinite(D["conf"])
    D = {k: v[ok] for k, v in D.items()}
    groups = [g for g in FAMILIES + list(SIMULATIONS) if (D["group"] == g).any()]
    analytic, real = [g for g in groups if g in FAMILIES], [g for g in groups if g in SIMULATIONS]
    wgt = np.zeros(len(D["group"]))
    for g in groups:                                   # the same weight for every group, whatever its number of lines
        wgt[D["group"] == g] = 1.0 / (D["group"] == g).sum()

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

    def score(p, y):
        acc = p <= 1 - T
        k = max(1, int(round(0.36 * len(p))))
        return {"n_lines": int(len(y)), "bad_rate": float(y.mean()), "predicted_bad": float(p.mean()), "auc": auc(p, y > 0), "ece": ece(p, y),
                "accepted": float(acc.mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None, "bad_best_36": float(y[np.argsort(p)[:k]].mean())}

    for g in groups:
        m = D["group"] == g
        out["groups"][g] = {"kind": "analytical" if g in FAMILIES else "realistic", "n_lines": int(m.sum()), "bad_rate_10": float(D["y10"][m].mean()),
                            "bad_rate_20": float(D["y20"][m].mean()), "potential_bad_rate_10": float(D["pot10"][m].mean()),
                            "published": {lab: score(1 - D["conf"][m], D[lab][m]) for lab in ("y10", "y20")}}
    COLS = {"five": slice(0, 5), "six": slice(0, 6)}
    SCHEMES = {"analytical_six_families": [g for g in SIX_FAMILIES if g in groups], "mixed": groups, "realistic": real}
    for scheme, train in SCHEMES.items():
        if len(train) < 2:
            continue
        out["models"][scheme] = {}
        for name, cols in COLS.items():
            res = {"by_group": {}, "pooled": {}}
            pred = {lab: np.full(len(D["group"]), np.nan) for lab in ("y10", "y20")}
            for lab in ("y10", "y20"):
                for g in groups:                       # the group evaluated never takes part in the fit that predicts it
                    tr = np.isin(D["group"], [t for t in train if t != g])
                    pred[lab][D["group"] == g] = predict(fit(D["X"][tr][:, cols], D[lab][tr], wgt[tr]), D["X"][D["group"] == g][:, cols])
            for g in groups:
                m = D["group"] == g
                res["by_group"][g] = {lab: score(pred[lab][m], D[lab][m]) for lab in ("y10", "y20")}
                res["by_group"][g]["in_training_scheme"] = g in train
            for kind, gs in (("analytical_calibration", [g for g in SIX_FAMILIES if g in groups]), ("analytical_external", [g for g in EXTERNAL if g in groups]),
                             ("realistic", real)):
                m = np.isin(D["group"], gs)
                if m.any():
                    res["pooled"][kind] = {lab: score(pred[lab][m], D[lab][m]) for lab in ("y10", "y20")}
            tr = np.isin(D["group"], train)
            res["coef_all_training_groups"] = {lab: dict(zip(F5 + ["departure"], fit(D["X"][tr][:, cols], D[lab][tr], wgt[tr])["coef"])) for lab in ("y10", "y20")}
            out["models"][scheme][name] = res
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

if parts:
    print("groups (wrong lines with the 10 % and 20 % definitions; potential field; published confidence: AUC, accepted, wrong among them)")
    for g, q in out["groups"].items():
        pb = q["published"]["y10"]
        print(f"  {g:17s} {q['kind']:10s} {q['n_lines']:5d} lines | {q['bad_rate_10']:.2f} / {q['bad_rate_20']:.2f} | potential {q['potential_bad_rate_10']:.2f} | "
              f"AUC {pb['auc']:.3f}, {pb['accepted']:.2f}, {pb['bad_accepted'] if pb['bad_accepted'] is not None else float('nan'):.3f}")
    for scheme, ms in out["models"].items():
        for name, res in ms.items():
            print(f"{scheme}, {name} terms: every group predicted by a fit without it")
            for kind, q in res["pooled"].items():
                print(f"  {kind:24s} " + " | ".join(f"{lab}: AUC {v['auc']:.3f}, ECE {v['ece']:.3f}, accepted {v['accepted']:.2f} with {v['bad_accepted'] if v['bad_accepted'] is not None else float('nan'):.3f} wrong, "
                                                f"best 36 %: {v['bad_best_36']:.2f} of {v['bad_rate']:.2f}" for lab, v in q.items()))
            for g in real + list(EXTERNAL):
                if g in res["by_group"]:
                    v = res["by_group"][g]["y10"]
                    w = res["by_group"][g]["y20"]
                    print(f"    {g:17s} y10: AUC {v['auc']:.3f}, predicted {v['predicted_bad']:.2f} of {v['bad_rate']:.2f}, accepted {v['accepted']:.2f} with "
                          f"{v['bad_accepted'] if v['bad_accepted'] is not None else float('nan'):.3f} | y20: AUC {w['auc']:.3f}, accepted {w['accepted']:.2f} with "
                          f"{w['bad_accepted'] if w['bad_accepted'] is not None else float('nan'):.3f} of {w['bad_rate']:.2f}")
            print("    coefficients (10 %):", {k: round(v, 2) for k, v in res["coef_all_training_groups"]["y10"].items()})
print("written", a.json)
