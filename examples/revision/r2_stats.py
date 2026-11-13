"""Revision R2: strict statistical validation of the per-line confidence.

From a calibration suite (by default the one of the paper: six families in boxes of four
proportions, 162 extrapolations) computes, with the logistic regression and leave-one-family-out validation:

* AUC, ECE, Brier, calibration slope and intercept, global, per family and per box proportion;
* bootstrap intervals grouped by extrapolation (case) and by family;
* confusion matrix, precision, recall and specificity with thresholds 0.7/0.8/0.9
  (positive = wrong line; flagged as unreliable if confidence < threshold);
* risk–coverage curve and reliability diagram;
* ablation: removing each indicator.

Label: ``--label rel`` (the one of the paper: shape error > 10 % of the length or
open/closed change), ``abs`` (mean error > ``--abs-mm`` Mm) or ``mixed`` (both).

    python examples/revision/r2_stats.py --tag nuevo
    python examples/revision/r2_stats.py --tag nuevo_l2_0 --no-ablation --l2 0

The figures of version 1.1 (four terms, 108 extrapolations) are those of
``--suite data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl --features alpha_spread buffer length length_sq``.
"""
import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from solarff.calibration import USED_FEATURES, add_box, box_name, cross_validate, ece  # noqa: E402
from solarff.confidence import auc  # noqa: E402

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax_boxes.pkl")
ap.add_argument("--tag", default="nuevo")
ap.add_argument("--label", default="rel", choices=["rel", "abs", "mixed"])
ap.add_argument("--abs-mm", type=float, default=2.0)
ap.add_argument("--exclude", nargs="*", default=[], help="familias que no se usan (p. ej. TD2-relax)")
ap.add_argument("--features", nargs="*", default=list(USED_FEATURES))
ap.add_argument("--no-ablation", action="store_true")
ap.add_argument("--l2", type=float, default=1e-2, help="penalización L2 de la logística (0 = sin penalizar)")
a = ap.parse_args()
OUT = "docs/revision/resultados"
os.makedirs(OUT, exist_ok=True)

S = pickle.load(open(a.suite, "rb"))
data = S["data"] if isinstance(S, dict) else S
data = add_box([d for d in data if d["field"] not in a.exclude])   # with length_h, also for suites saved without it


def relabel(d):
    e = dict(d)
    err = d["err"]
    rel = (err["shape"] > 0.1) | (err["topology"] > 0)
    if a.label == "rel":
        e["bad"] = rel
    else:
        ab = (np.nan_to_num(err["shape_abs"], nan=np.inf) > a.abs_mm) | (err["topology"] > 0)
        e["bad"] = ab if a.label == "abs" else (rel & ab) | (err["topology"] > 0)
    return e


data = [relabel(d) for d in data]


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def cal_slope_intercept(p, y):
    """Logistic regression of y on logit(p): slope 1 and intercept 0 = perfect calibration."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    b = np.zeros(2)
    X = np.column_stack([np.ones_like(z), z])
    for _ in range(50):
        q = 1 / (1 + np.exp(-X @ b))
        W = q * (1 - q) + 1e-9
        H = X.T @ (X * W[:, None])
        b = b + np.linalg.solve(H, X.T @ (y - q))
    return float(b[1]), float(b[0])


def metrics(p, y):
    s, i = cal_slope_intercept(p, y)
    return {"auc": float(auc(p, y)), "ece": float(ece(p, y)), "ece_15": float(ece(p, y, 15)), "brier": brier(p, y),
            "cal_slope": s, "cal_intercept": i, "bad_rate": float(y.mean()), "n": int(len(y))}


def at_threshold(p, y, t):
    conf = 1 - p
    acc = conf >= t
    tp = int(((~acc) & (y == 1)).sum()); fn = int((acc & (y == 1)).sum())
    fp = int(((~acc) & (y == 0)).sum()); tn = int((acc & (y == 0)).sum())
    return {"threshold": t, "accepted": float(acc.mean()), "bad_among_accepted": float(y[acc].mean()) if acc.any() else None,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1)}


cv = cross_validate(data, features=a.features, models=["logistic"], group="field", verbose=False, l2=a.l2)
p = cv["models"]["logistic"]["p"]
y = cv["y"].astype(float)
fam = np.asarray(cv["groups"])
case = np.asarray(cv["case"])
R = {"suite": a.suite, "label": a.label, "l2": a.l2, "abs_mm": a.abs_mm if a.label != "rel" else None,
     "features": a.features, "n_cases": int(len(np.unique(case))), "n_lines": int(len(y)),
     "families": sorted(set(fam.tolist())), "overall": metrics(p, y)}
rng = np.random.default_rng(0)


def boot(groups, n=1000):
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        sel = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y[sel])) < 2:
            continue
        vals.append([auc(p[sel], y[sel]), ece(p[sel], y[sel]), brier(p[sel], y[sel])])
    v = np.array(vals)
    return {k: [float(np.percentile(v[:, j], 2.5)), float(np.percentile(v[:, j], 97.5))]
            for j, k in enumerate(("auc", "ece", "brier"))}


R["ci_case_bootstrap"] = boot(case)
R["ci_family_bootstrap"] = boot(fam)
R["per_family"] = {}
for g in np.unique(fam):
    m = fam == g
    R["per_family"][str(g)] = {**metrics(p[m], y[m]), "thresholds": [at_threshold(p[m], y[m], t) for t in (0.7, 0.8, 0.9)]}
box = np.array([box_name(d) for d in data])[case]       # proportion of the box of each line
R["per_box"] = {}
for g in ("standard", "flat", "very flat", "tall"):
    m = box == g
    if m.any():
        R["per_box"][g] = {**metrics(p[m], y[m]), "n_cases": int(len(np.unique(case[m]))),
                           "thresholds": [at_threshold(p[m], y[m], t) for t in (0.7, 0.8, 0.9)]}
fa = np.array([v["auc"] for v in R["per_family"].values()])
R["family_auc_mean_sd_min"] = [float(fa.mean()), float(fa.std(ddof=1)), float(fa.min())]
R["thresholds"] = [at_threshold(p, y, t) for t in (0.7, 0.8, 0.9)]
# risk versus coverage
order = np.argsort(p)
cov = np.arange(1, len(p) + 1) / len(p)
risk = np.cumsum(y[order]) / np.arange(1, len(p) + 1)
R["aurc"] = float(np.trapezoid(risk, cov))
R["risk_at_coverage"] = {str(c): float(risk[int(c * len(p)) - 1]) for c in (0.2, 0.4, 0.5, 0.6, 0.8)}

if not a.no_ablation:
    R["ablation"] = {}
    for f in a.features:
        feats = [x for x in a.features if x != f and not (f == "length" and x == "length_sq")]
        c2 = cross_validate(data, features=feats, models=["logistic"], group="field", verbose=False)
        p2 = c2["models"]["logistic"]["p"]; y2 = c2["y"].astype(float)
        R["ablation"][f] = {"without": feats, "auc": float(auc(p2, y2)), "ece": float(ece(p2, y2))}

# figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
bins = np.linspace(0, 1, 11)
for g in list(np.unique(fam)) + ["all"]:
    m = np.ones_like(y, bool) if g == "all" else fam == g
    k = np.digitize(p[m], bins[1:-1])
    xs = [p[m][k == i].mean() for i in range(10) if (k == i).sum() > 20]
    ys = [y[m][k == i].mean() for i in range(10) if (k == i).sum() > 20]
    ax[0].plot(xs, ys, "-o" if g == "all" else "-", lw=2.2 if g == "all" else 0.9, ms=3,
               color="k" if g == "all" else None, label="all" if g == "all" else str(g))
ax[0].plot([0, 1], [0, 1], ":", color="#999")
ax[0].set_xlabel("predicted probability of a wrong line"); ax[0].set_ylabel("observed fraction of wrong lines")
ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("Reliability (leave one family out)", fontsize=10)
ax[1].plot(cov, risk, color="k")
for t in R["thresholds"]:
    ax[1].plot(t["accepted"], t["bad_among_accepted"], "o", label=f"confidence ≥ {t['threshold']}")
ax[1].axhline(y.mean(), ls=":", color="#999")
ax[1].set_xlabel("coverage (fraction of lines kept)"); ax[1].set_ylabel("wrong lines among those kept")
ax[1].legend(fontsize=8, frameon=False); ax[1].set_title("Risk versus coverage", fontsize=10)
fig.savefig(os.path.join(OUT, f"r2_{a.tag}.png"), dpi=140)
plt.close(fig)
json.dump(R, open(os.path.join(OUT, f"r2_{a.tag}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
o = R["overall"]
print(f"[{a.tag}] AUC {o['auc']:.3f} caso{np.round(R['ci_case_bootstrap']['auc'], 3).tolist()} familia{np.round(R['ci_family_bootstrap']['auc'], 3).tolist()} | "
      f"ECE {o['ece']:.3f} Brier {o['brier']:.3f} pendiente {o['cal_slope']:.2f} intercepto {o['cal_intercept']:+.2f}")
print("  AUC por familia:", {k: round(v["auc"], 3) for k, v in R["per_family"].items()})
for t in R["thresholds"]:
    print(f"  umbral {t['threshold']}: aceptadas {t['accepted']:.1%}, malas entre ellas {t['bad_among_accepted']:.1%}, "
          f"precisión {t['precision']:.2f}, recall {t['recall']:.2f}, especificidad {t['specificity']:.2f}")
for g, v in R["per_box"].items():
    t = v["thresholds"][1]
    print(f"  cajas {g}: {v['n_cases']} casos, AUC {v['auc']:.3f}, ECE {v['ece']:.3f}, aceptadas {t['accepted']:.1%}, malas entre ellas {t['bad_among_accepted']:.1%}")
if "ablation" in R:
    print("  ablación (AUC sin el indicador):", {k: round(v["auc"], 3) for k, v in R["ablation"].items()})
