"""Exploration: does refitting the model with boxes of several proportions correct what the published one misses?

The published model was fitted with boxes of a single proportion (``n x n x 0.7 n``). In flat boxes like those of the
regions it accepts 8 to 10 % of wrong lines instead of 4 % (``examples/27_box_proportion.py``), and more with the
complexity of a real region (``examples/28_model_as_truth.py``) or in strongly sheared linear fields on the observed
flux (``examples/25_observed_flux_fields.py``). Here the same recipe (logistic regression, L2 of 0.01, threshold 0.8)
is refitted with the six calibration families in four proportions, with and without two terms that know about the
height of the box, and evaluated on what it has not seen. It is an exploration: the published model is not changed.

    M0   the published model, not refitted (reference)
    M0s  the four published terms fitted with the boxes of the suite only (the published recipe, for the leave-one-out)
    M1   the four published terms, fitted with the four proportions
    M2   M1 plus the apex height relative to the height of the box
    M3   M1 plus the logarithm of the length in box heights
    M4   M1 plus both

Internal evaluation: one family left out at a time, with all its boxes. External evaluation, never used to fit:
TD2, LL-IV, LFFF-delta and TD3 in the boxes of the suite, LFFF-delta in the other boxes, the two magnetohydrostatic
families, the six observed-flux fields and the three region models taken as the truth.

The candidates and the criteria (RCa to RCc) were registered in the plan of the paper before fitting (13-Nov-2026).

Reads the per-line records saved by the other scripts (``data/``); writes ``docs/resultados/recalibration_boxes.json``.

    python examples/30_recalibration_boxes.py
"""
import csv
import gzip
import json
import os
import pickle

import numpy as np

from solarff.calibration import _nz, ece
from solarff.confidence import CALIBRATION_V1 as DEFAULT_CALIBRATION, _design, _fit_logistic, auc, confidence  # the default model when this test was made (version 1.1)
from solarff.utils import code_version

T = DEFAULT_CALIBRATION["threshold"]
BASE = ["alpha_spread", "buffer", "length", "length_sq"]
MODELS = {"M1": [], "M2": ["apex"], "M3": ["loglh"], "M4": ["apex", "loglh"]}
SIX = ["LL-I", "LL-II", "LL-III", "LFFF-bip", "LFFF-quad", "TD-relax"]
D = "data"


def ratio(nx, ny, nz):
    """Height / horizontal diagonal of a box, as ``line_indicators`` measures them."""
    return (nz - 1) / np.hypot(nx - 1, ny - 1)


def case(x, family, box, r, group):
    ind = {k: np.asarray(x["ind"][k], float) for k in ("alpha_spread", "buffer", "length", "apex")}
    return {"ind": ind, "bad": np.asarray(x["bad"]).astype(int), "family": family, "box": box, "ratio": float(r), "group": group}


def design(c, extra):
    X = _design(c["ind"], BASE)
    cols = [X]
    if "apex" in extra:
        cols.append(c["ind"]["apex"][:, None])
    if "loglh" in extra:
        cols.append(np.log10(np.maximum(c["ind"]["length"] / c["ratio"], 1e-4))[:, None])
    return np.column_stack(cols)


def fit(cases, extra):
    X = np.vstack([design(c, extra) for c in cases])
    y = np.concatenate([c["bad"] for c in cases]).astype(float)
    ok = np.all(np.isfinite(X), axis=1)
    cal = _fit_logistic(X[ok], y[ok])
    cal["extra"] = list(extra)
    return cal


def predict(cal, c):
    Z = (design(c, cal["extra"]) - np.array(cal["mu"])) / np.array(cal["sd"])
    conf = 1 - 1 / (1 + np.exp(-(Z @ np.array(cal["coef"]) + cal["intercept"])))
    conf[~np.isfinite(conf)] = np.nan
    return conf


def metrics(c, y):
    ok = np.isfinite(c)
    c, y = c[ok], y[ok]
    acc, a9 = c >= T, c >= 0.9
    return {"n_lines": int(ok.sum()), "bad_rate": float(y.mean()), "auc": auc(1 - c, y), "ece": ece(1 - c, y), "accepted": float(acc.mean()),
            "bad_accepted": float(y[acc].mean()) if acc.any() else None, "accepted_0.9": float(a9.mean()),
            "bad_accepted_0.9": float(y[a9].mean()) if a9.any() else None}


# ---------------------------------------------------------------- data
train, external = [], []
for x in pickle.load(open(os.path.join(D, "calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl"), "rb"))["data"]:
    n = x["size"]
    train.append(case(x, x["field"], "standard", ratio(n, n, _nz(n)), "suite"))
BOXNAME = {"flat": "flat", "very flat": "very flat", "tall": "tall"}
for k, x in pickle.load(open(os.path.join(D, "box_proportion.pkl"), "rb"))["cases"].items():
    c = case(x, x["field"], BOXNAME[x["box"]], ratio(x["size"], x["size"], x["nz"]), "boxes")
    (train if x["field"] in SIX else external).append(c)                       # LFFF-delta in the other boxes is external
for fam, cases in pickle.load(open(os.path.join(D, "external_families.pkl"), "rb")).items():
    for x in cases:
        n = x["size"]
        external.append(case(x, fam, "standard", ratio(n, n, _nz(n)), "external"))
for fam, cases in pickle.load(open(os.path.join(D, "mhs_families.pkl"), "rb")).items():
    for x in cases:
        n = x["size"]
        external.append(case(x, fam, "standard", ratio(n, n, _nz(n)), "external"))
# TD2-relax: from the published per-line table (its per-case records are not kept apart)
rows = [r for r in csv.DictReader(gzip.open(os.path.join("docs", "resultados", "line_table.csv.gz"), "rt", encoding="utf-8")) if r["family"] == "TD2-relax"]
for cid in sorted({r["case"] for r in rows}):
    rs = [r for r in rows if r["case"] == cid]
    n = int(rs[0]["size"])
    x = {"ind": {k: [float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rs] for k in ("alpha_spread", "buffer", "length", "apex")},
         "bad": [int(r["wrong"]) for r in rs]}
    external.append(case(x, "TD2-relax", "standard", ratio(n, n, _nz(n)), "external"))
st = pickle.load(open(os.path.join(D, "observed_flux_fields.pkl"), "rb"))
for k, x in st["cases"].items():
    name = k.split(" |")[0]
    external.append(case(x, name, "region box", ratio(*st["exact"][name]["grid"]), "observed flux"))
for tag, label in (("", ""), ("_post", ", post-flare"), ("_first", ", first time")):
    f = os.path.join(D, f"model_as_truth{tag}.pkl")
    if not os.path.exists(f):
        continue
    st = pickle.load(open(f, "rb"))
    for k, x in st["cases"].items():
        noaa = k.split(" |")[0]
        external.append(case(x, "twin " + noaa + label, "cropped region box", ratio(*st["truth"][noaa]["grid"]), "model as truth"))
fbis = os.path.join(D, "observed_flux_fields_a50.pkl")
if os.path.exists(fbis):
    st = pickle.load(open(fbis, "rb"))
    for k, x in st["cases"].items():
        name = k.split(" |")[0]
        external.append(case(x, name + " (alpha 0.5)", "region box", ratio(*st["exact"][name]["grid"]), "observed flux, moderate shear"))

out = {"code_version": code_version()["label"], "threshold": T, "n_train_cases": len(train), "n_train_lines": int(sum(len(c["bad"]) for c in train)),
       "models": {}, "leave_one_family_out": {}, "external": {}}

# sanity: the published recipe on the suite reproduces the published coefficients
chk = fit([c for c in train if c["group"] == "suite"], [])
out["published_refit_max_diff"] = float(max(np.abs(np.array(chk["coef"]) - np.array(DEFAULT_CALIBRATION["coef"])).max(),
                                             abs(chk["intercept"] - DEFAULT_CALIBRATION["intercept"])))
print(f"published recipe on the suite: max difference with the published coefficients {out['published_refit_max_diff']:.1e}")

# ---------------------------------------------------------------- leave one family out
ALL = ["M0", "M0s"] + list(MODELS)
pred = {m: [] for m in ALL}            # (box, family, conf, y)
for fam in SIX:
    tr = [c for c in train if c["family"] != fam]
    te = [c for c in train if c["family"] == fam]
    cals = {"M0s": fit([c for c in tr if c["group"] == "suite"], [])}
    cals.update({m: fit(tr, extra) for m, extra in MODELS.items()})
    for c in te:
        pred["M0"].append((c["box"], fam, confidence(c["ind"], DEFAULT_CALIBRATION), c["bad"]))
        for m, cal in cals.items():
            pred[m].append((c["box"], fam, predict(cal, c), c["bad"]))
for m in ALL:
    out["leave_one_family_out"][m] = {}
    for box in ("standard", "flat", "very flat", "tall"):
        sel = [p for p in pred[m] if p[0] == box]
        r = metrics(np.concatenate([p[2] for p in sel]), np.concatenate([p[3] for p in sel]))
        r["by_family"] = {f: metrics(np.concatenate([p[2] for p in sel if p[1] == f]), np.concatenate([p[3] for p in sel if p[1] == f])) for f in SIX}
        out["leave_one_family_out"][m][box] = r

# ---------------------------------------------------------------- external
final = {m: fit(train, extra) for m, extra in MODELS.items()}
final["M0s"] = chk
for m, cal in final.items():
    names = BASE + cal["extra"]
    out["models"][m] = {"features": names, "intercept": cal["intercept"], "coef": cal["coef"], "mu": cal["mu"], "sd": cal["sd"]}
groups = {}
for c in external:
    groups.setdefault((c["family"], c["box"]), []).append(c)
for (fam, box), cs in groups.items():
    y = np.concatenate([c["bad"] for c in cs])
    r = {"M0": metrics(np.concatenate([confidence(c["ind"], DEFAULT_CALIBRATION) for c in cs]), y)}
    for m, cal in final.items():
        r[m] = metrics(np.concatenate([predict(cal, c) for c in cs]), y)
    out["external"][f"{fam} | {box}"] = r
# ---------------------------------------------------------------- robustness of M3 (added after seeing the result)
# (a) intervals of the coefficients, resampling whole extrapolations
rng = np.random.default_rng(0)
boot = []
for _ in range(300):
    cal = fit([train[i] for i in rng.integers(0, len(train), len(train))], MODELS["M3"])
    boot.append([cal["intercept"]] + cal["coef"])
boot = np.array(boot)
out["models"]["M3"]["bootstrap"] = {"n": 300, "sd": boot.std(0).tolist(), "lo": np.percentile(boot, 2.5, 0).tolist(), "hi": np.percentile(boot, 97.5, 0).tolist(),
                                    "order": ["intercept"] + BASE + MODELS["M3"]}
# (b) one proportion left out: is the term learnt from the other boxes valid in a proportion it has not seen?
out["leave_one_proportion_out"] = {}
for held, keep in (("flat", ("standard", "very flat", "tall")), ("very flat", ("standard", "flat", "tall")), ("tall", ("standard", "flat", "very flat")),
                   ("flat and very flat", ("standard", "tall"))):
    tr = [c for c in train if c["box"] in keep]
    te = [c for c in train if c["box"] not in keep]
    r = {}
    for m in ("M1", "M3"):
        # the family is also left out, as in the main table
        cs, ys, bx = [], [], []
        for fam in SIX:
            cal = fit([c for c in tr if c["family"] != fam], MODELS[m])
            for c in te:
                if c["family"] == fam:
                    cs.append(predict(cal, c)); ys.append(c["bad"]); bx.append(np.full(len(c["bad"]), c["box"]))
        cs, ys, bx = np.concatenate(cs), np.concatenate(ys), np.concatenate(bx)
        r[m] = {b: metrics(cs[bx == b], ys[bx == b]) for b in sorted(set(bx))}
        r[m + "_coef"] = fit(tr, MODELS[m])["coef"]
    out["leave_one_proportion_out"][held] = r
json.dump(out, open(os.path.join("docs", "resultados", "recalibration_boxes.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def cell(r):
    return f"{r['auc']:.3f} {100 * r['accepted']:3.0f}% {'  - ' if r['bad_accepted'] is None else format(100 * r['bad_accepted'], '4.1f')}"


print("\ncoefficients (standardised): " + " | ".join(f"{m}: " + ", ".join(f"{n} {b:+.2f}" for n, b in zip(v["features"], v["coef"])) for m, v in out["models"].items()))
print("\nleave one family out (AUC, accepted, wrong among accepted %)")
print(f"  {'box':10s} " + " | ".join(f"{m:^15s}" for m in ALL))
for box in ("standard", "flat", "very flat", "tall"):
    print(f"  {box:10s} " + " | ".join(cell(out["leave_one_family_out"][m][box]) for m in ALL))
for box in ("flat", "very flat"):
    for f in SIX:
        print(f"  {box[:9]:9s} {f:9s} " + " | ".join(cell(out["leave_one_family_out"][m][box]["by_family"][f]) for m in ALL))
print("\nexternal (AUC, accepted, wrong among accepted %)")
print(f"  {'family | box':38s} " + " | ".join(f"{m:^15s}" for m in ALL))
for name, r in out["external"].items():
    print(f"  {name:38s} " + " | ".join(cell(r[m]) for m in ALL))
b = out["models"]["M3"]["bootstrap"]
print("\nM3, coefficients with their 95 % interval (300 replicates by extrapolation): " + " | ".join(
    f"{n} {v:+.2f} [{lo:+.2f}, {hi:+.2f}]" for n, v, lo, hi in zip(b["order"], [out["models"]["M3"]["intercept"]] + out["models"]["M3"]["coef"], b["lo"], b["hi"])))
print("\none proportion left out, and the family too (AUC, accepted, wrong among accepted %): M1 | M3")
for held, r in out["leave_one_proportion_out"].items():
    for box in r["M3"]:
        print(f"  without {held:18s} tested on {box:10s} {cell(r['M1'][box])} | {cell(r['M3'][box])}   (coefficient of the length in heights: {r['M3_coef'][-1]:+.2f})")
print("written docs/resultados/recalibration_boxes.json")
