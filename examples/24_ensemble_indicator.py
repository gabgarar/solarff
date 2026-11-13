"""Does the displacement of a line under a change of what is not known tell the wrong lines apart?

For every case of the suite three more extrapolations are made (``calibration_case(ensemble=...)``): on the
magnetogram rebinned by 2 (``coarse``), with the side and top faces fixed to the LFFF of best-fit alpha instead
of the potential field (``lfff``), and with three times the smoothing of the preprocessing (``smooth``). The
displacement of each line with respect to the nominal model is a candidate indicator: unlike the Monte Carlo
over the noise, it probes systematic errors. It is evaluated as the other indicators were: alone, and added to
the three of the published model, leaving out one calibration family at a time; and, fitted on the six
calibration families, on the families that took no part in the calibration.

A rule with no fitted parameter is also evaluated (the two-key rule): a line is accepted if the published
model gives it a confidence of at least 0.8 and a variant reproduces it, that is, the line moves by at most
10 % of its length, the definition of a wrong line, and does not change between open and closed.

Last, the displacement is tried as a global indicator of the extrapolation, as the five quantities of
``examples/18_applicability.py``: the fraction of the lines that a variant does not reproduce, against the
extrapolations in which the published model is optimistic (it accepts 10 % or more of wrong lines).

Twelve families (six of the calibration, four external, two magnetohydrostatic), three noise models, three
resolutions, one realization. The ``lfff`` variant is not a fair test in the families that are themselves linear
(LFFF-*, MHS-*), where it is close to the exact solution: the criteria registered in the plan of the paper
(12-Nov-2026) look at the Low & Lou and rope families.

Saves ``data/ensemble_<family>.pkl`` (with checkpoints) and ``docs/resultados/ensemble_indicator.json``.

    python examples/24_ensemble_indicator.py                  # ~4 h
    python examples/24_ensemble_indicator.py --analyze-only   # with what is already computed
"""
import argparse
import json
import os
import pickle
import warnings

import numpy as np
from scipy.stats import spearmanr

from solarff.analytic import CALIBRATION_FIELDS
from solarff.calibration import calibration_suite, ece
from solarff.confidence import CALIBRATION_V1 as DEFAULT_CALIBRATION, _design, _fit_logistic, auc, confidence  # the default model when this test was made (version 1.1)
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--fields", nargs="*")
ap.add_argument("--sizes", type=int, nargs="+", default=[32, 48, 64])
ap.add_argument("--out", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "ensemble_indicator.json"))
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

CALIBRATION = list(CALIBRATION_FIELDS) + ["TD-relax"]
EXTERNAL = ["TD2-relax", "LL-IV", "LFFF-delta", "TD3-relax", "MHS-bip", "MHS-quad"]
NONLINEAR = ["LL-I", "LL-II", "LL-III", "TD-relax", "TD2-relax", "LL-IV", "TD3-relax"]   # where the lfff variant is a fair test
ORDER = ["TD3-relax", "TD2-relax", "TD-relax", "LL-I", "LL-II", "LL-III", "LL-IV", "LFFF-bip", "LFFF-quad", "LFFF-delta",
         "MHS-bip", "MHS-quad"]
VARIANTS = ("coarse", "lfff", "smooth")
BASE = ["alpha_spread", "buffer", "length", "length_sq"]
SETS = {"published": (), "coarse": ("coarse",), "lfff": ("lfff",), "smooth": ("smooth",), "coarse+smooth": ("coarse", "smooth"),
        "all": VARIANTS}
RULES = {"published": (), "coarse": ("coarse",), "lfff": ("lfff",), "coarse+lfff": ("coarse", "lfff")}
T = 0.8
N_BOOT = 1000


def displacement(ind, v):
    """Displacement under the variant ``v``; a line that changes between open and closed counts as moved by at least half its length."""
    s = np.asarray(ind["ens_" + v], float)
    return np.where(np.asarray(ind["ens_" + v + "_topo"]) > 0, np.maximum(np.nan_to_num(s, nan=0.5), 0.5), s)


def design(d, extra):
    X = np.vstack([np.column_stack([_design(x["ind"], BASE)] + [np.log10(np.maximum(displacement(x["ind"], v), 1e-3)) for v in extra])
                   for x in d])
    y = np.concatenate([x["bad"] for x in d]).astype(float)
    g = np.concatenate([np.full(len(x["bad"]), i) for i, x in enumerate(d)])
    ok = np.isfinite(X).all(axis=1)
    return X[ok], y[ok], g[ok]


def predict(cal, X):
    z = (X - np.array(cal["mu"])) / np.array(cal["sd"]) @ np.array(cal["coef"]) + cal["intercept"]
    return 1 / (1 + np.exp(-z))


def metrics(p, y):
    acc = p <= 1 - T
    return {"n_lines": int(y.size), "bad_rate": float(y.mean()), "auc": auc(p, y), "ece": ece(p, y), "accepted": float(acc.mean()),
            "bad_accepted": float(y[acc].mean()) if acc.any() else None}


def two_key(data):
    """Published model alone and with the second key: the line is reproduced within 10 % by the variants of the rule."""
    out = {}
    rng = np.random.default_rng(0)
    for group, fams in {**{n: [n] for n in data}, "calibration": CALIBRATION}.items():
        if not all(f in data for f in fams):
            continue
        d = [x for f in fams for x in data[f]]
        out[group] = {"n_cases": len(d)}
        for label, variants in RULES.items():
            kept, bad, n = [], [], []
            for x in d:
                c = confidence(x["ind"], DEFAULT_CALIBRATION)
                ok = np.isfinite(c)
                keep = ok & (np.nan_to_num(c, nan=0.0) >= T)
                for v in variants:
                    with np.errstate(invalid="ignore"):
                        keep &= (np.asarray(x["ind"]["ens_" + v], float) <= 0.1) & (np.asarray(x["ind"]["ens_" + v + "_topo"]) == 0)
                kept.append(keep.sum()), bad.append((keep & np.asarray(x["bad"], bool)).sum()), n.append(ok.sum())
            kept, bad, n = np.array(kept, float), np.array(bad, float), np.array(n, float)
            i = rng.integers(0, len(d), (N_BOOT, len(d)))                # resampling whole extrapolations
            b = bad[i].sum(1) / np.maximum(kept[i].sum(1), 1)
            out[group][label] = {"accepted": float(kept.sum() / n.sum()), "n_accepted": int(kept.sum()),
                                 "bad_accepted": float(bad.sum() / kept.sum()) if kept.sum() else None,
                                 "bad_accepted_ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    return out


def global_indicator(data):
    """Per extrapolation: fraction of the lines that a variant does not reproduce, against the optimism of the published model."""
    rows = []
    for name, d in data.items():
        for x in d:
            c = confidence(x["ind"], DEFAULT_CALIBRATION)
            ok = np.isfinite(c)
            acc = ok & (np.nan_to_num(c, nan=0.0) >= T)
            y = np.asarray(x["bad"], bool)
            row = {"field": name, "bad_rate": float(y[ok].mean()), "predicted_bad": float((1 - c[ok]).mean()),
                   "accepted": float(acc[ok].mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None}
            for v in ("coarse", "lfff"):
                sh = np.asarray(x["ind"]["ens_" + v], float)
                val = np.isfinite(sh)
                with np.errstate(invalid="ignore"):
                    moved = (sh > 0.1) | (np.asarray(x["ind"]["ens_" + v + "_topo"]) > 0)
                row["G_" + v] = float(moved[val].mean())
                row["G_" + v + "_accepted"] = float(moved[val & acc].mean()) if (val & acc).any() else None
            rows.append(row)
    opt = np.array([r["bad_accepted"] is not None and r["bad_accepted"] >= 0.10 for r in rows])
    out = {"n_cases": len(rows), "n_optimistic": int(opt.sum()), "optimistic_definition": "bad_accepted >= 0.10",
           "optimistic_by_family": {f: int(sum(o for r, o in zip(rows, opt) if r["field"] == f)) for f in data},
           "metrics": {}, "rows": rows}
    bad_rate = np.array([r["bad_rate"] for r in rows])
    excess = bad_rate - np.array([r["predicted_bad"] for r in rows])
    for key in ("G_coarse", "G_lfff", "G_coarse_accepted", "G_lfff_accepted"):
        g = np.array([np.nan if r[key] is None else r[key] for r in rows], float)
        m = np.isfinite(g)
        out["metrics"][key] = {
            "auc_optimistic": auc(g[m], opt[m].astype(int)) if opt[m].any() and not opt[m].all() else None,
            "spearman_bad_rate": float(spearmanr(g[m], bad_rate[m])[0]), "spearman_excess": float(spearmanr(g[m], excess[m])[0]),
            "median_by_family": {f: float(np.median(g[m & np.array([r["field"] == f for r in rows])])) for f in data},
            "range": [float(g[m].min()), float(g[m].max())]}
    return out


def analyze(data):
    out = {"families": sorted(data), "single_auc": {}, "leave_one_family_out": {}, "external": {}}
    for name, d in data.items():
        y = np.concatenate([x["bad"] for x in d]).astype(int)
        out["single_auc"][name] = {v: auc(np.concatenate([displacement(x["ind"], v) for x in d]), y) for v in VARIANTS}
    cal_fams = [f for f in CALIBRATION if f in data]
    if len(cal_fams) == len(CALIBRATION):
        for label, extra in SETS.items():
            ps, ys, fam = [], [], []
            for held in CALIBRATION:
                Xtr, ytr, _ = design([x for f in CALIBRATION if f != held for x in data[f]], extra)
                Xte, yte, _ = design(data[held], extra)
                ps.append(predict(_fit_logistic(Xtr, ytr), Xte))
                ys.append(yte)
                fam.append(np.full(len(yte), held))
            p, y, fam = np.concatenate(ps), np.concatenate(ys), np.concatenate(fam)
            m = metrics(p, y)
            m["by_family"] = {f: metrics(p[fam == f], y[fam == f]) for f in CALIBRATION}
            m["auc_nonlinear"] = auc(p[np.isin(fam, NONLINEAR)], y[np.isin(fam, NONLINEAR)])
            out["leave_one_family_out"][label] = m
        for label, extra in SETS.items():
            Xtr, ytr, _ = design([x for f in CALIBRATION for x in data[f]], extra)
            cal = _fit_logistic(Xtr, ytr)
            out["external"][label] = {"coef": cal["coef"]}
            for name in EXTERNAL:
                if name in data:
                    Xte, yte, _ = design(data[name], extra)
                    out["external"][label][name] = metrics(predict(cal, Xte), yte)
    out["two_key"] = two_key(data)
    out["global"] = global_indicator(data)
    return out


if not a.analyze_only:
    with keep_awake():
        for name in (a.fields or ORDER):
            pk = os.path.join(a.out, f"ensemble_{name}.pkl")
            if os.path.exists(pk):
                continue
            d = calibration_suite(fields=[name], sizes=tuple(a.sizes), realizations=1, ensemble=VARIANTS,
                                  checkpoint=os.path.join(a.out, f"ensemble_{name}.ckpt"))
            pickle.dump(d, open(pk, "wb"))
            y = np.concatenate([x["bad"] for x in d]).astype(int)
            print(f"{name:11s} wrong {y.mean():.0%} | AUC of the displacement alone: " +
                  ", ".join(f"{v} {auc(np.concatenate([displacement(x['ind'], v) for x in d]), y):.2f}" for v in VARIANTS), flush=True)

data = {}
for name in ORDER:
    pk = os.path.join(a.out, f"ensemble_{name}.pkl")
    if os.path.exists(pk):
        data[name] = pickle.load(open(pk, "rb"))
out = {"code_version": code_version()["label"], "variants": list(VARIANTS), "sizes": list(a.sizes), "realizations": 1, **analyze(data)}
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for label, m in out["leave_one_family_out"].items():
    print(f"leave-one-family-out, {label:14s} AUC {m['auc']:.3f} (non-linear families {m['auc_nonlinear']:.3f}) | accepted {m['accepted']:.2f}, "
          f"wrong among them {m['bad_accepted']:.3f}")
for label, r in out["external"].items():
    print(f"external, {label:14s} " + " | ".join(f"{n} AUC {v['auc']:.2f} acc {v['accepted']:.2f} wrong {v['bad_accepted'] if v['bad_accepted'] is None else round(v['bad_accepted'], 3)}"
                                                for n, v in r.items() if n != "coef"))
for group, r in out["two_key"].items():
    print(f"two keys, {group:11s} " + " | ".join(
        f"{label}: acc {v['accepted']:.2f} wrong {'-' if v['bad_accepted'] is None else format(v['bad_accepted'], '.3f')}"
        for label, v in r.items() if label != "n_cases"))
g = out["global"]
print(f"global indicator: {g['n_optimistic']} optimistic cases of {g['n_cases']}")
for key, m in g["metrics"].items():
    print(f"  {key:18s} AUC for the optimistic cases {'-' if m['auc_optimistic'] is None else format(m['auc_optimistic'], '.3f')} | "
          f"Spearman with the wrong fraction {m['spearman_bad_rate']:+.2f}, with the excess {m['spearman_excess']:+.2f} | medians: " +
          ", ".join(f"{f} {v:.2f}" for f, v in m["median_by_family"].items()))
print("written", a.json)
