"""Feasibility of a confidence against a realistic truth: do new indicators tell the wrong lines apart?

Against the MURaM simulation (``examples/33_mhd_truth.py``) the published confidence does not rank the lines, and the
indicators it uses do not when they are refitted on the simulation itself. This script repeats the 18 cases of that
test and adds indicators that probe what is not known, not the noise:

- the displacement of each line between the nominal extrapolation and three repetitions: at half the resolution
  (``coarse``), with the side and top faces taken from the linear force-free field of best-fit alpha (``lfff``) and
  with three times the smoothing of the preprocessing (``smooth``);
- the displacement between the NLFFF line and the potential-field line traced from the same seed.

Logistic models with these indicators are refitted on the simulation and evaluated out of sample in two ways: leaving
one snapshot out, and training on the lines rooted in one half of the box and testing on the other half. A rule with
no fitted parameter is evaluated too: a line is accepted if the published confidence is at least 0.8 and the
repetitions reproduce it (it moves by at most 10 % of its length and keeps its topology).

The candidates, the models and the cut (V1: the best of models b and c reaches an AUC of 0.75 with both validations)
were written down in the plan of the paper before running (18-Nov-2026).

Needs the reduced truths and the checkpoint of ``examples/33_mhd_truth.py``. Saves ``data/mhd_indicators.pkl``
(checkpoint, with the per-line records) and ``docs/resultados/mhd_indicators.json``.

    python -u examples/34_mhd_indicators.py            # about 2 h
    python examples/34_mhd_indicators.py --analyze-only
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.analytic import field_magnetogram
from solarff.calibration import NOISE_MODELS
from solarff.confidence import ALL_FEATURES, DEFAULT_CALIBRATION, _design, _fit_logistic, auc, calibration_case, confidence, line_errors
from solarff.field import VectorField3D
from solarff.potential import potential_field
from solarff.tracing import seeds_from_magnetogram, trace_field_lines
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "mhd_indicators.json"))
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

SNAPSHOTS, RS = [290000, 310000, 270000], [8, 4]        # the order of examples/33: the seeds depend on it
VARIANTS = ("coarse", "lfff", "smooth")
T = DEFAULT_CALIBRATION["threshold"]
F5 = list(DEFAULT_CALIBRATION["features"])
OLD = list(ALL_FEATURES) + [f for f in F5 if f not in ALL_FEATURES]
CKPT = os.path.join(a.data, "mhd_indicators.pkl")
REF = pickle.load(open(os.path.join(a.data, "mhd_truth.pkl"), "rb"))["cases"]      # the registered test: same cases, same labels
N_BOOT = 1000
CUT = 0.75


def truth(snap, r):
    d = np.load(os.path.join(a.data, f"mhd_truth_{snap}_r{r}.npz"))
    return VectorField3D(d["bx"].astype(float), d["by"].astype(float), d["bz"].astype(float), float(d["dx"]), tuple(d["origin"]),
                         {"method": f"MURaM ar098192, snapshot {snap}, r = {r}"})


# ------------------------------------------------------------------ the cases, with the repetitions
state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {"cases": {}}
if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        for r in RS:                                    # the coarse pixel first: it is fast and gives a first look
            for si, snap in enumerate(SNAPSHOTS):
                tr = None
                for j, (nm, (sz, st)) in enumerate(NOISE_MODELS.items()):
                    key = f"{snap} | r{r} | {nm}"
                    if key in state["cases"]:
                        continue
                    tr = truth(snap, r) if tr is None else tr
                    nz = tr.shape[2]
                    tc = time.time()
                    k = 7919 * (si + 1) + 100 * r + j
                    mag = field_magnetogram(tr, noise=sz, noise_t=st, rng=k)
                    rec = calibration_case(tr, mag, nz, n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"max_iter": 4000}, keep_model=True,
                                           ensemble=VARIANTS)
                    model, pp = rec.pop("model"), rec.pop("pp")
                    assert np.array_equal(np.asarray(rec["bad"]), np.asarray(REF[key]["bad"])), f"{key}: the labels are not those of the registered test"
                    seeds = seeds_from_magnetogram(mag, n=300, method="random", polarity="both", threshold=0.05, rng=k + 7)
                    ep = line_errors(trace_field_lines(model, seeds), trace_field_lines(potential_field(pp, nz), seeds))
                    rec["ind"]["ens_pot"], rec["ind"]["ens_pot_topo"] = ep["shape"], ep["topology"]
                    rec.update(snapshot=int(snap), r=int(r), noise_model=nm, seed_x=np.asarray(seeds)[:, 0], x_mid=float(tr.x.mean()),
                               seconds=time.time() - tc)
                    state["cases"][key] = rec
                    pickle.dump(state, open(CKPT, "wb"))
                    print(f"  {key}: {rec['bad'].mean():.0%} wrong lines; not reproduced by " +
                          ", ".join(f"{v} {np.nanmean((np.asarray(rec['ind']['ens_' + v], float) > 0.1) | (np.asarray(rec['ind']['ens_' + v + '_topo']) > 0)):.0%}"
                                    for v in VARIANTS) + f" ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)", flush=True)


# ------------------------------------------------------------------ analysis
def displacement(ind, v):
    """Displacement under the variant ``v``; a line that changes between open and closed counts as moved by at least half its length."""
    s = np.asarray(ind["ens_" + v], float)
    return np.where(np.asarray(ind["ens_" + v + "_topo"]) > 0, np.maximum(np.nan_to_num(s, nan=0.5), 0.5), s)


def logd(x):
    return np.log10(np.maximum(x, 1e-3))


cases = list(state["cases"].values())
out = {"code_version": code_version()["label"], "note": "feasibility study registered in the plan of the paper on 18-Nov-2026; the cases are those of mhd_truth.json",
       "variants": list(VARIANTS), "n_cases": len(cases), "cut_auc": CUT}
if cases:
    cat = lambda f: np.concatenate([np.asarray(f(x), float) for x in cases])  # noqa: E731
    sh, tp = cat(lambda x: x["err"]["shape"]), cat(lambda x: x["err"]["topology"]) > 0
    LAB = {"rel10": ((sh > 0.10) | tp).astype(float), "rel20": ((sh > 0.20) | tp).astype(float)}
    snap = np.concatenate([np.full(len(x["bad"]), x["snapshot"]) for x in cases])
    rr = np.concatenate([np.full(len(x["bad"]), x["r"]) for x in cases])
    half = np.concatenate([np.asarray(x["seed_x"], float) >= x["x_mid"] for x in cases])
    case_id = np.concatenate([np.full(len(x["bad"]), i) for i, x in enumerate(cases)])
    Xold = np.vstack([_design(x["ind"], OLD) for x in cases])
    D = {v: cat(lambda x, v=v: displacement(x["ind"], v)) for v in VARIANTS + ("pot",)}
    NEW = {"ens_" + v: logd(D[v]) for v in VARIANTS} | {"ens_max": logd(np.nanmax(np.stack([D[v] for v in VARIANTS]), axis=0)), "ens_pot": logd(D["pot"])}
    col = lambda names: np.column_stack([Xold[:, OLD.index(n)] if n in OLD else NEW[n] for n in names])  # noqa: E731
    MODELS = {"a_five_terms": F5, "b_five_plus_max": F5 + ["ens_max"], "c_five_plus_four": F5 + ["ens_coarse", "ens_lfff", "ens_smooth", "ens_pot"],
              "d_max_alone": ["ens_max"], "e_everything": OLD + ["ens_coarse", "ens_lfff", "ens_smooth", "ens_pot"]}
    FOLDS = {"snapshot": [snap == s for s in np.unique(snap)], "space": [half, ~half]}

    def predict(fit, X):
        return 1 / (1 + np.exp(-((X - np.array(fit["mu"])) / np.array(fit["sd"]) @ np.array(fit["coef"]) + fit["intercept"])))

    def risk(p, y, cov):
        k = max(1, int(round(cov * len(p))))
        return float(y[np.argsort(p)[:k]].mean())

    out["single_indicator_auc"] = {n: {"all": auc(v[np.isfinite(v)], LAB["rel10"][np.isfinite(v)] > 0),
                                       **{f"r{r}": auc(v[np.isfinite(v) & (rr == r)], LAB["rel10"][np.isfinite(v) & (rr == r)] > 0) for r in RS}}
                                   for n, v in NEW.items()}
    out["not_reproduced"] = {v: float(np.nanmean(D[v] > 0.1)) for v in D}
    out["models"] = {}
    for name, feats in MODELS.items():
        X = col(feats)
        ok = np.isfinite(X).all(axis=1)
        m = {"features": feats, "n_lines": int(ok.sum())}
        for lab, y in LAB.items():
            m[lab] = {}
            for val, folds in FOLDS.items():
                p = np.full(len(y), np.nan)
                for te in folds:
                    fit = _fit_logistic(X[ok & ~te], y[ok & ~te])
                    p[ok & te] = predict(fit, X[ok & te])
                m[lab][val] = {"auc": auc(p[ok], y[ok] > 0), **{f"auc_r{r}": auc(p[ok & (rr == r)], y[ok & (rr == r)] > 0) for r in RS},
                               "bad_rate": float(y[ok].mean()), **{f"bad_best_{int(100 * c)}": risk(p[ok], y[ok], c) for c in (0.2, 0.36, 0.5)}}
        m["coef_all_data"] = dict(zip(feats, _fit_logistic(X[ok], LAB["rel10"][ok])["coef"]))
        out["models"][name] = m
    best = {val: max(out["models"][n]["rel10"][val]["auc"] for n in ("b_five_plus_max", "c_five_plus_four")) for val in FOLDS}
    out["cut_V1"] = {"best_of_b_and_c": best, "met": bool(all(v >= CUT for v in best.values()))}

    # the rule of two keys, with no fitted parameter: the published confidence and the repetitions
    c = cat(lambda x: confidence(x["ind"], DEFAULT_CALIBRATION))
    okc = np.isfinite(c)
    key1 = okc & (np.nan_to_num(c, nan=0.0) >= T)
    rep = {v: (np.nan_to_num(D[v], nan=1.0) <= 0.1) for v in VARIANTS}
    rules = {"published": key1, **{f"published and {v}": key1 & rep[v] for v in VARIANTS},
             "published and any repetition": key1 & np.any(np.stack(list(rep.values())), axis=0),
             "published and all the repetitions": key1 & np.all(np.stack(list(rep.values())), axis=0),
             "all the repetitions alone": okc & np.all(np.stack(list(rep.values())), axis=0)}
    rng = np.random.default_rng(0)
    out["two_keys"] = {}
    for name, keep in rules.items():
        q = {}
        for lab, y in LAB.items():
            kept = np.array([keep[case_id == i].sum() for i in range(len(cases))], float)
            bad = np.array([(keep & (y > 0))[case_id == i].sum() for i in range(len(cases))], float)
            i = rng.integers(0, len(cases), (N_BOOT, len(cases)))            # whole extrapolations are resampled
            b = bad[i].sum(1) / np.maximum(kept[i].sum(1), 1)
            q[lab] = {"bad_accepted": float(bad.sum() / kept.sum()) if kept.sum() else None,
                      "bad_accepted_ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "bad_rate": float(y[okc].mean())}
        q["accepted"] = float(keep.sum() / okc.sum())
        q["accepted_by_r"] = {str(r): float(keep[rr == r].sum() / (okc & (rr == r)).sum()) for r in RS}
        q["bad_accepted_by_r"] = {str(r): (float(LAB["rel10"][keep & (rr == r)].mean()) if (keep & (rr == r)).any() else None) for r in RS}
        out["two_keys"][name] = q
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

if cases:
    print(f"{len(cases)} cases, {int(okc.sum())} lines; wrong {LAB['rel10'][okc].mean():.2f} (10 %), {LAB['rel20'][okc].mean():.2f} (20 %)")
    print("lines not reproduced within 10 %:", ", ".join(f"{v} {q:.2f}" for v, q in out["not_reproduced"].items()))
    print("each new indicator alone, AUC (all, r = 8, r = 4):", "; ".join(f"{n} {q['all']:.3f}, {q['r8']:.3f}, {q['r4']:.3f}" for n, q in out["single_indicator_auc"].items()))
    for name, m in out["models"].items():
        for lab in LAB:
            print(f"  {name:18s} {lab}: " + " | ".join(
                f"{val}: AUC {q['auc']:.3f} (r8 {q['auc_r8']:.3f}, r4 {q['auc_r4']:.3f}), wrong in the best 20/36/50 %: "
                f"{q['bad_best_20']:.2f}/{q['bad_best_36']:.2f}/{q['bad_best_50']:.2f} of {q['bad_rate']:.2f}" for val, q in m[lab].items()))
    print("coefficients of model c with all the data:", {k: round(v, 2) for k, v in out["models"]["c_five_plus_four"]["coef_all_data"].items()})
    print("rule of two keys (accepted; wrong among them with the 10 % and the 20 % definitions):")
    for name, q in out["two_keys"].items():
        print(f"  {name:34s} accepted {q['accepted']:.2f} | {q['rel10']['bad_accepted'] if q['rel10']['bad_accepted'] is not None else float('nan'):.3f} "
              f"{[round(v, 3) for v in q['rel10']['bad_accepted_ci']]} | {q['rel20']['bad_accepted'] if q['rel20']['bad_accepted'] is not None else float('nan'):.3f}"
              f" | by pixel (10 %): " + ", ".join(f"r{r} {q['accepted_by_r'][r]:.2f} accepted, {q['bad_accepted_by_r'][r] if q['bad_accepted_by_r'][r] is not None else float('nan'):.3f}" for r in q["accepted_by_r"]))
    v1 = out["cut_V1"]
    print(f"cut V1 (AUC of at least {CUT} with both validations, best of b and c): " + ", ".join(f"{k} {v:.3f}" for k, v in v1["best_of_b_and_c"].items()) +
          (" -> MET" if v1["met"] else " -> NOT MET"))
print("written", a.json)
