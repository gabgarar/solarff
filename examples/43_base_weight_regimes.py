"""How much should the extrapolation trust its base? A test on snapshots and noise realisations never used before.

The weight of the data term of the relaxed base (``relax_bottom``) says how closely the base of the model follows
the magnetogram. An exploration on snapshots already used (19-Nov-2026, plan of the paper) suggested that:

- the weight that gives the fewest wrong lines changes by orders of magnitude from one case to another: high when the
  base is compatible with a force-free field and the noise is low, low when the base is a photosphere with forces or
  the noise is high;
- the misfit of the base in one reference run says roughly which weight to use;
- with the true field on the side and top faces the wrong lines fall, little at a photosphere and a lot above it;
- the lines of a model made with the configuration of ``examples/40_base_treatment.py`` can be ranked.

This script tests those four statements once, on data never opened before: three snapshots of the simulated active
region (280000, 300000 and 296000) and a new realisation of the analytical families. The criteria were written down
in the plan of the paper before running it.

Per case, with the same lines in every configuration:

- ``default``: the preprocessing and the weight of the library (``mu3`` = ``mu4`` = 0.01, weight 0.1);
- ``recommended``: the configuration of ``examples/40`` (``mu3`` 0.1, ``mu4`` 0.01, weight 0.01);
- a scan of the weight without preprocessing, from 1 to 0.001, and the weight given by the rule
  ``kappa / misfit`` with the misfit of the run with weight 0.1;
- at the photosphere with the larger pixel, the true side and top faces instead of those of the potential field
  (a diagnostic, not a method: in an observation those faces are not known).

Needs the raw cubes of the three snapshots (``--raw``), and ``data/base_scan.pkl`` and ``data/base_test.pkl`` of
``examples/40`` to fit the ranking. Saves ``data/base_weight.pkl`` (checkpoint) and
``docs/resultados/base_weight_regimes.json``.

    python -u examples/43_base_weight_regimes.py --raw C:/Users/.../solarff-data/muram_ar098192      # about two hours
    python examples/43_base_weight_regimes.py --analyze-only
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.analytic import CALIBRATION_FIELDS, EXTERNAL_TEST_FIELDS, field_magnetogram
from solarff.calibration import NOISE_MODELS, SIX_FAMILIES, _nz
from solarff.confidence import CALIBRATION_MHD, _design, _fit_logistic, auc, calibration_case, line_errors
from solarff.field import VectorField3D
from solarff.nlfff_torch import nlfff_lbfgs
from solarff.potential import potential_field
from solarff.simulations import SIMULATIONS, reduced_field
from solarff.tracing import seeds_from_magnetogram, trace_field_lines
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=os.path.join("data", "muram_ar098192"), help="folder with the raw cubes of the simulation")
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "base_weight_regimes.json"))
ap.add_argument("--analyze-only", action="store_true")
ap.add_argument("--smoke", action="store_true", help="check of the script on a snapshot already used (290000) and on the seeds of examples/36; its files have the suffix _smoke")
a = ap.parse_args()

SIM = "ar098192"
USED, FRESH = [290000, 310000, 270000], [280000, 300000, 296000]                # the snapshots of examples/35 to 42, and the ones never opened
NOISES = list(NOISE_MODELS.items())
EXACT = ["LL-I", "LL-II", "LL-III", "LFFF-bip", "LFFF-quad", "LL-IV", "LFFF-delta"]     # exact solutions
ROPES = ["TD-relax", "TD2-relax", "TD3-relax"]                                  # ropes relaxed with this code and a fixed base
FAMILIES = list(SIX_FAMILIES) + ["TD2-relax", "LL-IV", "LFFF-delta", "TD3-relax"]       # the order of examples/36, for the seeds
KNOWN = {**CALIBRATION_FIELDS, **EXTERNAL_TEST_FIELDS}
SEED_ANALYTIC = 900001                                                          # 500009 and 700001 were used in examples/36, 41 and 42
WEIGHTS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
REFERENCE, KAPPA, LIMITS = 0.1, 10 ** -5.4, (1e-3, 1.0)                        # the rule: weight = KAPPA / misfit of the run with the weight REFERENCE
DEFAULT, RECOMMENDED = (1e-2, 1e-2, 1e-1), (1e-1, 1e-2, 1e-2)                  # mu3, mu4, relax_bottom
if a.smoke:                                                                     # nothing new is opened: the figures must agree with those of the exploration
    FRESH, SEED_ANALYTIC, EXACT, ROPES = [290000], 500009, ["LL-I", "LFFF-quad"], ["TD-relax"]
    a.json = a.json.replace(".json", "_smoke.json")
CKPT = os.path.join(a.data, "base_weight_smoke.pkl" if a.smoke else "base_weight.pkl")
state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}


def truth(snap, r, z0):
    """Reduced truth of a snapshot with the base at the photosphere (z0 = 0) or at a height, as examples/35 and 39 write it."""
    path = os.path.join(a.data, f"truth_{SIM}_{snap}_r{r}{'' if z0 == 0 else f'_z{z0:g}'}.npz")
    if not os.path.exists(path):
        f = reduced_field(SIM, snap, r, a.raw, z0=z0)
        np.savez_compressed(path, bx=f.bx.astype(np.float32), by=f.by.astype(np.float32), bz=f.bz.astype(np.float32), dx=f.dx, origin=np.array(f.origin))
    d = np.load(path)
    return VectorField3D(d["bx"].astype(float), d["by"].astype(float), d["bz"].astype(float), float(d["dx"]), (float(d["origin"][0]), float(d["origin"][1]), 0.0))


def cases():
    """(key, function that gives the truth, seed of the noise, maximum of iterations) of every case of the test."""
    for r, js in ((8, (0, 1, 2)), (4, (2,))):                                   # with the smaller pixel, only the noise like that of HMI
        for z0 in (0.0, 1.5):
            for snap in FRESH:
                for j in js:
                    k = 7919 * ((USED + FRESH).index(snap) + 1) + 100 * r + j + 100000 * list(SIMULATIONS).index(SIM)
                    yield ("sim", snap, r, z0, j), (lambda snap=snap, r=r, z0=z0: truth(snap, r, z0)), k, 4000
    for fam in EXACT + ROPES:
        for j in range(len(NOISES)):
            yield ("analytic", fam, 32, 0.0, j), (lambda fam=fam: KNOWN[fam](32, 32, _nz(32))), SEED_ANALYTIC + 1000 * FAMILIES.index(fam) + 10 * 32 + j, 2000


def line_record(model, exact, seeds, le):
    e = line_errors(trace_field_lines(model, seeds), le)
    return {"shape": e["shape"], "topology": e["topology"], "L_force": float(model.meta["L_force"]), "L_data": float(model.meta["L_data"]),
            "iterations": int(model.meta["iterations"]), "converged": str(model.meta["message"]).startswith("convergido")}


def with_true_faces(mag, exact):
    """Potential field of the magnetogram with the side and top faces of the truth: only for the diagnostic of the box."""
    f = potential_field(mag, exact.shape[2])
    for c in ("bx", "by", "bz"):
        u, v = getattr(f, c), getattr(exact, c)
        u[0], u[-1], u[:, 0], u[:, -1], u[:, :, -1] = v[0], v[-1], v[:, 0], v[:, -1], v[:, :, -1]
    return f


def rule(rec):
    """The weight of the rule from the record of the run with the reference weight."""
    misfit = rec["L_data"] / REFERENCE
    return float(np.clip(KAPPA / misfit, *LIMITS)) if misfit > 0 else LIMITS[1]


if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        for key, get, k, mi in cases():
            kind, name, r, z0, j = key
            nm, (sz, st) = NOISES[j]
            photosphere = kind == "sim" and z0 == 0
            todo = [("scan", w) for w in WEIGHTS] + [("rule",)]
            if photosphere:
                todo += [("default",), ("recommended",)] + ([("fixed",), ("faces", "fixed"), ("faces", 0.01)] if r == 8 else [])
            if all((key, c) in state for c in todo):
                continue
            exact = get()
            nz = exact.shape[2]
            mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=k)
            seeds = seeds_from_magnetogram(mag, n=300, method="random", polarity="both", threshold=0.05, rng=k + 7)     # the seeds of calibration_case
            le = trace_field_lines(exact, seeds)
            for c in todo:
                if (key, c) in state:
                    continue
                tc = time.time()
                if c[0] in ("default", "recommended"):
                    mu3, mu4, w = DEFAULT if c[0] == "default" else RECOMMENDED
                    rec = calibration_case(exact, mag, nz, n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"max_iter": mi, "relax_bottom": w}, preprocess_kw={"mu3": mu3, "mu4": mu4},
                                           potential=True)
                    rec = {"shape": rec["err"]["shape"], "topology": rec["err"]["topology"], "ind": rec["ind"], "length_ref": rec["err"]["length_ref"]}
                    assert np.allclose(rec["length_ref"], line_errors(le, le)["length_ref"], rtol=1e-6, equal_nan=True), "the lines of calibration_case are not those of this script"
                elif c[0] == "scan":
                    rec = line_record(nlfff_lbfgs(mag, nz, verbose=False, max_iter=mi, relax_bottom=c[1]), exact, seeds, le)
                elif c[0] == "fixed":
                    rec = line_record(nlfff_lbfgs(mag, nz, verbose=False, max_iter=mi, relax_bottom=None), exact, seeds, le)
                elif c[0] == "rule":
                    w = rule(state[(key, ("scan", REFERENCE))])
                    rec = {**line_record(nlfff_lbfgs(mag, nz, verbose=False, max_iter=mi, relax_bottom=w), exact, seeds, le), "weight": w}
                else:
                    rec = line_record(nlfff_lbfgs(mag, nz, verbose=False, max_iter=mi, relax_bottom=None if c[1] == "fixed" else c[1], initial=with_true_faces(mag, exact)), exact, seeds, le)
                rec.update(noise_model=nm, seconds=time.time() - tc)
                state[(key, c)] = rec
                pickle.dump(state, open(CKPT, "wb"))
                bad = (np.asarray(rec["shape"], float) > 0.10) | (np.asarray(rec["topology"]) > 0)
                print(f"  {' '.join(str(q) for q in key[:4])} | {nm} | {' '.join(str(q) for q in c)}{' ' + format(rec['weight'], '.4f') if 'weight' in rec else ''}: {bad.mean():.1%} wrong lines "
                      f"({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)", flush=True)

# ------------------------------------------------------------------ analysis
bad = lambda rec, thr=0.10: (np.asarray(rec["shape"], float) > thr) | (np.asarray(rec["topology"]) > 0)  # noqa: E731
keys = sorted({k for k, _ in state}, key=str)


def rate(ks, c, thr=0.10):
    """Wrong lines of a configuration pooled over cases; None if a case is missing."""
    if not ks or any((k, c) not in state for k in ks):
        return None
    return float(np.concatenate([bad(state[(k, c)], thr) for k in ks]).mean())


def diff(ks, c_old, c_new, thr=0.10, n_boot=2000):
    """Change of the wrong lines from one configuration to another (the same lines), in points, with an interval resampling whole cases."""
    if not ks or any((k, c) not in state for k in ks for c in (c_old, c_new)):
        return None
    n = np.array([len(bad(state[(k, c_old)])) for k in ks], float)
    o, w = np.array([bad(state[(k, c_old)], thr).sum() for k in ks], float), np.array([bad(state[(k, c_new)], thr).sum() for k in ks], float)
    idx = np.random.default_rng(0).integers(0, len(ks), (n_boot, len(ks)))
    boot = 100 * (w[idx].sum(1) - o[idx].sum(1)) / n[idx].sum(1)
    return {"old": float(o.sum() / n.sum()), "new": float(w.sum() / n.sum()), "points": float(100 * (w.sum() - o.sum()) / n.sum()), "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "cases_that_fall": int((w < o).sum()), "cases_that_rise": int((w > o).sum()), "n_cases": len(ks), "n_lines": int(n.sum())}


GROUPS = {"photosphere, 1.54 Mm": lambda k: k[0] == "sim" and k[2] == 8 and k[3] == 0, "photosphere, 0.77 Mm": lambda k: k[0] == "sim" and k[2] == 4 and k[3] == 0,
          "base at 1.5 Mm, 1.54 Mm": lambda k: k[0] == "sim" and k[2] == 8 and k[3] > 0, "base at 1.5 Mm, 0.77 Mm": lambda k: k[0] == "sim" and k[2] == 4 and k[3] > 0}
for j, (nm, _) in enumerate(NOISES):
    GROUPS[f"exact families, {nm}"] = lambda k, j=j: k[0] == "analytic" and k[1] in EXACT and k[4] == j
    GROUPS[f"relaxed ropes, {nm}"] = lambda k, j=j: k[0] == "analytic" and k[1] in ROPES and k[4] == j
RULE_GROUPS = [g for g in GROUPS if not g.startswith("relaxed ropes")]          # the ropes are reported apart: see the docstring of the plan

out = {"code_version": code_version()["label"], "snapshots": FRESH, "weights": WEIGHTS, "rule": {"reference": REFERENCE, "kappa": KAPPA, "limits": list(LIMITS)}, "groups": {}, "criteria": {}}
for g, f in GROUPS.items():
    ks = [k for k in keys if f(k)]
    scan = {str(w): rate(ks, ("scan", w)) for w in WEIGHTS}
    if not ks or any(v is None for v in scan.values()) or rate(ks, ("rule",)) is None:
        continue
    best_case = float(np.mean([min(bad(state[(k, ("scan", w))]).mean() for w in WEIGHTS) for k in ks]))
    case_rate = lambda c: float(np.mean([bad(state[(k, c)]).mean() for k in ks]))  # noqa: E731
    out["groups"][g] = {"n_cases": len(ks), "scan": scan, "scan_20": {str(w): rate(ks, ("scan", w), 0.20) for w in WEIGHTS}, "rule": rate(ks, ("rule",)), "rule_20": rate(ks, ("rule",), 0.20),
                        "rule_weights": [state[(k, ("rule",))]["weight"] for k in ks], "best_weight_of_each_case": best_case,
                        "regret_points": {**{str(w): 100 * (case_rate(("scan", w)) - best_case) for w in WEIGHTS}, "rule": 100 * (case_rate(("rule",)) - best_case)},
                        "default": rate(ks, ("default",)), "recommended": rate(ks, ("recommended",)), "recommended_against_default": diff(ks, ("default",), ("recommended",)),
                        "recommended_against_default_20": diff(ks, ("default",), ("recommended",), 0.20), "tight_against_loose": diff(ks, ("scan", 1.0), ("scan", 0.01)),
                        "faces": {"loose": diff(ks, ("scan", 0.01), ("faces", 0.01)), "fixed": diff(ks, ("fixed",), ("faces", "fixed"))}}
G = out["groups"]


# the ranking of the lines: fitted on the snapshots already used, with the recommended configuration, and tested on the new ones
def score(cal, ind):
    X = _design(ind, cal["features"])
    return ((X - np.array(cal["mu"])) / np.array(cal["sd"])) @ np.array(cal["coef"]) + cal["intercept"]


def as_library(ind):
    """The indicators of a record with ``departure`` as ``solarff.confidence.potential_departure`` gives it in a real region:
    0.5 at least when the line changes between open and closed with respect to the potential field."""
    d = np.asarray(ind["departure"], float)
    return {**ind, "departure": np.where(np.asarray(ind["departure_topo"]) > 0, np.maximum(np.nan_to_num(d, nan=0.5), 0.5), d)}


def ranking_test(convert):
    """Fit on the 15 old cases and test on the new ones; ``convert`` prepares the indicators of a record."""
    feats = CALIBRATION_MHD["features"]
    Xo = np.vstack([_design(convert(x["ind"]), feats) for x in old])
    yo = np.concatenate([(np.asarray(x["err"]["shape"], float) > 0.10) | (np.asarray(x["err"]["topology"]) > 0) for x in old]).astype(float)
    ok = np.isfinite(Xo).all(1)
    cal = {**_fit_logistic(Xo[ok], yo[ok]), "features": feats}
    res = {"fit": {"n_cases": len(old), "n_lines": int(ok.sum()), "terms": feats, "definition": 0.10}, "calibration": cal, "test": {}}
    for g in ("photosphere, 1.54 Mm", "photosphere, 0.77 Mm"):
        ks = [k for k in keys if GROUPS[g](k) and (k, ("recommended",)) in state]
        if not ks:
            continue
        sc = {k: score(cal, convert(state[(k, ("recommended",))]["ind"])) for k in ks}
        allsc = np.concatenate([sc[k] for k in ks])
        fin = np.isfinite(allsc)
        third = np.argsort(allsc[fin])[: fin.sum() // 3]
        r = {"n_cases": len(ks), "n_lines": int(len(allsc))}
        for thr in (0.10, 0.20):
            y = np.concatenate([bad(state[(k, ("recommended",))], thr) for k in ks])
            r[f"{thr:.2f}"] = {"auc": auc(allsc, y), "wrong": float(y.mean()), "wrong_best_third": float(y[fin][third].mean()),
                               "auc_by_snapshot": {str(sn): auc(np.concatenate([sc[k] for k in ks if k[1] == sn]), np.concatenate([bad(state[(k, ("recommended",))], thr) for k in ks if k[1] == sn]))
                                                   for sn in FRESH}}
        res["test"][g] = r
    return res


scan40, test40 = (pickle.load(open(os.path.join(a.data, f), "rb")) for f in ("base_scan.pkl", "base_test.pkl"))
old = [scan40[RECOMMENDED + (j,)] for j in range(len(NOISES))] + [rec for (sim, snap, r, j), rec in test40["cases"].items() if sim == SIM]
assert test40["combo"] == RECOMMENDED and len(old) == 15, "the records of examples/40 are not those of the recommended configuration"
if a.smoke:                                                                     # the snapshot of the check is one of the fit: leave it out of it
    old = [rec for (sim, snap, r, j), rec in test40["cases"].items() if sim == SIM]
out["ranking"] = ranking_test(as_library)                                       # the one of the criterion: usable with solarff.confidence.rank_lines(calibration=...)
out["ranking_with_the_stored_indicator"] = ranking_test(lambda ind: ind)        # as the script was first run (see the plan of the paper)

# the criteria, as written down in the plan
p8, p4, c8, c4 = (G.get(g) for g in ("photosphere, 1.54 Mm", "photosphere, 0.77 Mm", "base at 1.5 Mm, 1.54 Mm", "base at 1.5 Mm, 0.77 Mm"))
e10, e30, ehmi = (G.get(f"exact families, {nm}") for nm, _ in NOISES)
C = out["criteria"]
if p8 and p4 and p8["recommended_against_default"] and p4["recommended_against_default"]:
    C["F1"] = bool(p8["recommended_against_default"]["points"] <= -10 and p8["recommended_against_default"]["cases_that_fall"] == p8["n_cases"] and p4["recommended_against_default"]["points"] <= -10)
if p8 and p4 and c8 and c4:
    C["F2a"] = bool(p8["tight_against_loose"]["points"] <= -10 and p4["tight_against_loose"]["points"] <= -10)
    C["F2b"] = bool(c8["tight_against_loose"]["points"] >= 5 and c4["tight_against_loose"]["points"] >= 5)
if e10 and ehmi:
    C["F2c"] = bool(e10["tight_against_loose"]["points"] >= 3 and ehmi["tight_against_loose"]["points"] <= -5)
if all(g in G for g in RULE_GROUPS):
    mean_regret = {c: float(np.mean([G[g]["regret_points"][c] for g in RULE_GROUPS])) for c in [str(w) for w in WEIGHTS] + ["rule"]}
    out["rule"]["mean_regret_points"] = mean_regret
    C["F3"] = bool(all(mean_regret["rule"] < mean_regret[str(w)] for w in WEIGHTS))
if p8 and p8["faces"]["loose"] and p8["faces"]["fixed"]:
    C["F4"] = bool(p8["faces"]["loose"]["points"] <= -5 and abs(p8["faces"]["fixed"]["points"]) < 5)
rk = out["ranking"]["test"]
if len(rk) == 2:
    C["F5"] = bool(all(v["0.20"]["auc"] >= 0.70 and v["0.20"]["wrong_best_third"] <= 0.5 * v["0.20"]["wrong"] for v in rk.values()))
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

fmt = lambda v: "   -  " if v is None else f"{100 * v:5.1f}%"  # noqa: E731
print("wrong lines at the 10 % definition")
print(f"{'group':32s} cases | " + " ".join(f"{w:>6g}" for w in WEIGHTS) + " |   rule (weights) | best of each case | default | recommended")
for g, v in G.items():
    ws = v["rule_weights"]
    print(f"{g:32s} {v['n_cases']:5d} | " + " ".join(fmt(v["scan"][str(w)]) for w in WEIGHTS) + f" | {fmt(v['rule'])} ({min(ws):.3f} to {max(ws):.3f}) | {fmt(v['best_weight_of_each_case'])}            | "
          f"{fmt(v['default'])}  | {fmt(v['recommended'])}")
if "mean_regret_points" in out["rule"]:
    print("mean regret over the groups without ropes, in points: " + ", ".join(f"{c} {v:.1f}" for c, v in out["rule"]["mean_regret_points"].items()))
for g in ("photosphere, 1.54 Mm", "photosphere, 0.77 Mm"):
    v = G.get(g)
    if v and v["recommended_against_default"]:
        d = v["recommended_against_default"]
        print(f"{g}: recommended against default {d['old']:.3f} -> {d['new']:.3f}, {d['points']:+.1f} points ({d['ci'][0]:+.1f} to {d['ci'][1]:+.1f}), falls in {d['cases_that_fall']} of {d['n_cases']}")
    if v and v["faces"]["loose"]:
        d = v["faces"]["loose"]
        e = v["faces"]["fixed"]
        print(f"{g}: true side and top faces, weight 0.01: {d['old']:.3f} -> {d['new']:.3f}, {d['points']:+.1f} points; fixed base: {e['old']:.3f} -> {e['new']:.3f}, {e['points']:+.1f} points")
for g, v in rk.items():
    print(f"ranking, {g}: " + "; ".join(f"{thr} definition: AUC {v[thr]['auc']:.3f}, wrong {v[thr]['wrong']:.1%}, in the best third {v[thr]['wrong_best_third']:.1%}" for thr in ("0.10", "0.20")))
print("criteria:", C)
print("written", a.json)
