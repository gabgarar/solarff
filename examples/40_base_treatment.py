"""Treatment of the base against a realistic truth: how much of the photospheric gap do the existing parameters close?

With the base at the photosphere of a radiative MHD simulation of an active region 62 % of the NLFFF lines are wrong;
with the base at 1.5 Mm, 29 % (``examples/39_chromospheric_base.py``). The first figure already includes the
preprocessing with its default weights. This script measures how much of that gap is closed by the three parameters
that set how the base is treated: ``mu3`` and ``mu4`` of the preprocessing (how far from the data it may move, how
much it is smoothed) and ``relax_bottom`` of the NLFFF (the weight with which the base follows the data).

- Scan: a grid of 48 combinations on one snapshot of the MURaM run, with the 1.54 Mm pixel and the three noise models.
- Choice, by a rule written before running: the fewest wrong lines at the 10 % definition; among the combinations
  within one point of the best, the closest to the defaults.
- Test, once and with the chosen combination: the other two snapshots, with the 1.54 and the 0.77 Mm pixel, and the
  two Bifrost runs with the large pixel.

The noise and the seeds of the lines are those of ``examples/35_realistic_truths.py``, so the lines are the same and
the comparison with the default parameters is paired. The default combination of the scan must reproduce the stored
cases of that example, or the script stops. The criteria were written down in the plan of the paper before running
this (19-Nov-2026).

Needs the reduced truths and ``data/realistic_<simulation>.pkl`` of ``examples/35`` and, for the reference with the
chromospheric base, ``data/chromo_<simulation>_z1.5.pkl`` of ``examples/39``. Saves ``data/base_scan.pkl`` and
``data/base_test.pkl`` (checkpoints) and ``docs/resultados/base_treatment.json``.

    python -u examples/40_base_treatment.py
    python examples/40_base_treatment.py --analyze-only
"""
import argparse
import itertools
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.analytic import field_magnetogram
from solarff.calibration import NOISE_MODELS
from solarff.confidence import CALIBRATION_MHD, _design, auc, calibration_case
from solarff.field import VectorField3D
from solarff.metrics import force_free_metrics
from solarff.simulations import SIMULATIONS, reduced_field
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
RUNS = {"ar098192": ("muram_ar098192", [290000, 310000, 270000]), "qs024048_by3363": ("qs024048_by3363", [950, 900]), "en024048_hion": ("en024048_hion", [385])}
GRID = {"mu3": [1e-3, 1e-2, 1e-1], "mu4": [1e-3, 1e-2, 1e-1, 1.0], "relax_bottom": [None, 1e-2, 1e-1, 1.0]}
DEFAULT = (1e-2, 1e-2, 1e-1)
COMBOS = [DEFAULT] + [c for c in itertools.product(*GRID.values()) if c != DEFAULT]
SIM, FIT, TEST = "ar098192", 290000, [310000, 270000]
TESTS = [(SIM, s, r) for r in (8, 4) for s in TEST] + [("qs024048_by3363", s, 8) for s in (950, 900)] + [("en024048_hion", 385, 8)]
NOISES = list(NOISE_MODELS.items())
ap = argparse.ArgumentParser()
ap.add_argument("--raw-root", default="data", help="folder with one folder of cubes per simulation (only if a reduced truth is missing)")
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "base_treatment.json"))
ap.add_argument("--check-only", action="store_true", help="compute the default combination, compare it with the stored cases and stop")
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()
SCAN, TESTF = os.path.join(a.data, "base_scan.pkl"), os.path.join(a.data, "base_test.pkl")
_truths, _stored = {}, {}


def truth(sim, snap, r):
    """The field of the snapshot above the photosphere, as ``examples/35_realistic_truths.py`` builds it."""
    if (sim, snap, r) not in _truths:
        path = os.path.join(a.data, f"truth_{sim}_{snap}_r{r}.npz")
        if not os.path.exists(path):
            f = reduced_field(sim, snap, r, os.path.join(a.raw_root, RUNS[sim][0]))
            np.savez_compressed(path, bx=f.bx.astype(np.float32), by=f.by.astype(np.float32), bz=f.bz.astype(np.float32), dx=f.dx, origin=np.array(f.origin),
                                elapsed=f.meta["elapsed_s"])
        d = np.load(path)
        _truths[(sim, snap, r)] = VectorField3D(d["bx"].astype(float), d["by"].astype(float), d["bz"].astype(float), float(d["dx"]), tuple(d["origin"]))
    return _truths[(sim, snap, r)]


def stored(sim, snap, r, j, kind="realistic"):
    """The case with the default parameters (``examples/35``) or with the base at 1.5 Mm (``examples/39``); None if it is not there."""
    path = os.path.join(a.data, f"realistic_{sim}.pkl" if kind == "realistic" else f"chromo_{sim}_z1.5.pkl")
    if path not in _stored:
        _stored[path] = pickle.load(open(path, "rb"))["cases"] if os.path.exists(path) else {}
    return _stored[path].get(f"{snap} | r{r} | {NOISES[j][0]}")


def case(sim, snap, r, j, combo):
    tr = truth(sim, snap, r)
    nm, (sz, st) = NOISES[j]
    k = 7919 * (RUNS[sim][1].index(snap) + 1) + 100 * r + j + 100000 * list(SIMULATIONS).index(sim)       # the noise and the lines of examples/35
    mag = field_magnetogram(tr, noise=sz, noise_t=st, rng=k)
    tc = time.time()
    rec = calibration_case(tr, mag, tr.shape[2], n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"max_iter": 4000, "relax_bottom": combo[2]},
                           preprocess_kw={"mu3": combo[0], "mu4": combo[1]}, potential=True, keep_model=True)
    model, pp = rec.pop("model"), rec.pop("pp")
    obs = np.stack([mag.bx, mag.by, mag.bz])
    shift = lambda b: float(np.sqrt(((b - obs) ** 2).sum(axis=0).mean() / (obs ** 2).sum(axis=0).mean()))  # noqa: E731
    rec["base"] = {"pp_shift": shift(np.stack([pp.bx, pp.by, pp.bz])), "model_shift": shift(np.stack([model.bx[:, :, 0], model.by[:, :, 0], model.bz[:, :, 0]])),
                   "cwsin": float(force_free_metrics(model, margin=int(model.meta.get("nd", 0)))["CWsin"]), "iterations": int(model.meta.get("iterations", 0))}
    rec.update(sim=sim, snapshot=int(snap), r=int(r), noise_model=nm, combo=combo, seconds=time.time() - tc)
    return rec


def bad(rec, thr=0.10):
    return (np.asarray(rec["err"]["shape"], float) > thr) | (np.asarray(rec["err"]["topology"]) > 0)


def same_lines(u, v):
    """Fraction of the lines whose true line has the same length within 0.1 % in the two cases: 1 if the seeds and the truth are the same.

    The stored Bifrost cases were computed with the truth before it was written in single precision, and a few lines differ."""
    p, q = np.asarray(u["err"]["length_ref"], float), np.asarray(v["err"]["length_ref"], float)
    return float(np.mean(np.abs(p - q) <= 1e-3 * np.maximum(np.abs(q), 1e-9))) if len(p) == len(q) else 0.0


def chain_check(scan):
    out = []
    for j, (nm, _) in enumerate(NOISES):
        new, old = scan[DEFAULT + (j,)], stored(SIM, FIT, 8, j)
        out.append({"noise": nm, "stored": float(np.mean(old["bad"])), "recomputed": float(np.mean(new["bad"])),
                    "same_labels": float(np.mean(np.asarray(new["bad"]) == np.asarray(old["bad"]))), "same_lines": same_lines(new, old)})
    return out


def rates(scan, thr=0.10):
    """Wrong lines of each complete combination over its three noise models."""
    return {c: float(np.concatenate([bad(scan[c + (j,)], thr) for j in range(len(NOISES))]).mean()) for c in COMBOS if all(c + (j,) in scan for j in range(len(NOISES)))}


def distance(c):
    """How far a combination is from the defaults: parameters changed, then decades of change (a fixed base counts as two)."""
    return sum(u != v for u, v in zip(c, DEFAULT)), float(sum(2.0 if u is None else abs(np.log10(u / v)) for u, v in zip(c, DEFAULT)))


def choose(rate):
    best = min(rate.values())
    near = [c for c in rate if rate[c] <= best + 0.01]
    return min(near, key=lambda c: distance(c) + (rate[c],)), best, len(near)


if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        scan = pickle.load(open(SCAN, "rb")) if os.path.exists(SCAN) else {}
        for ci, combo in enumerate(COMBOS):
            for j in range(len(NOISES)):
                if combo + (j,) in scan:
                    continue
                scan[combo + (j,)] = rec = case(SIM, FIT, 8, j, combo)
                pickle.dump(scan, open(SCAN, "wb"))
                print(f"  scan {ci + 1}/{len(COMBOS)} mu3 {combo[0]:g} mu4 {combo[1]:g} relax {combo[2]} | {rec['noise_model']}: {np.mean(rec['bad']):.1%} wrong lines "
                      f"({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)", flush=True)
            if ci == 0:
                chk = chain_check(scan)
                for q in chk:
                    print(f"  chain check, {q['noise']}: stored {q['stored']:.3f}, recomputed {q['recomputed']:.3f}, same labels {q['same_labels']:.3f}, same lines {q['same_lines']}", flush=True)
                if any(abs(q["stored"] - q["recomputed"]) > 0.01 or q["same_lines"] < 1 for q in chk):
                    raise SystemExit("the default combination does not reproduce the stored cases: stopping before anything else is looked at")
                if a.check_only:
                    raise SystemExit(0)
        chosen, best, near = choose(rates(scan))
        print(f"chosen: mu3 {chosen[0]:g}, mu4 {chosen[1]:g}, relax_bottom {chosen[2]} ({rates(scan)[chosen]:.1%} wrong; best {best:.1%}; {near} within one point)", flush=True)
        test = pickle.load(open(TESTF, "rb")) if os.path.exists(TESTF) else {"combo": chosen, "cases": {}}
        if test["combo"] != chosen:
            raise SystemExit("the test checkpoint was computed with another combination")
        if chosen != DEFAULT:
            for sim, snap, r in TESTS:
                for j in range(len(NOISES)):
                    if (sim, snap, r, j) in test["cases"]:
                        continue
                    test["cases"][(sim, snap, r, j)] = rec = case(sim, snap, r, j, chosen)
                    pickle.dump(test, open(TESTF, "wb"))
                    print(f"  test {sim} | {snap} | r{r} | {rec['noise_model']}: {np.mean(rec['bad']):.1%} wrong lines ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)",
                          flush=True)


# ------------------------------------------------------------------ analysis
def summary(cases):
    cat = lambda f: np.concatenate([np.asarray(f(x), float) for x in cases])  # noqa: E731
    name = lambda x, q: q if q in x["ind"] else {"departure": "ens_pot", "departure_topo": "ens_pot_topo"}[q]  # noqa: E731  (the stored cases use the older name)
    sh, tp = cat(lambda x: x["err"]["shape"]), cat(lambda x: x["err"]["topology"]) > 0
    psh, ptp = cat(lambda x: x["pot"]["pp"]["shape"]), cat(lambda x: x["pot"]["pp"]["topology"]) > 0
    dep = cat(lambda x: np.where(np.asarray(x["ind"][name(x, "departure_topo")]) > 0, np.maximum(np.nan_to_num(np.asarray(x["ind"][name(x, "departure")], float), nan=0.5), 0.5),
                                 np.asarray(x["ind"][name(x, "departure")], float)))
    X = np.column_stack([np.vstack([_design(x["ind"], CALIBRATION_MHD["features"][:5]) for x in cases]), np.log10(np.maximum(dep, 1e-3))])
    score = (X - np.array(CALIBRATION_MHD["mu"])) / np.array(CALIBRATION_MHD["sd"]) @ np.array(CALIBRATION_MHD["coef"])
    ok, same = np.isfinite(score), ~tp & ~ptp & np.isfinite(sh) & np.isfinite(psh)
    o = {"n_lines": int(len(sh)), "nlfff_closer_than_potential": float(np.mean(sh[same] < psh[same])), "departure_median": float(np.nanmedian(dep))}
    for lab, thr in (("10", 0.10), ("20", 0.20)):
        y, yp = (sh > thr) | tp, (psh > thr) | ptp
        o[lab] = {"nlfff_bad": float(y.mean()), "potential_bad": float(yp.mean()), "ranking_auc": auc(score[ok], y[ok]), "departure_auc": auc(dep[ok], y[ok])}
    if all("base" in x for x in cases):
        o["base"] = {k: float(np.mean([x["base"][k] for x in cases])) for k in ("pp_shift", "model_shift", "cwsin")}
        o["seconds"] = float(np.sum([x["seconds"] for x in cases]))
    return o


def combo_dict(c):
    return {"mu3": c[0], "mu4": c[1], "relax_bottom": c[2]}


scan = pickle.load(open(SCAN, "rb"))
rate = rates(scan)
chosen, best, near = choose(rate)
out = {"code_version": code_version()["label"], "grid": GRID, "default": combo_dict(DEFAULT), "fit": {"simulation": SIM, "snapshot": FIT, "r": 8, "combinations": len(rate)},
       "chain_check": chain_check(scan), "scan": [], "chosen": {**combo_dict(chosen), "bad_fit": rate[chosen], "best_bad_fit": best, "within_one_point": near,
                                                                 "is_default": chosen == DEFAULT}, "test": {}, "criteria": {}}
for c in COMBOS:
    if c in rate:
        cs = [scan[c + (j,)] for j in range(len(NOISES))]
        out["scan"].append({**combo_dict(c), **summary(cs), "by_noise": {nm: float(np.mean(x["bad"])) for (nm, _), x in zip(NOISES, cs)}})

test = pickle.load(open(TESTF, "rb")) if os.path.exists(TESTF) else {"combo": chosen, "cases": {}}
assert test["combo"] == chosen, "the test checkpoint was computed with another combination"
for sim, r in dict.fromkeys((s, r) for s, _, r in TESTS):
    keys = [(sim, snap, r, j) for s, snap, rr in TESTS if (s, rr) == (sim, r) for j in range(len(NOISES))]
    if not all(k in test["cases"] for k in keys):
        continue
    new, old = [test["cases"][k] for k in keys], [stored(*k) for k in keys]
    chromo = [stored(*k, kind="chromo") for k in keys]
    g = {"default": summary(old), "chosen": summary(new), "same_lines": min(same_lines(u, v) for u, v in zip(new, old)),
         "per_case": [{"snapshot": k[1], "noise": NOISES[k[3]][0], "default": float(np.mean(bad(u))), "chosen": float(np.mean(bad(v)))} for k, u, v in zip(keys, old, new)],
         "lines_fixed": int(sum((bad(u) & ~bad(v)).sum() for u, v in zip(old, new))), "lines_broken": int(sum((~bad(u) & bad(v)).sum() for u, v in zip(old, new)))}
    d, c = g["default"]["10"]["nlfff_bad"], g["chosen"]["10"]["nlfff_bad"]
    g["fall_points"], g["falls_in_all"] = 100 * (d - c), all(p["chosen"] < p["default"] for p in g["per_case"])
    if all(x is not None for x in chromo):
        g["chromospheric_base"] = {lab: float(np.concatenate([bad(x, thr) for x in chromo]).mean()) for lab, thr in (("10", 0.10), ("20", 0.20))}
        g["gap_closed"] = (d - c) / (d - g["chromospheric_base"]["10"])
    out["test"][f"{sim} | r{r}"] = g
g8, g4 = out["test"].get(f"{SIM} | r8"), out["test"].get(f"{SIM} | r4")
if g8 and g4:
    out["criteria"] = {"B1": bool(g8["fall_points"] >= 5 and g8["falls_in_all"]), "B2": bool(g8["gap_closed"] >= 0.5), "B3": bool(g4["fall_points"] >= 5)}
elif chosen == DEFAULT:
    out["criteria"] = {"B1": False, "B2": False, "B3": False, "note": "the defaults are within one point of the best of the grid: nothing to test"}
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

for q in out["chain_check"]:
    print(f"chain check, {q['noise']}: stored {q['stored']:.3f}, recomputed {q['recomputed']:.3f}, same labels {q['same_labels']:.3f}, same lines {q['same_lines']}")
print(f"scan on snapshot {FIT}, r = 8, {len(rate)} combinations; wrong lines at 10 % (20 %), potential of the preprocessed base, shift of the base, CWsin of the model")
for s in sorted(out["scan"], key=lambda s: s["10"]["nlfff_bad"]):
    mark = " <- chosen" if (s["mu3"], s["mu4"], s["relax_bottom"]) == chosen else (" <- default" if (s["mu3"], s["mu4"], s["relax_bottom"]) == DEFAULT else "")
    print(f"  mu3 {s['mu3']:<6g} mu4 {s['mu4']:<6g} relax {str(s['relax_bottom']):<5}: {s['10']['nlfff_bad']:.3f} ({s['20']['nlfff_bad']:.3f}) | potential {s['10']['potential_bad']:.3f} | "
          f"shift {s['base']['pp_shift']:.2f} / {s['base']['model_shift']:.2f} | CWsin {s['base']['cwsin']:.2f}{mark}")
for name, g in out["test"].items():
    d, c = g["default"], g["chosen"]
    print(f"test {name}, {c['n_lines']} lines (same lines: {g['same_lines']}): wrong at 10 % {d['10']['nlfff_bad']:.3f} -> {c['10']['nlfff_bad']:.3f}, at 20 % {d['20']['nlfff_bad']:.3f} -> "
          f"{c['20']['nlfff_bad']:.3f}; falls {g['fall_points']:.1f} points, in every case: {g['falls_in_all']}; fixed {g['lines_fixed']}, broken {g['lines_broken']}"
          + (f"; base at 1.5 Mm {g['chromospheric_base']['10']:.3f}, gap closed {g['gap_closed']:.2f}" if "gap_closed" in g else ""))
    print(f"   potential {d['10']['potential_bad']:.3f} -> {c['10']['potential_bad']:.3f} | NLFFF closer than the potential {d['nlfff_closer_than_potential']:.2f} -> "
          f"{c['nlfff_closer_than_potential']:.2f} | ranking AUC {d['10']['ranking_auc']:.3f} -> {c['10']['ranking_auc']:.3f} | shift of the base {c['base']['pp_shift']:.2f} / "
          f"{c['base']['model_shift']:.2f}, CWsin {c['base']['cwsin']:.2f}")
    print("   per case: " + ", ".join(f"{p['default']:.2f}->{p['chosen']:.2f}" for p in g["per_case"]))
print("criteria:", out["criteria"])
print("written", a.json)
