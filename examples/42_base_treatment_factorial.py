"""The two changes of the base treatment taken apart: is there one that gains against the simulation at no cost on exact fields?

``examples/40_base_treatment.py`` chose two changes together against a radiative MHD simulation: less smoothing in
the preprocessing (``mu3`` 0.1 instead of 0.01) and a freer base in the NLFFF (``relax_bottom`` 0.01 instead of 0.1).
``examples/41_base_treatment_analytic.py`` showed that together they cost wrong lines on the exact force-free fields of
the paper. This script takes the two changes apart: the four combinations, against both kinds of truth.

- Stage 1: the two single-change configurations on the 60 analytical cases of ``examples/41`` (same lines).
- Choice, by a rule written before running: among the three configurations other than the default, those that do not
  raise the wrong lines by more than 2 points on the six calibration families nor on the four external ones, and by no
  more than 5 points on any family; of them, the one with the fewest wrong lines on the snapshot of the simulation
  used for the scan of ``examples/40``.
- Stage 2, once and with the chosen configuration: a new realisation of the 60 analytical cases (another noise, other
  seeds) with the default and the chosen configuration, and the two test snapshots of the simulation.

The two single-change configurations are computed on the test snapshots in any case, to have the whole table on the
same snapshots. The criteria were written down in the plan of the paper before running this (19-Nov-2026).

Needs the checkpoints of ``examples/35``, ``40`` and ``41`` and the reduced truths ``data/truth_ar098192_*.npz``.
Saves ``data/base_factorial.pkl`` (checkpoint) and ``docs/resultados/base_treatment_factorial.json``.

    python -u examples/42_base_treatment_factorial.py           # about an hour
    python examples/42_base_treatment_factorial.py --analyze-only
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.analytic import CALIBRATION_FIELDS, EXTERNAL_TEST_FIELDS, field_magnetogram
from solarff.calibration import NOISE_MODELS, SIX_FAMILIES, _nz, ece
from solarff.confidence import DEFAULT_CALIBRATION, auc, calibration_case, confidence
from solarff.field import VectorField3D
from solarff.metrics import energy, force_free_metrics
from solarff.potential import potential_field
from solarff.simulations import SIMULATIONS
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "base_treatment_factorial.json"))
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

EXTERNAL = ("TD2-relax", "LL-IV", "LFFF-delta", "TD3-relax")
FAMILIES = list(SIX_FAMILIES) + list(EXTERNAL)
KNOWN = {**CALIBRATION_FIELDS, **EXTERNAL_TEST_FIELDS}
SIZES = (32, 48)
NOISES = list(NOISE_MODELS.items())
CONFIGS = {"default": (1e-2, 1e-2, 1e-1), "sharp": (1e-1, 1e-2, 1e-1), "free": (1e-2, 1e-2, 1e-2), "both": (1e-1, 1e-2, 1e-2)}     # mu3, mu4, relax_bottom
SINGLE = ("sharp", "free")                                                      # less smoothing only, freer base only
SEEDS = {"analytic": 500009, "fresh": 700001}                                   # the cases of examples/36 and 41, and a new realisation
SIM, SNAPSHOTS, FIT, TEST = "ar098192", [290000, 310000, 270000], 290000, [310000, 270000]
T = DEFAULT_CALIBRATION["threshold"]
KEYS = [(fam, size, j) for size in SIZES for fam in FAMILIES for j in range(len(NOISES))]
MKEYS = [(snap, r, j) for r in (8, 4) for snap in TEST for j in range(len(NOISES))]
CKPT = os.path.join(a.data, "base_factorial.pkl")
load = lambda name: pickle.load(open(os.path.join(a.data, name), "rb"))  # noqa: E731
prior, scan, btest, realistic = load("base_analytic.pkl"), load("base_scan.pkl"), load("base_test.pkl"), load(f"realistic_{SIM}.pkl")["cases"]
assert btest["combo"] == CONFIGS["both"], "the test checkpoint of examples/40 was computed with another combination"
state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
_truths = {}


def base_stats(rec, mag, model, pp):
    obs = np.stack([mag.bx, mag.by, mag.bz])
    shift = lambda b: float(np.sqrt(((b - obs) ** 2).sum(axis=0).mean() / (obs ** 2).sum(axis=0).mean()))  # noqa: E731
    nd = int(model.meta.get("nd", 0))
    return {"pp_shift": shift(np.stack([pp.bx, pp.by, pp.bz])), "model_shift": shift(np.stack([model.bx[:, :, 0], model.by[:, :, 0], model.bz[:, :, 0]])),
            "cwsin": float(force_free_metrics(model, margin=nd)["CWsin"]), "iterations": int(model.meta.get("iterations", 0)),
            "converged": str(model.meta.get("message", "")).startswith("convergido")}


def analytic_case(fam, size, j, combo, exact, base_seed):
    """One analytical case, as ``examples/41_base_treatment_analytic.py`` computes it."""
    nm, (sz, st) = NOISES[j]
    nz = _nz(size)
    k = base_seed + 1000 * FAMILIES.index(fam) + 10 * size + j
    mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=k)
    tc = time.time()
    rec = calibration_case(exact, mag, nz, n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"relax_bottom": combo[2], "max_iter": 2000},
                           preprocess_kw={"mu3": combo[0], "mu4": combo[1]}, potential=True, keep_model=True)
    model, pp = rec.pop("model"), rec.pop("pp")
    nd = int(model.meta.get("nd", 0))
    e_exact = energy(exact, nd) - energy(potential_field(field_magnetogram(exact, 0.0), nz), nd)
    e_model = energy(model, nd) - energy(potential_field(pp, nz), nd)
    rec["base"] = {**base_stats(rec, mag, model, pp), "efree_exact": float(e_exact), "efree_model": float(e_model), "efree_err": float(e_model / e_exact - 1)}
    rec.update(group=fam, size=int(size), noise_model=nm, combo=combo, seconds=time.time() - tc)
    return rec


def truth(snap, r):
    if (snap, r) not in _truths:
        d = np.load(os.path.join(a.data, f"truth_{SIM}_{snap}_r{r}.npz"))       # written by examples/35_realistic_truths.py
        _truths[(snap, r)] = VectorField3D(d["bx"].astype(float), d["by"].astype(float), d["bz"].astype(float), float(d["dx"]), tuple(d["origin"]))
    return _truths[(snap, r)]


def muram_case(snap, r, j, combo):
    """One case of the simulation, as ``examples/40_base_treatment.py`` computes it."""
    tr = truth(snap, r)
    nm, (sz, st) = NOISES[j]
    k = 7919 * (SNAPSHOTS.index(snap) + 1) + 100 * r + j + 100000 * list(SIMULATIONS).index(SIM)
    mag = field_magnetogram(tr, noise=sz, noise_t=st, rng=k)
    tc = time.time()
    rec = calibration_case(tr, mag, tr.shape[2], n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"max_iter": 4000, "relax_bottom": combo[2]},
                           preprocess_kw={"mu3": combo[0], "mu4": combo[1]}, potential=True, keep_model=True)
    model, pp = rec.pop("model"), rec.pop("pp")
    rec["base"] = base_stats(rec, mag, model, pp)
    rec.update(sim=SIM, snapshot=int(snap), r=int(r), noise_model=nm, combo=combo, seconds=time.time() - tc)
    return rec


def A(name, k, kind="analytic"):
    """The analytical case ``k`` = (family, size, noise) with a configuration; None if it is not computed."""
    if kind == "analytic" and name in ("default", "both"):
        return prior.get(("default" if name == "default" else "chosen",) + k)
    return state.get((kind, name) + k)


def M(name, k):
    """The case ``k`` = (snapshot, r, noise) of the test snapshots of the simulation with a configuration; None if it is not computed."""
    if name == "default":
        return realistic.get(f"{k[0]} | r{k[1]} | {NOISES[k[2]][0]}")
    return btest["cases"].get((SIM,) + k) if name == "both" else state.get(("muram", name) + k)


def bad(rec, thr=0.10):
    return (np.asarray(rec["err"]["shape"], float) > thr) | (np.asarray(rec["err"]["topology"]) > 0)


def same_lines(u, v):
    """Fraction of the lines whose true line has the same length within 0.1 % in the two cases: 1 if the seeds and the truth are the same."""
    p, q = np.asarray(u["err"]["length_ref"], float), np.asarray(v["err"]["length_ref"], float)
    return float(np.mean(np.abs(p - q) <= 1e-3 * np.maximum(np.abs(q), 1e-9))) if len(p) == len(q) else 0.0


def rise(old, new, thr=0.10, n_boot=2000):
    """Change of the wrong lines from ``old`` to ``new`` (the same lines), in points, with an interval resampling whole cases."""
    n = np.array([len(bad(u)) for u in old], float)
    d, c = np.array([bad(u, thr).sum() for u in old], float), np.array([bad(v, thr).sum() for v in new], float)
    idx = np.random.default_rng(0).integers(0, len(old), (n_boot, len(old)))
    boot = 100 * (c[idx].sum(axis=1) - d[idx].sum(axis=1)) / n[idx].sum(axis=1)
    return {"n_lines": int(n.sum()), "default": float(d.sum() / n.sum()), "config": float(c.sum() / n.sum()), "rise_points": float(100 * (c.sum() - d.sum()) / n.sum()),
            "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "fixed": int(sum((bad(u, thr) & ~bad(v, thr)).sum() for u, v in zip(old, new))),
            "broken": int(sum((~bad(u, thr) & bad(v, thr)).sum() for u, v in zip(old, new))), "cases_that_rise": int((c > d).sum()), "n_cases": len(old)}


def summary(cases):
    """Wrong lines, published confidence and state of the base of a set of analytical cases of one configuration."""
    cat = lambda f: np.concatenate([np.asarray(f(x), float) for x in cases])  # noqa: E731
    sh, tp = cat(lambda x: x["err"]["shape"]), cat(lambda x: x["err"]["topology"]) > 0
    conf = cat(lambda x: confidence(x["ind"], DEFAULT_CALIBRATION))
    ok = np.isfinite(conf)
    acc, y, y20 = conf[ok] >= T, (sh > 0.10) | tp, (sh > 0.20) | tp
    b = lambda k: np.array([x["base"][k] for x in cases], float)  # noqa: E731
    return {"n_lines": int(len(sh)), "bad_10": float(y.mean()), "bad_20": float(y20.mean()), "confidence_auc": auc(1 - conf[ok], y[ok]), "confidence_ece": ece(1 - conf[ok], y[ok]),
            "accepted": float(acc.mean()), "bad_accepted": float(y[ok][acc].mean()) if acc.any() else None, "bad_accepted_20": float(y20[ok][acc].mean()) if acc.any() else None,
            "cwsin": float(b("cwsin").mean()), "pp_shift": float(b("pp_shift").mean()), "model_shift": float(b("model_shift").mean()),
            "efree_err_median": float(np.median(b("efree_err"))), "efree_abs_err_median": float(np.median(np.abs(b("efree_err")))),
            "iterations_median": float(np.median(b("iterations"))), "not_converged": int((b("converged") == 0).sum())}


def analytic_compare(name, kind="analytic"):
    """A configuration against the default one on the analytical cases; None if they are not all computed."""
    if any(A("default", k, kind) is None or A(name, k, kind) is None for k in KEYS):
        return None
    assert all(same_lines(A("default", k, kind), A(name, k, kind)) == 1 for k in KEYS), "the lines of the two configurations are not the same"
    pick = lambda f, thr=0.10: rise([A("default", k, kind) for k in KEYS if f(k)], [A(name, k, kind) for k in KEYS if f(k)], thr)  # noqa: E731
    g = {"calibration": pick(lambda k: k[0] in SIX_FAMILIES), "external": pick(lambda k: k[0] in EXTERNAL), "ten": pick(lambda k: True)}
    g["at_20"] = {"calibration": pick(lambda k: k[0] in SIX_FAMILIES, 0.20), "external": pick(lambda k: k[0] in EXTERNAL, 0.20), "ten": pick(lambda k: True, 0.20)}
    g["by_family"] = {fam: pick(lambda k: k[0] == fam) for fam in FAMILIES}
    g["by_noise"] = {nm: pick(lambda k: k[2] == j) for j, (nm, _) in enumerate(NOISES)}
    g["by_size"] = {str(size): pick(lambda k: k[1] == size) for size in SIZES}
    worst = max(FAMILIES, key=lambda f: g["by_family"][f]["rise_points"])
    g.update(worst_family=worst, worst_family_rise_points=g["by_family"][worst]["rise_points"],
             meets_A1=bool(g["calibration"]["rise_points"] <= 2 and g["external"]["rise_points"] <= 2), meets_A2=bool(g["by_family"][worst]["rise_points"] <= 5))
    g["summary"] = {grp: {"default": summary([A("default", k, kind) for k in KEYS if f(k)]), "config": summary([A(name, k, kind) for k in KEYS if f(k)])}
                    for grp, f in (("calibration", lambda k: k[0] in SIX_FAMILIES), ("external", lambda k: k[0] in EXTERNAL), ("ten", lambda k: True))}
    return g


def sim_summary(cases):
    """Wrong lines, published confidence and state of the base of a set of cases of the simulation with one configuration."""
    cat = lambda f: np.concatenate([np.asarray(f(x), float) for x in cases])  # noqa: E731
    sh, tp = cat(lambda x: x["err"]["shape"]), cat(lambda x: x["err"]["topology"]) > 0
    conf = cat(lambda x: confidence(x["ind"], DEFAULT_CALIBRATION))
    ok = np.isfinite(conf)
    acc, y = conf[ok] >= T, (sh > 0.10) | tp
    s = {"n_lines": int(len(sh)), "bad_10": float(y.mean()), "confidence_auc": auc(1 - conf[ok], y[ok]), "accepted": float(acc.mean()),
         "bad_accepted": float(y[ok][acc].mean()) if acc.any() else None}
    if all("base" in x for x in cases):                                         # the records of examples/35 do not keep the state of the base
        s.update({q: float(np.mean([x["base"][q] for x in cases])) for q in ("pp_shift", "model_shift", "cwsin")})
    return s


def muram_compare(name, r):
    """A configuration against the default one on the test snapshots of the simulation with one pixel; None if not computed."""
    keys = [k for k in MKEYS if k[1] == r]
    if any(M("default", k) is None or M(name, k) is None for k in keys):
        return None
    old, new = [M("default", k) for k in keys], [M(name, k) for k in keys]
    assert min(same_lines(u, v) for u, v in zip(old, new)) == 1, "the lines of the two configurations are not the same"
    g = rise(old, new)
    g.update(fall_points=-g["rise_points"], falls_in_all=bool(all(bad(v).mean() < bad(u).mean() for u, v in zip(old, new))), at_20=rise(old, new, 0.20),
             per_case=[{"snapshot": k[0], "noise": NOISES[k[2]][0], "default": float(bad(u).mean()), "config": float(bad(v).mean())} for k, u, v in zip(keys, old, new)],
             base={q: float(np.mean([v["base"][q] for v in new])) for q in ("pp_shift", "model_shift", "cwsin")},
             by_noise={nm: rise([u for k, u in zip(keys, old) if k[2] == j], [v for k, v in zip(keys, new) if k[2] == j]) for j, (nm, _) in enumerate(NOISES)},
             summary={"default": sim_summary(old), "config": sim_summary(new)})
    return g


def fit_rate(name):
    """Wrong lines of a configuration on the snapshot of the scan of ``examples/40`` (10 % definition)."""
    return float(np.concatenate([bad(scan[CONFIGS[name] + (j,)]) for j in range(len(NOISES))]).mean())


def choose():
    """The registered rule: of the configurations that meet the bars of A1 and A2 on the analytical cases, the best on the fit snapshot."""
    comp = {n: analytic_compare(n) for n in CONFIGS if n != "default"}
    if any(c is None for c in comp.values()):
        return None, comp
    ok = [n for n, c in comp.items() if c["meets_A1"] and c["meets_A2"]]
    return (min(ok, key=fit_rate) if ok else None), comp


if not a.analyze_only:
    with keep_awake():
        t0 = time.time()

        def run(key, fn, label):
            if key not in state:
                state[key] = rec = fn()
                pickle.dump(state, open(CKPT, "wb"))
                print(f"  {label}: {np.mean(rec['bad']):.1%} wrong lines, {rec['base']['iterations']} iterations ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)", flush=True)

        def analytic_stage(kind, names):
            for name in names:
                exact = {}
                for fam, size, j in KEYS:
                    if (kind, name, fam, size, j) in state:
                        continue
                    if (fam, size) not in exact:
                        exact = {(fam, size): KNOWN[fam](size, size, _nz(size))}
                    run((kind, name, fam, size, j), lambda: analytic_case(fam, size, j, CONFIGS[name], exact[(fam, size)], SEEDS[kind]), f"{kind} | {name} | {fam} | {size} | {NOISES[j][0]}")

        analytic_stage("analytic", SINGLE)                                      # stage 1
        chosen, _ = choose()
        print(f"chosen by the rule: {chosen}", flush=True)
        for name in SINGLE:                                                     # the test snapshots, for the whole table; only the chosen one counts for the criterion
            for snap, r, j in MKEYS:
                run(("muram", name, snap, r, j), lambda: muram_case(snap, r, j, CONFIGS[name]), f"simulation | {name} | {snap} | r{r} | {NOISES[j][0]}")
        if chosen is not None:                                                  # stage 2: a new realisation of the analytical cases
            analytic_stage("fresh", ["default", chosen])

# ------------------------------------------------------------------ analysis
chosen, comp = choose()
out = {"code_version": code_version()["label"], "configurations": {n: dict(zip(("mu3", "mu4", "relax_bottom"), c)) for n, c in CONFIGS.items()},
       "simulation_fit": {"snapshot": FIT, "r": 8, "bad_10": {n: fit_rate(n) for n in CONFIGS},
                          "summary": {n: sim_summary([scan[CONFIGS[n] + (j,)] for j in range(len(NOISES))]) for n in CONFIGS}}, "analytic": comp, "chosen": chosen, "simulation_test": {}, "fresh": None,
       "additivity": {}, "criteria": {}}
for name in ("sharp", "free", "both"):
    out["simulation_test"][name] = {f"r{r}": muram_compare(name, r) for r in (8, 4)}
if all(c is not None for c in comp.values()):
    out["additivity"]["analytic_ten_rise_points"] = {"sharp": comp["sharp"]["ten"]["rise_points"], "free": comp["free"]["ten"]["rise_points"], "both": comp["both"]["ten"]["rise_points"],
                                                     "sum_of_singles": comp["sharp"]["ten"]["rise_points"] + comp["free"]["ten"]["rise_points"]}
    f = out["simulation_fit"]["bad_10"]
    out["additivity"]["simulation_fit_fall_points"] = {n: 100 * (f["default"] - f[n]) for n in ("sharp", "free", "both")}
    out["additivity"]["simulation_fit_fall_points"]["sum_of_singles"] = 100 * (2 * f["default"] - f["sharp"] - f["free"])
    out["criteria"]["D1"] = bool(any(comp[n]["meets_A1"] and comp[n]["meets_A2"] for n in SINGLE))
    out["criteria"]["D2"] = out["criteria"]["D3"] = None
    if chosen is not None:
        out["fresh"] = analytic_compare(chosen, "fresh")
        both8, test8 = out["simulation_test"]["both"]["r8"], out["simulation_test"][chosen]["r8"] if chosen != "both" else None
        if out["fresh"] is not None:
            out["criteria"]["D2"] = bool(out["fresh"]["meets_A1"] and out["fresh"]["meets_A2"])
        if test8 is not None:
            out["criteria"].update(D3=bool(test8["fall_points"] >= 0.5 * both8["fall_points"] and test8["falls_in_all"]), D3_bar_points=0.5 * both8["fall_points"])
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("wrong lines at the 10 % definition; analytical cases: change from the default configuration in points (interval), worst family; simulation: snapshot of the scan")
for name, c in comp.items():
    if c is None:
        print(f"  {name:6s} not computed on the analytical cases; simulation {out['simulation_fit']['bad_10'][name]:.3f}")
        continue
    s = c["summary"]["ten"]
    print(f"  {name:6s} calibration {c['calibration']['rise_points']:+.1f} ({c['calibration']['ci'][0]:+.1f} to {c['calibration']['ci'][1]:+.1f}) | external {c['external']['rise_points']:+.1f} "
          f"({c['external']['ci'][0]:+.1f} to {c['external']['ci'][1]:+.1f}) | ten {c['ten']['rise_points']:+.1f} ({c['ten']['ci'][0]:+.1f} to {c['ten']['ci'][1]:+.1f}), at 20 % "
          f"{c['at_20']['ten']['rise_points']:+.1f} | worst {c['worst_family']} {c['worst_family_rise_points']:+.1f} | A1 {c['meets_A1']}, A2 {c['meets_A2']} | simulation "
          f"{out['simulation_fit']['bad_10']['default']:.3f} -> {out['simulation_fit']['bad_10'][name]:.3f}")
    print(f"         confidence AUC {s['default']['confidence_auc']:.3f} -> {s['config']['confidence_auc']:.3f}, accepted {s['default']['accepted']:.3f} -> {s['config']['accepted']:.3f}, "
          f"wrong among them {s['default']['bad_accepted']:.3f} -> {s['config']['bad_accepted']:.3f} | CWsin {s['default']['cwsin']:.3f} -> {s['config']['cwsin']:.3f} | base of the model "
          f"{s['default']['model_shift']:.2f} -> {s['config']['model_shift']:.2f} | free energy, median absolute error {s['default']['efree_abs_err_median']:.3f} -> "
          f"{s['config']['efree_abs_err_median']:.3f}")
    print("         by family: " + ", ".join(f"{fam} {q['rise_points']:+.1f}" for fam, q in c["by_family"].items()))
print("test snapshots of the simulation: default -> configuration")
for name, g in out["simulation_test"].items():
    for r, q in g.items():
        if q is not None:
            print(f"  {name:6s} {r}: {q['default']:.3f} -> {q['config']:.3f}, falls {q['fall_points']:.1f} points ({-q['ci'][1]:.1f} to {-q['ci'][0]:.1f}), in every case: {q['falls_in_all']}; "
                  f"at 20 % {q['at_20']['default']:.3f} -> {q['at_20']['config']:.3f}; fixed {q['fixed']}, broken {q['broken']}; CWsin {q['base']['cwsin']:.2f}")
if out["fresh"] is not None:
    c = out["fresh"]
    print(f"new realisation of the analytical cases, {chosen}: calibration {c['calibration']['default']:.3f} -> {c['calibration']['config']:.3f} ({c['calibration']['rise_points']:+.1f}, "
          f"{c['calibration']['ci'][0]:+.1f} to {c['calibration']['ci'][1]:+.1f}) | external {c['external']['default']:.3f} -> {c['external']['config']:.3f} ({c['external']['rise_points']:+.1f}, "
          f"{c['external']['ci'][0]:+.1f} to {c['external']['ci'][1]:+.1f}) | worst {c['worst_family']} {c['worst_family_rise_points']:+.1f} | A1 {c['meets_A1']}, A2 {c['meets_A2']}")
    print("   by family: " + ", ".join(f"{fam} {q['rise_points']:+.1f}" for fam, q in c["by_family"].items()))
print("additivity:", {k: {n: round(v, 1) for n, v in q.items()} for k, q in out["additivity"].items()})
print("chosen:", chosen, "| criteria:", out["criteria"])
print("written", a.json)
