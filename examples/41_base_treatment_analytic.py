"""The base treatment chosen against a realistic truth, tested on the analytical truths of the paper.

``examples/40_base_treatment.py`` chose, against a radiative MHD simulation of an active region, a treatment of the
base that lowers the wrong lines there: ``mu3`` 0.1 and ``mu4`` 0.01 in the preprocessing and ``relax_bottom`` 0.01 in
the NLFFF (the defaults are 0.01, 0.01 and 0.1). This script measures what that configuration does on the exact
force-free fields the published confidence was fitted and tested with.

- Truths: the ten analytical families of the paper (six of the calibration, four external), grids of 32 and 48
  points, three noise models. They are the 60 cases of ``examples/36_mixed_model.py`` with the same noise and the same
  seeds, so the lines are the same and the comparison is paired.
- Two configurations, the default one and the chosen one; nothing else changes.
- The default configuration is computed again and must reproduce the stored cases, or the script stops.
- The published confidence is applied as is to the models of both configurations.

The criteria were written down in the plan of the paper before running this (19-Nov-2026).

Needs ``data/mixed_analytic.pkl`` of ``examples/36``. Saves ``data/base_analytic.pkl`` (checkpoint) and
``docs/resultados/base_treatment_analytic.json``.

    python -u examples/41_base_treatment_analytic.py            # about half an hour
    python examples/41_base_treatment_analytic.py --analyze-only
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
from solarff.confidence import CALIBRATION_MHD, DEFAULT_CALIBRATION, _design, auc, calibration_case, confidence
from solarff.metrics import energy, force_free_metrics
from solarff.potential import potential_field
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "base_treatment_analytic.json"))
ap.add_argument("--chosen-json", default=os.path.join("docs", "resultados", "base_treatment.json"), help="result of examples/40, to check the chosen combination")
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

EXTERNAL = ("TD2-relax", "LL-IV", "LFFF-delta", "TD3-relax")
FAMILIES = list(SIX_FAMILIES) + list(EXTERNAL)
KNOWN = {**CALIBRATION_FIELDS, **EXTERNAL_TEST_FIELDS}
SIZES = (32, 48)
NOISES = list(NOISE_MODELS.items())
CONFIGS = {"default": (1e-2, 1e-2, 1e-1), "chosen": (1e-1, 1e-2, 1e-2)}        # mu3, mu4, relax_bottom
MAX_ITER = 2000                                                                 # that of the suite of the paper
T = DEFAULT_CALIBRATION["threshold"]
CKPT, STORED = os.path.join(a.data, "base_analytic.pkl"), os.path.join(a.data, "mixed_analytic.pkl")
KEYS = [(fam, size, j) for size in SIZES for fam in FAMILIES for j in range(len(NOISES))]
if os.path.exists(a.chosen_json):
    q = json.load(open(a.chosen_json, encoding="utf-8"))["chosen"]
    assert (q["mu3"], q["mu4"], q["relax_bottom"]) == CONFIGS["chosen"], "the chosen combination is not that of examples/40"


def case(fam, size, j, combo, exact):
    nm, (sz, st) = NOISES[j]
    nz = _nz(size)
    k = 500009 + 1000 * FAMILIES.index(fam) + 10 * size + j                     # the noise and the lines of examples/36
    mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=k)
    tc = time.time()
    rec = calibration_case(exact, mag, nz, n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"relax_bottom": combo[2], "max_iter": MAX_ITER},
                           preprocess_kw={"mu3": combo[0], "mu4": combo[1]}, potential=True, keep_model=True)
    model, pp = rec.pop("model"), rec.pop("pp")
    obs = np.stack([mag.bx, mag.by, mag.bz])
    shift = lambda b: float(np.sqrt(((b - obs) ** 2).sum(axis=0).mean() / (obs ** 2).sum(axis=0).mean()))  # noqa: E731
    nd = int(model.meta.get("nd", 0))
    # free energy as in examples/06_robustness.py: each field against the potential field of its own base
    e_exact = energy(exact, nd) - energy(potential_field(field_magnetogram(exact, 0.0), nz), nd)
    e_model = energy(model, nd) - energy(potential_field(pp, nz), nd)
    rec["base"] = {"pp_shift": shift(np.stack([pp.bx, pp.by, pp.bz])), "model_shift": shift(np.stack([model.bx[:, :, 0], model.by[:, :, 0], model.bz[:, :, 0]])),
                   "cwsin": float(force_free_metrics(model, margin=nd)["CWsin"]), "iterations": int(model.meta.get("iterations", 0)),
                   "converged": str(model.meta.get("message", "")).startswith("convergido"), "efree_exact": float(e_exact), "efree_model": float(e_model),
                   "efree_exact_fraction": float(e_exact / energy(exact, nd)), "efree_err": float(e_model / e_exact - 1)}
    rec.update(group=fam, size=int(size), noise_model=nm, combo=combo, seconds=time.time() - tc)
    return rec


def bad(rec, thr=0.10):
    return (np.asarray(rec["err"]["shape"], float) > thr) | (np.asarray(rec["err"]["topology"]) > 0)


def same_lines(u, v):
    """Whether the true lines of two cases are the same: same number and same lengths."""
    p, q = np.asarray(u["err"]["length_ref"], float), np.asarray(v["err"]["length_ref"], float)
    return bool(len(p) == len(q) and np.allclose(p, q, rtol=1e-6, atol=1e-9, equal_nan=True))


def chain(new, old):
    return {"stored": float(np.mean(old["bad"])), "recomputed": float(np.mean(new["bad"])), "same_lines": same_lines(new, old),
            "same_labels": float(np.mean(np.asarray(new["bad"]) == np.asarray(old["bad"]))) if len(new["bad"]) == len(old["bad"]) else 0.0}


stored = pickle.load(open(STORED, "rb"))["cases"]
state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        for name, combo in CONFIGS.items():
            exact = {}
            for fam, size, j in KEYS:
                if (name, fam, size, j) in state:
                    continue
                if (fam, size) not in exact:
                    exact = {(fam, size): KNOWN[fam](size, size, _nz(size))}
                state[(name, fam, size, j)] = rec = case(fam, size, j, combo, exact[(fam, size)])
                pickle.dump(state, open(CKPT, "wb"))
                line = f"  {name} | {fam} | {size} | {rec['noise_model']}: {np.mean(rec['bad']):.1%} wrong lines, {rec['base']['iterations']} iterations"
                if name == "default":
                    q = chain(rec, stored[f"{fam} | {size} | {rec['noise_model']}"])
                    line += f"; stored {q['stored']:.3f}, same labels {q['same_labels']:.3f}, same lines {q['same_lines']}"
                    if abs(q["stored"] - q["recomputed"]) > 0.01 or not q["same_lines"]:
                        print(line, flush=True)
                        raise SystemExit("the default configuration does not reproduce the stored case: stopping before anything else is looked at")
                print(line + f" ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)", flush=True)


# ------------------------------------------------------------------ analysis
def summary(cases):
    """Wrong lines, published confidence and state of the base of a set of cases of one configuration."""
    cat = lambda f: np.concatenate([np.asarray(f(x), float) for x in cases])  # noqa: E731
    sh, tp = cat(lambda x: x["err"]["shape"]), cat(lambda x: x["err"]["topology"]) > 0
    psh, ptp = cat(lambda x: x["pot"]["pp"]["shape"]), cat(lambda x: x["pot"]["pp"]["topology"]) > 0
    dep = cat(lambda x: np.where(np.asarray(x["ind"]["departure_topo"]) > 0, np.maximum(np.nan_to_num(np.asarray(x["ind"]["departure"], float), nan=0.5), 0.5),
                                 np.asarray(x["ind"]["departure"], float)))
    X = np.column_stack([np.vstack([_design(x["ind"], CALIBRATION_MHD["features"][:5]) for x in cases]), np.log10(np.maximum(dep, 1e-3))])
    score = (X - np.array(CALIBRATION_MHD["mu"])) / np.array(CALIBRATION_MHD["sd"]) @ np.array(CALIBRATION_MHD["coef"])
    conf = cat(lambda x: confidence(x["ind"], DEFAULT_CALIBRATION))
    ok, oks, same = np.isfinite(conf), np.isfinite(score), ~tp & ~ptp & np.isfinite(sh) & np.isfinite(psh)
    acc = conf[ok] >= T
    o = {"n_lines": int(len(sh)), "nlfff_closer_than_potential": float(np.mean(sh[same] < psh[same])), "departure_median": float(np.nanmedian(dep)),
         "shape_error_median": float(np.nanmedian(sh)), "topology_changed": float(tp.mean()), "accepted": float(acc.mean())}
    for lab, thr in (("10", 0.10), ("20", 0.20)):
        y, yp = (sh > thr) | tp, (psh > thr) | ptp
        o[lab] = {"nlfff_bad": float(y.mean()), "potential_bad": float(yp.mean()), "confidence_auc": auc(1 - conf[ok], y[ok]), "confidence_ece": ece(1 - conf[ok], y[ok]),
                  "bad_accepted": float(y[ok][acc].mean()) if acc.any() else None, "ranking_auc": auc(score[oks], y[oks]), "departure_auc": auc(dep[oks], y[oks])}
    b = lambda k: np.array([x["base"][k] for x in cases], float)  # noqa: E731
    o["base"] = {"pp_shift": float(b("pp_shift").mean()), "model_shift": float(b("model_shift").mean()), "cwsin": float(b("cwsin").mean()),
                 "iterations_median": float(np.median(b("iterations"))), "iterations_max": int(b("iterations").max()), "not_converged": int((b("converged") == 0).sum()),
                 "efree_err_median": float(np.median(b("efree_err"))), "efree_abs_err_median": float(np.median(np.abs(b("efree_err")))),
                 "efree_abs_err_mean": float(np.mean(np.abs(b("efree_err")))), "efree_exact_fraction_min": float(b("efree_exact_fraction").min())}
    o["seconds"] = float(np.sum([x["seconds"] for x in cases]))
    return o


def compare(keys, n_boot=2000):
    """The default and the chosen configuration on the same cases: summaries, lines fixed and broken, and the rise of the wrong lines."""
    old, new = [state[("default",) + k] for k in keys], [state[("chosen",) + k] for k in keys]
    assert all(same_lines(u, v) for u, v in zip(old, new)), "the lines of the two configurations are not the same"
    g = {"n_cases": len(keys), "default": summary(old), "chosen": summary(new)}
    for lab, thr in (("10", 0.10), ("20", 0.20)):
        n = np.array([len(bad(u)) for u in old], float)
        d, c = np.array([bad(u, thr).sum() for u in old], float), np.array([bad(v, thr).sum() for v in new], float)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(keys), (n_boot, len(keys)))                   # whole cases are resampled: the lines of a case are not independent
        boot = 100 * (c[idx].sum(axis=1) - d[idx].sum(axis=1)) / n[idx].sum(axis=1)
        g[f"rise_points_{lab}"] = float(100 * (c.sum() - d.sum()) / n.sum())
        g[f"rise_ci_{lab}"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        g[f"lines_fixed_{lab}"] = int(sum((bad(u, thr) & ~bad(v, thr)).sum() for u, v in zip(old, new)))
        g[f"lines_broken_{lab}"] = int(sum((~bad(u, thr) & bad(v, thr)).sum() for u, v in zip(old, new)))
        g[f"cases_that_rise_{lab}"] = int((c > d).sum())
    return g


done = [k for k in KEYS if all((name,) + k in state for name in CONFIGS)]
out = {"code_version": code_version()["label"], "configurations": {n: dict(zip(("mu3", "mu4", "relax_bottom"), c)) for n, c in CONFIGS.items()}, "max_iter": MAX_ITER,
       "families": {"calibration": list(SIX_FAMILIES), "external": list(EXTERNAL)}, "sizes": list(SIZES), "n_cases": len(done), "complete": len(done) == len(KEYS),
       "chain_check": {}, "pooled": {}, "by_family": {}, "by_noise": {}, "by_size": {}, "by_family_and_noise": {}, "per_case": [], "criteria": {}}
if done:
    chk = [chain(state[("default",) + k], stored[f"{k[0]} | {k[1]} | {NOISES[k[2]][0]}"]) for k in done]
    out["chain_check"] = {"cases": len(chk), "same_lines": int(sum(q["same_lines"] for q in chk)), "same_labels_min": float(min(q["same_labels"] for q in chk)),
                          "largest_difference": float(max(abs(q["stored"] - q["recomputed"]) for q in chk))}
    sel = lambda f: [k for k in done if f(k)]  # noqa: E731
    out["pooled"] = {"calibration": compare(sel(lambda k: k[0] in SIX_FAMILIES)), "external": compare(sel(lambda k: k[0] in EXTERNAL)), "ten": compare(done)}
    out["by_family"] = {fam: compare(sel(lambda k: k[0] == fam)) for fam in FAMILIES if sel(lambda k: k[0] == fam)}
    out["by_noise"] = {nm: compare(sel(lambda k: k[2] == j)) for j, (nm, _) in enumerate(NOISES)}
    out["by_size"] = {str(s): compare(sel(lambda k: k[1] == s)) for s in SIZES if sel(lambda k: k[1] == s)}
    out["by_family_and_noise"] = {fam: {nm: compare(sel(lambda k: k[0] == fam and k[2] == j))["rise_points_10"] for j, (nm, _) in enumerate(NOISES)}
                                  for fam in out["by_family"]}
    for k in done:
        u, v = state[("default",) + k], state[("chosen",) + k]
        out["per_case"].append({"family": k[0], "size": k[1], "noise": NOISES[k[2]][0], "n_lines": int(len(bad(u))), "default": float(bad(u).mean()), "chosen": float(bad(v).mean()),
                                "default_base": u["base"], "chosen_base": v["base"]})
if out["complete"]:
    cal, ext, ten = out["pooled"]["calibration"], out["pooled"]["external"], out["pooled"]["ten"]["chosen"]
    worst = max(out["by_family"], key=lambda f: out["by_family"][f]["rise_points_10"])
    out["criteria"] = {"A1": bool(cal["rise_points_10"] <= 2 and ext["rise_points_10"] <= 2), "A2": bool(out["by_family"][worst]["rise_points_10"] <= 5),
                       "A3": bool(ten["10"]["confidence_auc"] >= 0.85 and ten["10"]["bad_accepted"] <= 0.10),
                       "worst_family": worst, "worst_family_rise_points": out["by_family"][worst]["rise_points_10"]}
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def show(name, g):
    d, c = g["default"], g["chosen"]
    f = lambda v: float("nan") if v is None else v  # noqa: E731
    print(f"{name:22s} {c['n_lines']:5d} lines | wrong {d['10']['nlfff_bad']:.3f} -> {c['10']['nlfff_bad']:.3f} ({g['rise_points_10']:+.1f} points, {g['rise_ci_10'][0]:+.1f} to "
          f"{g['rise_ci_10'][1]:+.1f}; fixed {g['lines_fixed_10']}, broken {g['lines_broken_10']}) | at 20 % {d['20']['nlfff_bad']:.3f} -> {c['20']['nlfff_bad']:.3f} | "
          f"AUC {d['10']['confidence_auc']:.3f} -> {c['10']['confidence_auc']:.3f} | accepted {d['accepted']:.3f} -> {c['accepted']:.3f}, wrong among them "
          f"{f(d['10']['bad_accepted']):.3f} -> {f(c['10']['bad_accepted']):.3f} | ECE {d['10']['confidence_ece']:.3f} -> {c['10']['confidence_ece']:.3f}")


if done:
    q = out["chain_check"]
    print(f"chain check: {q['cases']} cases, same lines in {q['same_lines']}, same labels at least {q['same_labels_min']:.3f}, largest difference of the wrong fraction {q['largest_difference']:.4f}")
    print("default -> chosen; wrong lines at the 10 % definition, the published confidence applied as is")
    for name, g in out["pooled"].items():
        show("pooled " + name, g)
    for group in ("by_family", "by_noise", "by_size"):
        for name, g in out[group].items():
            show(name, g)
    for name, g in out["pooled"].items():
        d, c = g["default"], g["chosen"]
        print(f"pooled {name}: potential {d['10']['potential_bad']:.3f} -> {c['10']['potential_bad']:.3f} | NLFFF closer than the potential {d['nlfff_closer_than_potential']:.2f} -> "
              f"{c['nlfff_closer_than_potential']:.2f} | shift of the base {d['base']['pp_shift']:.2f} / {d['base']['model_shift']:.2f} -> {c['base']['pp_shift']:.2f} / "
              f"{c['base']['model_shift']:.2f} | CWsin {d['base']['cwsin']:.3f} -> {c['base']['cwsin']:.3f} | iterations {d['base']['iterations_median']:.0f} (max "
              f"{d['base']['iterations_max']}, {d['base']['not_converged']} not converged) -> {c['base']['iterations_median']:.0f} (max {c['base']['iterations_max']}, "
              f"{c['base']['not_converged']} not converged) | free energy, median of the absolute error {d['base']['efree_abs_err_median']:.3f} -> "
              f"{c['base']['efree_abs_err_median']:.3f}, median error {d['base']['efree_err_median']:+.3f} -> {c['base']['efree_err_median']:+.3f} | departure AUC "
              f"{d['10']['departure_auc']:.3f} -> {c['10']['departure_auc']:.3f}, ranking AUC {d['10']['ranking_auc']:.3f} -> {c['10']['ranking_auc']:.3f}")
print("criteria:", out["criteria"])
print("written", a.json)
