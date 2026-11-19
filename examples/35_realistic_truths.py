"""Realistic truths: does the departure of an NLFFF line from the potential field tell whether the line is wrong?

The study of ``examples/34_mhd_indicators.py`` on the MURaM simulation found that repeating the extrapolation does
not tell the wrong lines apart, and that one quantity does: how far the NLFFF line is from the potential-field line
traced from the same seed. That was found on one simulation. This script tests it on simulations that took no part
in finding it, the public Bifrost runs of the Hinode Science Data Centre Europe, built as the MURaM truth was:

- the field of a snapshot above the photosphere, averaged to the working pixel (``solarff.simulations``);
- each case as in the calibration suite: the base with noise (three noise models), the same preprocessing and code,
  300 random seeds, lines traced in the model, in the truth and in the potential field of the preprocessed base;
- per line: the published indicators, the departure from the potential line, the true error of the NLFFF line and
  the true error of the potential line, which says whether the extrapolation improves on the potential field.

The analysis applies, without refitting, a model fitted on the MURaM cases only, and the departure from the
potential line on its own, which has no fitted parameter. The hypothesis, the frozen model and the criteria were
written down in the plan of the paper before running the Bifrost cases (19-Nov-2026).

Saves ``data/realistic_<simulation>.pkl`` (checkpoints, with the per-line records), the reduced truths
``data/truth_<simulation>_<snapshot>_r<r>.npz``, the models of the cases ``data/realistic_models/`` and
``docs/resultados/realistic_truths.json``.

    python -u examples/35_realistic_truths.py --sim qs024048_by3363 --snapshots 950 900
    python examples/35_realistic_truths.py --analyze-only
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.analytic import field_magnetogram
from solarff.calibration import NOISE_MODELS, ece
from solarff.confidence import DEFAULT_CALIBRATION, _design, _fit_logistic, auc, calibration_case, confidence, line_errors
from solarff.field import VectorField3D
from solarff.helicity import energy_decomposition
from solarff.metrics import force_free_metrics
from solarff.potential import potential_field
from solarff.preprocess import preprocess
from solarff.simulations import SIMULATIONS, reduced_field
from solarff.tracing import seeds_from_magnetogram, trace_field_lines
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--sim", default=None, choices=list(SIMULATIONS), help="simulation to compute (the analysis uses every one already computed)")
ap.add_argument("--snapshots", nargs="*", type=int, default=[])
ap.add_argument("--r", nargs="*", type=int, default=[8, 4], help="pixels of the simulation averaged into one of the model")
ap.add_argument("--raw", default=None, help="folder of the cubes of the simulation (default: data/<simulation>)")
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "realistic_truths.json"))
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

T = DEFAULT_CALIBRATION["threshold"]
F5 = list(DEFAULT_CALIBRATION["features"])
TRAIN = "ar098192"                      # the simulation on which the hypothesis was found: the frozen model is fitted on it alone
MURAM_SNAPSHOTS = [290000, 310000, 270000]
assert list(SIMULATIONS)[0] == TRAIN, "the seeds of the MURaM cases must be those of examples/33"


def ckpt(sim):
    return os.path.join(a.data, f"realistic_{sim}.pkl")


def truth(sim, snap, r, raw):
    path = os.path.join(a.data, f"truth_{sim}_{snap}_r{r}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return VectorField3D(d["bx"].astype(float), d["by"].astype(float), d["bz"].astype(float), float(d["dx"]), tuple(d["origin"]),
                             {"method": f"{sim}, snapshot {snap}, r = {r}", "elapsed_s": float(d["elapsed"])})
    f = reduced_field(sim, snap, r, raw)
    np.savez_compressed(path, bx=f.bx.astype(np.float32), by=f.by.astype(np.float32), bz=f.bz.astype(np.float32), dx=f.dx,
                        origin=np.array(f.origin), elapsed=f.meta["elapsed_s"])
    return f


def seed_number(sim, si, r, j):
    """Seed of the noise of a case. For the MURaM run it is the one of ``examples/33_mhd_truth.py``, so that the cases are the same."""
    return 7919 * (si + 1) + 100 * r + j + 100000 * list(SIMULATIONS).index(sim)


def potential_errors(tr, mag, pp, seeds, nz):
    """True error of the potential-field lines: with the potential field of the preprocessed base, the one the NLFFF starts
    from and keeps on its side and top faces, and with that of the base as observed."""
    le = trace_field_lines(tr, seeds)
    out = {}
    for name, m in (("pp", pp), ("obs", mag)):
        e = line_errors(trace_field_lines(potential_field(m, nz), seeds), le)
        out[name] = {"shape": e["shape"], "topology": e["topology"], "shape_abs": e["shape_abs"]}
    return out


# ------------------------------------------------------------------ the cases of one simulation
if a.sim and not a.analyze_only:
    sim = a.sim
    raw = a.raw or os.path.join(a.data, sim)
    state = pickle.load(open(ckpt(sim), "rb")) if os.path.exists(ckpt(sim)) else {"cases": {}, "truth": {}}
    # the MURaM cases were computed by examples/34: their records are taken from there and only the potential lines are added
    prev = pickle.load(open(os.path.join(a.data, "mhd_indicators.pkl"), "rb"))["cases"] if sim == TRAIN else {}
    snaps = a.snapshots or (MURAM_SNAPSHOTS if sim == TRAIN else [])
    os.makedirs(os.path.join(a.data, "realistic_models"), exist_ok=True)
    with keep_awake():
        t0 = time.time()
        for r in a.r:
            for si, snap in enumerate(snaps):
                tr = None
                for j, (nm, (sz, st)) in enumerate(NOISE_MODELS.items()):
                    key = f"{snap} | r{r} | {nm}"
                    if key in state["cases"]:
                        continue
                    tr = truth(sim, snap, r, raw) if tr is None else tr
                    nz = tr.shape[2]
                    group = f"{snap} | r{r}"
                    if group not in state["truth"]:
                        state["truth"][group] = {"snapshot": int(snap), "r": int(r), "elapsed_s": float(tr.meta.get("elapsed_s", np.nan)),
                                                 "grid": [int(n) for n in tr.shape], "dx_Mm": float(tr.dx),
                                                 "height_over_diagonal": float((tr.shape[2] - 1) / np.hypot(tr.shape[0] - 1, tr.shape[1] - 1)),
                                                 "cwsin": float(force_free_metrics(tr)["CWsin"]), "bz_max_G": float(np.abs(tr.bz[:, :, 0]).max()),
                                                 "bz_mean_G": float(np.abs(tr.bz[:, :, 0]).mean())}
                    tc = time.time()
                    k = seed_number(sim, MURAM_SNAPSHOTS.index(snap) if sim == TRAIN else si, r, j)
                    mag = field_magnetogram(tr, noise=sz, noise_t=st, rng=k)
                    seeds = seeds_from_magnetogram(mag, n=300, method="random", polarity="both", threshold=0.05, rng=k + 7)
                    if key in prev:
                        rec = {q: prev[key][q] for q in ("ind", "err", "bad", "noise", "bad_threshold")}
                        pp = preprocess(mag, verbose=False)
                        metrics = None
                    else:
                        rec = calibration_case(tr, mag, nz, n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"max_iter": 4000}, keep_model=True)
                        model, pp = rec.pop("model"), rec.pop("pp")
                        e = line_errors(trace_field_lines(model, seeds), trace_field_lines(potential_field(pp, nz), seeds))
                        rec["ind"]["ens_pot"], rec["ind"]["ens_pot_topo"] = e["shape"], e["topology"]
                        ed = energy_decomposition(model)
                        metrics = {"cwsin": float(force_free_metrics(model, margin=int(model.meta.get("nd", 0)))["CWsin"]),
                                   "e_div_frac": float(ed["E_div_frac"]), "iterations": int(model.meta.get("iterations", 0))}
                        np.savez_compressed(os.path.join(a.data, "realistic_models", f"{sim}_{snap}_r{r}_{j}.npz"), bx=model.bx.astype(np.float32),
                                            by=model.by.astype(np.float32), bz=model.bz.astype(np.float32), dx=model.dx, origin=np.array(model.origin))
                    rec["pot"] = potential_errors(tr, mag, pp, seeds, nz)
                    rec.update(sim=sim, snapshot=int(snap), r=int(r), noise_model=nm, seed_x=np.asarray(seeds)[:, 0], x_mid=float(tr.x.mean()),
                               metrics=metrics, seconds=time.time() - tc)
                    state["cases"][key] = rec
                    pickle.dump(state, open(ckpt(sim), "wb"))
                    bp = (np.asarray(rec["pot"]["pp"]["shape"], float) > 0.1) | (np.asarray(rec["pot"]["pp"]["topology"]) > 0)
                    print(f"  {sim} | {key}: {np.mean(rec['bad']):.0%} wrong lines; potential field {bp.mean():.0%} ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)",
                          flush=True)


# ------------------------------------------------------------------ analysis, with every simulation already computed
def logd(x):
    return np.log10(np.maximum(x, 1e-3))


def displacement(ind, v="pot"):
    s = np.asarray(ind["ens_" + v], float)
    return np.where(np.asarray(ind["ens_" + v + "_topo"]) > 0, np.maximum(np.nan_to_num(s, nan=0.5), 0.5), s)


def table(d):
    """Arrays of all the lines of the cases ``d``."""
    cat = lambda f: np.concatenate([np.asarray(f(x), float) for x in d])  # noqa: E731
    sh, tp = cat(lambda x: x["err"]["shape"]), cat(lambda x: x["err"]["topology"]) > 0
    psh, ptp = cat(lambda x: x["pot"]["pp"]["shape"]), cat(lambda x: x["pot"]["pp"]["topology"]) > 0
    osh, otp = cat(lambda x: x["pot"]["obs"]["shape"]), cat(lambda x: x["pot"]["obs"]["topology"]) > 0
    X5 = np.vstack([_design(x["ind"], F5) for x in d])
    pot = logd(cat(lambda x: displacement(x["ind"])))
    return {"y10": ((sh > 0.10) | tp).astype(float), "y20": ((sh > 0.20) | tp).astype(float), "shape": sh, "topo": tp,
            "pot_y10": ((psh > 0.10) | ptp).astype(float), "pot_y20": ((psh > 0.20) | ptp).astype(float), "pot_shape": psh, "pot_topo": ptp,
            "obs_y10": ((osh > 0.10) | otp).astype(float), "obs_shape": osh, "obs_topo": otp,
            "X": np.column_stack([X5, pot]), "pot": pot, "conf": cat(lambda x: confidence(x["ind"], DEFAULT_CALIBRATION)),
            "r": np.concatenate([np.full(len(x["bad"]), x["r"]) for x in d]),
            "case": np.concatenate([np.full(len(x["bad"]), i) for i, x in enumerate(d)])}


def predict(fit, X):
    return 1 / (1 + np.exp(-((X - np.array(fit["mu"])) / np.array(fit["sd"]) @ np.array(fit["coef"]) + fit["intercept"])))


def risk(p, y, cov):
    k = max(1, int(round(cov * len(p))))
    return float(y[np.argsort(p)[:k]].mean())


def scores(p, t, lab="y10"):
    ok = np.isfinite(p) & np.isfinite(t["X"]).all(axis=1)
    y = t[lab]
    o = {"n_lines": int(ok.sum()), "bad_rate": float(y[ok].mean()), "auc": auc(p[ok], y[ok] > 0),
         **{f"auc_r{r}": auc(p[ok & (t["r"] == r)], y[ok & (t["r"] == r)] > 0) for r in np.unique(t["r"]).astype(int)},
         **{f"bad_best_{int(100 * c)}": risk(p[ok], y[ok], c) for c in (0.2, 0.36, 0.5)}}
    rng = np.random.default_rng(0)             # interval of the AUC: whole extrapolations are resampled
    ids = np.unique(t["case"][ok])
    b = []
    for _ in range(200):
        pick = np.concatenate([np.flatnonzero(ok & (t["case"] == i)) for i in rng.choice(ids, len(ids))])
        b.append(auc(p[pick], y[pick] > 0))
    o["auc_ci"] = [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
    return o


data = {s: list(pickle.load(open(ckpt(s), "rb"))["cases"].values()) for s in SIMULATIONS if os.path.exists(ckpt(s))}
truths = {s: pickle.load(open(ckpt(s), "rb"))["truth"] for s in data}
out = {"code_version": code_version()["label"], "note": "hypothesis found on the MURaM run (examples/34) and tested on the Bifrost runs; plan of the paper, 19-Nov-2026",
       "trained_on": TRAIN, "features": F5 + ["ens_pot"], "simulations": {}, "truth": truths, "cross": {}}
tabs = {s: table(d) for s, d in data.items() if d}
fits = {}
for s, t in tabs.items():
    ok = np.isfinite(t["X"]).all(axis=1)
    fits[s] = {lab: {"five_plus_pot": _fit_logistic(t["X"][ok], t[lab][ok]), "five": _fit_logistic(t["X"][ok][:, :5], t[lab][ok])} for lab in ("y10", "y20")}
for s, t in tabs.items():
    ok = np.isfinite(t["X"]).all(axis=1) & np.isfinite(t["conf"])
    same = ok & ~t["topo"] & ~t["pot_topo"]
    o = {"what": SIMULATIONS[s]["what"], "reference": SIMULATIONS[s]["reference"], "n_cases": len(data[s]), "n_lines": int(ok.sum()),
         "nlfff": {"bad_rate_10": float(t["y10"][ok].mean()), "bad_rate_20": float(t["y20"][ok].mean()), "open_closed": float(t["topo"][ok].mean()),
                   "shape_median": float(np.nanmedian(t["shape"][ok & ~t["topo"]]))},
         "potential": {"bad_rate_10": float(t["pot_y10"][ok].mean()), "bad_rate_20": float(t["pot_y20"][ok].mean()), "open_closed": float(t["pot_topo"][ok].mean()),
                       "shape_median": float(np.nanmedian(t["pot_shape"][ok & ~t["pot_topo"]])), "bad_rate_10_observed_base": float(t["obs_y10"][ok].mean())},
         "nlfff_closer_than_potential": float(np.mean(t["shape"][same] < t["pot_shape"][same])),
         "nlfff_closer_by_departure_tercile": [float(np.mean((t["shape"] < t["pot_shape"])[same & (np.digitize(t["pot"], np.quantile(t["pot"][same], [1 / 3, 2 / 3])) == i)]))
                                               for i in range(3)],
         "published_confidence": {lab: scores(1 - t["conf"], t, lab) for lab in ("y10", "y20")},
         "departure_alone": {lab: scores(t["pot"], t, lab) for lab in ("y10", "y20")}}
    acc = ok & (t["conf"] >= T)
    o["published_confidence"]["accepted"], o["published_confidence"]["bad_accepted_10"] = float(acc[ok].mean()), float(t["y10"][acc].mean()) if acc.any() else None
    if TRAIN in fits:
        for name, cols in (("five_plus_pot", slice(None)), ("five", slice(0, 5))):
            o[f"frozen_{name}"] = {lab: scores(predict(fits[TRAIN][lab][name], t["X"][:, cols]), t, lab) for lab in ("y10", "y20")}
            if s == TRAIN:
                o[f"frozen_{name}"]["note"] = "in sample: this is the simulation the model was fitted on"
    out["simulations"][s] = o
# every simulation as the training set of every other one
for s in tabs:
    for u in tabs:
        if s != u:
            out["cross"][f"{s} -> {u}"] = {lab: scores(predict(fits[s][lab]["five_plus_pot"], tabs[u]["X"]), tabs[u], lab) for lab in ("y10", "y20")}
if TRAIN in fits:
    out["frozen_model"] = {lab: {"five_plus_pot": fits[TRAIN][lab]["five_plus_pot"]} for lab in ("y10", "y20")}
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

for s, o in out["simulations"].items():
    print(f"{s} ({o['what']}): {o['n_cases']} cases, {o['n_lines']} lines")
    n, p = o["nlfff"], o["potential"]
    print(f"  wrong lines (10 % / 20 %): NLFFF {n['bad_rate_10']:.2f} / {n['bad_rate_20']:.2f}, potential field {p['bad_rate_10']:.2f} / {p['bad_rate_20']:.2f} "
          f"(of the observed base {p['bad_rate_10_observed_base']:.2f}) | median shape error {n['shape_median']:.3f} against {p['shape_median']:.3f} | "
          f"the NLFFF line is closer to the truth in {o['nlfff_closer_than_potential']:.2f} of the lines; by tercile of departure from the potential line: "
          + ", ".join(f"{v:.2f}" for v in o["nlfff_closer_by_departure_tercile"]))
    for name in ("published_confidence", "departure_alone", "frozen_five", "frozen_five_plus_pot"):
        if name in o:
            print(f"  {name:22s} " + " | ".join(f"{lab}: AUC {q['auc']:.3f} {[round(v, 3) for v in q['auc_ci']]}, wrong in the best 20/36/50 %: "
                                               f"{q['bad_best_20']:.2f}/{q['bad_best_36']:.2f}/{q['bad_best_50']:.2f} of {q['bad_rate']:.2f}" for lab, q in o[name].items() if isinstance(q, dict)))
for k, q in out["cross"].items():
    print(f"trained on one, tested on the other, {k}: " + " | ".join(f"{lab}: AUC {v['auc']:.3f}, best 36 %: {v['bad_best_36']:.2f} of {v['bad_rate']:.2f}" for lab, v in q.items()))
print("written", a.json)
