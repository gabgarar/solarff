"""Realistic truths with the base in the chromosphere: is the photosphere what breaks the confidence?

Against radiative MHD simulations the confidence calibrated on exact force-free fields does not tell the wrong lines
apart (``examples/33_mhd_truth.py``, ``35_realistic_truths.py``). The base of those tests is the photosphere, which
is not force-free. Here the same simulations, snapshots, pixels, noise models and code are used with the base at a
height in the chromosphere, 1.5 Mm by default, where the field is closer to force-free: the truth is the field
above that height and the extrapolation starts from its lowest plane.

Nothing is refitted. Per simulation: the wrong lines of the NLFFF and of the potential field, the published
confidence, the departure from the potential line on its own, and the ranking ``CALIBRATION_MHD``, which was fitted
on the cases with the photospheric base. The criteria were written down in the plan of the paper before running
this (19-Nov-2026). The wrong lines by noise model and by pixel, and their comparison with the photospheric base
(the cases of ``examples/35_realistic_truths.py``, when their files are in the data folder), were added after
seeing the result.

Saves ``data/chromo_<simulation>_z<height>.pkl`` (checkpoints), the reduced truths
``data/truth_<simulation>_<snapshot>_r<r>_z<height>.npz`` and ``docs/resultados/chromospheric_base.json``.

    python -u examples/39_chromospheric_base.py --raw-root C:/Users/.../solarff-data
    python examples/39_chromospheric_base.py --analyze-only
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
from solarff.confidence import CALIBRATION_MHD, DEFAULT_CALIBRATION, _design, auc, calibration_case, confidence
from solarff.field import VectorField3D
from solarff.metrics import force_free_metrics
from solarff.simulations import SIMULATIONS, reduced_field
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
RUNS = {"ar098192": ("muram_ar098192", [290000, 310000, 270000]), "qs024048_by3363": ("qs024048_by3363", [950, 900]), "en024048_hion": ("en024048_hion", [385])}
ap = argparse.ArgumentParser()
ap.add_argument("--z0", type=float, default=1.5, help="height of the base in Mm")
ap.add_argument("--sims", nargs="*", default=list(RUNS), choices=list(RUNS))
ap.add_argument("--r", nargs="*", type=int, default=[8, 4])
ap.add_argument("--raw-root", default="data", help="folder with one folder of cubes per simulation")
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "chromospheric_base.json"))
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()
TAG = f"z{a.z0:g}"
T = DEFAULT_CALIBRATION["threshold"]


def ckpt(sim):
    return os.path.join(a.data, f"chromo_{sim}_{TAG}.pkl")


def truth(sim, snap, r):
    path = os.path.join(a.data, f"truth_{sim}_{snap}_r{r}_{TAG}.npz")
    if not os.path.exists(path):
        f = reduced_field(sim, snap, r, os.path.join(a.raw_root, RUNS[sim][0]), z0=a.z0)
        np.savez_compressed(path, bx=f.bx.astype(np.float32), by=f.by.astype(np.float32), bz=f.bz.astype(np.float32), dx=f.dx, origin=np.array(f.origin))
    d = np.load(path)
    # heights are counted from the base, as the extrapolation does; the height of the base in the simulation is kept in meta
    return VectorField3D(d["bx"].astype(float), d["by"].astype(float), d["bz"].astype(float), float(d["dx"]), (float(d["origin"][0]), float(d["origin"][1]), 0.0),
                         {"base_height_Mm": float(d["origin"][2])})


if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        for sim in a.sims:
            state = pickle.load(open(ckpt(sim), "rb")) if os.path.exists(ckpt(sim)) else {"cases": {}, "truth": {}}
            for r in a.r:
                for si, snap in enumerate(RUNS[sim][1]):
                    tr = None
                    for j, (nm, (sz, st)) in enumerate(NOISE_MODELS.items()):
                        key = f"{snap} | r{r} | {nm}"
                        if key in state["cases"]:
                            continue
                        tr = truth(sim, snap, r) if tr is None else tr
                        if f"{snap} | r{r}" not in state["truth"]:
                            state["truth"][f"{snap} | r{r}"] = {"grid": [int(n) for n in tr.shape], "dx_Mm": float(tr.dx), "base_height_Mm": float(tr.meta["base_height_Mm"]),
                                                                "cwsin": float(force_free_metrics(tr)["CWsin"]), "bz_mean_G": float(np.abs(tr.bz[:, :, 0]).mean()),
                                                                "bz_max_G": float(np.abs(tr.bz[:, :, 0]).max())}
                        tc = time.time()
                        k = 7919 * (si + 1) + 100 * r + j + 100000 * list(SIMULATIONS).index(sim)        # the noise seeds of examples/35
                        mag = field_magnetogram(tr, noise=sz, noise_t=st, rng=k)
                        rec = calibration_case(tr, mag, tr.shape[2], n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"max_iter": 4000}, potential=True)
                        rec.update(sim=sim, snapshot=int(snap), r=int(r), noise_model=nm, seconds=time.time() - tc)
                        state["cases"][key] = rec
                        pickle.dump(state, open(ckpt(sim), "wb"))
                        print(f"  {sim} | {key}: {np.mean(rec['bad']):.0%} wrong lines ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)", flush=True)


# ------------------------------------------------------------------ analysis
def errors(cases):
    """Errors of each line of the NLFFF and of the potential field, with the noise model, the pixel and the case it comes from."""
    cat = lambda f: np.concatenate([np.asarray(f(x), float) for x in cases])  # noqa: E731
    tag = lambda f: np.concatenate([np.full(len(x["bad"]), f(x)) for x in cases])  # noqa: E731
    return {"sh": cat(lambda x: x["err"]["shape"]), "tp": cat(lambda x: x["err"]["topology"]) > 0, "psh": cat(lambda x: x["pot"]["pp"]["shape"]),
            "ptp": cat(lambda x: x["pot"]["pp"]["topology"]) > 0, "noise": tag(lambda x: x["noise_model"]), "r": tag(lambda x: int(x["r"])),
            "case": np.concatenate([np.full(len(x["bad"]), i) for i, x in enumerate(cases)])}


def table(cases):
    cat = lambda f: np.concatenate([np.asarray(f(x), float) for x in cases])  # noqa: E731
    t = errors(cases)
    dep = cat(lambda x: np.where(np.asarray(x["ind"]["departure_topo"]) > 0, np.maximum(np.nan_to_num(np.asarray(x["ind"]["departure"], float), nan=0.5), 0.5),
                                 np.asarray(x["ind"]["departure"], float)))
    X = np.column_stack([np.vstack([_design(x["ind"], CALIBRATION_MHD["features"][:5]) for x in cases]), np.log10(np.maximum(dep, 1e-3))])
    score = (X - np.array(CALIBRATION_MHD["mu"])) / np.array(CALIBRATION_MHD["sd"]) @ np.array(CALIBRATION_MHD["coef"])
    t.update(dep=dep, score=score, conf=cat(lambda x: confidence(x["ind"], DEFAULT_CALIBRATION)))
    ok = np.isfinite(score) & np.isfinite(t["conf"]) & np.isfinite(t["psh"])
    return {k: v[ok] for k, v in t.items()}


def rates(t, sel=slice(None)):
    """Wrong lines of the NLFFF and of the potential field at the two definitions."""
    return {"n_lines": int(len(t["sh"][sel])), **{lab: {"nlfff_bad": float(((t["sh"] > thr) | t["tp"])[sel].mean()), "potential_bad": float(((t["psh"] > thr) | t["ptp"])[sel].mean())}
                                                  for lab, thr in (("10", 0.10), ("20", 0.20))}}


def split(t):
    return {name: {str(v): rates(t, t[name] == v) for v in dict.fromkeys(t[name].tolist())} for name in ("noise", "r")}


def auc_ci(p, y, case):
    rng, b = np.random.default_rng(0), []
    ids = np.unique(case)
    for _ in range(200):                                # whole extrapolations are resampled
        pick = np.concatenate([np.flatnonzero(case == i) for i in rng.choice(ids, len(ids))])
        b.append(auc(p[pick], y[pick]))
    return [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]


out = {"code_version": code_version()["label"], "base_height_Mm": a.z0, "simulations": {}, "truth": {}}
for sim in RUNS:
    if not os.path.exists(ckpt(sim)):
        continue
    state = pickle.load(open(ckpt(sim), "rb"))
    t = table(list(state["cases"].values()))
    out["truth"][sim] = state["truth"]
    o = {"n_cases": len(state["cases"]), "n_lines": int(len(t["sh"])), "departure_median": float(np.median(t["dep"]))}
    same = ~t["tp"] & ~t["ptp"]
    o["nlfff_closer_than_potential"] = float(np.mean(t["sh"][same] < t["psh"][same]))
    acc = t["conf"] >= T
    for lab, thr in (("10", 0.10), ("20", 0.20)):
        y, yp = (t["sh"] > thr) | t["tp"], (t["psh"] > thr) | t["ptp"]
        o[lab] = {"nlfff_bad": float(y.mean()), "potential_bad": float(yp.mean()),
                  "published_confidence": {"auc": auc(1 - t["conf"], y), "auc_ci": auc_ci(1 - t["conf"], y, t["case"]), "accepted": float(acc.mean()),
                                           "bad_accepted": float(y[acc].mean()) if acc.any() else None},
                  "departure_alone": {"auc": auc(t["dep"], y)}, "mhd_ranking": {"auc": auc(t["score"], y), "bad_best_36": float(y[np.argsort(t["score"])[:max(1, int(round(0.36 * len(y))))]].mean())}}
    # added after seeing the result: the wrong lines by noise model and by pixel, and the same for the photospheric base
    o["split"] = split(t)
    photo = os.path.join(a.data, f"realistic_{sim}.pkl")                    # the cases of examples/35_realistic_truths.py
    if os.path.exists(photo):
        tp = errors(list(pickle.load(open(photo, "rb"))["cases"].values()))
        tp = {k: v[np.isfinite(tp["psh"])] for k, v in tp.items()}
        o["photospheric_base"] = {**rates(tp), "split": split(tp)}
    out["simulations"][sim] = o
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

for sim, o in out["simulations"].items():
    tr = list(out["truth"][sim].values())
    print(f"{sim}: {o['n_cases']} cases, {o['n_lines']} lines; base at {tr[0]['base_height_Mm']:.2f} Mm, CWsin of the truth {min(q['cwsin'] for q in tr):.2f} to {max(q['cwsin'] for q in tr):.2f}, "
          f"mean |Bz| {min(q['bz_mean_G'] for q in tr):.0f} to {max(q['bz_mean_G'] for q in tr):.0f} G; NLFFF closer than the potential field in {o['nlfff_closer_than_potential']:.2f}; "
          f"median departure {o['departure_median']:.3f}")
    for lab in ("10", "20"):
        q = o[lab]
        pc = q["published_confidence"]
        print(f"  {lab} %: wrong NLFFF {q['nlfff_bad']:.3f}, potential {q['potential_bad']:.3f} | published confidence AUC {pc['auc']:.3f} {[round(v, 3) for v in pc['auc_ci']]}, "
              f"accepted {pc['accepted']:.2f} with {pc['bad_accepted'] if pc['bad_accepted'] is not None else float('nan'):.3f} wrong | departure alone AUC {q['departure_alone']['auc']:.3f} | "
              f"MHD ranking AUC {q['mhd_ranking']['auc']:.3f}, wrong in its best 36 %: {q['mhd_ranking']['bad_best_36']:.3f}")
    ph = o.get("photospheric_base")
    for name in ("noise", "r"):
        for v, q in o["split"][name].items():
            p = ph["split"][name].get(v) if ph else None
            print(f"  {name} {v}: wrong NLFFF / potential at 10 % {q['10']['nlfff_bad']:.3f} / {q['10']['potential_bad']:.3f}, at 20 % {q['20']['nlfff_bad']:.3f} / {q['20']['potential_bad']:.3f}"
                  + (f" | photospheric base {p['10']['nlfff_bad']:.3f} / {p['10']['potential_bad']:.3f}, {p['20']['nlfff_bad']:.3f} / {p['20']['potential_bad']:.3f}" if p else ""))
    if ph:
        print(f"  photospheric base, {ph['n_lines']} lines: wrong NLFFF / potential at 10 % {ph['10']['nlfff_bad']:.3f} / {ph['10']['potential_bad']:.3f}, "
              f"at 20 % {ph['20']['nlfff_bad']:.3f} / {ph['20']['potential_bad']:.3f}")
print("written", a.json)
