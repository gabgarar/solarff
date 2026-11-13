"""A region model as the truth: does the confidence hold in a field with the localised currents of a real region?

The exact fields of the suite are smooth; those built on the observed flux (``examples/25_observed_flux_fields.py``)
have the flux of a region but a constant alpha. Neither has the localised currents of a real region. A field that
has them is the published extrapolation of each region. It is taken here as the truth of a twin experiment:

- the published NLFFF model of NOAA 11158, 11429 and 12673 is cropped (10 % of each side and the upper third are
  removed), so that the side and top faces of the truth are not the potential field the extrapolation assumes;
- its base, with noise, is extrapolated in the cropped box as in the suite (300 random seeds, three noise models),
  with the ``coarse`` variant to measure how many lines half the resolution does not reproduce;
- the published coefficients are applied without refitting, with the length of the line relative to the horizontal
  diagonal (published definition) and relative to the height of the box (``examples/26_box_height_length.py``).

What this test is not: the truth is not an exact force-free field (it keeps the residual force and divergence of
the published model) and it comes out of the same code, which favours the extrapolation. It is a consistency
experiment with realistic complexity, not a test against an exact solution.

The criteria (T1 to T3) were registered in the plan of the paper before running (13-Nov-2026).

Saves ``data/model_as_truth.pkl`` (checkpoint, with the per-line records) and ``docs/resultados/model_as_truth.json``.

    python -u examples/28_model_as_truth.py                 # ~10 min
    python -u examples/28_model_as_truth.py --set post      # the post-flare models as the truth (files with the suffix _post)
    python examples/28_model_as_truth.py --analyze-only
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.analytic import field_magnetogram
from solarff.calibration import NOISE_MODELS, _nz, ece
from solarff.confidence import DEFAULT_CALIBRATION, auc, calibration_case, confidence
from solarff.field import VectorField3D
from solarff.magnetogram import aly_criteria
from solarff.metrics import force_free_metrics
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--regions", nargs="*", default=["11429", "11158", "12673"])
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=None)
ap.add_argument("--set", choices=("pre", "post", "first"), default="pre",
                help="which published model of each region is the truth: just before the flare (the registered test), "
                     "after it, or the first of the earlier times")
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

SETS = {"pre": {"11158": "harp377_study_mc10.pkl", "11429": "harp1449_20120306_2336_r7_nz32_mc10.pkl",
                "12673": "harp7115_20170906_1136_r4_nz56_mc10.pkl"},
        "post": {"11158": "harp377_20110215_0300_r4_nz48_mc10.pkl", "11429": "harp1449_20120307_0200_r7_nz32_mc10.pkl",
                 "12673": "harp7115_20170906_1300_r4_nz56_mc10.pkl"},
        "first": {"11158": "harp377_20110214_2100_r4_nz48_mc0.pkl", "11429": "harp1449_20120306_2036_r7_nz32_mc0.pkl",
                  "12673": "harp7115_20170906_0948_r4_nz56_mc0.pkl"}}
REGIONS = SETS[a.set]
TAG = "" if a.set == "pre" else "_" + a.set
a.json = a.json or os.path.join("docs", "resultados", f"model_as_truth{TAG}.json")
MARGIN, HEIGHT = 0.10, 2.0 / 3.0          # fraction removed on each side; fraction of the height kept
T = DEFAULT_CALIBRATION["threshold"]
CKPT = os.path.join(a.data, f"model_as_truth{TAG}.pkl")
N_BOOT = 1000


def ratio(nx, ny, nz):
    return (nz - 1) / np.hypot(nx - 1, ny - 1)


C0 = float(np.mean([ratio(n, n, _nz(n)) for n in (32, 48, 64)]))


def crop(f):
    nx, ny, nz = f.shape
    mx, my, kz = int(round(MARGIN * nx)), int(round(MARGIN * ny)), int(round(HEIGHT * nz))
    sl = (slice(mx, nx - mx), slice(my, ny - my), slice(0, kz))
    origin = (f.origin[0] + mx * f.dx, f.origin[1] + my * f.dx, f.origin[2])
    return VectorField3D(f.bx[sl].copy(), f.by[sl].copy(), f.bz[sl].copy(), f.dx, origin, {"method": "published NLFFF model, cropped"})


def evaluate(d, grid, rule):
    factor = 1.0 if rule == "published" else C0 / ratio(*grid)
    cs = []
    for x in d:
        ind = dict(x["ind"])
        ind["length"] = np.asarray(ind["length"], float) * factor
        cs.append(confidence(ind, DEFAULT_CALIBRATION))
    c = np.concatenate(cs)
    y = np.concatenate([x["bad"] for x in d]).astype(int)
    ok = np.isfinite(c)
    acc = ok & (np.nan_to_num(c, nan=0.0) >= T)
    p = 1 - c[ok]
    out = {"n_lines": int(ok.sum()), "bad_rate": float(y[ok].mean()), "auc": auc(p, y[ok]), "ece": ece(p, y[ok]),
           "accepted": float(acc[ok].mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None}
    if acc.any():
        rng = np.random.default_rng(0)
        ya = y[acc]
        b = [ya[rng.integers(0, ya.size, ya.size)].mean() for _ in range(N_BOOT)]
        out["bad_accepted_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    m = ok & (np.nan_to_num(c, nan=0.0) >= 0.9)
    out["accepted_0.9"], out["bad_accepted_0.9"] = float(m[ok].mean()), (float(y[m].mean()) if m.any() else None)
    sh = np.concatenate([np.asarray(x["ind"]["ens_coarse"], float) for x in d])
    topo = np.concatenate([np.asarray(x["ind"]["ens_coarse_topo"]) for x in d]) > 0
    val = np.isfinite(sh)
    with np.errstate(invalid="ignore"):
        moved = (sh > 0.1) | topo
    out["not_reproduced_10"] = float(moved[val].mean())
    out["not_reproduced_10_accepted"] = float(moved[val & acc].mean()) if (val & acc).any() else None
    # size of the errors of the accepted lines
    shape = np.concatenate([np.asarray(x["err"]["shape"], float) for x in d])
    tp = np.concatenate([np.asarray(x["err"]["topology"], float) for x in d]) > 0
    if acc.any():
        out["accepted_wrong_at_20"] = float(((shape > 0.2) | tp)[acc].mean())
        out["accepted_open_closed"] = float(tp[acc].mean())
    return out


state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {"cases": {}, "truth": {}}
if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        for noaa in a.regions:
            study = pickle.load(open(os.path.join(a.data, REGIONS[noaa]), "rb"))
            study = study[0] if isinstance(study, tuple) else study
            full = study.nlfff
            truth = crop(full)
            del study
            nz = truth.shape[2]
            if noaa not in state["truth"]:
                ff = force_free_metrics(truth)
                state["truth"][noaa] = {"grid_published": [int(n) for n in full.shape], "grid": [int(n) for n in truth.shape],
                                        "dx_Mm": float(truth.dx), "height_over_diagonal": float(ratio(*truth.shape)),
                                        "cwsin": float(ff["CWsin"]),
                                        "eps_force": float(aly_criteria(field_magnetogram(truth, 0.0, 0.0))["eps_force"])}
            for j, (nm, (sz, st)) in enumerate(NOISE_MODELS.items()):
                key = f"{noaa} | {nm}"
                if key in state["cases"]:
                    continue
                tc = time.time()
                k = 104729 * (list(REGIONS).index(noaa) + 1) + j + 1000 * list(SETS).index(a.set)
                mag = field_magnetogram(truth, noise=sz, noise_t=st, rng=k)
                rec = calibration_case(truth, mag, nz, n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"max_iter": 4000},
                                       ensemble=("coarse",))
                rec.update(region=noaa, noise_model=nm, seconds=time.time() - tc)
                state["cases"][key] = rec
                pickle.dump(state, open(CKPT, "wb"))
                print(f"  {key}: {rec['bad'].mean():.0%} wrong lines ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)", flush=True)

out = {"code_version": code_version()["label"], "calibration": "DEFAULT_CALIBRATION, coefficients not refitted", "c0": C0,
       "margin": MARGIN, "height_kept": HEIGHT, "noise_models": list(NOISE_MODELS), "truth": state["truth"], "regions": {}, "pooled": {}}
for noaa in state["truth"]:
    d = [x for k, x in state["cases"].items() if k.startswith(noaa + " |")]
    if d:
        out["regions"][noaa] = {rule: evaluate(d, state["truth"][noaa]["grid"], rule) for rule in ("published", "height")}
        out["regions"][noaa]["by_noise"] = {x["noise_model"]: {rule: evaluate([x], state["truth"][noaa]["grid"], rule) for rule in ("published", "height")}
                                            for x in d}
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def fmt(v, f=".3f"):
    return "-" if v is None else format(v, f)


for noaa, r in out["regions"].items():
    e = out["truth"][noaa]
    print(f"NOAA {noaa}: truth {e['grid']} (published {e['grid_published']}), height / diagonal {e['height_over_diagonal']:.3f}, "
          f"CWsin {e['cwsin']:.2f}, force residual of its base {e['eps_force']:.2f}")
    for rule in ("published", "height"):
        q = r[rule]
        print(f"  {rule:9s} wrong {q['bad_rate']:.2f} | AUC {q['auc']:.3f} | ECE {q['ece']:.3f} | accepted {q['accepted']:.2f}, wrong among them "
              f"{fmt(q['bad_accepted'])} {[round(v, 3) for v in q.get('bad_accepted_ci', [])]} (0.9: {fmt(q['bad_accepted_0.9'])}, accepted {q['accepted_0.9']:.2f}) | "
              f"at the 20 % criterion {fmt(q.get('accepted_wrong_at_20'))}, open/closed {fmt(q.get('accepted_open_closed'))} | "
              f"not reproduced at half resolution: all {q['not_reproduced_10']:.2f}, accepted {fmt(q['not_reproduced_10_accepted'], '.2f')}")
print("written", a.json)
