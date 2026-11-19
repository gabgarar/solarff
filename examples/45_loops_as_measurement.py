"""What an image of loops can and cannot say about an extrapolation: a test with synthetic loops.

In an observed region the only external reference is the loops seen by AIA (``examples/08_aia_validation.py``,
``44_recommended_regions_aia.py``). Whether the angle between a line and the loops says anything about the line
being right cannot be checked there, because there is no truth. Here it is checked in a simulated active region:
a number of true field lines, seen from above, are drawn as an image of bright loops, and the angle of the lines
of several models with the ridges of that image is measured with the detector used with AIA
(``solarff.aia.ridge_orientation`` and ``line_misalignment``). The loops come from other seeds than the lines
that are tested, so nothing of the tested lines goes into the image.

Four questions, on snapshots never opened before; the criteria were written down in the plan of the paper first:

- does the per-line difference of the angle between two models, the statistic of ``examples/44``, follow the
  difference of their wrong lines?
- is it zero between two models that are equally good?
- does the angle tell the wrong lines of one model?
- does picking, for every seed, the model that is best aligned give fewer wrong lines than one model alone?

Needs the raw cubes of the snapshots (``--raw``). Saves ``data/loops_measurement.pkl`` (checkpoint) and
``docs/resultados/loops_as_measurement.json``.

    python -u examples/45_loops_as_measurement.py --raw C:/Users/.../solarff-data/muram_ar098192
    python examples/45_loops_as_measurement.py --smoke       # check on two snapshots already used; files with the suffix _smoke
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

from solarff.aia import line_misalignment, ridge_orientation
from solarff.analytic import field_magnetogram
from solarff.calibration import NOISE_MODELS
from solarff.confidence import auc, line_errors
from solarff.field import VectorField3D
from solarff.nlfff_torch import nlfff_lbfgs
from solarff.potential import potential_field
from solarff.preprocess import preprocess
from solarff.simulations import SIMULATIONS, reduced_field
from solarff.tracing import seeds_from_magnetogram, trace_field_lines
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=os.path.join("data", "muram_ar098192"))
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "loops_as_measurement.json"))
ap.add_argument("--analyze-only", action="store_true")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

SIM, R = "ar098192", 8
ORDER = [290000, 310000, 270000, 280000, 300000, 296000, 292000, 294000, 298000]        # fixes the seeds; the first six were used in examples/35 to 43
TEST = [290000, 280000] if a.smoke else [292000, 294000, 298000]
NOISES = list(NOISE_MODELS.items())
MODELS = {"recommended": ("smooth less", 0.01), "default": ("default", 0.1), "weight 0.03": ("smooth less", 0.03), "weight 0.003": ("smooth less", 0.003),
          "no preprocessing, 0.03": ("none", 0.03), "no preprocessing, 0.003": ("none", 0.003), "potential": ("smooth less", None)}     # preprocessing and weight of the base term
UP = 4                                                                          # pixels of the image per pixel of the model
IMAGES = {"60 loops": 60, "300 loops": 300}
PRIMARY = "60 loops"
CKPT = os.path.join(a.data, "loops_measurement_smoke.pkl" if a.smoke else "loops_measurement.pkl")
if a.smoke:
    a.json = a.json.replace(".json", "_smoke.json")
state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}


def truth(snap):
    path = os.path.join(a.data, f"truth_{SIM}_{snap}_r{R}.npz")
    if not os.path.exists(path):
        f = reduced_field(SIM, snap, R, a.raw)
        np.savez_compressed(path, bx=f.bx.astype(np.float32), by=f.by.astype(np.float32), bz=f.bz.astype(np.float32), dx=f.dx, origin=np.array(f.origin))
    d = np.load(path)
    return VectorField3D(d["bx"].astype(float), d["by"].astype(float), d["bz"].astype(float), float(d["dx"]), (float(d["origin"][0]), float(d["origin"][1]), 0.0))


def pixels(lines, f):
    """Image pixels (x, y) of the points of each line, seen from above."""
    return [(np.asarray(l.points)[:, :2] - np.array(f.origin[:2])) / f.dx * UP for l in lines]


def loop_image(exact, mag0, n, rng):
    """Image of n closed true lines seen from above; each is as bright as the inverse of its length, and a noisy background is added."""
    seeds = seeds_from_magnetogram(mag0, n=4 * n, method="random", polarity="both", threshold=0.05, rng=rng)
    lines = [l for l in trace_field_lines(exact, seeds) if l.closed and len(l.points) > 5][:n]
    img = np.zeros((exact.shape[1] * UP, exact.shape[0] * UP))
    for l, P in zip(lines, pixels(lines, exact)):
        seg = np.hypot(*np.diff(P, axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        si = np.arange(0, s[-1], 0.25)
        x, y = np.rint(np.interp(si, s, P[:, 0])).astype(int), np.rint(np.interp(si, s, P[:, 1])).astype(int)
        ok = (x >= 0) & (y >= 0) & (x < img.shape[1]) & (y < img.shape[0])
        length = float(np.linalg.norm(np.diff(np.asarray(l.points), axis=0), axis=1).sum())
        np.add.at(img, (y[ok], x[ok]), 1.0 / max(length, 1.0))
    img = gaussian_filter(img, 1.5)
    g = np.random.default_rng(rng + 1)
    return img + 0.05 * np.percentile(img, 99) * (1 + 0.2 * g.standard_normal(img.shape)), len(lines)


if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        for snap in TEST:
            exact = truth(snap)
            nz = exact.shape[2]
            for j, (nm, (sz, st)) in enumerate(NOISES):
                key = (snap, j)
                if key in state:
                    continue
                k = 7919 * (ORDER.index(snap) + 1) + 100 * R + j + 100000 * list(SIMULATIONS).index(SIM)
                mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=k)
                seeds = seeds_from_magnetogram(mag, n=300, method="random", polarity="both", threshold=0.05, rng=k + 7)
                le = trace_field_lines(exact, seeds)
                pps = {"smooth less": preprocess(mag, mu3=1e-1, mu4=1e-2, verbose=False), "default": preprocess(mag, verbose=False), "none": mag}
                rec, lines = {"models": {}, "images": {}, "noise_model": nm}, {}
                for name, (ppn, w) in MODELS.items():
                    f = potential_field(pps[ppn], nz) if w is None else nlfff_lbfgs(pps[ppn], nz, verbose=False, max_iter=4000, relax_bottom=w)
                    lines[name] = trace_field_lines(f, seeds)
                    e = line_errors(lines[name], le)
                    rec["models"][name] = {"shape": e["shape"], "topology": e["topology"]}
                mag0 = field_magnetogram(exact, 0.0)
                for iname, n in IMAGES.items():
                    img, n_used = loop_image(exact, mag0, n, k + 101)
                    th, strength = ridge_orientation(img, sigma=2.0)                # the detector and the threshold of examples/08
                    thr = float(np.percentile(strength, 90))
                    rec["images"][iname] = {"n_loops": n_used, "angles": {name: line_misalignment(pixels(ls, exact), th, strength, thr) for name, ls in lines.items()},
                                            "truth": line_misalignment(pixels(le, exact), th, strength, thr)}
                state[key] = rec
                pickle.dump(state, open(CKPT, "wb"))
                print(f"  {snap} | {nm}: wrong lines " + ", ".join(f"{n} {100 * ((np.asarray(e['shape'], float) > 0.1) | (np.asarray(e['topology']) > 0)).mean():.0f}" for n, e in rec["models"].items())
                      + f"  (total {(time.time() - t0) / 60:.0f} min)", flush=True)

# ------------------------------------------------------------------ analysis
keys = sorted(k for k in state if k[0] in TEST)
bad = lambda e, thr=0.10: (np.asarray(e["shape"], float) > thr) | (np.asarray(e["topology"]) > 0)  # noqa: E731
pooled = lambda f: np.concatenate([f(state[k]) for k in keys])  # noqa: E731
rng = np.random.default_rng(0)
NAMES = list(MODELS)


def paired(d, n_boot=2000):
    d = d[np.isfinite(d)]
    b = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    return {"n": int(d.size), "mean": float(d.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


out = {"code_version": code_version()["label"], "snapshots": TEST, "pixel": R, "models": {n: {"preprocessing": p, "weight": w} for n, (p, w) in MODELS.items()}, "primary_image": PRIMARY,
       "wrong_lines": {f"{thr:.2f}": {n: float(pooled(lambda r, n=n: bad(r["models"][n], thr)).mean()) for n in NAMES} for thr in (0.10, 0.20)}, "images": {}, "criteria": {}}
for iname in IMAGES:
    A = {n: pooled(lambda r, n=n: r["images"][iname]["angles"][n]) for n in NAMES}
    T = pooled(lambda r: r["images"][iname]["truth"])
    B = {n: pooled(lambda r, n=n: bad(r["models"][n])) for n in NAMES}
    B20 = {n: pooled(lambda r, n=n: bad(r["models"][n], 0.20)) for n in NAMES}
    res = {"median_angle": {"true lines": float(np.nanmedian(T)), **{n: float(np.nanmedian(A[n])) for n in NAMES}}, "measured": {"true lines": float(np.isfinite(T).mean()), **{n: float(np.isfinite(A[n]).mean()) for n in NAMES}},
           "paired": {f"{x} minus {y}": paired(A[x] - A[y]) for x, y in (("recommended", "default"), ("recommended", "weight 0.03"), ("recommended", "potential"), ("default", "potential"),
                                                                       ("recommended", "no preprocessing, 0.003"), ("recommended", "weight 0.003"))},
           "recommended_minus_default_by_case": [float(np.nanmean(state[k]["images"][iname]["angles"]["recommended"] - state[k]["images"][iname]["angles"]["default"])) for k in keys]}
    a_, m = A["recommended"], np.isfinite(A["recommended"])
    q = np.nanpercentile(a_, [25, 50, 75])
    quart = [m & (a_ <= q[0]), m & (a_ > q[0]) & (a_ <= q[1]), m & (a_ > q[1]) & (a_ <= q[2]), m & (a_ > q[2])]
    res["indicator"] = {"auc_10": auc(a_, B["recommended"]), "auc_20": auc(a_, B20["recommended"]), "auc_by_model_10": {n: auc(A[n], B[n]) for n in NAMES}, "quartile_limits_deg": [float(x) for x in q],
                        "wrong_by_quartile_10": [float(B["recommended"][x].mean()) for x in quart], "wrong_by_quartile_20": [float(B20["recommended"][x].mean()) for x in quart],
                        "wrong_best_half_10": float(B["recommended"][m & (a_ <= q[1])].mean()), "wrong_not_measured_10": float(B["recommended"][~m].mean())}
    ang = np.stack([np.where(np.isfinite(A[n]), A[n], np.inf) for n in NAMES])
    pick = np.argmin(ang, axis=0)
    none = ~np.isfinite(ang.min(axis=0))
    pick[none] = NAMES.index("recommended")                                     # without a measurement the line of the recommended model is kept
    sel, sel20 = np.choose(pick, [B[n] for n in NAMES]), np.choose(pick, [B20[n] for n in NAMES])
    case = np.concatenate([np.full(len(state[k]["models"]["recommended"]["shape"]), i) for i, k in enumerate(keys)])
    d_case = np.array([100 * (sel[case == i].mean() - B["recommended"][case == i].mean()) for i in range(len(keys))])
    boot = np.array([d_case[rng.integers(0, len(keys), len(keys))].mean() for _ in range(2000)])
    res["selection"] = {"selected_10": float(sel.mean()), "recommended_10": float(B["recommended"].mean()), "perfect_choice_10": float(np.stack([B[n] for n in NAMES]).all(0).mean()), "selected_20": float(sel20.mean()),
                        "recommended_20": float(B20["recommended"].mean()), "perfect_choice_20": float(np.stack([B20[n] for n in NAMES]).all(0).mean()), "with_a_measurement": float((~none).mean()),
                        "change_points_10": float(100 * (sel.mean() - B["recommended"].mean())), "change_ci_10": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        "cases_worse": int((d_case > 0).sum()), "n_cases": len(keys), "share": {n: float((pick == i).mean()) for i, n in enumerate(NAMES)}}
    med = {k: {n: float(np.nanmedian(state[k]["images"][iname]["angles"][n])) for n in NAMES} for k in keys}
    chosen = [min(med[k], key=med[k].get) for k in keys]
    res["whole_model"] = {"picked": {n: chosen.count(n) for n in NAMES}, "wrong_10": float(np.mean([bad(state[k]["models"][c]).mean() for k, c in zip(keys, chosen)])),
                          "recommended_10": float(np.mean([bad(state[k]["models"]["recommended"]).mean() for k in keys]))}
    out["images"][iname] = res

P, W = out["images"][PRIMARY], out["wrong_lines"]["0.10"]
C = out["criteria"]
C["H1"] = bool(P["paired"]["recommended minus default"]["ci"][1] < 0)
C["H2"] = bool(abs(P["paired"]["recommended minus weight 0.03"]["mean"]) < 0.5) if abs(W["recommended"] - W["weight 0.03"]) <= 0.02 else None
C["H3"] = bool(P["indicator"]["auc_10"] >= 0.65 and P["indicator"]["wrong_by_quartile_10"][3] >= 1.5 * P["indicator"]["wrong_best_half_10"])
C["H4"] = bool(P["selection"]["selected_10"] > P["selection"]["recommended_10"])
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"{len(keys)} cases, snapshots {TEST}; wrong lines at 10 %: " + ", ".join(f"{n} {100 * v:.1f}" for n, v in W.items()))
for iname, r in out["images"].items():
    print(f"image with {iname}{' (primary)' if iname == PRIMARY else ''}: median angle " + ", ".join(f"{n} {v:.1f}" for n, v in r["median_angle"].items()))
    for n, v in r["paired"].items():
        print(f"   {n}: {v['mean']:+.2f} deg ({v['ci'][0]:+.2f} to {v['ci'][1]:+.2f}), {v['n']} lines")
    i, s_ = r["indicator"], r["selection"]
    print(f"   angle as an indicator in the recommended model: AUC {i['auc_10']:.3f} at 10 % and {i['auc_20']:.3f} at 20 %; wrong by quartile of the angle " + ", ".join(f"{100 * v:.0f}" for v in i["wrong_by_quartile_10"])
          + f" % (limits {', '.join(f'{v:.0f}' for v in i['quartile_limits_deg'])} deg); not measured {100 * i['wrong_not_measured_10']:.0f} %")
    print(f"   the best aligned of the seven for every seed: {100 * s_['selected_10']:.1f} % wrong against {100 * s_['recommended_10']:.1f} % of the recommended alone ({s_['change_points_10']:+.1f} points, "
          f"{s_['change_ci_10'][0]:+.1f} to {s_['change_ci_10'][1]:+.1f}; worse in {s_['cases_worse']} of {s_['n_cases']} cases); a perfect choice would give {100 * s_['perfect_choice_10']:.1f} %")
    print(f"   the whole model with the smallest median angle, case by case: {100 * r['whole_model']['wrong_10']:.1f} % wrong against {100 * r['whole_model']['recommended_10']:.1f} %; picked {r['whole_model']['picked']}")
print("criteria:", C)
print("written", a.json)
