"""Does the published confidence survive systematic (non-random) errors in the magnetogram?

The calibration suite adds only random noise to bases that are exactly force-free. Real vector
magnetograms also carry systematic errors that break the force-free consistency of the base. Here the
base of each of the six calibration families is contaminated coherently, the same HMI-like noise as in
the suite (10 G in B_z, 60 G in B_t) is added, the field is extrapolated with the same pipeline and the
published model (``DEFAULT_CALIBRATION``) is applied **without refitting**. The true error of each line
is measured against the exact field, as in the suite.

Perturbations (the control uses the same noise seeds, so the difference is the perturbation alone):

* ``control``      — none (reproduces the suite conditions);
* ``bt_x0.7``      — transverse field scaled by 0.7 (underestimated B_t, e.g. unresolved fields);
* ``bt_x1.3``      — transverse field scaled by 1.3;
* ``azimuth_10``   — transverse field rotated by +10° everywhere (systematic azimuth error; also 3°, 5°, −5°);
* ``bt_height``    — transverse field taken one cell above the base (B_t and B_z from different heights).

For each perturbation: fraction of wrong lines, AUC, ECE, fraction accepted and wrong lines among the
accepted (≥ 0.8), over the 18 cases (6 families × 3 grids) and per family, plus the Aly force and torque
residuals of the contaminated magnetogram and the relative change of the base by the preprocessing.
The predictions were registered before running.

Saves ``data/systematic_base.pkl`` (checkpoint) and ``docs/resultados/systematic_base.json``.

    python -u examples/20_systematic_base.py        # ~1 h
"""
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solarff.analytic import CALIBRATION_FIELDS, EXTERNAL_TEST_FIELDS, field_magnetogram  # noqa: E402
from solarff.calibration import add_box, NOISE_MODELS, _nz, bootstrap_ci, ece  # noqa: E402
from solarff.confidence import CALIBRATION_V1, DEFAULT_CALIBRATION, auc, calibration_case, confidence  # noqa: E402
from solarff.magnetogram import aly_criteria  # noqa: E402
from solarff.preprocess import preprocess  # noqa: E402
from solarff.utils import code_version, keep_awake  # noqa: E402

warnings.filterwarnings("ignore")
OUT_PKL = os.path.join("data", "systematic_base.pkl")
OUT_JSON = os.path.join("docs", "resultados", "systematic_base.json")
FAMILIES = list(CALIBRATION_FIELDS) + ["TD-relax"]
KNOWN = {**CALIBRATION_FIELDS, **EXTERNAL_TEST_FIELDS}
SIZES = (32, 48, 64)
NOISE = "HMI (10/60 G)"
PERTURBATIONS = ["control", "bt_x0.7", "bt_x1.3", "azimuth_10", "bt_height"]
# Added after seeing that the 10° rotation breaks the confidence, to locate where it starts (not preregistered)
PERTURBATIONS += ["azimuth_3", "azimuth_5"]
# Registered in the plan before running: the opposite sense of rotation
PERTURBATIONS += ["azimuth_-5"]


def contaminate(mag, exact, kind):
    """Apply the systematic error ``kind`` to the transverse field of ``mag`` (noise already added)."""
    bx, by = mag.bx.copy(), mag.by.copy()
    if kind == "bt_x0.7":
        bx, by = 0.7 * bx, 0.7 * by
    elif kind == "bt_x1.3":
        bx, by = 1.3 * bx, 1.3 * by
    elif kind.startswith("azimuth_"):
        deg = float(kind.split("_")[1])
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        bx, by = c * bx - s * by, s * bx + c * by
    elif kind == "bt_height":  # exact transverse field one cell higher, with the same noise realization
        bx = bx - exact.bx[:, :, 0] + exact.bx[:, :, 1]
        by = by - exact.by[:, :, 0] + exact.by[:, :, 1]
    elif kind != "control":
        raise ValueError(kind)
    mag.bx, mag.by = bx, by
    return mag


def rel_change(a, b):
    return float(np.sqrt(sum(((x - y) ** 2).sum() for x, y in zip(a, b)) / sum((y ** 2).sum() for y in b)))


def evaluate(d, cal=DEFAULT_CALIBRATION):
    c = np.concatenate([confidence(x["ind"], cal) for x in d])
    y = np.concatenate([x["bad"] for x in d]).astype(int)
    g = np.concatenate([np.full(len(x["bad"]), i) for i, x in enumerate(d)])
    ok = np.isfinite(c)
    c, y, g = c[ok], y[ok], g[ok]
    p = 1 - c
    acc = c >= cal["threshold"]
    return {"n_cases": int(len(d)), "n_lines": int(y.size), "bad_rate": float(y.mean()), "auc": auc(p, y),
            "auc_ci": list(bootstrap_ci(p, y, auc, groups=g, n_boot=500)), "ece": ece(p, y),
            "accepted": float(acc.mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None,
            "mean_predicted_bad": float(p.mean())}


cases = [(f, n, k) for k in PERTURBATIONS for f in FAMILIES for n in SIZES]
data = add_box(pickle.load(open(OUT_PKL, "rb"))) if os.path.exists(OUT_PKL) else []   # with length_h, also the cases saved without it
done = {(d["field"], d["size"], d["perturbation"]) for d in data}
sz, st = NOISE_MODELS[NOISE]
cache = {}
t0 = time.time()
with keep_awake():
    for i, (fname, n, kind) in enumerate(cases):
        if (fname, n, kind) in done:
            continue
        tc = time.time()
        if (fname, n) not in cache:
            cache = {(fname, n): KNOWN[fname](n, n, _nz(n))}
        exact = cache[(fname, n)]
        seed = 7919 * (FAMILIES.index(fname) * 10 + SIZES.index(n)) + 11   # same noise for every perturbation
        clean = field_magnetogram(exact, noise=sz, noise_t=st, rng=seed)
        ref = (clean.bx.copy(), clean.by.copy(), clean.bz.copy())
        mag = contaminate(clean, exact, kind)
        rec = calibration_case(exact, mag, _nz(n), n_seeds=300, seed=seed + 7, noise_bt=st)
        pp = preprocess(mag, verbose=False)
        aly = aly_criteria(mag)
        rec.update(field=fname, size=n, perturbation=kind, noise_model=NOISE, seconds=time.time() - tc,
                   eps_force=float(aly["eps_force"]), eps_torque=float(aly["eps_torque"]),
                   base_change=rel_change((mag.bx, mag.by, mag.bz), ref),
                   pp_change=rel_change((pp.bx, pp.by, pp.bz), (mag.bx, mag.by, mag.bz)))
        data.append(rec)
        pickle.dump(data, open(OUT_PKL, "wb"))
        print(f"caso {i + 1}/{len(cases)}: {fname}, {n}², {kind}: {rec['bad'].mean():.0%} malas "
              f"({rec['seconds']:.0f}s; total {(time.time() - t0) / 60:.0f} min)", flush=True)

out = {"code_version": code_version()["label"], "calibration": "DEFAULT_CALIBRATION (three indicators in five terms, six families in boxes of four proportions), not refitted", "noise_model": NOISE,
       "sizes": list(SIZES), "families": FAMILIES, "perturbations": {}}
for kind in PERTURBATIONS:
    d = [x for x in data if x["perturbation"] == kind]
    r = evaluate(d)
    r["by_family"] = {f: {k: v for k, v in evaluate([x for x in d if x["field"] == f]).items() if k != "auc_ci"} for f in FAMILIES}
    for k in ("eps_force", "eps_torque", "base_change", "pp_change"):
        v = [x[k] for x in d]
        r[k] = {"min": float(min(v)), "median": float(np.median(v)), "max": float(max(v))}
    def _bad_acc(x):
        c = confidence(x["ind"], DEFAULT_CALIBRATION)
        acc = np.isfinite(c) & (c >= DEFAULT_CALIBRATION["threshold"])
        return float(np.asarray(x["bad"])[acc].mean()) if acc.any() else 0.0
    r["cases_optimistic"] = int(sum(_bad_acc(x) >= 0.10 for x in d))
    r["v1"] = {k: v for k, v in evaluate(d, CALIBRATION_V1).items() if k in ("auc", "ece", "accepted", "bad_accepted")}   # the model of version 1.1
    out["perturbations"][kind] = r
    print(f"{kind:11s} malas {r['bad_rate']:.0%} | AUC {r['auc']:.3f} ECE {r['ece']:.3f} | ≥ 0.8: {r['accepted']:.0%} aceptadas, "
          f"{r['bad_accepted']:.1%} malas | casos optimistas {r['cases_optimistic']}/18 | eps_force {r['eps_force']['median']:.3f} "
          f"| versión 1.1: AUC {r['v1']['auc']:.3f}, {r['v1']['bad_accepted']:.1%} malas")

# How much of a scale error of the transverse field survives the preprocessing: relative difference of the
# transverse field between the scaled base and the control, before (30 % by construction) and after preprocessing.
def bt_diff(m, ref):
    return rel_change((m.bx, m.by), (ref.bx, ref.by))


scale = {"bt_x0.7": {"before": [], "after": []}, "bt_x1.3": {"before": [], "after": []}}
for fname in FAMILIES:
    for n in SIZES:
        exact = KNOWN[fname](n, n, _nz(n))
        seed = 7919 * (FAMILIES.index(fname) * 10 + SIZES.index(n)) + 11
        ctrl = field_magnetogram(exact, noise=sz, noise_t=st, rng=seed)
        pp_ctrl = preprocess(ctrl, verbose=False)
        for kind in scale:
            m = contaminate(field_magnetogram(exact, noise=sz, noise_t=st, rng=seed), exact, kind)
            scale[kind]["before"].append(bt_diff(m, ctrl))
            scale[kind]["after"].append(bt_diff(preprocess(m, verbose=False), pp_ctrl))
out["scale_after_preprocess"] = {
    k: {w: {"min": float(min(v[w])), "median": float(np.median(v[w])), "max": float(max(v[w]))} for w in v}
    for k, v in scale.items()}
for k, v in out["scale_after_preprocess"].items():
    print(f"{k}: B_t difference from control {v['before']['median']:.0%} before, "
          f"{v['after']['min']:.0%}-{v['after']['max']:.0%} after preprocessing")

# Reference for the control: the same noise model in the calibration suite (other noise realizations)
SUITE = os.path.join("data", "calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl")
if os.path.exists(SUITE):
    hmi = add_box([x for x in pickle.load(open(SUITE, "rb"))["data"] if x["noise_model"] == NOISE])   # the boxes of standard proportion
    out["suite_same_noise"] = {k: v for k, v in evaluate(hmi).items() if k != "auc_ci"}
    print(f"suite, {NOISE}: {out['suite_same_noise']['bad_accepted']:.1%} wrong among accepted ({len(hmi)} cases)")
json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("guardado", OUT_JSON)
