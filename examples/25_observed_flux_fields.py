"""Exact fields built on the observed flux: does the published confidence hold with the fine structure of a real magnetogram?

The exact fields of the suite are smooth and well resolved; a real magnetogram has structure down to the pixel.
The difference shows in the stability of the lines: at half the resolution the accepted lines of the suite are
reproduced almost without exception, and two out of three of those of the regions are not
(``examples/22_resolution_check.py --coarser``, ``examples/24_ensemble_indicator.py``).

An exact field with the flux distribution of a region can be built, because the linear solution (force-free
with ``a = 0``, magnetohydrostatic of Low 1991 with ``a > 0``; :func:`solarff.lfff.linear_mhs`) is computed on
any ``B_z``, also the observed one. Here it is computed on the pre-flare ``B_z`` of NOAA 11158, 11429 and 12673,
with the grid and the box of the published extrapolations, ``|alpha| = 0.9 alpha_max`` and the sign of the
best-fit alpha of the region. Its base, with noise, is extrapolated as in the suite (300 random seeds, three
noise models) and the published model is applied without refitting. The ``coarse`` variant gives the fraction
of the lines that half the resolution does not reproduce. These fields have the flux of a region but not its
localised currents: alpha is constant.

The criteria (F1, F2) were registered in the plan of the paper before running (13-Nov-2026).

Saves ``data/observed_flux_fields.pkl`` (checkpoint, with the per-line records) and
``docs/resultados/observed_flux_fields.json``.

    python -u examples/25_observed_flux_fields.py                 # ~1 h
    python -u examples/25_observed_flux_fields.py --alpha-frac 0.5 --fields LFFF-obs    # moderate shear, files with the suffix _a50
    python examples/25_observed_flux_fields.py --analyze-only
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.alpha import alpha_best_fit
from solarff.analytic import field_magnetogram
from solarff.calibration import NOISE_MODELS, bootstrap_ci, ece
from solarff.confidence import CALIBRATION_V1, DEFAULT_CALIBRATION, auc, calibration_case, confidence, length_in_heights
from solarff.lfff import alpha_max, linear_mhs
from solarff.magnetogram import Magnetogram, aly_criteria
from solarff.metrics import force_free_metrics
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--regions", nargs="*", default=["11429", "11158", "12673"])
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=None)
ap.add_argument("--alpha-frac", type=float, default=0.9, help="alpha as a fraction of alpha_max (0.9: the registered test)")
ap.add_argument("--fields", nargs="*", default=None)
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

# region: cache of the published study (its magnetogram and its box are used)
REGIONS = {"11158": "harp377_study_mc10.pkl", "11429": "harp1449_20120306_2336_r7_nz32_mc10.pkl",
           "12673": "harp7115_20170906_1136_r4_nz56_mc10.pkl"}
FIELDS = {"LFFF-obs": 0.0, "MHS-obs": 0.6}      # amplitude a of the non-force-free current at the base
if a.fields:
    FIELDS = {k: FIELDS[k] for k in a.fields}
ALPHA_FRAC, SCALE_MM = a.alpha_frac, 2.0
T = DEFAULT_CALIBRATION["threshold"]
TAG = "" if ALPHA_FRAC == 0.9 else f"_a{round(100 * ALPHA_FRAC):02d}"      # the other values of alpha keep their own files
CKPT = os.path.join(a.data, f"observed_flux_fields{TAG}.pkl")
a.json = a.json or os.path.join("docs", "resultados", f"observed_flux_fields{TAG}.json")


def evaluate(d, cal=DEFAULT_CALIBRATION):
    """Metrics of the model ``cal`` on the cases ``d``; the intervals resample lines within the pooled cases."""
    c = np.concatenate([confidence(x["ind"], cal) for x in d])
    y = np.concatenate([x["bad"] for x in d]).astype(int)
    ok = np.isfinite(c)
    acc = ok & (np.nan_to_num(c, nan=0.0) >= T)
    p = 1 - c[ok]
    out = {"n_cases": len(d), "n_lines": int(ok.sum()), "bad_rate": float(y[ok].mean()), "auc": auc(p, y[ok]),
           "auc_ci": list(bootstrap_ci(p, y[ok], auc, n_boot=500)), "ece": ece(p, y[ok]), "mean_predicted_bad": float(p.mean()),
           "accepted": float(acc[ok].mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None,
           "bad_accepted_by_case": [float(x["bad"][np.nan_to_num(confidence(x["ind"], cal), nan=0.0) >= T].mean())
                                    for x in d]}
    if acc.any():
        rng = np.random.default_rng(0)
        ya = y[acc]
        b = [ya[rng.integers(0, ya.size, ya.size)].mean() for _ in range(1000)]
        out["bad_accepted_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    for t in (0.9, 0.95):
        m = ok & (np.nan_to_num(c, nan=0.0) >= t)
        out[f"accepted_{t}"], out[f"bad_accepted_{t}"] = float(m[ok].mean()), (float(y[m].mean()) if m.any() else None)
    sh = np.concatenate([np.asarray(x["ind"]["ens_coarse"], float) for x in d])
    topo = np.concatenate([np.asarray(x["ind"]["ens_coarse_topo"]) for x in d]) > 0
    val = np.isfinite(sh)
    with np.errstate(invalid="ignore"):
        moved = (sh > 0.1) | topo
    out["not_reproduced_10"] = float(moved[val].mean())
    out["not_reproduced_10_accepted"] = float(moved[val & acc].mean()) if (val & acc).any() else None
    out["auc_displacement_alone"] = auc(np.where(topo, np.maximum(sh, 0.5), sh)[val & ok], y[val & ok])
    # second key: accepted and reproduced within 10 % at half the resolution
    two = acc & val & ~moved
    out["two_keys"] = {"accepted": float(two[ok].mean()), "bad_accepted": float(y[two].mean()) if two.any() else None}
    return out


state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {"cases": {}, "exact": {}}
if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        for noaa in a.regions:
            study, _ = pickle.load(open(os.path.join(a.data, REGIONS[noaa]), "rb"))
            obs, nz = study.mag, study.nlfff.shape[2]
            sign = np.sign(alpha_best_fit(obs)) or 1.0
            alpha = float(sign * ALPHA_FRAC * alpha_max(obs))
            del study
            for fname, amp in FIELDS.items():
                exact = None
                for j, (nm, (sz, st)) in enumerate(NOISE_MODELS.items()):
                    key = f"{fname} {noaa} | {nm}"
                    if key in state["cases"]:
                        continue
                    if exact is None:
                        exact = linear_mhs(Magnetogram(bz=obs.bz, dx=obs.dx, origin=obs.origin), alpha, nz, a=amp, kappa=1.0 / SCALE_MM)
                        ff = force_free_metrics(exact)
                        state["exact"][f"{fname} {noaa}"] = {
                            "grid": [int(n) for n in exact.shape], "dx_Mm": float(exact.dx), "alpha": alpha, "a": amp, "scale_mm": SCALE_MM,
                            "eps_force": float(aly_criteria(field_magnetogram(exact, 0.0, 0.0))["eps_force"]),
                            "eps_force_observed": float(aly_criteria(obs)["eps_force"]), "cwsin": float(ff["CWsin"])}
                    tc = time.time()
                    k = 7919 * int(noaa) + 100 * ["LFFF-obs", "MHS-obs"].index(fname) + j + (0 if ALPHA_FRAC == 0.9 else round(1000 * ALPHA_FRAC))
                    mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=k)
                    rec = calibration_case(exact, mag, nz, n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"max_iter": 4000},
                                           ensemble=("coarse",))
                    rec.update(field=fname, region=noaa, noise_model=nm, seconds=time.time() - tc)
                    state["cases"][key] = rec
                    pickle.dump(state, open(CKPT, "wb"))
                    print(f"  {key}: {rec['bad'].mean():.0%} wrong lines ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)", flush=True)

for key, x in state["cases"].items():      # cases saved before the length in box heights existed
    if "length_h" not in x["ind"]:
        x["ind"]["length_h"] = length_in_heights(x["ind"]["length"], state["exact"][key.split(" |")[0]]["grid"])
# ``fields`` and ``pooled``: the default model; ``fields_v1`` and ``pooled_v1``: the model of version 1.1
out = {"code_version": code_version()["label"], "calibration": "DEFAULT_CALIBRATION (five terms) and CALIBRATION_V1 (four terms, version 1.1), not refitted",
       "alpha_frac": ALPHA_FRAC, "noise_models": list(NOISE_MODELS), "exact": state["exact"], "fields": {}, "pooled": {},
       "fields_v1": {}, "pooled_v1": {}}
for name in state["exact"]:
    d = [x for k, x in state["cases"].items() if k.startswith(name + " |")]
    if d:
        out["fields"][name], out["fields_v1"][name] = evaluate(d), evaluate(d, CALIBRATION_V1)
for fname in FIELDS:
    d = [x for k, x in state["cases"].items() if k.startswith(fname + " ")]
    if d:
        out["pooled"][fname], out["pooled_v1"][fname] = evaluate(d), evaluate(d, CALIBRATION_V1)
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for name, r in {**out["fields"], **{k + " (three regions)": v for k, v in out["pooled"].items()}}.items():
    e = out["exact"].get(name, {})
    print(f"{name:26s} force {e.get('eps_force', float('nan')):.2f} | wrong {r['bad_rate']:.2f} | AUC {r['auc']:.3f} | ECE {r['ece']:.3f} | "
          f"accepted {r['accepted']:.2f}, wrong among them {'-' if r['bad_accepted'] is None else format(r['bad_accepted'], '.3f')} "
          f"(0.9: {'-' if r['bad_accepted_0.9'] is None else format(r['bad_accepted_0.9'], '.3f')}) | not reproduced at half resolution: "
          f"all {r['not_reproduced_10']:.2f}, accepted {'-' if r['not_reproduced_10_accepted'] is None else format(r['not_reproduced_10_accepted'], '.2f')} | "
          f"two keys: accepted {r['two_keys']['accepted']:.2f}, wrong {'-' if r['two_keys']['bad_accepted'] is None else format(r['two_keys']['bad_accepted'], '.3f')}")
for name, r in {**out["fields_v1"], **{k + " (three regions)": v for k, v in out["pooled_v1"].items()}}.items():
    print(f"version 1.1  {name:26s} AUC {r['auc']:.3f} | accepted {r['accepted']:.2f}, wrong among them "
          f"{'-' if r['bad_accepted'] is None else format(r['bad_accepted'], '.3f')}")
print("written", a.json)
