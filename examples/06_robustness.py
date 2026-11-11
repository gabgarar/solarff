"""Robustness of the per-line confidence (and of the energy) against the arbitrary choices.

A. **"Wrong line" threshold** (5, 10, 20 % of the length): with the suite used to fit
   the published calibration (6 families, ``04_calibration_suite.py``), measures
   (i) whether the default calibration, fitted with 10 %, still ranks well with
   other definitions and (ii) how much it changes when refitted with each one.
B. **NLFFF and preprocessing parameters** (ν, λ, ε, buffer layer, μ3, μ4) and
   **resolution**, one at a time around the default values, on several cases
   with known truth: AUC and ECE of the default confidence, fraction of reliable
   lines, wrong lines among the reliable ones and error with respect to the exact
   field of the free energy (base and volume potential) and of the relative
   helicity. Feeds the uncertainty budget of the report (``solarff.budget``).

Saves ``data/robustness.pkl`` and the summary for the report in
``solarff/robustness.json``.

    python examples/06_robustness.py            # ~15-20 min, with checkpoints
    python examples/06_robustness.py --quick --out <tmp> --json <tmp>/r.json   # 1-2 min test
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

import solarff as sff
from solarff.analytic import CALIBRATION_FIELDS, field_magnetogram
from solarff.calibration import NOISE_MODELS, cross_validate, ece, fit_default, relabel
from solarff.confidence import DEFAULT_CALIBRATION, auc, calibration_case, confidence
from solarff.helicity import helicity_report
from solarff.metrics import energy
from solarff.potential import potential_field
from solarff.utils import code_version, keep_awake, load_checkpoint, save_checkpoint

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data", help="carpeta de los .pkl y del punto de control")
ap.add_argument("--json", default=os.path.join(os.path.dirname(sff.__file__), "robustness.json"),
                help="resumen para el informe (por defecto, el del paquete)")
ap.add_argument("--quick", action="store_true", help="prueba rápida: 1 caso, 2 variantes, malla 24")
a = ap.parse_args()
OUT = a.out
os.makedirs(OUT, exist_ok=True)
summary = {"code_version": code_version()["label"]}

# ------------------------------------------------------------------ A. threshold
suite = os.path.join("data", "calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl")  # the published one
if os.path.exists(suite):
    data = pickle.load(open(suite, "rb"))["data"]
    thr_rows = {}
    for t in ((0.1,) if a.quick else (0.05, 0.1, 0.2)):
        d = relabel(data, t)
        y = np.concatenate([x["bad"] for x in d])
        c = np.concatenate([confidence(x["ind"], DEFAULT_CALIBRATION) for x in d])
        ok = np.isfinite(c)
        acc = c[ok] >= DEFAULT_CALIBRATION["threshold"]
        cv = cross_validate(d, models=["logistic"], n_boot=20 if a.quick else 200, verbose=False)
        refit = fit_default(d, cv)
        thr_rows[str(t)] = {
            "bad_rate": float(y.mean()),
            "default_auc": auc(1 - c[ok], y[ok]), "default_ece": ece(1 - c[ok], y[ok]),
            "default_bad_accepted": float(y[ok][acc].mean()),
            "refit_auc": cv["models"]["logistic"]["auc"], "refit_ece": cv["models"]["logistic"]["ece"],
            "refit_auc_ci": cv["models"]["logistic"]["auc_ci"],
            "refit_coef": dict(zip(refit["calibration"]["features"], refit["calibration"]["coef"])),
        }
        print(f"umbral {t:.0%}: líneas malas {y.mean():.0%}; calibración por defecto AUC "
              f"{thr_rows[str(t)]['default_auc']:.3f}, ECE {thr_rows[str(t)]['default_ece']:.3f}; "
              f"reajustada AUC {thr_rows[str(t)]['refit_auc']:.3f}, ECE {thr_rows[str(t)]['refit_ece']:.3f}",
              flush=True)
    summary["bad_threshold"] = thr_rows

# --------------------------------------------------------------- B. parameters
SIZE = 40
VARIANTS = [("default", {}, {}, SIZE)]
for k, vals in (("relax_bottom", (0.05, 0.2)), ("smooth", (0.003, 0.03)), ("eps", (0.01, 0.1)),
                ("nd", (2, 8))):
    VARIANTS += [(f"{k}={v:g}", {k: v}, {}, SIZE) for v in vals]
for k, vals in (("mu3", (1e-3, 1e-1)), ("mu4", (1e-3, 1e-1))):
    VARIANTS += [(f"{k}={v:g}", {}, {k: v}, SIZE) for v in vals]
VARIANTS += [(f"size={n}", {}, {}, n) for n in (28, 56)]  # same field, another resolution
CASES = [("LL-I", "uniforme-30G"), ("LL-II", "HMI (10/60 G)"), ("LFFF-bip", "HMI (10/60 G)"),
         ("LFFF-quad", "uniforme-30G")]
if a.quick:
    SIZE = 24
    VARIANTS = [("default", {}, {}, SIZE), ("mu4=0.1", {}, {"mu4": 0.1}, SIZE)]
    CASES = CASES[:1]
ck = os.path.join(OUT, "robustness.ckpt")
jobs = [(v[0], f, nm) for v in VARIANTS for f, nm in CASES]
sig = {"jobs": jobs}
state = load_checkpoint(ck, sig) or {"signature": sig, "rows": []}
t0 = time.time()
exacts = {}
with keep_awake():
    for i in range(len(state["rows"]), len(jobs)):
        name, f, nm = jobs[i]
        nk, pk, n = next((a, b, c) for v, a, b, c in VARIANTS if v == name)
        NZ = int(round(0.7 * n))
        if (f, n) not in exacts:
            exacts[(f, n)] = CALIBRATION_FIELDS[f](n, n, NZ)
        ex = exacts[(f, n)]
        sz, st = NOISE_MODELS[nm]
        j = [c for c in CASES].index((f, nm))
        mag = field_magnetogram(ex, noise=sz, noise_t=st, rng=500 + j)  # same observation in every variant
        rec = calibration_case(ex, mag, NZ, method="lbfgs", seed=900 + j, noise_bt=st,
                               nlfff_kw=nk, preprocess_kw=pk, keep_model=True)
        c = confidence(rec["ind"], DEFAULT_CALIBRATION)
        ok = np.isfinite(c)
        y = rec["bad"][ok]
        acc = c[ok] >= DEFAULT_CALIBRATION["threshold"]
        nd = int(rec["model"].meta.get("nd", 4))
        clean = field_magnetogram(ex, 0.0)
        e_ex = energy(ex, nd) - energy(potential_field(clean, NZ), nd)
        # potential of the same (preprocessed) magnetogram as the NLFFF: E − E_pot is a free energy
        # only if both share B_z at the base (revision R13)
        e_mod = energy(rec["model"], nd) - energy(potential_field(rec["pp"], NZ), nd)
        row = {"variant": name, "field": f, "noise": nm, "auc": auc(1 - c[ok], y),
               "ece": ece(1 - c[ok], y), "bad_rate": float(y.mean()), "reliable": float(acc.mean()),
               "bad_reliable": float(y[acc].mean()) if acc.any() else float("nan"),
               "efree_err": float(e_mod / e_ex - 1)}
        # free energy with respect to the volume potential and relative helicity (Valori et al. 2012)
        h_ex, h_mod = helicity_report(ex), helicity_report(rec["model"])
        row["efree_vol_err"] = float(h_mod["energy"]["E_free"] / h_ex["energy"]["E_free"] - 1)
        row["hv_err"] = float(h_mod["helicity"]["H_V"] / h_ex["helicity"]["H_V"] - 1)
        state["rows"].append(row)
        save_checkpoint(ck, state)
        print(f"{i + 1}/{len(jobs)} {name:16s} {f:9s} {nm:14s} AUC {row['auc']:.3f} malas {row['bad_rate']:.0%} "
              f"fiables {row['reliable']:.0%} (malas entre ellas {row['bad_reliable']:.0%}) "
              f"E_free {row['efree_err']:+.0%}, vol {row['efree_vol_err']:+.0%}, H_V {row['hv_err']:+.0%} "
              f"({time.time() - t0:.0f}s)", flush=True)

rows = state["rows"]
par = {}
for name, *_ in VARIANTS:
    rs = [r for r in rows if r["variant"] == name]
    par[name] = {k: float(np.nanmean([r[k] for r in rs])) for k in
                 ("auc", "ece", "bad_rate", "reliable", "bad_reliable", "efree_err", "efree_vol_err", "hv_err")}
    for k in ("efree", "efree_vol", "hv"):
        par[name][f"{k}_abs"] = float(np.mean([abs(r[f"{k}_err"]) for r in rs]))
summary["parameters"] = par
summary["cases"] = [f"{f} ({nm})" for f, nm in CASES]
summary["size"] = SIZE
with open(os.path.join(OUT, "robustness.pkl"), "wb") as fh:
    pickle.dump({"rows": rows, "summary": summary}, fh)
with open(a.json, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=1, ensure_ascii=False)
for k, v in par.items():
    print(f"{k:16s}", {a: round(b, 3) for a, b in v.items()})
print("guardado")
