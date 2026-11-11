"""Does the bootstrap correction remove the bias that noise introduces in the free energy?

For each exact field, noise level and independent noisy observation:

1. The full study (preprocessing + NLFFF) is run on the observation.
2. Monte Carlo: ``n`` further realizations with noise of the same level (σ of the
   error maps), which give the median p50.
3. Bootstrap bias correction: ``corrected = 2·nominal − p50``.

It is compared with two references: the **noise-free model** (same method on the
exact base: isolates the bias due to noise) and the **exact field** (also includes
the error of the method). Saved in ``data/bias_validation.pkl`` and the summary,
used by the report, in ``solarff/bias_validation.json``.

    python examples/05_bias_validation.py          # ~1-1.5 h, with checkpoints
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
from solarff.montecarlo import _quantities, base_seeds, monte_carlo
from solarff.potential import potential_field
from solarff.utils import code_version, keep_awake, load_checkpoint, save_checkpoint

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--fields", nargs="+", default=["LL-I", "LL-II", "LFFF-bip"])
ap.add_argument("--noises", nargs="+", default=["10", "30", "60", "10/60"],
                help="σ en G; 'a/b' = σ(Bz)/σ(Bx,By)")
ap.add_argument("--obs", type=int, default=3, help="observaciones ruidosas independientes por caso")
ap.add_argument("--mc", type=int, default=12, help="realizaciones Monte Carlo por observación")
ap.add_argument("--size", type=int, default=40)
ap.add_argument("--out", default="data")
ap.add_argument("--json", default=None, help="resumen para el informe (por defecto, el del paquete)")
a = ap.parse_args()

Q = ("E_free", "E_free_vol", "E_ratio", "H_V")
nz = int(round(0.7 * a.size))
ck = os.path.join(a.out, "bias_validation.ckpt")
cases = [(f, nm, o) for f in a.fields for nm in a.noises for o in range(a.obs)]
sig = {"cases": cases, "mc": a.mc, "size": a.size}
state = load_checkpoint(ck, sig) or {"signature": sig, "refs": {}, "rows": []}
t0 = time.time()
with keep_awake():
    for f in a.fields:
        if f in state["refs"]:
            continue
        exact = CALIBRATION_FIELDS[f](a.size, a.size, nz)
        clean = field_magnetogram(exact, 0.0)
        st = sff.run_study(clean, nz, nlfff_method="lbfgs", verbose=False)
        nd = int(st.nlfff.meta.get("nd", 4))
        seeds = base_seeds(st.nlfff, max(1, a.size // 40))
        q_model, _ = _quantities(st.nlfff, sff.reference_potential(st), clean, nz, nd, seeds)
        q_exact, _ = _quantities(exact, potential_field(clean, nz), clean, nz, nd, seeds)
        state["refs"][f] = {"model_clean": {k: float(q_model[k]) for k in Q},
                            "exact": {k: float(q_exact[k]) for k in Q}}
        save_checkpoint(ck, state)
        print(f"{f}: referencia sin ruido E_free modelo {q_model['E_free']:.3e}, exacto {q_exact['E_free']:.3e}",
              flush=True)
    for i in range(len(state["rows"]), len(cases)):
        f, nm, o = cases[i]
        sz, st_ = (float(x) for x in nm.split("/")) if "/" in nm else (float(nm),) * 2
        exact = CALIBRATION_FIELDS[f](a.size, a.size, nz)
        obs = field_magnetogram(exact, noise=sz, noise_t=st_, rng=10_000 + i)
        study = sff.run_study(obs, nz, nlfff_method="lbfgs", verbose=False)
        mc = monte_carlo(study, n=a.mc, seed=i, verbose=False)
        row = {"field": f, "noise": nm, "obs": o}
        for k in Q:
            s = mc["summary"][k]
            row[k] = {x: s.get(x) for x in ("nominal", "p16", "p50", "p84", "corrected", "bias")}
        state["rows"].append(row)
        save_checkpoint(ck, state)
        r, ref = row["E_free"], state["refs"][f]["model_clean"]["E_free"]
        print(f"caso {i + 1}/{len(cases)} {f} σ={nm} obs {o}: E_free nominal {r['nominal'] / ref - 1:+.1%}, "
              f"corregido {r['corrected'] / ref - 1:+.1%} respecto al modelo sin ruido "
              f"({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------ summary
rows, refs = state["rows"], state["refs"]
summary = {"fields": a.fields, "noises": a.noises, "obs": a.obs, "mc": a.mc, "size": a.size,
           "code_version": code_version()["label"], "refs": refs, "by_case": {}, "overall": {}}
for k in Q:
    tab = {}
    for f in a.fields:
        for nm in a.noises:
            rs = [r for r in rows if r["field"] == f and r["noise"] == nm]
            ref = refs[f]["model_clean"][k]
            ex = refs[f]["exact"][k]
            nom = np.array([r[k]["nominal"] for r in rs]) / ref - 1
            cor = np.array([r[k]["corrected"] for r in rs]) / ref - 1
            p50 = np.array([r[k]["p50"] for r in rs]) / ref - 1
            covered = np.array([r[k]["p16"] <= ref <= r[k]["p84"] for r in rs])
            covered_c = np.array([abs(r[k]["corrected"] - ref) <= 0.5 * (r[k]["p84"] - r[k]["p16"]) for r in rs])
            tab[f"{f}|{nm}"] = {
                "nominal_err": float(np.mean(nom)), "corrected_err": float(np.mean(cor)),
                "p50_err": float(np.mean(p50)),
                "nominal_abs": float(np.mean(np.abs(nom))), "corrected_abs": float(np.mean(np.abs(cor))),
                "nominal_vs_exact": float(np.mean([r[k]["nominal"] for r in rs]) / ex - 1),
                "corrected_vs_exact": float(np.mean([r[k]["corrected"] for r in rs]) / ex - 1),
                "covered": float(covered.mean()), "covered_corrected": float(covered_c.mean())}
    summary["by_case"][k] = tab
    summary["overall"][k] = {
        "nominal_abs": float(np.mean([v["nominal_abs"] for v in tab.values()])),
        "corrected_abs": float(np.mean([v["corrected_abs"] for v in tab.values()])),
        "improved_frac": float(np.mean([v["corrected_abs"] < v["nominal_abs"] for v in tab.values()])),
        "covered": float(np.mean([v["covered"] for v in tab.values()])),
        "covered_corrected": float(np.mean([v["covered_corrected"] for v in tab.values()]))}
with open(os.path.join(a.out, "bias_validation.pkl"), "wb") as fh:
    pickle.dump({"state": state, "summary": summary}, fh)
with open(a.json or os.path.join(os.path.dirname(sff.__file__), "bias_validation.json"), "w",
          encoding="utf-8") as fh:
    json.dump(summary, fh, indent=1, ensure_ascii=False)
for k in Q:
    print(k, summary["overall"][k])
print("guardado")
