"""Global applicability indicator: does any quantity of the extrapolation itself, computable without
knowing the truth, warn that the per-line probability is optimistic?

Motivation (examples/17): in LFFF-δ and TD3 the NLFFF fails on 61-74 % of the lines and the published
model accepts 18-21 % of wrong lines (in the suite, 4 %). No per-line indicator sees it.
Here the NLFFF of every case of the suite (5 families + TD-relax), of the TD2 rope
(``data/revision/suite_r0.pkl``) and of the three external families (``data/external_families.pkl``) is
recomputed with the same noise seeds, and five global quantities are measured:

* ``cwsin`` (without the buffer layer, as in the reports) and ``e_div_frac`` of the NLFFF solution
  (Wheatland et al. 2000; Valori et al. 2013);
* ``boundary_weight``: mean potential field on the side and top faces divided by that of the
  base (how much the unobserved boundaries weigh), and its parts ``side_weight`` and ``top_weight``;
* ``pp_change``: relative change of the base by the preprocessing;
* ``open_frac``: fraction of open lines (from the record itself).

For each case the observed fraction of wrong lines, the one predicted by the model and the wrong
lines among the accepted ones (≥ 0.8) are saved too. A case is "optimistic" if it accepts ≥ 10 % wrong lines. For each
quantity the AUC with which it separates the optimistic cases and the Spearman correlation with the excess
(observed − predicted) are given. The same quantities are computed for the three regions of the paper. The
reproduction is verified by recomputing the labels of the first case of each source.

Saves ``docs/resultados/applicability.json`` and ``data/applicability.pkl`` (checkpoint).

    python -u examples/18_applicability.py         # ~1.5 h
"""
import json
import os
import pickle
import time
import warnings

import numpy as np
from scipy.stats import spearmanr

from solarff.analytic import CALIBRATION_FIELDS, EXTERNAL_TEST_FIELDS, field_magnetogram
from solarff.calibration import NOISE_MODELS, _nz
from solarff.confidence import DEFAULT_CALIBRATION, auc, confidence, line_errors
from solarff.helicity import energy_decomposition
from solarff.metrics import force_free_metrics
from solarff.nlfff_torch import nlfff_lbfgs
from solarff.potential import potential_field
from solarff.preprocess import preprocess
from solarff.tracing import seeds_from_magnetogram, trace_field_lines
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
OUT_JSON = os.path.join("docs", "resultados", "applicability.json")
CKPT = os.path.join("data", "applicability.pkl")
SIZES = (32, 48, 64)
KNOWN = {**CALIBRATION_FIELDS, **EXTERNAL_TEST_FIELDS}


def load(p):
    d = pickle.load(open(p, "rb"))
    return d["data"] if isinstance(d, dict) and "data" in d else d


# Sources: (name, list of families with which the suite was enumerated, records, families to process)
SOURCES = []
S5 = load("data/calibration_suite_s32-48-64_r2_mc0.pkl")
SOURCES.append(("bateria", list(CALIBRATION_FIELDS), S5, list(CALIBRATION_FIELDS)))
S6 = [d for d in load("data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl") if d["field"] == "TD-relax"]
SOURCES.append(("TD-relax", ["TD-relax"], S6, ["TD-relax"]))
R0 = load("data/revision/suite_r0.pkl")
SOURCES.append(("suite_r0", list(CALIBRATION_FIELDS) + ["TD-relax", "TD2-relax"], R0, ["TD2-relax"]))
EXT = pickle.load(open("data/external_families.pkl", "rb"))
for name, recs in EXT.items():
    SOURCES.append((name, [name], recs, [name]))


def global_metrics(mag, pp, model, nz):
    ff = force_free_metrics(model, margin=int(model.meta.get("nd", 0)))  # without the buffer layer, like the regions table
    ed = energy_decomposition(model)
    P = potential_field(pp, nz)
    B = np.sqrt(P.bx ** 2 + P.by ** 2 + P.bz ** 2)
    base = B[:, :, 0].mean()
    side = np.mean([B[0].mean(), B[-1].mean(), B[:, 0].mean(), B[:, -1].mean()])
    top = B[:, :, -1].mean()
    b0 = np.stack([mag.bx, mag.by, mag.bz])
    b1 = np.stack([pp.bx, pp.by, pp.bz])
    return {"cwsin": float(ff["CWsin"]), "e_div_frac": float(ed["E_div_frac"]),
            "boundary_weight": float((4 * side + top) / 5 / base), "side_weight": float(side / base),
            "top_weight": float(top / base), "pp_change": float(np.linalg.norm(b1 - b0) / np.linalg.norm(b0))}


def record_metrics(rec):
    c = confidence(rec["ind"], DEFAULT_CALIBRATION)
    ok = np.isfinite(c)
    y = np.asarray(rec["bad"], bool)[ok]
    acc = c[ok] >= DEFAULT_CALIBRATION["threshold"]
    return {"bad_rate": float(y.mean()), "predicted_bad": float(np.mean(1 - c[ok])),
            "accepted": float(acc.mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None,
            "open_frac": float(np.nanmean(np.asarray(rec["ind"]["open"], float)))}


done = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
rows = done.get("rows", [])
verified = done.get("verified", {})
have = {(r["source"], r["field"], r["noise_model"], r["size"], r["realization"]) for r in rows}
t_start = time.time()
with keep_awake():
    for src, fields, recs, wanted in SOURCES:
        cases = [(f, nm, s, r) for f in fields for nm in NOISE_MODELS for s in SIZES for r in range(2)]
        assert len(cases) == len(recs), (src, len(cases), len(recs))
        first_checked = src in verified
        for i, (fname, nm, size, r) in enumerate(cases):
            rec = recs[i]
            assert (rec["field"], rec["noise_model"], rec["size"], rec["realization"]) == (fname, nm, size, r), (src, i)
            if fname not in wanted or (src, fname, nm, size, r) in have:
                continue
            t0 = time.time()
            exact = KNOWN[fname](size, size, _nz(size))
            sz, st = NOISE_MODELS[nm]
            k = i  # seed = 0 in every suite
            mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=k)
            pp = preprocess(mag, verbose=False)
            model = nlfff_lbfgs(pp, _nz(size), verbose=False)
            if not first_checked:  # the labels of the first case must be reproduced exactly
                seeds = seeds_from_magnetogram(mag, n=300, method="random", polarity="both", threshold=0.05, rng=k + 7)
                err = line_errors(trace_field_lines(model, seeds), trace_field_lines(exact, seeds))
                bad = (err["shape"] > rec["bad_threshold"]) | (err["topology"] > 0)
                same = bool(len(bad) == len(rec["bad"]) and np.array_equal(bad, np.asarray(rec["bad"], bool)))
                verified[src] = {"case": [fname, nm, size, r], "labels_identical": same,
                                 "n_diff": int(np.sum(bad != np.asarray(rec["bad"], bool))) if len(bad) == len(rec["bad"]) else None}
                print(f"  verificación {src}: etiquetas {'idénticas' if same else 'DISTINTAS'} ({verified[src]['n_diff']} diferencias)")
                first_checked = True
            row = {"source": src, "field": fname, "noise_model": nm, "size": size, "realization": r,
                   **global_metrics(mag, pp, model, _nz(size)), **record_metrics(rec), "seconds": time.time() - t0}
            rows.append(row)
            pickle.dump({"rows": rows, "verified": verified}, open(CKPT, "wb"))
            print(f"  {src:10s} {fname:10s} {nm:13s} {size:2d}² r{r}: CWsin {row['cwsin']:.3f} E_div/E {row['e_div_frac']:.3f} "
                  f"contornos {row['boundary_weight']:.3f} pp {row['pp_change']:.3f} | malas {row['bad_rate']:.0%} "
                  f"pred {row['predicted_bad']:.0%} malas acept {row['bad_accepted']:.1%} ({row['seconds']:.0f}s, total {(time.time() - t_start) / 60:.0f} min)")

# ---------------------------------------------------------------- regions of the paper
REGIONS = {"11158": "data/harp377_study_mc10.pkl", "11429": "data/harp1449_20120306_2336_r7_nz32_mc10.pkl",
           "12673": "data/harp7115_20170906_1136_r4_nz56_mc10.pkl"}
regions = {}
for noaa, p in REGIONS.items():
    if not os.path.exists(p):
        continue
    study, _ = pickle.load(open(p, "rb"))
    pp = study.mag_pp if study.mag_pp is not None else preprocess(study.mag, verbose=False)
    regions[noaa] = global_metrics(study.mag, pp, study.nlfff, study.nlfff.bz.shape[2])
    print(f"  región {noaa}: " + " ".join(f"{k} {v:.3f}" for k, v in regions[noaa].items()))

# ---------------------------------------------------------------- analysis
METRICS = ["cwsin", "e_div_frac", "boundary_weight", "side_weight", "top_weight", "pp_change", "open_frac"]
opt = np.array([r["bad_accepted"] is not None and r["bad_accepted"] >= 0.10 for r in rows])
excess = np.array([r["bad_rate"] - r["predicted_bad"] for r in rows])
analysis = {"n_cases": len(rows), "n_optimistic": int(opt.sum()), "optimistic_definition": "bad_accepted >= 0.10",
            "optimistic_by_family": {}, "metrics": {}}
for f in sorted({r["field"] for r in rows}):
    m = np.array([r["field"] == f for r in rows])
    analysis["optimistic_by_family"][f] = {"n": int(m.sum()), "optimistic": int((opt & m).sum()),
                                           "median_excess": float(np.median(excess[m]))}
for key in METRICS:
    x = np.array([r[key] for r in rows], float)
    rho, pval = spearmanr(x, excess)
    a_opt = auc(x, opt.astype(int)) if 0 < opt.sum() < len(opt) else None
    fam_med = {f: float(np.median(x[[r["field"] == f for r in rows]])) for f in analysis["optimistic_by_family"]}
    train = np.array([r["source"] in ("bateria", "TD-relax") for r in rows])
    analysis["metrics"][key] = {"auc_optimistic": a_opt, "spearman_excess": float(rho), "spearman_p": float(pval),
                                "train_range": [float(x[train].min()), float(x[train].max())],
                                "train_p95": float(np.percentile(x[train], 95)),
                                "median_by_family": fam_med, "regions": {k: v[key] for k, v in regions.items()} if key != "open_frac" else None}
    print(f"{key:16s} AUC optimistas {a_opt if a_opt is None else round(a_opt, 3)}  Spearman(exceso) {rho:+.2f} (p {pval:.1e})  "
          f"batería [{x[train].min():.3f}, {x[train].max():.3f}]  medianas: " +
          ", ".join(f"{f} {v:.3f}" for f, v in fam_med.items()))
out = {"code_version": code_version()["label"], "verified": verified, "rows": rows, "regions": regions, "analysis": analysis}
json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("guardado", OUT_JSON, f"({(time.time() - t_start) / 60:.0f} min)")
