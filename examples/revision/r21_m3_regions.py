"""The refitted model M3 in the three regions: what would change with respect to the published model.

M3 is the published model with a fifth term, the logarithm of the length of the line in box heights, fitted with the
six calibration families in four box proportions (``examples/30_recalibration_boxes.py``). Its coefficients are read
from ``docs/resultados/recalibration_boxes.json`` and applied here, without touching them, to the lines of the paper:

- fraction of reliable lines;
- stability under a change of resolution (agreement of the decision between the published and the double resolution,
  equal-height test H6; per-line arrays of ``examples/22_resolution_check.py``);
- alignment with the AIA loops at 171 A;
- dependence on the unobserved side and top faces (fields of ``examples/29_boundary_sensitivity.py``).

For comparison, the same quantities with the published model and with the conservative variant (length relative to
the height of the box). Descriptive; registered in the plan of the paper before computing (13-Nov-2026).

    python examples/revision/r21_m3_regions.py
"""
import json
import os
import pickle
import warnings

import numpy as np
from scipy.stats import spearmanr

from solarff.calibration import _nz
from solarff.confidence import DEFAULT_CALIBRATION, _design, auc, confidence, line_errors
from solarff.montecarlo import base_seeds
from solarff.tracing import trace_field_lines

warnings.filterwarnings("ignore")
AIA = {"11158": "aia_validation_harp377.pkl", "11429": "harp1449_20120306_2336_aia.pkl", "12673": "harp7115_20170906_1136_aia.pkl"}
STUDY = {"11158": "harp377_study_mc10.pkl", "11429": "harp1449_20120306_2336_r7_nz32_mc10.pkl", "12673": "harp7115_20170906_1136_r4_nz56_mc10.pkl"}
T = DEFAULT_CALIBRATION["threshold"]
N_BOOT = 2000
OUT = os.path.join("docs", "revision", "resultados", "r21_m3_regions.json")
M3 = json.load(open(os.path.join("docs", "resultados", "recalibration_boxes.json"), encoding="utf-8"))["models"]["M3"]
assert M3["features"] == ["alpha_spread", "buffer", "length", "length_sq", "loglh"]


def ratio(nx, ny, nz):
    return (nz - 1) / np.hypot(nx - 1, ny - 1)


C0 = float(np.mean([ratio(n, n, _nz(n)) for n in (32, 48, 64)]))


def conf(d, prefix, grid, rule):
    ind = {k: np.asarray(d[f"{prefix}_{k}"], float) for k in ("alpha_spread", "buffer", "length")}
    r = ratio(*grid)
    if rule == "published":
        return confidence(ind, DEFAULT_CALIBRATION)
    if rule == "height":
        ind["length"] = ind["length"] * C0 / r
        return confidence(ind, DEFAULT_CALIBRATION)
    X = np.column_stack([_design(ind, ["alpha_spread", "buffer", "length", "length_sq"]), np.log10(np.maximum(ind["length"] / r, 1e-4))])
    Z = (X - np.array(M3["mu"])) / np.array(M3["sd"])
    c = 1 - 1 / (1 + np.exp(-(Z @ np.array(M3["coef"]) + M3["intercept"])))
    c[~np.isfinite(c)] = np.nan
    return c


def equal_height(y, reliable, apex, rng):
    def coef(i):
        X = np.column_stack([np.ones(len(i)), reliable[i], np.log(1 + apex[i])])
        return float(np.linalg.lstsq(X, y[i], rcond=None)[0][1])
    idx = np.arange(len(y))
    b = [coef(rng.choice(idx, len(idx))) for _ in range(N_BOOT)]
    return {"value": coef(idx), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "sigma": float(np.std(b))}


def pooled(rs):
    v, s = np.array([r["value"] for r in rs]), np.array([r["sigma"] for r in rs])
    w = 1 / s ** 2
    m, sd = float((w * v).sum() / w.sum()), float(w.sum() ** -0.5)
    return {"value": m, "ci": [m - 1.96 * sd, m + 1.96 * sd], "sigma": sd}


RULES = ("published", "height", "M3")
res = json.load(open(os.path.join("docs", "resultados", "resolution_check.json"), encoding="utf-8"))
out = {"threshold": T, "model": M3, "regions": {}, "pooled": {}}
for noaa in AIA:
    fine = np.load(os.path.join("data", f"resolution_check_lines_{noaa}.npz"))
    coarse = np.load(os.path.join("data", f"resolution_check_lines_{noaa}_coarser.npz"))
    grids = {"published": res["regions"][noaa]["published"]["grid"], "double": res["regions"][noaa]["double"]["grid"],
             "half": res["regions_coarser"][noaa]["double"]["grid"]}
    aia = pickle.load(open(os.path.join("data", AIA[noaa]), "rb"))
    wl = {str(k): np.asarray(w["angles"], float) for k, w in aia["plot"]["wavelengths"].items()}
    a171 = wl[[k for k in wl if k.startswith("171")][0]]
    # lines that move when the side and top faces change (fields of examples/29_boundary_sensitivity.py)
    study, _ = pickle.load(open(os.path.join("data", STUDY[noaa]), "rb"))
    seeds = base_seeds(study.nlfff, 2)
    assert np.allclose(np.asarray(seeds), fine["seeds"]), "the seeds are not those of the resolution check"
    lines = trace_field_lines(study.nlfff, seeds)
    fb = pickle.load(open(os.path.join("data", f"boundary_sensitivity_{noaa}.pkl"), "rb"))["lfff"]
    eb = line_errors(trace_field_lines(fb, seeds), lines)
    del study, fb
    bval = np.isfinite(eb["shape"])
    bmoved = (eb["shape"] > 0.1) | (eb["topology"] > 0)
    r = {"grids": grids}
    for rule in RULES:
        c_lo, c_hi = conf(fine, "published", grids["published"], rule), conf(fine, "double", grids["double"], rule)
        c_half = conf(coarse, "double", grids["half"], rule)
        acc_lo, acc_hi, acc_half = c_lo >= T, c_hi >= T, c_half >= T
        q = {"frac_reliable": float(np.mean(acc_lo)), "frac_reliable_double": float(np.mean(acc_hi)), "frac_reliable_half": float(np.mean(acc_half)),
             "frac_0.9": float(np.mean(c_lo >= 0.9))}
        ok = np.isfinite(c_lo) & np.isfinite(c_hi)
        q["same_decision_double"] = float(np.mean(acc_lo[ok] == acc_hi[ok]))
        chance = np.mean(acc_lo[ok]) * np.mean(acc_hi[ok]) + (1 - np.mean(acc_lo[ok])) * (1 - np.mean(acc_hi[ok]))
        q["kappa_double"] = float((q["same_decision_double"] - chance) / (1 - chance))
        q["spearman_double"] = float(spearmanr(c_lo[ok], c_hi[ok])[0])
        shape, topo = fine["err_shape"], fine["err_topology"] > 0
        val = np.isfinite(shape)
        use = val & np.isfinite(c_lo)
        q["equal_height_moved_20"] = equal_height(((shape > 0.2) | topo)[use].astype(float), acc_lo[use].astype(float), fine["apex"][use], np.random.default_rng(0))
        both = ok & acc_lo & acc_hi
        q["accepted_both_open_closed"] = float(np.mean(topo[both]))
        q["open_closed_accepted"] = float(np.mean(topo[val & acc_lo]))
        q["open_closed_rejected"] = float(np.mean(topo[val & ~acc_lo]))
        m = np.isfinite(a171) & np.isfinite(c_lo)
        q["aia_171"] = {"median_reliable": float(np.median(a171[m & acc_lo])), "median_unreliable": float(np.median(a171[m & ~acc_lo])),
                        "equal_height": equal_height(a171[m], acc_lo[m].astype(float), fine["apex"][m], np.random.default_rng(0))}
        q["apex_median_accepted_Mm"] = float(np.median(fine["apex"][acc_lo]))
        q["spearman_confidence_apex"] = float(spearmanr(c_lo[np.isfinite(c_lo)], fine["apex"][np.isfinite(c_lo)])[0])
        ub = bval & np.isfinite(c_lo)
        q["faces"] = {"moved_accepted": float(np.mean(bmoved[bval & acc_lo])), "moved_rejected": float(np.mean(bmoved[bval & ~acc_lo])),
                      "auc": auc(1 - c_lo[ub], bmoved[ub].astype(int)),
                      "equal_height": equal_height(bmoved[ub].astype(float), acc_lo[ub].astype(float), fine["apex"][ub], np.random.default_rng(0))}
        r[rule] = q
    cp, cm = conf(fine, "published", grids["published"], "published"), conf(fine, "published", grids["published"], "M3")
    okc = np.isfinite(cp) & np.isfinite(cm)
    r["accepted_by_published_kept_by_M3"] = float(np.mean((cm >= T)[okc & (cp >= T)]))
    r["accepted_by_M3_not_by_published"] = int(((cm >= T) & ~(cp >= T))[okc].sum())
    r["spearman_published_M3"] = float(spearmanr(cp[okc], cm[okc])[0])
    # added after seeing the result: do the lines that M3 drops depend on the unobserved faces?
    drop, keep = bval & okc & (cp >= T) & ~(cm >= T), bval & okc & (cp >= T) & (cm >= T)
    r["faces_moved_dropped_by_M3"] = {"n": int(drop.sum()), "moved": float(np.mean(bmoved[drop])) if drop.any() else None}
    r["faces_moved_kept_by_M3"] = {"n": int(keep.sum()), "moved": float(np.mean(bmoved[keep])) if keep.any() else None}
    sh, tp = fine["err_shape"], fine["err_topology"] > 0
    vr = np.isfinite(sh)
    r["resolution_moved_20_dropped_by_M3"] = float(np.mean(((sh > 0.2) | tp)[vr & okc & (cp >= T) & ~(cm >= T)]))
    r["resolution_moved_20_kept_by_M3"] = float(np.mean(((sh > 0.2) | tp)[vr & okc & (cp >= T) & (cm >= T)]))
    out["regions"][noaa] = r
for rule in RULES:
    out["pooled"][rule] = {"equal_height_moved_20": pooled([out["regions"][n][rule]["equal_height_moved_20"] for n in AIA]),
                           "aia_171_equal_height": pooled([out["regions"][n][rule]["aia_171"]["equal_height"] for n in AIA]),
                           "faces_equal_height": pooled([out["regions"][n][rule]["faces"]["equal_height"] for n in AIA])}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)


def ci(e):
    return f"{e['value']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"


for n in AIA:
    r = out["regions"][n]
    print(f"NOAA {n}: accepted by the published model that M3 keeps {r['accepted_by_published_kept_by_M3']:.2f}; new {r['accepted_by_M3_not_by_published']}; "
          f"Spearman published-M3 {r['spearman_published_M3']:.3f} | lines of the published model that M3 drops: {r['faces_moved_dropped_by_M3']['n']}, "
          f"of which {r['faces_moved_dropped_by_M3']['moved']:.2f} move when the faces change (kept: {r['faces_moved_kept_by_M3']['moved']:.2f}) and "
          f"{r['resolution_moved_20_dropped_by_M3']:.2f} move by more than 20 % when the resolution doubles (kept: {r['resolution_moved_20_kept_by_M3']:.2f})")
    for rule in RULES:
        q = r[rule]
        print(f"  {rule:9s} reliable {100 * q['frac_reliable']:.1f} % (double {100 * q['frac_reliable_double']:.1f}, half {100 * q['frac_reliable_half']:.1f}; at 0.9: {100 * q['frac_0.9']:.1f}) | "
              f"same decision {100 * q['same_decision_double']:.1f} % (kappa {q['kappa_double']:.2f}), Spearman {q['spearman_double']:.3f} | H6 {ci(q['equal_height_moved_20'])} | "
              f"open/closed accepted {100 * q['open_closed_accepted']:.1f} %, rejected {100 * q['open_closed_rejected']:.1f} % | "
              f"AIA 171: {q['aia_171']['median_reliable']:.1f} / {q['aia_171']['median_unreliable']:.1f}, equal height {ci(q['aia_171']['equal_height'])} | "
              f"faces: moved accepted {q['faces']['moved_accepted']:.2f}, rejected {q['faces']['moved_rejected']:.2f}, AUC {q['faces']['auc']:.3f}, equal height {ci(q['faces']['equal_height'])} | "
              f"apex median accepted {q['apex_median_accepted_Mm']:.1f} Mm | Spearman conf-apex {q['spearman_confidence_apex']:.2f}")
for rule in RULES:
    p = out["pooled"][rule]
    print(f"three regions, {rule}: H6 {ci(p['equal_height_moved_20'])} | AIA 171 at equal height {ci(p['aia_171_equal_height'])} | faces at equal height {ci(p['faces_equal_height'])}")
print("written", OUT)
