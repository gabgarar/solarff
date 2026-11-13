"""What changes in the three regions if the length is measured relative to the height of the box (N5 to N7 of the plan).

The published model measures the length of a line in units of the horizontal diagonal of the box; in flat boxes it
can accept long lines that are wrong, and the length relative to the height is a conservative variant that does not
(``examples/26_box_height_length.py``, ``examples/27_box_proportion.py``). The boxes of the three regions are flat.
Here the confidence of the lines of the paper is recomputed with that variant, from the per-line indicators saved by
``examples/22_resolution_check.py``, and three things are compared with the published definition:

- N5: the fraction of reliable lines;
- N6: the stability under a change of resolution (lines accepted that half the resolution does not reproduce;
  agreement of the decision between the published and the double resolution; the equal-height test H6);
- N7: the alignment with the AIA loops at 171 A (medians of reliable and unreliable lines; difference at equal height).

Registered in the plan of the paper before computing (13-Nov-2026). No extrapolation is run.

    python examples/revision/r20_height_rule_regions.py
"""
import json
import os
import pickle

import numpy as np
from scipy.stats import spearmanr

from solarff.calibration import _nz
from solarff.confidence import CALIBRATION_V1 as DEFAULT_CALIBRATION, confidence  # the default model when this test was made (version 1.1)

REGIONS = {"11158": "aia_validation_harp377.pkl", "11429": "harp1449_20120306_2336_aia.pkl", "12673": "harp7115_20170906_1136_aia.pkl"}
T = DEFAULT_CALIBRATION["threshold"]
N_BOOT = 2000
OUT = os.path.join("docs", "revision", "resultados", "r20_height_rule_regions.json")


def ratio(nx, ny, nz):
    return (nz - 1) / np.hypot(nx - 1, ny - 1)


C0 = float(np.mean([ratio(n, n, _nz(n)) for n in (32, 48, 64)]))


def conf(d, prefix, factor):
    ind = {k: np.asarray(d[f"{prefix}_{k}"], float) for k in ("alpha_spread", "buffer", "length")}
    ind["length"] = ind["length"] * factor
    return confidence(ind, DEFAULT_CALIBRATION)


def equal_height(y, reliable, apex, rng):
    """As in ``examples/22_resolution_check.py``: coefficient of ``reliable`` with log(1 + apex height) as control."""
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


res = json.load(open(os.path.join("docs", "resultados", "resolution_check.json"), encoding="utf-8"))
out = {"threshold": T, "c0": C0, "n_boot": N_BOOT, "regions": {}, "pooled": {}}
for noaa, f in REGIONS.items():
    fine = np.load(os.path.join("data", f"resolution_check_lines_{noaa}.npz"))
    coarse = np.load(os.path.join("data", f"resolution_check_lines_{noaa}_coarser.npz"))
    grids = {"published": res["regions"][noaa]["published"]["grid"], "double": res["regions"][noaa]["double"]["grid"],
             "half": res["regions_coarser"][noaa]["double"]["grid"]}
    assert np.allclose(conf(fine, "published", 1.0), fine["conf_published"], equal_nan=True), "the published confidence is not reproduced"
    r = {"grids": grids, "height_over_diagonal": {k: float(ratio(*g)) for k, g in grids.items()},
         "factor": {k: float(C0 / ratio(*g)) for k, g in grids.items()}}
    aia = pickle.load(open(os.path.join("data", f), "rb"))
    wl = {str(k): np.asarray(w["angles"], float) for k, w in aia["plot"]["wavelengths"].items()}
    a171 = wl[[k for k in wl if k.startswith("171")][0]]
    assert a171.size == fine["conf_published"].size, "the lines of the two analyses are not the same"
    for rule in ("published", "height"):
        fac = (lambda k: 1.0) if rule == "published" else (lambda k: C0 / ratio(*grids[k]))
        c_lo, c_hi, c_half = conf(fine, "published", fac("published")), conf(fine, "double", fac("double")), conf(coarse, "double", fac("half"))
        acc_lo, acc_hi, acc_half = c_lo >= T, c_hi >= T, c_half >= T
        q = {"frac_reliable": float(np.mean(acc_lo)), "frac_reliable_double": float(np.mean(acc_hi)), "frac_reliable_half": float(np.mean(acc_half))}
        # N6 (i): accepted lines that half the resolution does not reproduce within 10 % (or open/closed change)
        shape, topo = coarse["err_shape"], coarse["err_topology"] > 0
        val = np.isfinite(shape)
        moved = (shape > 0.1) | topo
        q["not_reproduced_half"] = {"all": float(np.mean(moved[val])), "accepted": float(np.mean(moved[val & acc_lo])),
                                    "rejected": float(np.mean(moved[val & ~acc_lo])), "n_accepted": int((val & acc_lo).sum())}
        # the same towards the double resolution
        shape, topo = fine["err_shape"], fine["err_topology"] > 0
        val = np.isfinite(shape)
        moved = (shape > 0.1) | topo
        q["not_reproduced_double"] = {"all": float(np.mean(moved[val])), "accepted": float(np.mean(moved[val & acc_lo])),
                                      "rejected": float(np.mean(moved[val & ~acc_lo]))}
        # N6 (ii): agreement of the decision between the published and the double resolution (and the half)
        ok = np.isfinite(c_lo) & np.isfinite(c_hi)
        q["same_decision_double"] = float(np.mean(acc_lo[ok] == acc_hi[ok]))
        q["spearman_double"] = float(spearmanr(c_lo[ok], c_hi[ok])[0])
        # agreement beyond chance (Cohen's kappa): the raw agreement rises by itself when fewer lines are accepted
        chance = np.mean(acc_lo[ok]) * np.mean(acc_hi[ok]) + (1 - np.mean(acc_lo[ok])) * (1 - np.mean(acc_hi[ok]))
        q["kappa_double"] = float((q["same_decision_double"] - chance) / (1 - chance))
        ok2 = np.isfinite(c_lo) & np.isfinite(c_half)
        q["same_decision_half"] = float(np.mean(acc_lo[ok2] == acc_half[ok2]))
        # N6 (iii): equal-height test of H6 (moved by more than 20 % or open/closed change, towards the double resolution)
        use = val & np.isfinite(c_lo)
        moved20 = (shape > 0.2) | topo
        q["equal_height_moved_20"] = equal_height(moved20[use].astype(float), acc_lo[use].astype(float), fine["apex"][use], np.random.default_rng(0))
        both = ok & acc_lo & acc_hi
        rel_long = fine["err_shape_abs"] / np.maximum(fine["err_length_ref"], fine["err_length_model"])
        q["accepted_both_differ_20_longer_line"] = float(np.mean(((rel_long > 0.2) | topo)[both & val]))
        q["accepted_both_open_closed"] = float(np.mean(topo[both]))
        # N7: AIA at 171 A
        m = np.isfinite(a171) & np.isfinite(c_lo)
        q["aia_171"] = {"n": int(m.sum()), "median_reliable": float(np.median(a171[m & acc_lo])), "median_unreliable": float(np.median(a171[m & ~acc_lo])),
                        "equal_height": equal_height(a171[m], acc_lo[m].astype(float), fine["apex"][m], np.random.default_rng(0))}
        # where the lines are: apex height of the accepted and of the rejected lines (Mm)
        q["apex_median_Mm"] = {"accepted": float(np.median(fine["apex"][acc_lo])), "rejected": float(np.median(fine["apex"][~acc_lo]))}
        q["spearman_confidence_apex"] = float(spearmanr(c_lo[np.isfinite(c_lo)], fine["apex"][np.isfinite(c_lo)])[0])
        r[rule] = q
    cp, ch = conf(fine, "published", 1.0), conf(fine, "published", C0 / ratio(*grids["published"]))
    okc = np.isfinite(cp) & np.isfinite(ch)
    r["accepted_by_published_kept"] = float(np.mean((ch >= T)[okc & (cp >= T)]))
    r["accepted_by_rule_not_by_published"] = int(((ch >= T) & ~(cp >= T))[okc].sum())
    out["regions"][noaa] = r
for rule in ("published", "height"):
    out["pooled"][rule] = {"equal_height_moved_20": pooled([out["regions"][n][rule]["equal_height_moved_20"] for n in REGIONS]),
                           "aia_171_equal_height": pooled([out["regions"][n][rule]["aia_171"]["equal_height"] for n in REGIONS])}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)


def ci(e):
    return f"{e['value']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"


for n in REGIONS:
    r = out["regions"][n]
    print(f"NOAA {n}: height / diagonal {r['height_over_diagonal']['published']:.3f}, factor {r['factor']['published']:.2f}; "
          f"accepted by the published rule that the height rule keeps {r['accepted_by_published_kept']:.2f}; new {r['accepted_by_rule_not_by_published']}")
    for rule in ("published", "height"):
        q = r[rule]
        print(f"  {rule:9s} reliable {100 * q['frac_reliable']:.1f} % (double {100 * q['frac_reliable_double']:.1f}, half {100 * q['frac_reliable_half']:.1f}) | "
              f"not reproduced at half: accepted {q['not_reproduced_half']['accepted']:.2f}, rejected {q['not_reproduced_half']['rejected']:.2f} | "
              f"at double: accepted {q['not_reproduced_double']['accepted']:.2f}, rejected {q['not_reproduced_double']['rejected']:.2f} | "
              f"same decision double {100 * q['same_decision_double']:.1f} % (kappa {q['kappa_double']:.2f}), half {100 * q['same_decision_half']:.1f} % | Spearman {q['spearman_double']:.3f} | "
              f"equal height {ci(q['equal_height_moved_20'])} | accepted both differ > 20 %: {100 * q['accepted_both_differ_20_longer_line']:.1f} %, "
              f"open/closed {100 * q['accepted_both_open_closed']:.1f} % | AIA 171: {q['aia_171']['median_reliable']:.1f} / {q['aia_171']['median_unreliable']:.1f}, "
              f"equal height {ci(q['aia_171']['equal_height'])} | apex median accepted {q['apex_median_Mm']['accepted']:.1f} Mm, rejected {q['apex_median_Mm']['rejected']:.1f} Mm | "
              f"Spearman conf-apex {q['spearman_confidence_apex']:.2f}")
for rule in ("published", "height"):
    p = out["pooled"][rule]
    print(f"three regions, {rule}: equal height (moved > 20 %) {ci(p['equal_height_moved_20'])} | AIA 171 equal height {ci(p['aia_171_equal_height'])}")
print("written", OUT)
