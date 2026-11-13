"""Do the lines that are stable under a change of resolution align better with the AIA loops? (H7 of the plan)

The comparison with the loops of AIA is the only external reference on real data, and there the confidence adds
nothing beyond the height of the line. The displacement of each line when the resolution doubles
(``examples/22_resolution_check.py``) is the real-data analogue of the ``coarse`` ensemble indicator
(``examples/24_ensemble_indicator.py``). The test, registered before computing: at equal apex height, the
stable lines (they move by at most 20 % of their length and do not change between open and closed) are less
misaligned with the loops than the others.

Per region: angle ~ stable + log(1 + apex height), the regression the paper uses with the confidence; the angle
of a line is the mean of the 171 and 193 A channels it has. Interval by bootstrap over lines; the three regions
are combined weighting by 1/sigma^2.

    python examples/revision/r19_stability_vs_aia.py
"""
import json
import os
import pickle

import numpy as np
from scipy.stats import spearmanr

from solarff.confidence import DEFAULT_CALIBRATION

REGIONS = {"11158": "aia_validation_harp377.pkl", "11429": "harp1449_20120306_2336_aia.pkl", "12673": "harp7115_20170906_1136_aia.pkl"}
T = DEFAULT_CALIBRATION["threshold"]
N_BOOT = 2000
OUT = os.path.join("docs", "revision", "resultados", "r19_stability_vs_aia.json")


def coefficients(y, cols, controls, rng):
    """Coefficients of the columns ``cols`` in the regression of ``y`` on them and the controls, with their intervals."""
    X = np.column_stack([np.ones(len(y))] + cols + controls)
    k = len(cols)
    def coef(i):
        return np.linalg.lstsq(X[i], y[i], rcond=None)[0][1:1 + k]
    idx = np.arange(len(y))
    b = np.array([coef(rng.choice(idx, len(idx))) for _ in range(N_BOOT)])
    v = coef(idx)
    return [{"value": float(v[j]), "ci": [float(np.percentile(b[:, j], 2.5)), float(np.percentile(b[:, j], 97.5))],
             "sigma": float(b[:, j].std())} for j in range(k)]


def pooled(rs):
    v, s = np.array([r["value"] for r in rs]), np.array([r["sigma"] for r in rs])
    w = 1 / s ** 2
    m, sd = float((w * v).sum() / w.sum()), float(w.sum() ** -0.5)
    return {"value": m, "ci": [m - 1.96 * sd, m + 1.96 * sd], "sigma": sd}


out = {"threshold": T, "n_boot": N_BOOT, "regions": {}, "pooled": {}}
for noaa, f in REGIONS.items():
    d = np.load(os.path.join("data", f"resolution_check_lines_{noaa}.npz"))
    aia = pickle.load(open(os.path.join("data", f), "rb"))
    ang = np.array([np.asarray(w["angles"], float) for w in aia["plot"]["wavelengths"].values()])
    assert ang.shape[1] == len(d["conf_published"]), "the lines of the two analyses are not the same"
    rel_aia = np.asarray(aia["plot"]["reliable"], bool)
    c = d["conf_published"]
    assert np.array_equal(rel_aia, np.nan_to_num(c, nan=0.0) >= T), "the confidence of the two analyses differs"
    with np.errstate(all="ignore"):
        angle = np.nanmean(ang, axis=0)
    shape, topo, apex, length = d["err_shape"], d["err_topology"] > 0, d["apex"], d["err_length_ref"]
    use = np.isfinite(angle) & np.isfinite(shape) & np.isfinite(c) & (length > 0)
    angle, shape, topo, apex, length, c = (v[use] for v in (angle, shape, topo, apex, length, c))
    stable = (~((shape > 0.2) | topo)).astype(float)
    reliable = (c >= T).astype(float)
    disp = np.log10(np.maximum(np.where(topo, np.maximum(shape, 0.5), shape), 1e-3))
    la, ll = np.log(1 + apex), np.log(length)
    rng = np.random.default_rng(0)
    r = {"n_lines": int(use.sum()), "stable_frac": float(stable.mean()), "reliable_frac": float(reliable.mean()),
         "median_angle": {"stable": float(np.median(angle[stable > 0])), "unstable": float(np.median(angle[stable == 0])),
                          "reliable": float(np.median(angle[reliable > 0])), "unreliable": float(np.median(angle[reliable == 0])),
                          "reliable_and_stable": float(np.median(angle[(reliable > 0) & (stable > 0)])),
                          "rest": float(np.median(angle[~((reliable > 0) & (stable > 0))]))},
         "spearman_displacement_angle": float(spearmanr(disp, angle)[0]),
         "stable": coefficients(angle, [stable], [la], rng)[0],                       # the registered test
         "reliable": coefficients(angle, [reliable], [la], rng)[0],                   # reference: the confidence
         "stable_length": coefficients(angle, [stable], [la, ll, ll ** 2], rng)[0],
         "log_displacement": coefficients(angle, [disp], [la], rng)[0],
         "reliable_and_stable": coefficients(angle, [reliable * stable], [la], rng)[0]}
    both = coefficients(angle, [stable, reliable], [la], rng)
    r["joint_stable"], r["joint_reliable"] = both
    out["regions"][noaa] = r
KEYS = ("stable", "reliable", "stable_length", "log_displacement", "reliable_and_stable", "joint_stable", "joint_reliable")
for k in KEYS:
    out["pooled"][k] = pooled([out["regions"][n][k] for n in REGIONS])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
for k in KEYS:
    row = " | ".join(f"{n} {out['regions'][n][k]['value']:+.2f} [{out['regions'][n][k]['ci'][0]:+.2f}, {out['regions'][n][k]['ci'][1]:+.2f}]"
                     for n in REGIONS)
    p = out["pooled"][k]
    print(f"{k:20s} {row} | three {p['value']:+.2f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]")
for n in REGIONS:
    r = out["regions"][n]
    print(n, f"n {r['n_lines']}, stable {r['stable_frac']:.2f}, reliable {r['reliable_frac']:.2f} | median angle: " +
          ", ".join(f"{k} {v:.1f}" for k, v in r["median_angle"].items()))
print("written", OUT)
