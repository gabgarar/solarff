"""Does the AIA measure penalise long lines? Control of the number of points.

The angle of each line is the median over the points that fall on clear ridges.
Long, high lines (mostly unreliable) may have few valid points, and a median with
few points is noisier and tends to 45°. Here we measure whether that explains part
of the difference between reliable and unreliable lines:

1. number of valid points per line, by group, and its relation with the angle;
2. the reliable − unreliable difference using **the same number of points** in
   every line (k random points, repeated many times);
3. regression angle ~ reliable + log(1 + height) + log(points).

Saves ``docs/resultados/aia_npoints.json``.

    python examples/14_aia_npoints_bias.py
"""
import glob
import json
import os
import pickle
import warnings

import numpy as np

from solarff.aia import _sample, ridge_orientation
from solarff.confidence import confidence_map

warnings.filterwarnings("ignore")
K = 8          # points per line in the matched comparison
N_REP = 200    # subsampling repetitions


def local_angles(P, th, st, thr, step_px=2.0):
    """Local angles (degrees) of a projected line at the points on clear ridges."""
    if len(P) < 3:
        return np.array([])
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 3 * step_px:
        return np.array([])
    si = np.arange(0, s[-1], step_px)
    x, y = np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])
    tx, ty = np.gradient(x), np.gradient(y)
    ang = np.mod(np.arctan2(ty, tx), np.pi)
    t, w = _sample(th, x, y), _sample(st, x, y)
    ok = np.isfinite(t) & (w >= thr) & (np.hypot(tx, ty) > 1e-6)
    d = np.abs(ang[ok] - t[ok])
    return np.degrees(np.minimum(d, np.pi - d))


def study_for(aia_path):
    if os.path.basename(aia_path) == "aia_validation_harp377.pkl":
        return "data/harp377_study_mc10.pkl"
    stem = aia_path[:-len("_aia.pkl")]
    c = sorted(glob.glob(stem + "_r*_mc*.pkl"))
    return c[0] if c else None


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = r @ r / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se


out = {}
for path in ["data/aia_validation_harp377.pkl"] + sorted(glob.glob("data/harp*_aia.pkl")):
    sp = study_for(path)
    if not os.path.exists(path) or not sp:
        continue
    res = pickle.load(open(path, "rb"))
    S, plot = res["summary"], res["plot"]
    rel = np.asarray(plot["reliable"], bool)
    study, _ = pickle.load(open(sp, "rb"))
    src = study.mag_pp if study.mag_pp is not None else study.mag
    step = max(1, int(round(max(study.mag.shape) / 80)))
    cm = confidence_map(study.nlfff, src, step=step, noise_bt=float(study.mag.meta.get("noise") or 100.0))
    lines = cm["lines"]
    if len(lines) != len(rel):
        print(f"{path}: {len(lines)} líneas frente a {len(rel)} en AIA; se omite")
        continue
    apex = np.array([l.apex for l in lines])
    rng = np.random.default_rng(0)
    R = {}
    for w, d in plot["wavelengths"].items():
        th, st = ridge_orientation(np.asarray(d["cutout"], float), sigma=float(S.get("sigma_px", 2.0)))
        thr = float(np.percentile(st, float(S.get("ridge_pct", 90))))
        loc = [local_angles(np.asarray(p, float), th, st, thr) for p in d["pix"]]
        n = np.array([len(a) for a in loc])
        med = np.array([np.median(a) if len(a) >= 5 else np.nan for a in loc])  # as in the original analysis
        ok = np.isfinite(med)
        r = {"n_points_median": {"reliable": float(np.median(n[ok & rel])), "unreliable": float(np.median(n[ok & ~rel]))},
             "corr_npoints_angle": float(np.corrcoef(np.log(n[ok]), med[ok])[0, 1]),
             "diff_original": float(np.median(med[ok & rel]) - np.median(med[ok & ~rel]))}
        # 2. same number of points in every line
        elig = np.array([len(a) >= K for a in loc])
        diffs = []
        for _ in range(N_REP):
            m = np.full(len(loc), np.nan)
            for i in np.where(elig)[0]:
                m[i] = np.median(rng.choice(loc[i], K, replace=False))
            diffs.append(np.nanmedian(m[rel]) - np.nanmedian(m[~rel]))
        r["equal_points"] = {"k": K, "n_reliable": int((elig & rel).sum()), "n_unreliable": int((elig & ~rel).sum()),
                             "diff": float(np.median(diffs)), "ci": [float(np.percentile(diffs, 2.5)),
                                                                      float(np.percentile(diffs, 97.5))]}
        # 3. regression with height and number of points
        X1 = np.column_stack([rel[ok].astype(float), np.log1p(apex[ok])])
        X2 = np.column_stack([X1, np.log(n[ok])])
        b1, s1 = ols(med[ok], X1)
        b2, s2 = ols(med[ok], X2)
        r["ols_height"] = {"reliable": float(b1[1]), "ci": [float(b1[1] - 1.96 * s1[1]), float(b1[1] + 1.96 * s1[1])]}
        r["ols_height_npoints"] = {"reliable": float(b2[1]), "ci": [float(b2[1] - 1.96 * s2[1]), float(b2[1] + 1.96 * s2[1])],
                                   "log_npoints": float(b2[3])}
        R[str(w)] = r
        print(f"{os.path.basename(path)} {w} Å: puntos (mediana) fiables {r['n_points_median']['reliable']:.0f} / "
              f"no fiables {r['n_points_median']['unreliable']:.0f}; corr(log n, ángulo) {r['corr_npoints_angle']:+.2f}\n"
              f"   fiables − no fiables: original {r['diff_original']:+.1f}°, mismos {K} puntos {r['equal_points']['diff']:+.1f}° "
              f"{np.round(r['equal_points']['ci'], 1).tolist()}; regresión con altura {r['ols_height']['reliable']:+.1f}° "
              f"{np.round(r['ols_height']['ci'], 1).tolist()}, + puntos {r['ols_height_npoints']['reliable']:+.1f}° "
              f"{np.round(r['ols_height_npoints']['ci'], 1).tolist()}", flush=True)
    out[os.path.basename(path)] = {"region": S.get("region"), "by_wavelength": R}

with open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "resultados"), "aia_npoints.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
