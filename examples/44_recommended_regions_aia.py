"""The recommended treatment of the base in three observed regions: do its lines follow the AIA loops better?

Against a simulated active region the configuration of ``examples/40_base_treatment.py`` (``mu3`` 0.1, ``mu4`` 0.01,
weight of the base term 0.01) gives fewer wrong lines than the default one (``examples/43_base_weight_regimes.py``).
In an observation there is no truth. The only external reference is the loops seen by SDO/AIA, which the lines
should follow. Here the same seeds are traced in the model of each region made with the default configuration (the
one of the reports) and in a model made with the recommended one, and the angle of every line with the loops of
AIA 171 and 193 A is measured exactly as in ``examples/08_aia_validation.py``, on the same cut-out of the image.

Per region: the per-line difference of the angle (recommended minus default; the angle of a line is the mean of the
channels it has), with an interval by bootstrap over lines; the same against the potential field; and whether the
lines that the ranking of ``examples/43`` puts first are better aligned at equal height. The three regions are
combined weighting by 1/sigma^2. The criteria were written down in the plan of the paper before running this.

Needs the saved studies of the three regions (``examples/02_hmi_sharp.py``), their AIA images in ``data/aia`` and
``docs/resultados/base_weight_regimes.json``. Saves ``data/regions_recommended.pkl`` (the models and the angles) and
``docs/resultados/recommended_regions_aia.json``.

    python -u examples/44_recommended_regions_aia.py
    python examples/44_recommended_regions_aia.py --check        # only the default model: must reproduce the saved validation
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.aia import line_misalignment, load_aia, project_points, ridge_orientation
from solarff.confidence import rank_lines
from solarff.nlfff_torch import nlfff_lbfgs
from solarff.preprocess import preprocess
from solarff.report.analysis import analyze
from solarff.tracing import trace_field_lines
from solarff.utils import code_version

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true", help="only the default model, compared with the saved AIA validation")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "recommended_regions_aia.json"))
ap.add_argument("--ranking", default=os.path.join("docs", "resultados", "base_weight_regimes.json"))
a = ap.parse_args()

REGIONS = {  # NOAA: saved study, Br segment of the SHARP, saved AIA validation
    "11158": ("data/harp377_study_mc10.pkl", "data/harp377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits", "data/aia_validation_harp377.pkl"),
    "11429": ("data/harp1449_20120306_2336_r7_nz32_mc10.pkl", "data/harp1449_20120306_2336/hmi.sharp_cea_720s.1449.20120306_233600_TAI.Br.fits", "data/harp1449_20120306_2336_aia.pkl"),
    "12673": ("data/harp7115_20170906_1136_r4_nz56_mc10.pkl", "data/harp7115_20170906_1136/hmi.sharp_cea_720s.7115.20170906_113600_TAI.Br.fits", "data/harp7115_20170906_1136_aia.pkl"),
}
RECOMMENDED = {"mu3": 1e-1, "mu4": 1e-2, "relax_bottom": 1e-2}
SIGMA, PCT, N_BOOT = 2.0, 90, 2000                                              # the loop detection of examples/08
CKPT = "data/regions_recommended.pkl"
state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) and not a.check else {}


def angles(sets, files, sharp):
    """Angle with the loops (degrees) of every line of every set, by wavelength, on one cut-out common to the sets."""
    out = {}
    for w, path in files.items():
        m = load_aia(path)
        pix = {k: [project_points(np.asarray(l.points), sharp, m) for l in ls] for k, ls in sets.items()}
        allp = np.vstack([p for ps in pix.values() for p in ps])
        x0, y0 = np.maximum(np.floor(allp.min(0) - 30).astype(int), 0)
        x1, y1 = np.ceil(allp.max(0) + 30).astype(int)
        th, st = ridge_orientation(m.data[y0:y1, x0:x1], sigma=SIGMA)
        thr = float(np.percentile(st, PCT))
        out[str(w)] = {k: line_misalignment([p - np.array([x0, y0]) for p in ps], th, st, thr) for k, ps in pix.items()}
    return out


def line_angle(ang, k):
    """Angle of each line: the mean of the channels in which it could be measured."""
    return np.nanmean(np.stack([ang[w][k] for w in ang]), axis=0)


def paired(d, rng):
    d = d[np.isfinite(d)]
    b = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(N_BOOT)])
    return {"n": int(d.size), "mean": float(d.mean()), "median": float(np.median(d)), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "sigma": float(b.std())}


def combine(rows):
    w = np.array([1 / r["sigma"] ** 2 for r in rows])
    m, s = float((w * np.array([r["mean"] for r in rows])).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
    return {"mean": m, "sigma": s, "ci": [m - 1.96 * s, m + 1.96 * s]}


def slope(y, x, apex, rng):
    """Coefficient of x in the regression of the angle on x and log(1 + apex height), with its interval by bootstrap over lines."""
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(apex)
    X, y = np.column_stack([np.ones(ok.sum()), x[ok], np.log1p(apex[ok])]), y[ok]
    f = lambda i: np.linalg.lstsq(X[i], y[i], rcond=None)[0][1]  # noqa: E731
    idx = np.arange(len(y))
    b = np.array([f(rng.integers(0, len(y), len(y))) for _ in range(N_BOOT)])
    return {"n": int(len(y)), "value": float(f(idx)), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "sigma": float(b.std())}


t0 = time.time()
out = {"code_version": code_version()["label"], "recommended": RECOMMENDED, "regions": {}}
ranking = None if a.check else json.load(open(a.ranking, encoding="utf-8"))["ranking"]["calibration"]
rng = np.random.default_rng(0)
for noaa, (cache, sharp, saved) in REGIONS.items():
    study, mc = pickle.load(open(cache, "rb"))
    S = pickle.load(open(saved, "rb"))["summary"]
    files = {w: os.path.join("data", "aia", r["file"]) for w, r in S["by_wavelength"].items()}
    cm = analyze(study, monte_carlo=mc)["_maps"]["confidence"]
    lines = cm["lines"]                                                         # the lines of the saved validation, traced in the default model
    seeds = np.array([l.seed for l in lines], dtype=float)
    assert len(lines) == S["n_lines"], f"{noaa}: {len(lines)} lines against {S['n_lines']} in the saved validation"
    if a.check:
        ang = angles({"default": lines}, files, sharp)
        rel = np.asarray(cm["assess"]["reliable"], bool)
        for w, r in S["by_wavelength"].items():
            x = ang[str(w)]["default"]
            print(f"NOAA {noaa}, AIA {w}: {int(np.isfinite(x).sum())} lines measured (saved {r['n_measured']}); median of the reliable {np.nanmedian(x[rel]):.2f} (saved {r['median_reliable']:.2f}), "
                  f"of the others {np.nanmedian(x[~rel]):.2f} (saved {r['median_unreliable']:.2f})  [{time.time() - t0:.0f} s]", flush=True)
        continue
    if noaa not in state:
        tc = time.time()
        nz = study.nlfff.shape[2]
        pp = preprocess(study.mag, mu3=RECOMMENDED["mu3"], mu4=RECOMMENDED["mu4"], verbose=False)
        model = nlfff_lbfgs(pp, nz, verbose=False, max_iter=4000, relax_bottom=RECOMMENDED["relax_bottom"])
        sets = {"default": lines, "recommended": trace_field_lines(model, seeds), "potential": trace_field_lines(study.potential, seeds)}
        noise_bt = float(study.mag.meta.get("noise") or 100.0)
        rk = rank_lines(model, sets["recommended"], pp, noise_bt=noise_bt, calibration=ranking)
        state[noaa] = {"angles": angles(sets, files, sharp), "apex": {k: np.array([l.apex for l in ls]) for k, ls in sets.items()}, "score": rk["score"], "percentile": rk["percentile"],
                       "iterations": int(model.meta["iterations"]), "converged": str(model.meta["message"]).startswith("convergido"), "grid": list(model.shape), "dx": float(model.dx),
                       "misfit": float(model.meta["L_data"] / RECOMMENDED["relax_bottom"]), "seconds": time.time() - tc}
        pickle.dump(state, open(CKPT, "wb"))
    st = state[noaa]
    A = {k: line_angle(st["angles"], k) for k in ("default", "recommended", "potential")}
    r = {"n_lines": int(len(A["default"])), "grid": st["grid"], "dx_Mm": st["dx"], "iterations": st["iterations"], "converged": st["converged"],
         "median_angle": {k: float(np.nanmedian(v)) for k, v in A.items()}, "measured": {k: int(np.isfinite(v).sum()) for k, v in A.items()},
         "recommended_minus_default": paired(A["recommended"] - A["default"], rng), "recommended_minus_potential": paired(A["recommended"] - A["potential"], rng),
         "default_minus_potential": paired(A["default"] - A["potential"], rng),
         "by_wavelength": {w: paired(v["recommended"] - v["default"], rng) for w, v in st["angles"].items()},
         "ranking": slope(A["recommended"], st["percentile"], st["apex"]["recommended"], rng)}
    third = st["percentile"] <= 1 / 3
    r["ranking"]["median_angle_best_third"], r["ranking"]["median_angle_rest"] = float(np.nanmedian(A["recommended"][third])), float(np.nanmedian(A["recommended"][~third & np.isfinite(st["percentile"])]))
    out["regions"][noaa] = r
    d = r["recommended_minus_default"]
    print(f"NOAA {noaa}: {r['measured']} lines measured; median angle {r['median_angle']}; recommended minus default {d['mean']:+.2f} deg ({d['ci'][0]:+.2f} to {d['ci'][1]:+.2f}); "
          f"ranking slope {r['ranking']['value']:+.2f} ({r['ranking']['ci'][0]:+.2f} to {r['ranking']['ci'][1]:+.2f})  [{time.time() - t0:.0f} s]", flush=True)

if not a.check:
    R = out["regions"]
    out["combined"] = {k: combine([R[n][k] for n in R]) for k in ("recommended_minus_default", "recommended_minus_potential", "default_minus_potential")}
    out["combined"]["ranking"] = combine([{"mean": R[n]["ranking"]["value"], "sigma": R[n]["ranking"]["sigma"]} for n in R])
    c = out["combined"]
    out["criteria"] = {"G1": bool(all(R[n]["recommended_minus_default"]["ci"][0] <= 0 for n in R)), "G2": bool(c["recommended_minus_default"]["ci"][1] < 0),
                       "G3": bool(c["ranking"]["ci"][0] > 0)}
    json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("combined:", {k: (round(v["mean"], 2), [round(x, 2) for x in v["ci"]]) for k, v in c.items()})
    print("criteria:", out["criteria"])
    print("written", a.json)
