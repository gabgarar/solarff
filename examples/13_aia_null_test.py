"""Null test of the AIA comparison: how much does a line align by chance?

In an active region the loops form a fan, so any line placed in the area already
tends to follow them somewhat. To measure that chance level, the projected NLFFF
lines are **shifted** (10–40 Mm in a random direction) or **rotated** (a random
angle about their centre) on the same image and their angle with the loops is
measured exactly as for the lines in place.

For each saved AIA validation (``data/aia_validation_harp377.pkl`` and
``data/harp*_aia.pkl``) adds ``summary["null"]`` and saves it again; the
report uses it. Summary in ``docs/resultados/aia_null.json``.

    python examples/13_aia_null_test.py --n 200
    # sensitivity to the shift, without touching the saved validations:
    python examples/13_aia_null_test.py --n 200 --shift-mm 5 15 --no-save --json <output>.json
"""
import argparse
import glob
import json
import os
import pickle
import warnings

import numpy as np

from solarff.aia import line_misalignment, ridge_orientation

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100, help="realizaciones por tipo de prueba")
ap.add_argument("--shift-mm", type=float, nargs=2, default=[10.0, 40.0])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs",
                                              "resultados", "aia_null.json"))
ap.add_argument("--no-save", action="store_true", help="no añadir el resultado a las validaciones AIA")
a = ap.parse_args()

AIA_PX_MM = 0.6 / 3600 * np.pi / 180 * 1.496e5  # 0.6" at 1 AU, in Mm (~0.435)
files = ["data/aia_validation_harp377.pkl"] + sorted(glob.glob("data/harp*_aia.pkl"))
rng = np.random.default_rng(a.seed)
summary = {}


def med(ang, m):
    v = ang[m & np.isfinite(ang)]
    return float(np.median(v)) if v.size else np.nan


for path in files:
    if not os.path.exists(path):
        continue
    res = pickle.load(open(path, "rb"))
    S, plot = res["summary"], res["plot"]
    rel = np.asarray(plot["reliable"], bool)
    sigma, pct = float(S.get("sigma_px", 2.0)), float(S.get("ridge_pct", 90))
    out = {"n": a.n, "shift_mm": a.shift_mm, "by_wavelength": {}}
    for w, d in plot["wavelengths"].items():
        th, st = ridge_orientation(np.asarray(d["cutout"], float), sigma=sigma)
        thr = float(np.percentile(st, pct))
        pix = [np.asarray(p, float) for p in d["pix"]]
        ang0 = np.asarray(d["angles"], float)
        obs = {"all": med(ang0, np.ones_like(rel)), "reliable": med(ang0, rel), "unreliable": med(ang0, ~rel)}
        nulls = {}
        for kind in ("shift", "rotate"):
            vals = {"all": [], "reliable": [], "unreliable": []}
            for _ in range(a.n):
                moved = []
                for p in pix:
                    if kind == "shift":
                        r = rng.uniform(*a.shift_mm) / AIA_PX_MM
                        phi = rng.uniform(0, 2 * np.pi)
                        moved.append(p + r * np.array([np.cos(phi), np.sin(phi)]))
                    else:
                        c = p.mean(0)
                        t = rng.uniform(0, np.pi)
                        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                        moved.append((p - c) @ R.T + c)
                ang = line_misalignment(moved, th, st, thr)
                for g, m in (("all", np.ones_like(rel)), ("reliable", rel), ("unreliable", ~rel)):
                    vals[g].append(med(ang, m))
            nulls[kind] = {g: {"median": float(np.nanmedian(v)), "p2.5": float(np.nanpercentile(v, 2.5)),
                               "p97.5": float(np.nanpercentile(v, 97.5)),
                               "p_value": float((np.sum(np.asarray(v) <= obs[g]) + 1) / (len(v) + 1))}
                           for g, v in vals.items()}
        out["by_wavelength"][w] = {"observed": obs, "null": nulls}
        print(f"{os.path.basename(path)} {w} Å: observado {obs['all']:.1f}° (fiables {obs['reliable']:.1f}°, "
              f"no fiables {obs['unreliable']:.1f}°); desplazadas {nulls['shift']['all']['median']:.1f}° "
              f"[{nulls['shift']['all']['p2.5']:.1f}, {nulls['shift']['all']['p97.5']:.1f}]; giradas "
              f"{nulls['rotate']['all']['median']:.1f}° [{nulls['rotate']['all']['p2.5']:.1f}, "
              f"{nulls['rotate']['all']['p97.5']:.1f}]", flush=True)
    if not a.no_save:
        S["null"] = out
        with open(path, "wb") as fh:
            pickle.dump(res, fh)
    summary[os.path.basename(path)] = {"region": S.get("region"), **out}

with open(a.json, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=1, ensure_ascii=False)
