"""The proportion of the box under control: the calibration fields in flat and tall boxes.

All the boxes of the calibration suite are ``n x n x 0.7 n``. The published model measures the length of a line in
units of the horizontal diagonal, and in the flat boxes of the regions it accepts long lines that are wrong
(``examples/25_observed_flux_fields.py``); with the length relative to the height of the box it does not
(``examples/26_box_height_length.py``). That rule came out of looking at those fields. This is the test on cases
that had not been looked at: the six exact families of the calibration, with the same settings as the suite,
changing only the height of the box.

    flat        64 x 64 x 22   (height / diagonal 0.236; NOAA 11158 has 0.227)
    very flat   64 x 64 x 16   (0.168; NOAA 11429 has 0.175)
    tall        48 x 48 x 48   (0.707)

The reference is the suite itself (64 x 64 x 45 and 48 x 48 x 34, per-line table). Three definitions of the relative
length are compared with the published coefficients, without refitting: the published one (horizontal diagonal),
the height rule and the smallest-dimension rule (see ``examples/26_box_height_length.py``).

The criteria (A1 to A3) were registered in the plan of the paper before running (13-Nov-2026).

Saves ``data/box_proportion.pkl`` (checkpoint, with the per-line records) and ``docs/resultados/box_proportion.json``.

    python -u examples/27_box_proportion.py                 # ~1.5 h
    python examples/27_box_proportion.py --analyze-only
"""
import argparse
import csv
import gzip
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.analytic import CALIBRATION_FIELDS, EXTERNAL_TEST_FIELDS, field_magnetogram
from solarff.calibration import NOISE_MODELS, _nz, ece
from solarff.confidence import DEFAULT_CALIBRATION, auc, calibration_case, confidence
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--boxes", nargs="*", default=["flat", "very flat", "tall"])
ap.add_argument("--families", nargs="*", default=None)
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "box_proportion.json"))
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

BOXES = {"flat": (64, 22), "very flat": (64, 16), "tall": (48, 48)}      # (n, nz)
# The six families of the calibration (TD-relax is defined with the external ones) and, apart, LFFF-delta: the
# strongly sheared linear field in which the published model is optimistic also in the boxes of the suite.
KNOWN = {**CALIBRATION_FIELDS, **EXTERNAL_TEST_FIELDS}
SIX = list(CALIBRATION_FIELDS) + ["TD-relax"]
FAMILIES = SIX + ["LFFF-delta"]
SIZES = (32, 48, 64)
T = DEFAULT_CALIBRATION["threshold"]
CKPT = os.path.join(a.data, "box_proportion.pkl")
N_BOOT = 1000


def ratio(nx, ny, nz):
    """Height / horizontal diagonal of a box, as ``line_indicators`` measures them."""
    return (nz - 1) / np.hypot(nx - 1, ny - 1)


def min_scale(nx, ny, nz):
    """Published diagonal / (sqrt(2) times the smallest of Lx, Ly and Lz / 0.7)."""
    return np.hypot(nx - 1, ny - 1) / (np.sqrt(2) * min(nx - 1, ny - 1, (nz - 1) / 0.7))


C0 = float(np.mean([ratio(n, n, _nz(n)) for n in SIZES]))


def conf(ind, factor):
    d = dict(ind)
    d["length"] = np.asarray(ind["length"], float) * factor
    return confidence(d, DEFAULT_CALIBRATION)


def evaluate(c, y, rng):
    ok = np.isfinite(c)
    c, y = c[ok], y[ok].astype(int)
    acc = c >= T
    out = {"n_lines": int(ok.sum()), "bad_rate": float(y.mean()), "auc": auc(1 - c, y), "ece": ece(1 - c, y),
           "accepted": float(acc.mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None}
    if acc.any():
        ya = y[acc]
        b = [ya[rng.integers(0, ya.size, ya.size)].mean() for _ in range(N_BOOT)]
        out["bad_accepted_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    m = c >= 0.9
    out["accepted_0.9"], out["bad_accepted_0.9"] = float(m.mean()), (float(y[m].mean()) if m.any() else None)
    return out


def three(ind, y, n, nz):
    rng = np.random.default_rng(0)
    return {"published": evaluate(conf(ind, 1.0), y, rng), "height": evaluate(conf(ind, C0 / ratio(n, n, nz)), y, rng),
            "min_dimension": evaluate(conf(ind, min_scale(n, n, nz)), y, rng)}


def pool(d):
    ind = {k: np.concatenate([np.asarray(x["ind"][k], float) for x in d]) for k in ("alpha_spread", "buffer", "length")}
    return ind, np.concatenate([x["bad"] for x in d])


a.families = a.families or FAMILIES
state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {"cases": {}}
if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        for bi, box in enumerate(BOXES):
            if box not in a.boxes:
                continue
            n, nz = BOXES[box]
            for fi, fname in enumerate(FAMILIES):
                if fname not in a.families:
                    continue
                exact = None
                for j, (nm, (sz, st)) in enumerate(NOISE_MODELS.items()):
                    key = f"{box} | {fname} | {nm}"
                    if key in state["cases"]:
                        continue
                    if exact is None:
                        exact = KNOWN[fname](n, n, nz)
                    tc = time.time()
                    k = 600011 + 1000 * bi + 10 * fi + j
                    mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=k)
                    rec = calibration_case(exact, mag, nz, n_seeds=300, seed=k + 7, noise_bt=st)
                    rec.update(field=fname, box=box, size=n, nz=nz, noise_model=nm, seconds=time.time() - tc)
                    state["cases"][key] = rec
                    pickle.dump(state, open(CKPT, "wb"))
                    print(f"  {key}: {rec['bad'].mean():.0%} wrong lines ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)", flush=True)

out = {"code_version": code_version()["label"], "calibration": "DEFAULT_CALIBRATION, coefficients not refitted", "c0": C0,
       "boxes": {}, "reference": {}}
for box, (n, nz) in BOXES.items():
    every = [x for k, x in state["cases"].items() if k.startswith(box + " |")]
    d = [x for x in every if x["field"] in SIX]
    if not d:
        continue
    r = {"grid": [n, n, nz], "height_over_diagonal": float(ratio(n, n, nz)), "n_cases": len(d),
         "families_pooled": sorted({x["field"] for x in d}), "pooled": three(*pool(d), n, nz), "families": {}}
    d = every
    for fname in FAMILIES:
        df = [x for x in d if x["field"] == fname]
        if df:
            r["families"][fname] = three(*pool(df), n, nz)
    out["boxes"][box] = r

# reference: the boxes of the suite with the same horizontal size, from the published per-line table
rows = [r for r in csv.DictReader(gzip.open(os.path.join("docs", "resultados", "line_table.csv.gz"), "rt", encoding="utf-8"))
        if r["set"] == "calibration"]
for n in (64, 48):
    rs = [r for r in rows if int(r["size"]) == n]
    ind = {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rs]) for k in ("alpha_spread", "buffer", "length")}
    y = np.array([int(r["wrong"]) for r in rs])
    out["reference"][f"{n} x {n} x {_nz(n)}"] = {"height_over_diagonal": float(ratio(n, n, _nz(n))), "pooled": three(ind, y, n, _nz(n))}
    rs5 = [r for r in rs if r["family"] != "TD-relax"]
    ind = {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rs5]) for k in ("alpha_spread", "buffer", "length")}
    out["reference"][f"{n} x {n} x {_nz(n)}"]["pooled_without_TD-relax"] = three(ind, np.array([int(r["wrong"]) for r in rs5]), n, _nz(n))
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def fmt(v):
    return "-" if v is None else format(v, ".3f")


def line(name, r):
    return (f"  {name:14s} " + " | ".join(
        f"{k}: AUC {r[k]['auc']:.3f}, accepted {r[k]['accepted']:.2f}, wrong among them {fmt(r[k]['bad_accepted'])}"
        for k in ("published", "height", "min_dimension")) + f" | wrong lines {r['published']['bad_rate']:.2f}")


for name, r in out["reference"].items():
    print(f"reference {name} (height / diagonal {r['height_over_diagonal']:.3f})")
    print(line("six families", r["pooled"]))
    print(line("without TD", r["pooled_without_TD-relax"]))
for box, r in out["boxes"].items():
    print(f"{box}: {r['grid']} (height / diagonal {r['height_over_diagonal']:.3f}), {r['n_cases']} cases")
    print(line(f"{len(r['families_pooled'])} families", r["pooled"]))
    for fname, rf in r["families"].items():
        print(line(fname, rf))
print("written", a.json)
