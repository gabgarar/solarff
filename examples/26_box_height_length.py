"""Length of the line relative to the height of the box: does the published model transfer to flat boxes?

The published model measures the length of a line in units of the horizontal diagonal of the box. All the boxes of
the suite are ``n x n x 0.7 n`` (height / diagonal = 0.48 to 0.50), so the calibration never saw another proportion.
The boxes of the three regions are flat (height / diagonal = 0.18, 0.23 and 0.27), and in the exact fields built on
their observed flux (``examples/25_observed_flux_fields.py``) the published model accepts long lines that are wrong.

Here the published coefficients are applied, without refitting anything, with the length measured relative to the
height and scaled to coincide with the published definition in the boxes of the suite:

    length_h = length * (diagonal / height) * C0,    C0 = 0.4898 (mean height / diagonal of the three suite sizes)

and the result is compared with the published definition on the six observed-flux fields and on the ten families of
the suite (per-line table). A second rule, with the smallest dimension of the box (reference diagonal = sqrt(2) times
the minimum of Lx, Ly and Lz / 0.7), is reported as a check: in the three boxes the height is the smallest.

This variant is conservative, not a better scale of the length: with the fields of the suite in flat boxes it
rejects half the lines that the published model accepts, most of them correct, and in a tall box it accepts more
wrong lines than the published definition (``examples/27_box_proportion.py``).

The criteria (N1 to N4) were registered in the plan of the paper before computing (13-Nov-2026).

Reads ``data/observed_flux_fields.pkl`` and ``docs/resultados/line_table.csv.gz``; writes
``docs/resultados/box_height_length.json``. No extrapolation is run (seconds).

    python examples/26_box_height_length.py
"""
import csv
import gzip
import json
import os
import pickle

import numpy as np

from solarff.calibration import _nz, ece
from solarff.confidence import CALIBRATION_V1 as DEFAULT_CALIBRATION, auc, confidence  # the default model when this test was made (version 1.1)
from solarff.utils import code_version

T = DEFAULT_CALIBRATION["threshold"]
SIZES = (32, 48, 64)
N_BOOT = 1000


def ratio(nx, ny, nz):
    """Height / horizontal diagonal of a box, as ``line_indicators`` measures them."""
    return (nz - 1) / np.hypot(nx - 1, ny - 1)


def min_scale(nx, ny, nz):
    """Factor of the second rule: published diagonal / (sqrt(2) times the smallest of Lx, Ly and Lz / 0.7)."""
    return np.hypot(nx - 1, ny - 1) / (np.sqrt(2) * min(nx - 1, ny - 1, (nz - 1) / 0.7))


C0 = float(np.mean([ratio(n, n, _nz(n)) for n in SIZES]))


def conf(ind, factor):
    """Published confidence with the relative length multiplied by ``factor`` (1 = published definition)."""
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
    for t in (0.9,):
        m = c >= t
        out[f"accepted_{t}"], out[f"bad_accepted_{t}"] = float(m.mean()), (float(y[m].mean()) if m.any() else None)
    return out


def three(ind, y, r, ms):
    """Published definition, height rule and smallest-dimension rule on the same lines."""
    rng = np.random.default_rng(0)
    cp, ch, cm = conf(ind, 1.0), conf(ind, C0 / r), conf(ind, ms)
    out = {"published": evaluate(cp, y, rng), "height": evaluate(ch, y, rng), "min_dimension": evaluate(cm, y, rng)}
    ok = np.isfinite(cp) & np.isfinite(ch)
    ap, ah = cp[ok] >= T, ch[ok] >= T
    out["accepted_kept"] = float((ap & ah).sum() / max(ap.sum(), 1))    # accepted by both / accepted by the published
    out["accepted_ratio"] = float(ah.sum() / max(ap.sum(), 1))
    return out


out = {"code_version": code_version()["label"], "calibration": "DEFAULT_CALIBRATION, coefficients not refitted", "c0": C0,
       "suite_ratio_by_size": {str(n): float(ratio(n, n, _nz(n))) for n in SIZES}, "observed": {}, "suite": {}}

# --- observed-flux fields
st = pickle.load(open(os.path.join("data", "observed_flux_fields.pkl"), "rb"))
pooled = {}
for name, e in st["exact"].items():
    d = [x for k, x in st["cases"].items() if k.startswith(name + " |")]
    ind = {k: np.concatenate([np.asarray(x["ind"][k], float) for x in d]) for k in ("alpha_spread", "buffer", "length")}
    y = np.concatenate([x["bad"] for x in d])
    r, ms = ratio(*e["grid"]), min_scale(*e["grid"])
    res = three(ind, y, r, ms)
    res.update(height_over_diagonal=float(r), factor_height=float(C0 / r), factor_min_dimension=float(ms))
    out["observed"][name] = res
    f = name.split()[0]
    pooled.setdefault(f, []).append((ind, y, r, ms))
for f, parts in pooled.items():
    rng = np.random.default_rng(0)
    y = np.concatenate([p[1] for p in parts])
    out["observed"][f + " (three regions)"] = {
        "published": evaluate(np.concatenate([conf(p[0], 1.0) for p in parts]), y, rng),
        "height": evaluate(np.concatenate([conf(p[0], C0 / p[2]) for p in parts]), y, rng)}

# --- suite: per-line table of the ten families
rows = list(csv.DictReader(gzip.open(os.path.join("docs", "resultados", "line_table.csv.gz"), "rt", encoding="utf-8")))
# as in the table of version 1.1, with which this test was made: three sets, all in boxes of standard proportion
rows = [r for r in rows if r["set"] in ("calibration", "rerun", "external") and int(r.get("nz") or _nz(int(r["size"]))) == _nz(int(r["size"]))]


def num(r, k):
    try:
        return float(r[k])
    except ValueError:
        return np.nan


def block(rs):
    ind = {k: np.array([num(r, k) for r in rs]) for k in ("alpha_spread", "buffer", "length")}
    y = np.array([int(r["wrong"]) for r in rs])
    size = np.array([int(r["size"]) for r in rs])
    rr = np.array([ratio(n, n, _nz(n)) for n in size])
    ms = np.array([min_scale(n, n, _nz(n)) for n in size])
    res = three(ind, y, rr, ms)
    stored = np.array([num(r, "confidence_v1" if "confidence_v1" in r else "confidence") for r in rs])   # the model of version 1.1
    both = np.isfinite(stored) & np.isfinite(conf(ind, 1.0))
    res["max_diff_with_table"] = float(np.abs(stored[both] - conf(ind, 1.0)[both]).max())
    return res


cal = [r for r in rows if r["set"] == "calibration"]
out["suite"]["calibration (six families)"] = block(cal)
for fam in sorted({r["family"] for r in rows if r["set"] != "calibration"}):
    out["suite"][fam] = block([r for r in rows if r["set"] != "calibration" and r["family"] == fam])
for fam in sorted({r["family"] for r in cal}):
    out["suite"]["calibration: " + fam] = block([r for r in cal if r["family"] == fam])

json.dump(out, open(os.path.join("docs", "resultados", "box_height_length.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def fmt(v, f=".3f"):
    return "-" if v is None else format(v, f)


print(f"C0 = {C0:.4f}; suite height / diagonal by size: {out['suite_ratio_by_size']}")
for sec in ("observed", "suite"):
    print(f"\n{sec}")
    for name, r in out[sec].items():
        p, h = r["published"], r["height"]
        extra = f" | factor {r['factor_height']:.2f}" if "factor_height" in r else ""
        extra += f" | min-dimension rule: AUC {r['min_dimension']['auc']:.3f}, wrong accepted {fmt(r['min_dimension']['bad_accepted'])}" if "min_dimension" in r else ""
        extra += f" | max diff with the table {r['max_diff_with_table']:.1e}" if "max_diff_with_table" in r else ""
        print(f"  {name:28s} AUC {p['auc']:.3f} -> {h['auc']:.3f} | ECE {p['ece']:.3f} -> {h['ece']:.3f} | accepted {p['accepted']:.2f} -> {h['accepted']:.2f} | "
              f"wrong among accepted {fmt(p['bad_accepted'])} -> {fmt(h['bad_accepted'])} {[round(x, 3) for x in h.get('bad_accepted_ci', [])]} | "
              f"at 0.9: {fmt(p['bad_accepted_0.9'])} -> {fmt(h['bad_accepted_0.9'])} (accepted {h['accepted_0.9']:.2f}){extra}")
print("written docs/resultados/box_height_length.json")
