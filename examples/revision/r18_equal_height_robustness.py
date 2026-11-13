"""Robustness of the equal-height test of the resolution check (post hoc, declared as such).

``examples/22_resolution_check.py`` finds that, at equal apex height, the lines accepted at the published
resolution move less when the resolution doubles than the rejected ones. The registered test controls the
height with one term, log(1 + apex). This script asks whether the difference survives harder controls, on the
per-line arrays that script 22 leaves in ``data/resolution_check_lines_<NOAA>.npz``:

- ``registered``: the registered regression (reference).
- ``height_bins``: apex height in ten bins (fixed effects) instead of one logarithmic term.
- ``height_length``: log height, log length and its square. The confidence uses the length, and the
  displacement is relative to the length, so this is the control that could remove the difference.
- ``height_length_bins``: ten bins of height and ten of length.
- ``double``: the decision taken at the finer resolution instead of the published one.
- ``continuous``: change per 0.1 of confidence instead of the accepted / rejected decision.

The outcome is the registered one (the line moves by more than 20 % of its length or changes between open
and closed) and, apart, the change between open and closed alone. Intervals by bootstrap over lines; the
three regions are combined weighting by 1/sigma^2.

    python examples/revision/r18_equal_height_robustness.py
"""
import json
import os

import numpy as np

from solarff.confidence import DEFAULT_CALIBRATION

REGIONS = ("11158", "11429", "12673")
T = DEFAULT_CALIBRATION["threshold"]
N_BOOT = 2000
OUT = os.path.join("docs", "revision", "resultados", "r18_equal_height_robustness.json")


def bins(x, n=10):
    """Indicator columns of ``n`` quantile bins of ``x`` (the first one dropped)."""
    q = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)[1:-1]))
    k = np.searchsorted(q, x)
    return np.column_stack([(k == j).astype(float) for j in range(1, len(q) + 1)])


def coefficient(y, x, controls, rng):
    X = np.column_stack([np.ones(len(y)), x] + controls)
    def coef(i):
        return float(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
    idx = np.arange(len(y))
    b = [coef(rng.choice(idx, len(idx))) for _ in range(N_BOOT)]
    return {"value": coef(idx), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "sigma": float(np.std(b))}


def pooled(rs):
    v, s = np.array([r["value"] for r in rs]), np.array([r["sigma"] for r in rs])
    w = 1 / s ** 2
    m, sd = float((w * v).sum() / w.sum()), float(w.sum() ** -0.5)
    return {"value": m, "ci": [m - 1.96 * sd, m + 1.96 * sd], "sigma": sd}


out = {"threshold": T, "n_boot": N_BOOT, "regions": {}, "pooled": {}}
for noaa in REGIONS:
    d = np.load(os.path.join("data", f"resolution_check_lines_{noaa}.npz"))
    c_lo, c_hi, shape, topo, apex = d["conf_published"], d["conf_double"], d["err_shape"], d["err_topology"] > 0, d["apex"]
    length = d["err_length_ref"]
    use = np.isfinite(shape) & np.isfinite(c_lo) & np.isfinite(c_hi) & (length > 0)
    c_lo, c_hi, shape, topo, apex, length = (v[use] for v in (c_lo, c_hi, shape, topo, apex, length))
    la, ll = np.log(1 + apex), np.log(length)
    outcomes = {"moved_20": ((shape > 0.2) | topo).astype(float), "open_closed": topo.astype(float)}
    rel_lo, rel_hi = (c_lo >= T).astype(float), (c_hi >= T).astype(float)
    specs = {"registered": (rel_lo, [la]),
             "height_bins": (rel_lo, [bins(apex)]),
             "height_length": (rel_lo, [la, ll, ll ** 2]),
             "height_length_bins": (rel_lo, [bins(apex), bins(length)]),
             "double": (rel_hi, [la]),
             "double_height_length": (rel_hi, [la, ll, ll ** 2]),
             "continuous": (c_lo / 0.1, [la]),
             "continuous_height_length": (c_lo / 0.1, [la, ll, ll ** 2])}
    rng = np.random.default_rng(0)
    r = {"n_lines": int(use.sum()),
         "by_confidence": {f"{lo:.1f}-{hi:.1f}": {"n": int(((c_lo >= lo) & (c_lo < hi)).sum()),
                                                 "moved_20": float(outcomes["moved_20"][(c_lo >= lo) & (c_lo < hi)].mean()),
                                                 "open_closed": float(topo[(c_lo >= lo) & (c_lo < hi)].mean())}
                           for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.01))}}
    for name, y in outcomes.items():
        r[name] = {k: coefficient(y, x, ctl, rng) for k, (x, ctl) in specs.items()}
    out["regions"][noaa] = r
for name in ("moved_20", "open_closed"):
    out["pooled"][name] = {k: pooled([out["regions"][n][name][k] for n in REGIONS]) for k in out["regions"][REGIONS[0]][name]}
    out["pooled"][name + "_new_regions"] = {k: pooled([out["regions"][n][name][k] for n in REGIONS[1:]])
                                            for k in out["regions"][REGIONS[0]][name]}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
for name in ("moved_20", "open_closed"):
    print(name)
    for k in out["pooled"][name]:
        row = " | ".join(f"{n} {out['regions'][n][name][k]['value']:+.3f} [{out['regions'][n][name][k]['ci'][0]:+.3f}, "
                         f"{out['regions'][n][name][k]['ci'][1]:+.3f}]" for n in REGIONS)
        p, q = out["pooled"][name][k], out["pooled"][name + "_new_regions"][k]
        print(f"  {k:26s} {row} | three {p['value']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}] | two new {q['value']:+.3f} "
              f"[{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}]")
for n in REGIONS:
    print(n, " | ".join(f"conf {k}: n {v['n']}, moved {v['moved_20']:.2f}, open/closed {v['open_closed']:.2f}"
                        for k, v in out["regions"][n]["by_confidence"].items()))
print("written", OUT)
