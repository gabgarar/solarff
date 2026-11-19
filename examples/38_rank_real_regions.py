"""The ranking fitted on MHD simulations, applied to the three observed regions of the paper.

There is no truth in a real region, so this is a description, not a test:

- how far the NLFFF lines of each region are from the potential lines (the indicator ``departure``), next to the
  same quantity in the simulations and in the analytical families, which says where the real regions sit;
- how much the order of ``solarff.rank_lines`` agrees with the default confidence, calibrated on exact force-free
  fields: their rank correlation, and where the lines accepted by the default confidence fall in the new order.

It uses the studies saved by the examples of the regions (``data/*.pkl``) and the per-line records of
``examples/35_realistic_truths.py`` and ``36_mixed_model.py``. Saves ``docs/resultados/rank_real_regions.json``.

    python examples/38_rank_real_regions.py
"""
import argparse
import json
import os
import pickle

import numpy as np
from scipy.stats import spearmanr

import solarff as sff
from solarff.simulations import SIMULATIONS
from solarff.utils import code_version

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "rank_real_regions.json"))
a = ap.parse_args()

REGIONS = {"11158": "harp377_study_mc10.pkl", "11429": "harp1449_20120306_2336_r7_nz32_mc10.pkl", "12673": "harp7115_20170906_1136_r4_nz56_mc10.pkl"}


def quartiles(x):
    return [float(v) for v in np.nanquantile(x, [0.25, 0.5, 0.75])]


def saved_departure(ind):
    key = "departure" if "departure" in ind else "ens_pot"
    s = np.asarray(ind[key], float)
    return np.where(np.asarray(ind[key + "_topo"]) > 0, np.maximum(np.nan_to_num(s, nan=0.5), 0.5), s)


out = {"code_version": code_version()["label"], "regions": {}, "reference_departure": {}}
# where the truths sit: the same indicator in the simulations and in the analytical families
for sim in SIMULATIONS:
    path = os.path.join(a.data, f"realistic_{sim}.pkl")
    if os.path.exists(path):
        out["reference_departure"][sim] = quartiles(np.concatenate([saved_departure(c["ind"]) for c in pickle.load(open(path, "rb"))["cases"].values()]))
path = os.path.join(a.data, "mixed_analytic.pkl")
if os.path.exists(path):
    cases = list(pickle.load(open(path, "rb"))["cases"].values())
    for fam in dict.fromkeys(c["group"] for c in cases):
        out["reference_departure"][fam] = quartiles(np.concatenate([saved_departure(c["ind"]) for c in cases if c["group"] == fam]))

for noaa, name in REGIONS.items():
    path = os.path.join(a.data, name)
    if not os.path.exists(path):
        continue
    st, ex = pickle.load(open(path, "rb"))
    lines = ex["lines_nominal"]
    src = st.mag_pp if st.mag_pp is not None else st.mag
    noise = float(st.mag.meta.get("noise") or 100.0)
    old = sff.assess_lines(st.nlfff, lines, mag=src, noise_bt=noise)
    new = sff.rank_lines(st.nlfff, lines, src, noise_bt=noise)
    conf, pct, dep = old["confidence"], new["percentile"], new["indicators"]["departure"]
    ok = np.isfinite(conf) & np.isfinite(pct)
    acc = old["reliable"] & ok
    third = [ok & (pct <= 1 / 3), ok & (pct > 1 / 3) & (pct <= 2 / 3), ok & (pct > 2 / 3)]
    out["regions"][noaa] = {
        "n_lines": int(ok.sum()), "grid": [int(n) for n in st.nlfff.shape], "dx_Mm": float(st.nlfff.dx),
        "departure_quartiles": quartiles(dep[ok]), "open_or_closed_differently": float(np.mean(dep[ok] >= 0.5)),
        "accepted_by_default_confidence": float(acc.sum() / ok.sum()),
        "spearman_confidence_vs_new_order": float(spearmanr(conf[ok], -new["score"][ok])[0]),
        "accepted_in_each_third_of_new_order": [float(acc[t].sum() / max(acc.sum(), 1)) for t in third],
        "best_third_accepted_by_default": float(acc[third[0]].mean()), "worst_third_accepted_by_default": float(acc[third[2]].mean()),
        "median_confidence_by_third": [float(np.median(conf[t])) for t in third]}
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("departure from the potential line, quartiles")
for k, q in out["reference_departure"].items():
    print(f"  {k:17s} {q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f}")
for noaa, r in out["regions"].items():
    q = r["departure_quartiles"]
    print(f"  NOAA {noaa:12s} {q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f}   ({r['n_lines']} lines, grid {r['grid']}, {r['dx_Mm']:.2f} Mm)")
print("default confidence against the new order")
for noaa, r in out["regions"].items():
    print(f"  NOAA {noaa}: accepted by the default confidence {r['accepted_by_default_confidence']:.2f}; rank correlation {r['spearman_confidence_vs_new_order']:+.2f}; "
          f"the accepted lines fall in the best, middle and worst third of the new order: " + ", ".join(f"{v:.2f}" for v in r["accepted_in_each_third_of_new_order"])
          + f"; accepted within the best third {r['best_third_accepted_by_default']:.2f}, within the worst {r['worst_third_accepted_by_default']:.2f}")
print("written", a.json)
