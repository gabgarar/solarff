"""Revision R3: natural variation of the free energy before each flare.

The paper compared the drop at the flare with a single pair of earlier times.
Here all the available earlier times (3–4 per region, every 36–72 min) are used to
measure how much the energy changes without large flares, with the two definitions
(E − E_p and E_J, volume potential), and this is compared with the drop at the flare.

    python examples/revision/r3_natural_variation.py
"""
import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.helicity import energy_decomposition  # noqa: E402

warnings.filterwarnings("ignore")
OUT = "docs/revision/resultados"
REG = {"11158": ("377", "20110214", "20110215_0000", "20110215_0300"),
       "11429": ("1449", "20120306", "20120306_2336", "20120307_0200"),
       "12673": ("7115", "20170906", "20170906_1136", "20170906_1300")}
res = {}
for noaa, (harp, day, pre, post) in REG.items():
    files = sorted(glob.glob(f"data/harp{harp}_{day}_*_r*_mc*.pkl"))
    if harp == "377":
        files.append("data/harp377_study_mc10.pkl")
    files += sorted(glob.glob(f"data/harp{harp}_{post}_r*_mc*.pkl"))
    rows = []
    for p in sorted(set(files)):
        t = "20110215_0000" if p.endswith("harp377_study_mc10.pkl") else os.path.basename(p).split("_r")[0].split(f"harp{harp}_")[1]
        st, _ = pickle.load(open(p, "rb"))
        e = energy_decomposition(st.nlfff)
        rows.append({"time": t, "E_free": float(e["E_free"]), "E_J": float(e["E_J"]), "post": t == post})
    rows.sort(key=lambda r: r["time"])
    prev = [r for r in rows if not r["post"] and r["time"] <= pre]
    ipre = next(r for r in rows if r["time"] == pre)
    ipost = next(r for r in rows if r["post"])
    R = {"times": rows}
    for k in ("E_free", "E_J"):
        v = np.array([r[k] for r in prev])
        steps = np.abs(np.diff(v))
        R[k] = {"flare_drop": ipost[k] - ipre[k], "flare_drop_rel": (ipost[k] - ipre[k]) / ipre[k],
                "natural_range": float(v.max() - v.min()), "natural_max_step": float(steps.max()) if steps.size else None,
                "natural_std": float(v.std(ddof=1)) if v.size > 1 else None, "n_pre": int(v.size)}
        R[k]["drop_over_range"] = abs(R[k]["flare_drop"]) / R[k]["natural_range"] if R[k]["natural_range"] else None
    res[noaa] = R
    print(f"NOAA {noaa}: instantes previos {[r['time'][-4:] for r in prev]}")
    for k in ("E_free", "E_J"):
        r = R[k]
        print(f"  {k}: caída {r['flare_drop']:+.2e} ({r['flare_drop_rel']:+.1%}); rango previo {r['natural_range']:.2e}, "
              f"mayor salto {r['natural_max_step']:.2e}; caída/rango = {r['drop_over_range']:.1f}")
json.dump(res, open(os.path.join(OUT, "r3_natural_variation.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
