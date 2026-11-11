"""Revision R4: physical checks of free energy and helicity.

1. The volume potential reproduces the normal component of B on the six faces.
2. In an exact solenoidal field (Low & Lou) the helicity hardly depends on the gauge
   and E − E_p ≈ E_J: this separates the implementation error from the effect of the divergence.
3. In HARP 377, relation between E_div/E and the gauge error when w_d varies (data from
   ``solarff/divergence_weight.json``): if they grow together, the gauge error measures the
   non-solenoidal part and is not a statistical uncertainty.
4. For each extrapolated time: E − E_p versus E_J and E_J,s, and the gauge error of the helicity.

    python examples/revision/r4_physics.py
"""
import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.analytic import low_lou_field  # noqa: E402
from solarff.helicity import helicity_report, volume_potential  # noqa: E402

warnings.filterwarnings("ignore")
OUT = "docs/revision/resultados"
R = {}

# 1 and 2: exact field
ex = low_lou_field(48, 48, 34, n=1, m=1, l=0.3, phi=np.pi / 4)
pv = volume_potential(ex)
faces = [(ex.bx[0], pv.bx[0]), (ex.bx[-1], pv.bx[-1]), (ex.by[:, 0], pv.by[:, 0]),
         (ex.by[:, -1], pv.by[:, -1]), (ex.bz[:, :, 0], pv.bz[:, :, 0]), (ex.bz[:, :, -1], pv.bz[:, :, -1])]
R["normal_bc_rel_error"] = float(max(np.abs(a - b).mean() / (np.abs(a).mean() + 1e-12) for a, b in faces))
h = helicity_report(ex)
ed = h["energy"]
R["low_lou"] = {"gauge": h["gauge_diff"], "E_div_frac": ed["E_div_frac"],
                "E_free_minus_E_J_rel": (ed["E_free"] - ed["E_J"]) / ed["E_J"]}
print("BC normal (error relativo medio, peor cara): %.2e" % R["normal_bc_rel_error"])
print("Low & Lou exacto: gauge %.2f %%, E_div/E %.4f, (E−E_p − E_J)/E_J %+.2f %%" %
      (100 * h["gauge_diff"], ed["E_div_frac"], 100 * R["low_lou"]["E_free_minus_E_J_rel"]))

# 3: divergence versus gauge (HARP 377)
D = json.load(open("solarff/divergence_weight.json", encoding="utf-8"))
ed_ = np.array([r["E_div_frac"] for r in D["scan"]])
ga = np.array([r["gauge"] for r in D["scan"]])
R["wd_scan_corr_Ediv_gauge"] = float(np.corrcoef(ed_, ga)[0, 1])
R["wd_scan"] = [{"w_d": r["div_weight"], "E_div_frac": r["E_div_frac"], "gauge": r["gauge"]} for r in D["scan"]]
print("HARP 377, barrido de w_d: correlación E_div/E – error de gauge = %.3f" % R["wd_scan_corr_Ediv_gauge"])

# 4: free-energy definitions at each time
R["instants"] = {}
for p in sorted(glob.glob("data/harp*_r*_mc*.pkl")) + ["data/harp377_study_mc10.pkl"]:
    st, _ = pickle.load(open(p, "rb"))
    hr = helicity_report(st.nlfff)
    e = hr["energy"]
    R["instants"][os.path.basename(p)] = {k: float(e[k]) for k in ("E", "E_p", "E_free", "E_J", "E_J_s", "E_div", "E_div_frac", "E_mix")}
    R["instants"][os.path.basename(p)]["gauge_diff"] = float(hr["gauge_diff"])
    print("%-42s E−E_p %.3e  E_J %.3e  E_J,s %.3e  (E−E_p)/E_J %.2f  E_div/E %.3f  gauge %.1f %%" %
          (os.path.basename(p), e["E_free"], e["E_J"], e["E_J_s"], e["E_free"] / e["E_J"], e["E_div_frac"], 100 * hr["gauge_diff"]))
json.dump(R, open(os.path.join(OUT, "r4_physics.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
