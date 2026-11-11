"""Revision R10: why is E − E_p ≠ E_J in an exact solenoidal field?

In the continuum, if ∇·B = 0 and B_p has the same normal component as B on the six faces,
∫ B_p·(B − B_p) dV = 0 and therefore E − E_p = E_J. Here we measure, for an exact Low & Lou
field on grids of 32 to 96 points:

* the relative difference (E − E_p − E_J)/E_J;
* the cross term 2∫B_p·B_J/8π, which is exactly that difference;
* the discrete divergence of B and of B_p (E_div/E) and the normal-flux mismatch on the faces.

If everything decreases when the grid is refined, the difference is a discretisation error.

    python examples/revision/r10_energy_identity.py
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.analytic import low_lou_field  # noqa: E402
from solarff.helicity import energy_decomposition, trapezoid_weights, volume_potential  # noqa: E402

warnings.filterwarnings("ignore")
MM3 = 1e24
rows = []
for n in (32, 48, 64, 96):
    f = low_lou_field(n, n, int(round(0.7 * n)), n=1, m=1, l=0.3, phi=np.pi / 4)
    pv = volume_potential(f)
    e = energy_decomposition(f, pv)
    c = f.dx ** 3 * MM3 / (8 * np.pi)
    w = trapezoid_weights(f.shape)
    cross = 2 * float((w * ((pv.bx * (f.bx - pv.bx)) + (pv.by * (f.by - pv.by)) + (pv.bz * (f.bz - pv.bz)))).sum() * c)
    faces = [(f.bx[0], pv.bx[0]), (f.bx[-1], pv.bx[-1]), (f.by[:, 0], pv.by[:, 0]), (f.by[:, -1], pv.by[:, -1]),
             (f.bz[:, :, 0], pv.bz[:, :, 0]), (f.bz[:, :, -1], pv.bz[:, :, -1])]
    flux_err = [float(np.abs(a - b).sum() / np.abs(a).sum()) for a, b in faces]
    ep = energy_decomposition(pv, pv)
    r = {"n": n, "rel_diff": (e["E_free"] - e["E_J"]) / e["E_J"], "cross_over_EJ": cross / e["E_J"],
         "E_div_frac_B": e["E_div_frac"], "E_div_frac_Bp": ep["E_div_frac"],
         "normal_flux_err_faces": flux_err, "E_free_over_E": e["E_free"] / e["E"]}
    rows.append(r)
    print(f"n={n:3d}: (E−E_p−E_J)/E_J {r['rel_diff']:+.2%} | cruzado/E_J {r['cross_over_EJ']:+.2%} | E_div/E: B {r['E_div_frac_B']:.4f}, "
          f"B_p {r['E_div_frac_Bp']:.4f} | flujo normal (peor cara) {max(flux_err):.2%} | E_free/E {r['E_free_over_E']:.3f}", flush=True)
json.dump(rows, open("docs/revision/resultados/r10_energy_identity.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
