"""Revision R12: is the relative helicity correct after the quadrature fix (R10)?

In the continuum, if ∇·B = 0 and B_p has the same normal component as B on the six faces, the
relative helicity H_V does not depend on the gauge: with the DeVore gauge referred to the top or to
the bottom face the result must be the same. For an exact Low & Lou field on grids of 32 to 96 points
we measure, with the current implementation and with the one before R10 (sum of nodes without weights and without
imposing the normal component of B_p on the faces):

* H_V/Φ² (Φ: unsigned flux at the base), which must converge to a limit when the grid is refined;
* the difference between gauges, which must tend to zero.

If the current implementation converges and the earlier one does not, the change of H_V in the regions after R10
corrects an error rather than introducing one.

    python examples/revision/r12_helicity_check.py
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import solarff.helicity as hel  # noqa: E402
from solarff.analytic import low_lou_field  # noqa: E402
from solarff.field import VectorField3D  # noqa: E402

warnings.filterwarnings("ignore")
CASES = {"LL-a (n=1, m=1, l=0.3, Φ=π/4)": dict(n=1, m=1, l=0.3, phi=np.pi / 4),
         "LL-b (n=1, m=1, l=0.3, Φ=π/2)": dict(n=1, m=1, l=0.3, phi=np.pi / 2)}


def old_potential(field):
    """Volume potential as before R10: gradient without imposing the normal component on the faces."""
    h = field.dx
    flux = {"x0": field.bx[0], "x1": field.bx[-1], "y0": field.by[:, 0], "y1": field.by[:, -1],
            "z0": field.bz[:, :, 0], "z1": field.bz[:, :, -1]}
    phi = hel.poisson_neumann(np.zeros(field.shape), h, flux)
    gx, gy, gz = hel._grad(phi, h)
    return VectorField3D(gx, gy, gz, dx=h, origin=field.origin)


def helicity(field, variant):
    real_w = hel.trapezoid_weights
    try:
        if variant == "anterior":
            hel.trapezoid_weights = lambda shape: np.ones(tuple(shape))
            pot = old_potential(field)
        else:
            pot = hel.volume_potential(field)
        top = hel.relative_helicity(field, pot, "top")["H_V"]
        bot = hel.relative_helicity(field, pot, "bottom")["H_V"]
    finally:
        hel.trapezoid_weights = real_w
    return top, bot


rows = []
for name, kw in CASES.items():
    for n in (32, 48, 64, 96):
        f = low_lou_field(n, n, int(round(0.7 * n)), **kw)
        h = f.dx
        b = f.bz[:, :, 0]
        wx = np.ones(n); wx[[0, -1]] = 0.5
        phi_flux = float((np.abs(b) * wx[:, None] * wx[None, :]).sum() * (h * hel.MM) ** 2)
        for variant in ("actual", "anterior"):
            top, bot = helicity(f, variant)
            r = {"case": name, "n": n, "variant": variant, "H_top_over_flux2": top / phi_flux ** 2,
                 "H_bottom_over_flux2": bot / phi_flux ** 2,
                 "gauge_diff": abs(top - bot) / max(abs(top), abs(bot))}
            rows.append(r)
            print(f"{name} n={n:3d} {variant:8s}: H_V/Φ² arriba {r['H_top_over_flux2']:+.5f}, abajo "
                  f"{r['H_bottom_over_flux2']:+.5f}, diferencia entre gauges {r['gauge_diff']:.2%}", flush=True)
json.dump(rows, open("docs/revision/resultados/r12_helicity_check.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
