"""Estimation of the global α for the LFFF from a vector magnetogram."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .lfff import lfff, alpha_max
from .magnetogram import Magnetogram


def _mask(mag: Magnetogram, bz_min: float, bt_min: float):
    m = np.abs(mag.bz) > bz_min
    if mag.has_vector:
        m &= mag.bt > bt_min
    # The derivatives at the edge are of poorer quality
    m[:2, :] = m[-2:, :] = False
    m[:, :2] = m[:, -2:] = False
    return m


def alpha_from_jz(mag: Magnetogram, bz_min: float = 100.0, bt_min: float = 100.0) -> float:
    r"""$\alpha_{\rm av} = \sum J_zB_z / \sum B_z^2$ (Hagino & Sakurai 2004), with $J_z = (\nabla\times\mathbf{B})_z$.

    Only uses pixels with |Bz| > ``bz_min`` and |Bt| > ``bt_min`` (G) to
    avoid the noise of the weak fields.
    """
    mag._require_vector()
    m = _mask(mag, bz_min, bt_min)
    jz = mag.jz()
    return float((jz[m] * mag.bz[m]).sum() / (mag.bz[m] ** 2).sum())


def alpha_best_fit(mag: Magnetogram, pad: float = 1.0, bt_min: float = 100.0,
                   n_scan: int = 41, return_curve: bool = False):
    r"""α that best reproduces the observed transverse field with an LFFF.

    For each α, (Bx, By) of the LFFF at z = 0 is computed and
    $\sum |\mathbf{B}_{t,\rm obs} - \mathbf{B}_{t,\rm LFFF}|^2$ is minimised over the pixels with |Bt| > ``bt_min``.
    Requires a vector magnetogram with the 180° ambiguity resolved (SHARP
    already is).
    """
    mag._require_vector()
    amax = 0.98 * alpha_max(mag, pad)
    m = _mask(mag, 0.0, bt_min)
    if m.sum() < 10:
        m = np.ones_like(mag.bz, dtype=bool)

    def cost(a):
        f = lfff(mag, a, nz=1, pad=pad)
        dx = f.bx[:, :, 0] - mag.bx
        dy = f.by[:, :, 0] - mag.by
        return float(((dx ** 2 + dy ** 2)[m]).sum() / ((mag.bx ** 2 + mag.by ** 2)[m]).sum())

    grid = np.linspace(-amax, amax, n_scan)
    costs = np.array([cost(a) for a in grid])
    i = int(np.argmin(costs))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_scan - 1)]
    res = minimize_scalar(cost, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    best = float(res.x)
    if return_curve:
        return best, grid, costs
    return best


def estimate_alpha(mag: Magnetogram, method: str = "fit", **kw) -> float:
    """Estimate α (Mm^-1). ``method``: ``"fit"`` (fit of Bt) or ``"jz"`` (α_av).

    The value is clipped to ±0.98·α_max so that the LFFF is computable.
    """
    if method == "fit":
        return alpha_best_fit(mag, **kw)
    if method == "jz":
        a = alpha_from_jz(mag, **kw)
        amax = 0.98 * alpha_max(mag)
        return float(np.clip(a, -amax, amax))
    raise ValueError("method debe ser 'fit' o 'jz'")
