"""Decay index of the confining field (torus instability).

A flux rope is held down by the external field passing over it.
If that field decays too fast with height, the rope becomes
unstable and may erupt (Kliem & Török 2006):

    n(z) = − d ln B_h / d ln z

where B_h is the horizontal component of the *potential* field (the "external"
field). For a thin torus the theoretical threshold is n_crit = 1.5; in
practice values between ~1.1 and ~1.75 are quoted. The critical height h_crit is
the height at which n reaches n_crit above the polarity inversion line.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from .field import VectorField3D


def pil_mask(bz: np.ndarray, frac: float = 0.15, width: int = 1) -> np.ndarray:
    """Pixels on the polarity inversion line, only between strong fields.

    A pixel is PIL if (smoothed) Bz changes sign with respect to a neighbour and
    there is field > ``frac``·max|Bz| in both polarities within 2 pixels.
    """
    s = gaussian_filter(bz, 1.0)
    sign = np.sign(s)
    m = np.zeros_like(bz, dtype=bool)
    m[:-1, :] |= sign[:-1, :] != sign[1:, :]
    m[:, :-1] |= sign[:, :-1] != sign[:, 1:]
    thr = frac * np.abs(bz).max()
    pos = maximum_filter(np.where(bz > 0, bz, 0), size=5) > thr
    neg = maximum_filter(np.where(bz < 0, -bz, 0), size=5) > thr
    m &= pos & neg
    if width > 1:
        m = maximum_filter(m, size=width)
    return m


def decay_index(field: VectorField3D) -> np.ndarray:
    """Local decay index n(x, y, z) of the horizontal component (NaN at z = 0)."""
    bh = np.hypot(field.bx, field.by)
    z = field.z
    with np.errstate(divide="ignore", invalid="ignore"):
        lb = np.log(np.where(bh > 0, bh, np.nan))
    n = np.full(bh.shape, np.nan)
    lz = np.log(z[1:])
    n[:, :, 1:] = -np.gradient(lb[:, :, 1:], lz, axis=2, edge_order=2)
    return n


def decay_index_profile(field: VectorField3D, mask: np.ndarray | None = None,
                        n_crit: float = 1.5) -> dict:
    """Profile n(z) above the PIL (or above the pixels of ``mask``).

    B_h is averaged over the selected columns at each height and that
    mean profile is differentiated. Returns ``z``, ``n``, ``bh`` and ``h_crit`` (first
    height with n ≥ ``n_crit``, interpolated; NaN if not reached).

    It must be applied to the **potential** field (the external confining field).
    """
    if mask is None:
        mask = pil_mask(field.bz[:, :, 0])
    if not mask.any():
        raise ValueError("La máscara (PIL) está vacía: no hay línea de inversión entre campos fuertes")
    bh = np.hypot(field.bx, field.by)[mask].mean(axis=0)  # (nz,)
    z = field.z
    n = np.full(z.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        n[1:] = -np.gradient(np.log(bh[1:]), np.log(z[1:]), edge_order=2)
    h_crit = np.nan
    ok = np.flatnonzero(np.isfinite(n) & (n >= n_crit))
    if ok.size:
        i = ok[0]
        if i > 1 and np.isfinite(n[i - 1]):
            t = (n_crit - n[i - 1]) / (n[i] - n[i - 1])
            h_crit = float(z[i - 1] + t * (z[i] - z[i - 1]))
        else:
            h_crit = float(z[i])
    return {"z": z, "n": n, "bh": bh, "h_crit": h_crit, "n_crit": n_crit,
            "mask": mask, "n_pil_pixels": int(mask.sum())}
