r"""Linear force-free field (LFFF): $\nabla\times\mathbf{B} = \alpha\mathbf{B}$ with constant $\alpha$.

Fourier solution (Nakagawa & Raadu 1972; Alissandrakis 1981) with
periodic lateral conditions. For each wavenumber $\mathbf{k} = (k_x, k_y)$:

$$
\begin{aligned}
l &= \sqrt{k^2 - \alpha^2} \\
\hat B_z(z) &= \hat B_z(0)\, e^{-l z} \\
\hat B_x(z) &= -i\,\frac{k_x l - \alpha k_y}{k^2}\, \hat B_z(z) \\
\hat B_y(z) &= -i\,\frac{k_y l + \alpha k_x}{k^2}\, \hat B_z(z)
\end{aligned}
$$

Only requires $B_z$ at the photosphere. Solutions that decay with height
require $|\alpha| < k_{\min} = 2\pi / L_{\max}$ ($L_{\max}$: longer side of the periodic domain).
$\alpha = 0$ gives the potential field.
"""

from __future__ import annotations

import numpy as np

from .field import VectorField3D
from .magnetogram import Magnetogram


def alpha_max(mag: Magnetogram, pad: float = 1.0) -> float:
    """Maximum admissible |α| (Mm^-1) for the domain (with the given padding factor)."""
    L = max(mag.shape) * pad * mag.dx
    return 2 * np.pi / L


def _padded(bz: np.ndarray, pad: float):
    nx, ny = bz.shape
    px, py = int(round(nx * pad)), int(round(ny * pad))
    if px == nx and py == ny:
        return bz, (0, 0)
    out = np.zeros((px, py))
    ox, oy = (px - nx) // 2, (py - ny) // 2
    out[ox:ox + nx, oy:oy + ny] = bz
    return out, (ox, oy)


def lfff(mag: Magnetogram, alpha: float, nz: int, pad: float = 1.0,
         heights: np.ndarray | None = None) -> VectorField3D:
    """Linear force-free extrapolation.

    **Parameters**

    - ``mag``: magnetogram (only ``bz`` is used).
    - ``alpha``: force-free parameter in Mm^-1 (positive = positive helicity /
        right-handed chirality).
    - ``nz``: number of levels in height (step = ``mag.dx``).
    - ``pad``: zero-padding factor (>= 1). Reduces the interaction with the
        periodic copies, but also reduces the maximum admissible α.
    - ``heights``: explicit heights in Mm (if given, ``nz`` is ignored; the returned
        field no longer has a uniform step in z — useful only to evaluate
        specific layers).
    """
    if pad < 1:
        raise ValueError("pad debe ser >= 1")
    bz0, (ox, oy) = _padded(mag.bz, pad)
    nx, ny = mag.shape
    px, py = bz0.shape
    h = mag.dx

    amax = 2 * np.pi / (max(px, py) * h)
    if abs(alpha) >= amax:
        raise ValueError(
            f"|alpha| = {abs(alpha):.4g} Mm^-1 >= alpha_max = {amax:.4g} Mm^-1 para este "
            f"dominio: la solución no decae con la altura. Reduce alpha o pad.")

    kx = 2 * np.pi * np.fft.fftfreq(px, d=h)
    ky = 2 * np.pi * np.fft.fftfreq(py, d=h)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    k2_safe = np.where(k2 == 0, 1.0, k2)
    l = np.sqrt(np.maximum(k2 - alpha ** 2, 0.0))

    bz_hat = np.fft.fft2(bz0)
    fx = -1j * (KX * l - alpha * KY) / k2_safe
    fy = -1j * (KY * l + alpha * KX) / k2_safe
    fx[0, 0] = fy[0, 0] = 0.0
    if alpha != 0:
        # A uniform field does not satisfy ∇×B = αB with α ≠ 0: the k=0 mode is removed.
        bz_hat[0, 0] = 0.0

    z = np.asarray(heights, dtype=float) if heights is not None else h * np.arange(nz)
    bx = np.empty((nx, ny, z.size))
    by = np.empty_like(bx)
    bz = np.empty_like(bx)
    for iz, zz in enumerate(z):
        decay = bz_hat * np.exp(-l * zz)
        bz[:, :, iz] = np.fft.ifft2(decay).real[ox:ox + nx, oy:oy + ny]
        bx[:, :, iz] = np.fft.ifft2(fx * decay).real[ox:ox + nx, oy:oy + ny]
        by[:, :, iz] = np.fft.ifft2(fy * decay).real[ox:ox + nx, oy:oy + ny]

    method = "potencial" if alpha == 0 else "LFFF"
    return VectorField3D(bx, by, bz, dx=h, origin=(*mag.origin, 0.0),
                         meta={"method": method, "alpha": float(alpha), "pad": pad})
