r"""Preprocessing of the vector magnetogram (Wiegelmann, Inhester & Sakurai 2006).

The photosphere is not force-free. Before the NLFFF the magnetogram is
modified slightly so that it meets the Aly criteria (zero net force and
torque), resembles the data and is smooth (closer to the
chromospheric field). We minimise

$$
L = \mu_1 L_1 + \mu_2 L_2 + \mu_3 L_3 + \mu_4 L_4
$$

$$
\begin{aligned}
L_1 &= \Big(\sum B_xB_z\Big)^2 + \Big(\sum B_yB_z\Big)^2 + \Big(\sum D\Big)^2 && \text{(net force)} \\
L_2 &= \Big(\sum x\,D\Big)^2 + \Big(\sum y\,D\Big)^2 + \Big(\sum (y\,B_xB_z - x\,B_yB_z)\Big)^2 && \text{(net torque)} \\
L_3 &= \sum |\mathbf{B} - \mathbf{B}_{\rm obs}|^2 && \text{(data)} \\
L_4 &= \sum |\Delta\mathbf{B}|^2 && \text{(smoothness)}
\end{aligned}
$$

with $D = B_z^2 - B_x^2 - B_y^2$ and the sums over the pixels of the magnetogram.

Here each term is normalised (sums → means, divided by $\langle B_{\rm obs}^2\rangle$) so that
the weights $\mu_i$ are independent of the size and intensity of the
magnetogram. Minimised with L-BFGS-B and analytical gradient.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .magnetogram import Magnetogram, aly_criteria


def _laplacian(a):
    """5-point Laplacian at the interior nodes (shape (nx-2, ny-2))."""
    return a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2] - 4 * a[1:-1, 1:-1]


def _laplacian_adjoint(r, shape):
    g = np.zeros(shape)
    g[2:, 1:-1] += r
    g[:-2, 1:-1] += r
    g[1:-1, 2:] += r
    g[1:-1, :-2] += r
    g[1:-1, 1:-1] -= 4 * r
    return g


def preprocess(mag: Magnetogram, mu1: float = 1.0, mu2: float = 1.0,
               mu3: float = 1e-2, mu4: float = 1e-2, max_iter: int = 500,
               fix_bz: bool = False, verbose: bool = True) -> Magnetogram:
    """Return a preprocessed magnetogram (does not modify the original).

    ``mu3`` controls how far from the data it may move and ``mu4`` how much it is
    smoothed. ``fix_bz=True`` only modifies the transverse components (noisier
    in HMI).
    """
    mag._require_vector()
    nx, ny = mag.shape
    n = nx * ny
    s = float(np.abs(np.stack([mag.bx, mag.by, mag.bz])).max())
    obs = np.stack([mag.bx, mag.by, mag.bz]) / s
    b2m = float((obs ** 2).sum(axis=0).mean())
    L = max(nx, ny) / 2
    x, y = np.meshgrid((np.arange(nx) - (nx - 1) / 2) / L,
                       (np.arange(ny) - (ny - 1) / 2) / L, indexing="ij")

    def fun(v):
        b = v.reshape(3, nx, ny)
        if fix_bz:
            b = np.concatenate([b[:2], obs[2:3]])
        bx, by, bz = b
        d = bz ** 2 - bx ** 2 - by ** 2
        c = 1.0 / (n * b2m)
        # L1
        sxz, syz, sd = c * (bx * bz).sum(), c * (by * bz).sum(), c * d.sum()
        L1 = sxz ** 2 + syz ** 2 + sd ** 2
        g1x = 2 * c * (sxz * bz - 2 * sd * bx)
        g1y = 2 * c * (syz * bz - 2 * sd * by)
        g1z = 2 * c * (sxz * bx + syz * by + 2 * sd * bz)
        # L2
        tx, ty = c * (x * d).sum(), c * (y * d).sum()
        tz = c * (y * bx * bz - x * by * bz).sum()
        L2 = tx ** 2 + ty ** 2 + tz ** 2
        g2x = 2 * c * (-2 * (tx * x + ty * y) * bx + tz * y * bz)
        g2y = 2 * c * (-2 * (tx * x + ty * y) * by - tz * x * bz)
        g2z = 2 * c * (2 * (tx * x + ty * y) * bz + tz * (y * bx - x * by))
        # L3
        diff = b - obs
        L3 = (diff ** 2).sum() / (n * b2m)
        g3 = 2 * diff / (n * b2m)
        # L4
        lap = np.stack([_laplacian(c_) for c_ in b])
        L4 = (lap ** 2).sum() / (n * b2m)
        g4 = 2 * np.stack([_laplacian_adjoint(l_, (nx, ny)) for l_ in lap]) / (n * b2m)

        total = mu1 * L1 + mu2 * L2 + mu3 * L3 + mu4 * L4
        grad = (mu1 * np.stack([g1x, g1y, g1z]) + mu2 * np.stack([g2x, g2y, g2z])
                + mu3 * g3 + mu4 * g4)
        if fix_bz:
            grad[2] = 0.0
        return total, grad.ravel()

    res = minimize(fun, obs.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter})
    b = res.x.reshape(3, nx, ny) * s
    if fix_bz:
        b[2] = mag.bz
    out = Magnetogram(bz=b[2], bx=b[0], by=b[1], dx=mag.dx,
                      origin=mag.origin, meta={**mag.meta, "preprocessed": True})
    if verbose:
        a0, a1 = aly_criteria(mag), aly_criteria(out)
        print(f"  Preprocesado ({res.nit} it): eps_force {a0['eps_force']:.3f} -> "
              f"{a1['eps_force']:.3f}, eps_torque {a0['eps_torque']:.3f} -> {a1['eps_torque']:.3f}")
    return out
