r"""NLFFF by the Grad–Rubin method (Grad & Rubin 1958; Amari et al. 2006; Wheatland 2007).

It belongs to a different family from the optimisation method: instead of minimising a
functional, it imposes the equations by iterating current and field:

1. $\alpha$ at the base: $\alpha_0 = (\nabla\times\mathbf{B})_z / B_z$ on ONE polarity (P: $B_z > 0$, N: $B_z < 0$).
2. $\alpha(\mathbf{r})$: each point inherits $\alpha_0$ from the footpoint of its field line on that polarity
   ($\alpha = 0$ if the line does not reach it).
3. $\mathbf{J} = \alpha\,\mathbf{B}$, cleaned so that $\nabla\cdot\mathbf{J} = 0$.
4. $\mathbf{B} = \mathbf{B}_p + \nabla\times\mathbf{A}$, with $\nabla^2\mathbf{A} = -\mathbf{J}$, $\mathbf{A}_t = 0$ and
   $\partial A_n/\partial n = 0$ on the faces, so $\nabla\times\mathbf{A}$ has no normal component and
   $\mathbf{B}$ keeps that of $\mathbf{B}_p$.
5. Repeat 2–4 until $\mathbf{B}$ stops changing.

The P and N solutions use different halves of the α data. If the data
were perfectly consistent with a force-free field, they would coincide;
their difference measures that inconsistency (Wheatland & Régnier 2009).

Internal units: $\nabla\times\mathbf{B}$ in G/Mm, $\alpha$ in Mm⁻¹ ($\mu_0$ is absorbed in $\mathbf{J}$).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.fft import dctn, dstn, idctn, idstn
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

from .field import VectorField3D, curl, divergence
from .helicity import poisson_neumann
from .magnetogram import Magnetogram
from .potential import potential_field
from .tracing import integrate_lines


def _eig(n, h, kind):
    """Eigenvalues of the 3-point 1D Laplacian: Neumann (DCT-I, n nodes) or Dirichlet (DST-I, n−2 interior)."""
    k = np.arange(n) if kind == "N" else np.arange(1, n - 1)
    return (2 * np.cos(np.pi * k / (n - 1)) - 2) / (h * h)


def _poisson_mixed(f, h, kinds):
    """∇²u = f with, on each axis, homogeneous Neumann ("N") or homogeneous Dirichlet ("D").

    On the Dirichlet axes the solution is 0 on the faces.
    """
    sl = tuple(slice(None) if k == "N" else slice(1, -1) for k in kinds)
    g = np.array(f, dtype=float)[sl]
    for ax, k in enumerate(kinds):
        g = dctn(g, type=1, axes=[ax]) if k == "N" else dstn(g, type=1, axes=[ax])
    lam = 0
    for ax, k in enumerate(kinds):
        e = _eig(f.shape[ax], h, k)
        shape = [1, 1, 1]
        shape[ax] = e.size
        lam = lam + e.reshape(shape)
    lam = np.where(lam == 0, 1.0, lam)
    g = g / lam
    if all(k == "N" for k in kinds):
        g.flat[0] = 0.0
    for ax, k in enumerate(kinds):
        g = idctn(g, type=1, axes=[ax]) if k == "N" else idstn(g, type=1, axes=[ax])
    u = np.zeros(f.shape)
    u[sl] = g
    return u


def current_field(jx, jy, jz, h):
    """Field B_J = ∇×A of a current distribution, with B_J·n = 0 on the six faces."""
    ax = _poisson_mixed(-jx, h, ("N", "D", "D"))
    ay = _poisson_mixed(-jy, h, ("D", "N", "D"))
    az = _poisson_mixed(-jz, h, ("D", "D", "N"))
    return curl(ax, ay, az, h)


def boundary_alpha(mag: Magnetogram, threshold: float = 0.05, smooth: float = 1.0,
                   clip_percentile: float = 99.0) -> np.ndarray:
    """α_0 = J_z / B_z at the base (Mm⁻¹), 0 where |B_z| < ``threshold``·max|B_z|.

    Slightly smoothed (``smooth`` pixels) and the extreme values (noise where
    B_z is small) clipped at the ``clip_percentile`` percentile of |α|.
    """
    mag._require_vector()
    bx = gaussian_filter(mag.bx, smooth) if smooth else mag.bx
    by = gaussian_filter(mag.by, smooth) if smooth else mag.by
    bz = gaussian_filter(mag.bz, smooth) if smooth else mag.bz
    jz = np.gradient(by, mag.dx, axis=0, edge_order=2) - np.gradient(bx, mag.dx, axis=1, edge_order=2)
    strong = np.abs(mag.bz) > threshold * np.abs(mag.bz).max()
    a = np.zeros_like(bz)
    a[strong] = jz[strong] / bz[strong]
    if strong.any():
        lim = np.percentile(np.abs(a[strong]), clip_percentile)
        a = np.clip(a, -lim, lim)
    return a


def _assign_alpha(field, alpha0, polarity, strong, stride, ds):
    """α in the volume: each point inherits α_0 from the footpoint of its line on the chosen polarity."""
    nx, ny, nz = field.shape
    gx = np.arange(0, nx, stride)
    gy = np.arange(0, ny, stride)
    gz = np.arange(0, nz, stride)
    for g, n in ((gx, nx), (gy, ny), (gz, nz)):
        if g[-1] != n - 1:
            g.resize(g.size + 1, refcheck=False)
            g[-1] = n - 1
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    start = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()]).astype(float)
    # P: the field leaves B_z > 0 → the footpoint is reached by following −B. N: following +B.
    sign = -1.0 if polarity == "P" else 1.0
    end, _, _ = integrate_lines(field, start, sign, ds=ds)
    at_base = end[2] <= 0.5
    a_end = map_coordinates(alpha0, end[:2], order=1, mode="nearest")
    ok_end = map_coordinates(strong.astype(float), end[:2], order=1, mode="nearest") > 0.5
    a = np.where(at_base & ok_end, a_end, 0.0).reshape(GX.shape)
    if stride == 1:
        return a
    return zoom(a, (nx / a.shape[0], ny / a.shape[1], nz / a.shape[2]), order=1)[:nx, :ny, :nz]


def nlfff_grad_rubin(mag: Magnetogram, nz: int, polarity: str = "P", initial: VectorField3D | None = None,
                     max_iter: int = 40, tol: float = 1e-3, relax: float = 0.5, stride: int | None = None,
                     alpha_threshold: float = 0.05, alpha_smooth: float = 1.0, ds: float = 0.5,
                     verbose: bool = True, return_result: bool = False):
    """NLFFF extrapolation by Grad–Rubin.

    **Parameters**

    - ``mag``: vector magnetogram (B_z is used as boundary condition and J_z/B_z
        on the chosen polarity).
    - ``nz``: levels in height.
    - ``polarity``: ``"P"`` (α of the positive polarity) or ``"N"`` (negative).
    - ``initial``: initial / reference field (by default the potential); its
        normal component on the six faces is kept.
    - ``max_iter``: Grad–Rubin iterations.
    - ``tol``: stop when the relative rms change of B between iterations < ``tol``.
    - ``relax``: under-relaxation B ← (1−r) B + r B_new (stabilises the iteration).
    - ``stride``: α is assigned by tracing from 1 of every ``stride`` points per axis and
        interpolated; 1 = every point (more accurate: in Low & Lou C_vec 0.980/0.987
        for P/N against 0.957/0.984 with 2). Default 1 up to ~2·10⁵ points and 2 above.
    """
    from .nlfff import NLFFFResult
    mag._require_vector()
    h = mag.dx
    pot = initial if initial is not None else potential_field(mag, nz)
    if stride is None:
        stride = 1 if mag.shape[0] * mag.shape[1] * nz <= 200_000 else 2
    bp = [pot.bx, pot.by, pot.bz]
    alpha0 = boundary_alpha(mag, alpha_threshold, alpha_smooth)
    pol = mag.bz > 0 if polarity == "P" else mag.bz < 0
    strong = pol & (np.abs(mag.bz) > alpha_threshold * np.abs(mag.bz).max())
    B = [c.copy() for c in bp]
    hist = []
    t0 = time.time()
    message, converged = "max_iter alcanzado", False
    for it in range(1, max_iter + 1):
        cur = VectorField3D(*B, dx=h, origin=pot.origin)
        a = _assign_alpha(cur, alpha0, polarity, strong, stride, ds)
        J = [a * B[i] for i in range(3)]
        # Cleaning: J ← J − ∇ψ, ∇²ψ = ∇·J, ∂ψ/∂n = 0 (∇·J = 0 without touching J_n)
        psi = poisson_neumann(divergence(*J, h), h)
        gpsi = np.gradient(psi, h, edge_order=2)
        J = [J[i] - gpsi[i] for i in range(3)]
        bj = current_field(*J, h)
        new = [bp[i] + bj[i] for i in range(3)]
        num = sum(((new[i] - B[i]) ** 2).sum() for i in range(3))
        den = sum((new[i] ** 2).sum() for i in range(3))
        change = float(np.sqrt(num / den))
        B = [(1 - relax) * B[i] + relax * new[i] for i in range(3)]
        e = float(sum((c ** 2).sum() for c in B))
        hist.append((it, change, e))
        if verbose:
            print(f"  Grad-Rubin ({polarity}) it {it:3d}  cambio rms {change:.2e}  "
                  f"E/E0 {e / float(sum((c ** 2).sum() for c in bp)):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if change < tol:
            converged, message = True, f"convergido: cambio rms = {change:.1e}"
            break
    out = VectorField3D(*B, dx=h, origin=pot.origin,
                        meta={"method": "NLFFF", "backend": f"grad-rubin-{polarity}", "polarity": polarity,
                              "iterations": it, "message": message, "nd": max(4, min(*mag.shape, nz) // 10),
                              "history": hist})
    if return_result:
        return NLFFFResult(out, [(i, c, np.nan) for i, c, _ in hist], converged, message)
    return out
