r"""Nonlinear force-free field (NLFFF): $(\nabla\times\mathbf{B})\times\mathbf{B} = 0$, $\nabla\cdot\mathbf{B} = 0$, $\alpha = \alpha(\mathbf{r})$.

Optimisation method (Wheatland, Sturrock & Roumeliotis 2000) with a weighting
function at the boundaries (Wiegelmann 2004). We minimise

$$
L = \int_V w\left[\frac{|(\nabla\times\mathbf{B})\times\mathbf{B}|^2}{B^2} + |\nabla\cdot\mathbf{B}|^2\right] dV
$$

iterating $\mathbf{B} \leftarrow \mathbf{B} + \mu\,\mathbf{F}$, with

$$
\begin{aligned}
\mathbf{\Omega} &= \frac{(\nabla\times\mathbf{B})\times\mathbf{B} - (\nabla\cdot\mathbf{B})\,\mathbf{B}}{B^2} \\
\mathbf{F} &= w\left[\nabla\times(\mathbf{\Omega}\times\mathbf{B}) - \mathbf{\Omega}\times(\nabla\times\mathbf{B})
  - \nabla(\mathbf{\Omega}\cdot\mathbf{B}) + \mathbf{\Omega}\,(\nabla\cdot\mathbf{B}) + \Omega^2\,\mathbf{B}\right] \\
&\quad + (\mathbf{\Omega}\times\mathbf{B})\times\nabla w + (\mathbf{\Omega}\cdot\mathbf{B})\,\nabla w
\end{aligned}
$$

Boundary conditions: the vector magnetogram at z = 0; sides and
top taken from the initial (potential) field. The function $w = 1$ in the interior
and drops to 0 (cosine profile) in a layer of ``nd`` points next to the side
and top faces, so that the unobserved conditions have little influence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .field import VectorField3D, curl, divergence, cross, dot
from .magnetogram import Magnetogram
from .potential import potential_field


def weight_function(shape, nd: int) -> np.ndarray:
    """w(x,y,z): 1 in the interior, decays as a cosine to 0 over ``nd`` points at the side
    and top edges (not at the base)."""
    nx, ny, nz = shape

    def profile(n, both=True):
        p = np.ones(n)
        if nd <= 0:
            return p
        i = np.arange(nd)
        ramp = 0.5 * (1 - np.cos(np.pi * i / nd))  # 0 -> ~1
        p[n - nd:] = ramp[::-1]
        if both:
            p[:nd] = ramp
        return p

    return (profile(nx)[:, None, None] * profile(ny)[None, :, None]
            * profile(nz, both=False)[None, None, :])


def _functional_and_force(b, w, gw, h, need_force=True, eps2=0.0):
    bx, by, bz = b
    jx, jy, jz = curl(bx, by, bz, h)
    d = divergence(bx, by, bz, h)
    b2 = bx * bx + by * by + bz * bz
    # Regularisation near nulls: Ω = X/(B²+ε²). With L_ε = ∫ w |X|²/(B²+ε²) the
    # Wheatland force F with this Ω is still the exact gradient.
    b2s = np.where(b2 + eps2 > 0, b2 + eps2, 1e-30)
    fx, fy, fz = cross(jx, jy, jz, bx, by, bz)
    ox = (fx - d * bx) / b2s
    oy = (fy - d * by) / b2s
    oz = (fz - d * bz) / b2s
    o2 = ox * ox + oy * oy + oz * oz
    L = float((w * (b2 + eps2) * o2)[1:-1, 1:-1, 1:-1].sum())
    if not need_force:
        return L, None

    ob = dot(ox, oy, oz, bx, by, bz)
    oxb = cross(ox, oy, oz, bx, by, bz)
    c1 = curl(*oxb, h)
    c2 = cross(ox, oy, oz, jx, jy, jz)
    g = np.gradient(ob, h, edge_order=2)
    F = [w * (c1[i] - c2[i] - g[i] + (ox, oy, oz)[i] * d + o2 * b[i]) for i in range(3)]
    # Terms of the gradient of the weighting function
    t = cross(*oxb, *gw)
    for i in range(3):
        F[i] += t[i] + ob * gw[i]
    return L, F


@dataclass
class NLFFFResult:
    field: VectorField3D
    history: list = field(default_factory=list)  # (iteration, L, μ)
    converged: bool = False
    message: str = ""


def nlfff(mag: Magnetogram, nz: int, initial: VectorField3D | None = None,
          nd: int | None = None, max_iter: int = 5000, tol: float = 1e-5,
          patience: int = 50, mu0: float | None = None, eps: float = 0.03,
          verbose: bool = True, return_result: bool = False):
    """NLFFF extrapolation by optimisation.

    **Parameters**

    - ``mag``: *vector* magnetogram (ideally preprocessed with
        ``solarff.preprocess``).
    - ``nz``: levels in height.
    - ``initial``: initial field (by default the potential). Its values on the
        side and top faces are kept fixed.
    - ``nd``: width (in points) of the buffer layer. By default
        ~10 % of the shorter side, minimum 4.
    - ``max_iter``: maximum number of accepted iterations.
    - ``tol``: stopping criterion: relative change of L per iteration averaged
        over ``patience`` iterations smaller than ``tol``.
    - ``mu0``: initial step (in units of h²); adapted automatically.
    - ``eps``: regularisation near magnetic nulls, as a fraction of max|Bz|.
        In the functional B² + ε² is used instead of B²: without it, the points with
        |B| ≈ 0 (frequent in real data) make the force blow up and the step
        μ drops to ~1e-7 h², stalling the optimisation. 0 = original method.
    - ``return_result``: if True returns ``NLFFFResult`` with the history.
    """
    mag._require_vector()
    nx, ny = mag.shape
    h = mag.dx
    if initial is None:
        initial = potential_field(mag, nz)
    if not np.allclose(initial.origin[:2], mag.origin):
        raise ValueError("El campo inicial y el magnetograma tienen distinto origen")
    if initial.shape != (nx, ny, nz):
        raise ValueError(f"campo inicial con forma {initial.shape}, se esperaba {(nx, ny, nz)}")
    if nd is None:
        nd = max(4, min(nx, ny, nz) // 10)

    # Normalisation: the functional is scale invariant in Ω, but it is convenient
    # to work with B ~ O(1).
    scale = float(np.abs(mag.bz).max()) or 1.0
    b = [initial.bx / scale, initial.by / scale, initial.bz / scale]
    b = [c.copy() for c in b]
    b[0][:, :, 0] = mag.bx / scale
    b[1][:, :, 0] = mag.by / scale
    b[2][:, :, 0] = mag.bz / scale

    w = weight_function((nx, ny, nz), nd)
    gw = np.gradient(w, h, edge_order=2)

    eps2 = float(eps) ** 2  # B is normalised to max|Bz| = 1
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    mu = (mu0 if mu0 is not None else 0.1) * h * h

    L, F = _functional_and_force(b, w, gw, h, eps2=eps2)
    history = [(0, L, mu)]
    it, rejected = 0, 0
    t0 = time.time()
    converged, message = False, "max_iter alcanzado"

    while it < max_iter:
        trial = [c.copy() for c in b]
        for i in range(3):
            trial[i][interior] += mu * F[i][interior]
        L_new, F_new = _functional_and_force(trial, w, gw, h, eps2=eps2)
        if np.isfinite(L_new) and L_new < L:
            b, F = trial, F_new
            it += 1
            history.append((it, L_new, mu))
            L = L_new
            mu *= 1.01
            rejected = 0
            if it > patience:
                L_old = history[-patience - 1][1]
                rel = (L_old - L) / (L_old * patience)
                if rel < tol:
                    converged, message = True, f"convergido: dL/L por iteracion = {rel:.2e}"
                    break
            if verbose and it % 200 == 0:
                print(f"  NLFFF it {it:5d}  L = {L:.4e}  mu/h^2 = {mu / h / h:.3e}  "
                      f"({time.time() - t0:.1f}s)")
        else:
            mu *= 0.5
            rejected += 1
            if rejected > 60 or mu < 1e-12 * h * h:
                converged, message = True, "paso mu demasiado pequeno (minimo local)"
                break

    if verbose:
        print(f"  NLFFF terminado: {message}; {it} iteraciones, L: "
              f"{history[0][1]:.3e} -> {L:.3e} ({time.time() - t0:.1f}s)")

    out = VectorField3D(b[0] * scale, b[1] * scale, b[2] * scale, dx=h, origin=(*mag.origin, 0.0),
                        meta={"method": "NLFFF", "iterations": it, "L_final": L,
                              "L_initial": history[0][1], "nd": nd, "eps": eps, "message": message})
    if return_result:
        return NLFFFResult(out, history, converged, message)
    return out
