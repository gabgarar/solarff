r"""NLFFF with exact gradient (automatic differentiation, PyTorch) and L-BFGS.

Minimises the discrete functional

$$
L = \Big\langle w\Big[\frac{|\mathbf{J}\times\mathbf{B}|^2}{B^2+\varepsilon^2} + (\nabla\cdot\mathbf{B})^2\Big]\Big\rangle_V
  + \nu\,\Big\langle W\,|\mathbf{B}_t - \mathbf{B}_{t,\rm obs}|^2\Big\rangle_{\rm base}
$$

with the same finite differences as the numpy version (2nd order, also
at the edges). PyTorch computes the exact gradient of L with respect to ALL the
unknowns, including the transverse field of the base when it is relaxed.
This allows:

* **Relaxed base** (Wiegelmann et al. 2010, 2012): B_t at z = 0 is no longer
  fixed and may adjust within the measurement noise, with more freedom where
  the transverse field is weak ($W = |\mathbf{B}_{t,\rm obs}| / \max|\mathbf{B}_{t,\rm obs}|$). B_z stays fixed.
* **L-BFGS** with Wolfe line search: converges in far fewer
  iterations than gradient descent with a fixed step.

The side and top faces are fixed to the initial (potential) field and the
weighting function w is the same as in ``solarff.nlfff``.
"""

from __future__ import annotations

import time

import numpy as np

from .field import VectorField3D
from .magnetogram import Magnetogram
from .nlfff import NLFFFResult, weight_function
from .potential import potential_field


def _require_torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("El método 'lbfgs' necesita PyTorch: pip install torch") from e
    return torch


def _diff(torch, f, axis):
    """2nd-order derivative on cells (equivalent to np.gradient(edge_order=2) with h = 1)."""
    def sl(a, b):
        idx = [slice(None)] * f.dim()
        idx[axis] = slice(a, b)
        return tuple(idx)
    first = (-3 * f[sl(0, 1)] + 4 * f[sl(1, 2)] - f[sl(2, 3)]) / 2
    inner = (f[sl(2, None)] - f[sl(None, -2)]) / 2
    last = (3 * f[sl(-1, None)] - 4 * f[sl(-2, -1)] + f[sl(-3, -2)]) / 2
    return torch.cat([first, inner, last], dim=axis)


def _density(torch, B, eps2, wd=1.0):
    bx, by, bz = B[0], B[1], B[2]
    d = lambda f, a: _diff(torch, f, a)
    jx = d(bz, 1) - d(by, 2)
    jy = d(bx, 2) - d(bz, 0)
    jz = d(by, 0) - d(bx, 1)
    div = d(bx, 0) + d(by, 1) + d(bz, 2)
    fx = jy * bz - jz * by
    fy = jz * bx - jx * bz
    fz = jx * by - jy * bx
    b2 = bx * bx + by * by + bz * bz
    return (fx * fx + fy * fy + fz * fz) / (b2 + eps2) + wd * div * div


def nlfff_lbfgs(mag: Magnetogram, nz: int, initial: VectorField3D | None = None,
                nd: int | None = None, relax_bottom: float | None = 0.1, eps: float = 0.03,
                smooth: float = 1e-2, div_weight: float = 1.0,
                max_iter: int = 2000, tol: float = 1e-6, chunk: int = 50,
                threads: int | None = None, verbose: bool = True,
                return_result: bool = False):
    """NLFFF extrapolation with exact gradient and L-BFGS.

    **Parameters**

    - ``mag``: vector magnetogram (ideally preprocessed).
    - ``nz``: levels in height.
    - ``initial``: initial field; its side and top faces stay fixed
        (by default the potential).
    - ``nd``: width of the buffer layer (by default ~10 % of the shorter side).
    - ``relax_bottom``: weight ν of the data term. With ``None`` the base stays fixed
        (as in the classic method). Higher values = base closer to the
        data; lower = more freedom to make it consistent.
    - ``eps``: regularisation near nulls (fraction of max|Bz|).
    - ``smooth``: weight λ of the smoothness penalty λ <|∇²B|²>. Centred
        differences do not "see" grid-scale oscillations
        (checkerboard); an optimiser with exact gradient exploits them
        to lower L without improving the physics. This term suppresses them.
    - ``div_weight``: weight w_d of the divergence term relative to the force term
        (w_f = 1), as in Wiegelmann et al. (2012). More weight = more
        solenoidal field (helicity less gauge dependent) at the cost of somewhat more
        Lorentz force. Default 1 (standard form). With 4 the
        thresholds of Thalmann et al. (2019) are met in HARP 377, but the
        lines reproduce the exact field worse and neither the energy nor the helicity
        get closer to the truth (``examples/09`` and ``11``), so it is used
        only as a sensitivity test.
    - ``max_iter``: maximum number of L-BFGS iterations.
    - ``tol``: stop if L decreases relatively by less than ``tol`` per iteration
        (average over a block of ``chunk`` iterations).
    - ``threads``: CPU threads for PyTorch (None = whatever PyTorch decides).
    """
    torch = _require_torch()
    if threads:
        torch.set_num_threads(threads)
    mag._require_vector()
    nx, ny = mag.shape
    h = mag.dx
    if initial is None:
        initial = potential_field(mag, nz)
    if initial.shape != (nx, ny, nz):
        raise ValueError(f"campo inicial con forma {initial.shape}, se esperaba {(nx, ny, nz)}")
    if nd is None:
        nd = max(4, min(nx, ny, nz) // 10)
    relax = relax_bottom is not None and relax_bottom > 0
    nu = float(relax_bottom) if relax else 0.0

    scale = float(np.abs(mag.bz).max()) or 1.0
    B0 = np.stack([initial.bx, initial.by, initial.bz]) / scale
    B0[0, :, :, 0] = mag.bx / scale
    B0[1, :, :, 0] = mag.by / scale
    B0[2, :, :, 0] = mag.bz / scale
    dt = torch.float64
    fixed = torch.tensor(B0, dtype=dt)
    w = torch.tensor(weight_function((nx, ny, nz), nd), dtype=dt)
    obs = torch.tensor(B0[:2, 1:-1, 1:-1, 0].copy(), dtype=dt)
    bt = np.hypot(B0[0, 1:-1, 1:-1, 0], B0[1, 1:-1, 1:-1, 0])
    W = torch.tensor(bt / bt.max() if bt.max() > 0 else np.ones_like(bt), dtype=dt)

    P = torch.tensor(B0[:, 1:-1, 1:-1, 1:-1].copy(), dtype=dt, requires_grad=True)
    params = [P]
    if relax:
        Pb = torch.tensor(B0[:2, 1:-1, 1:-1, 0].copy(), dtype=dt, requires_grad=True)
        params.append(Pb)
    z0 = 0 if relax else 1
    region = (slice(1, -1), slice(1, -1), slice(z0, -1))
    eps2 = float(eps) ** 2

    def assemble():
        B = fixed.clone()
        B[:, 1:-1, 1:-1, 1:-1] = P
        if relax:
            B[:2, 1:-1, 1:-1, 0] = Pb
        return B

    def loss_terms():
        B = assemble()
        dens = _density(torch, B, eps2, div_weight)
        lff = (w[region] * dens[region]).mean()
        ldata = nu * (W * ((Pb - obs) ** 2).sum(0)).mean() if relax else torch.zeros((), dtype=dt)
        if smooth:
            c = B[:, 1:-1, 1:-1, 1:-1]
            lap = (B[:, 2:, 1:-1, 1:-1] + B[:, :-2, 1:-1, 1:-1] + B[:, 1:-1, 2:, 1:-1]
                   + B[:, 1:-1, :-2, 1:-1] + B[:, 1:-1, 1:-1, 2:] + B[:, 1:-1, 1:-1, :-2] - 6 * c)
            lff = lff + smooth * (lap ** 2).sum(0).mean()
        return lff, ldata

    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=chunk, history_size=20,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-14, tolerance_change=1e-16)
    n_eval = [0]

    def closure():
        opt.zero_grad()
        lff, ld = loss_terms()
        loss = lff + ld
        loss.backward()
        n_eval[0] += 1
        return loss

    with torch.no_grad():
        lff, ld = loss_terms()
    history = [(0, float(lff + ld), float(lff), float(ld))]
    t0 = time.time()
    it, converged, message = 0, False, "max_iter alcanzado"
    while it < max_iter:
        L_prev = history[-1][1]
        opt.step(closure)
        it = int(opt.state[opt._params[0]].get("n_iter", it + chunk))
        with torch.no_grad():
            lff, ld = loss_terms()
        L = float(lff + ld)
        history.append((it, L, float(lff), float(ld)))
        if verbose:
            print(f"  NLFFF(L-BFGS) it {it:5d}  L = {L:.4e}  (fuerza+div {float(lff):.3e}, datos {float(ld):.3e})"
                  f"  ({time.time() - t0:.1f}s)", flush=True)
        done_iters = max(it - history[-2][0], 1)
        rel = (L_prev - L) / (abs(L_prev) * done_iters) if L_prev else 0.0
        if rel < tol:
            converged, message = True, f"convergido: dL/L por iteracion = {rel:.2e}"
            break
    with torch.no_grad():
        B = assemble().numpy() * scale
    if verbose:
        print(f"  NLFFF(L-BFGS) terminado: {message}; {it} iteraciones, {n_eval[0]} evaluaciones, "
              f"L: {history[0][1]:.3e} -> {history[-1][1]:.3e} ({time.time() - t0:.1f}s)")
    out = VectorField3D(B[0], B[1], B[2], dx=h, origin=(*mag.origin, 0.0),
                        meta={"method": "NLFFF", "backend": "lbfgs", "iterations": it,
                              "evaluations": n_eval[0], "L_final": history[-1][1],
                              "L_initial": history[0][1], "nd": nd, "eps": eps,
                              "relax_bottom": relax_bottom if relax else None, "smooth": smooth,
                              "div_weight": div_weight,
                              "message": message})
    if return_result:
        # History compatible with the classic method: (iteration, L, step) with step = NaN
        hist = [(i, L, np.nan) for i, L, _, _ in history]
        res = NLFFFResult(out, hist, converged, message)
        res.terms = history
        return res
    return out
