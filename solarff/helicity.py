r"""Energy decomposition and relative magnetic helicity.

**1. Volume potential.** $\mathbf{B}_p = \nabla\phi$ with $\nabla^2\phi = 0$ in $V$ and
$\partial\phi/\partial n = \mathbf{B}\cdot\mathbf{n}$ on the
SIX faces of the domain. It is the right reference potential for
Thomson's theorem ($E \ge E_p$) and for the relative helicity. Solved with
a discrete cosine transform (DCT-I), exact for the 7-point Laplacian
with Neumann conditions.

**2. Energy decomposition** (Valori et al. 2013). Each part
(potential $\mathbf{B}_p$ and current-carrying $\mathbf{B}_J = \mathbf{B} - \mathbf{B}_p$) is split into solenoidal and
non-solenoidal ($\mathbf{X}_{ns} = \nabla\chi$, $\nabla^2\chi = \nabla\cdot\mathbf{X}$, $\partial\chi/\partial n = 0$):

$$
\begin{aligned}
E &= E_{p,s} + E_{J,s} + E_{p,ns} + E_{J,ns} + E_{\rm mix} \\
E_{\rm div} &= E_{p,ns} + E_{J,ns} + |E_{\rm mix}|
\end{aligned}
$$

$E_{\rm div}/E$ measures how much energy is due to the numerical divergence. If it is
large, the energy (and above all the helicity) are not reliable.

**3. Relative helicity** (Finn & Antonsen 1985; Berger & Field 1984) in the
DeVore gauge ($A_z = 0$; Valori et al. 2012):

$$
\begin{aligned}
H_V &= \int_V (\mathbf{A} + \mathbf{A}_p)\cdot(\mathbf{B} - \mathbf{B}_p)\,dV \\
H_J &= \int_V (\mathbf{A} - \mathbf{A}_p)\cdot(\mathbf{B} - \mathbf{B}_p)\,dV && \text{(helicity of the current-carrying part)} \\
H_{PJ} &= 2\int_V \mathbf{A}_p\cdot(\mathbf{B} - \mathbf{B}_p)\,dV && \text{(mutual helicity)}
\end{aligned}
$$

The index $|H_J|/|H_V|$ (Pariat et al. 2017) has been proposed as an
indicator of the eruptive capacity of a region.

Units: energy in erg, helicity in Mx² (B in G, lengths in Mm).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn
from scipy.integrate import cumulative_trapezoid

from .field import VectorField3D, divergence, trapezoid_weights  # noqa: F401 (reexportada)

MM = 1e8  # cm


# ------------------------------------------------------------- Neumann
def _eig(n, h):
    k = np.arange(n)
    return (2 * np.cos(np.pi * k / (n - 1)) - 2) / (h * h)


def poisson_neumann(f: np.ndarray, h: float, flux: dict | None = None) -> np.ndarray:
    """Solve ∇²φ = f on a grid of nodes with ∂φ/∂n prescribed on the faces.

    ``flux`` = {"x0": g, "x1": g, "y0": ..., "y1": ..., "z0": ..., "z1": ...}
    with g = ∂φ/∂x (or ∂y, ∂z) on that face (Cartesian component, not outward
    normal). Without ``flux``: homogeneous Neumann. The constant component is
    removed (the solution is defined up to a constant); if the problem
    is not compatible (net flux ≠ ∫f) it is projected onto the closest compatible one.
    """
    rhs = np.array(f, dtype=float, copy=True)
    nx, ny, nz = rhs.shape
    if flux:
        # Ghost points: φ_-1 = φ_1 − 2h g0  →  term −2 g0/h at node 0
        if "x0" in flux:
            rhs[0] += 2 * flux["x0"] / h
        if "x1" in flux:
            rhs[-1] -= 2 * flux["x1"] / h
        if "y0" in flux:
            rhs[:, 0] += 2 * flux["y0"] / h
        if "y1" in flux:
            rhs[:, -1] -= 2 * flux["y1"] / h
        if "z0" in flux:
            rhs[:, :, 0] += 2 * flux["z0"] / h
        if "z1" in flux:
            rhs[:, :, -1] -= 2 * flux["z1"] / h
    F = dctn(rhs, type=1)
    lam = (_eig(nx, h)[:, None, None] + _eig(ny, h)[None, :, None] + _eig(nz, h)[None, None, :])
    lam[0, 0, 0] = 1.0
    F = F / lam
    F[0, 0, 0] = 0.0
    return idctn(F, type=1)


def _grad(phi, h):
    return np.gradient(phi, h, edge_order=2)


def volume_potential(field: VectorField3D) -> VectorField3D:
    """Potential field with the same normal component as ``field`` on the six faces."""
    h = field.dx
    bx, by, bz = field.bx, field.by, field.bz
    flux = {"x0": bx[0], "x1": bx[-1], "y0": by[:, 0], "y1": by[:, -1],
            "z0": bz[:, :, 0], "z1": bz[:, :, -1]}
    phi = poisson_neumann(np.zeros(field.shape), h, flux)
    gx, gy, gz = _grad(phi, h)
    # The normal component on the faces is the imposed one (the gradient with one-sided
    # differences does not reproduce it exactly, and that breaks the identity E − E_p = E_J; revision R10)
    gx[0], gx[-1] = bx[0], bx[-1]
    gy[:, 0], gy[:, -1] = by[:, 0], by[:, -1]
    gz[:, :, 0], gz[:, :, -1] = bz[:, :, 0], bz[:, :, -1]
    return VectorField3D(gx, gy, gz, dx=h, origin=field.origin,
                         meta={"method": "potencial (volumen)", "phi": phi})


def _split(X, h):
    """X = X_s + X_ns with X_ns = ∇χ, ∇²χ = ∇·X, ∂χ/∂n = 0."""
    d = divergence(*X, h)
    chi = poisson_neumann(d, h)
    ns = _grad(chi, h)
    s = [X[i] - ns[i] for i in range(3)]
    return s, list(ns)


def _e(X, h):
    w = trapezoid_weights(X[0].shape)
    return float(sum((w * c * c).sum() for c in X) / (8 * np.pi) * (h * MM) ** 3)


def _dot(X, Y, h):
    w = trapezoid_weights(X[0].shape)
    return float(sum((w * X[i] * Y[i]).sum() for i in range(3)) / (4 * np.pi) * (h * MM) ** 3)


def energy_decomposition(field: VectorField3D, potential: VectorField3D | None = None) -> dict:
    """Decomposition of Valori et al. (2013). Returns the energies (erg) and E_div/E."""
    h = field.dx
    if potential is None:
        potential = volume_potential(field)
    B = [field.bx, field.by, field.bz]
    P = [potential.bx, potential.by, potential.bz]
    J = [B[i] - P[i] for i in range(3)]
    Ps, Pns = _split(P, h)
    Js, Jns = _split(J, h)
    E = _e(B, h)
    out = {"E": E, "E_p": _e(P, h), "E_J": _e(J, h),
           "E_p_s": _e(Ps, h), "E_J_s": _e(Js, h), "E_p_ns": _e(Pns, h), "E_J_ns": _e(Jns, h)}
    out["E_mix"] = (_dot(Ps, Jns, h) + _dot(Pns, Js, h) + _dot(Ps, Pns, h) + _dot(Js, Jns, h))
    out["E_div"] = out["E_p_ns"] + out["E_J_ns"] + abs(out["E_mix"])
    out["E_div_frac"] = out["E_div"] / E
    out["E_free"] = E - out["E_p"]
    return out


# ------------------------------------------------------------- helicity
def _a0_plane(bz_plane, h):
    """A_0 in a plane with (∇×A_0)_z = B_z and ∇·A_0 = 0:  A_0 = (−∂u/∂y, ∂u/∂x), ∇²u = B_z."""
    nx, ny = bz_plane.shape
    # The net flux (mean of B_z) cannot be represented with homogeneous Neumann:
    # it is treated apart with u_m = m (x² + y²)/4, which satisfies ∇²u_m = m exactly.
    wx = np.ones(nx)
    wx[[0, -1]] = 0.5
    wy = np.ones(ny)
    wy[[0, -1]] = 0.5
    w2 = wx[:, None] * wy[None, :]
    m = float((bz_plane * w2).sum() / w2.sum())  # mean with trapezoid weights (mode 0 of the DCT-I)
    F = dctn(bz_plane - m, type=1)
    # Eigenvalues of the operator that is actually applied afterwards: two
    # centred derivatives (np.gradient) on a cosine mode give −sin²(θ)/h², not the
    # 5-point Laplacian. Using the same ones guarantees ∇×A_0 = B_z in the interior.
    # For θ > π/2 the centred differences hardly see the mode (sin θ → 0 at
    # Nyquist) and dividing by sin²θ would amplify the noise: there the
    # 5-point Laplacian, which is stable, is used.
    tx = np.pi * np.arange(nx) / (nx - 1)
    ty = np.pi * np.arange(ny) / (ny - 1)

    def ev(t):
        return np.where(t <= np.pi / 2, -np.sin(t) ** 2, 2 * np.cos(t) - 2)
    lam = (ev(tx)[:, None] + ev(ty)[None, :]) / (h * h)
    null = np.abs(lam) < 1e-12 * np.abs(lam).max()  # constant and Nyquist modes
    lam[null] = 1.0
    F = F / lam
    F[null] = 0.0
    x = h * np.arange(nx)
    y = h * np.arange(ny)
    u = idctn(F, type=1) + m * (x[:, None] ** 2 + y[None, :] ** 2) / 4
    ux, uy = np.gradient(u, h, edge_order=2)
    return -uy, ux


def devore_potential(field: VectorField3D, z0: str = "top"):
    """Vector potential in the DeVore gauge (A_z = 0) referred to the ``z0`` face."""
    h = field.dx
    bx, by, bz = field.bx, field.by, field.bz
    k0 = -1 if z0 == "top" else 0
    a0x, a0y = _a0_plane(bz[:, :, k0], h)
    ibx = cumulative_trapezoid(bx, dx=h, axis=2, initial=0)
    iby = cumulative_trapezoid(by, dx=h, axis=2, initial=0)
    if z0 == "top":
        ibx = ibx - ibx[:, :, -1:]
        iby = iby - iby[:, :, -1:]
    # A = A_0 − ẑ × ∫_{z0}^{z} B dz'
    ax = a0x[:, :, None] + iby
    ay = a0y[:, :, None] - ibx
    az = np.zeros_like(ax)
    return ax, ay, az


def relative_helicity(field: VectorField3D, potential: VectorField3D | None = None,
                      z0: str = "top") -> dict:
    """Relative helicity $H_V$, $H_J$, $H_{PJ}$ (Mx²) and the index $|H_J|/|H_V|$."""
    h = field.dx
    if potential is None:
        potential = volume_potential(field)
    A = devore_potential(field, z0)
    Ap = devore_potential(potential, z0)
    B = [field.bx, field.by, field.bz]
    P = [potential.bx, potential.by, potential.bz]
    D = [B[i] - P[i] for i in range(3)]
    # units: A [G·Mm], B [G], dV [Mm³]  →  ×1e8 (A) ×1e24 (dV) = 1e32 → Mx²
    fac = h ** 3 * MM * MM ** 3
    w = trapezoid_weights(D[0].shape)
    hv = float(sum((w * (A[i] + Ap[i]) * D[i]).sum() for i in range(3)) * fac)
    hj = float(sum((w * (A[i] - Ap[i]) * D[i]).sum() for i in range(3)) * fac)
    hpj = float(2 * sum((w * Ap[i] * D[i]).sum() for i in range(3)) * fac)
    return {"H_V": hv, "H_J": hj, "H_PJ": hpj,
            "ratio_HJ_HV": abs(hj) / abs(hv) if hv else float("nan"), "gauge": f"DeVore ({z0})"}


def helicity_report(field: VectorField3D) -> dict:
    """Decomposed energy + helicity in two gauges (the difference measures the numerical error)."""
    pot = volume_potential(field)
    e = energy_decomposition(field, pot)
    ht = relative_helicity(field, pot, "top")
    hb = relative_helicity(field, pot, "bottom")
    gauge_diff = abs(ht["H_V"] - hb["H_V"]) / max(abs(ht["H_V"]), abs(hb["H_V"]), 1e-300)
    return {"energy": e, "helicity": ht, "helicity_bottom_gauge": hb, "gauge_diff": gauge_diff,
            "potential": pot}
