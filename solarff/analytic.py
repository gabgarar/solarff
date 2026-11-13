r"""Analytical NLFFF solution of Low & Lou (1990) to validate the methods.

Self-similar axisymmetric field in spherical coordinates $(r, \theta, \varphi)$:

$$
\begin{aligned}
A &= \frac{P(\mu)}{r^n}, \qquad \mu = \cos\theta, \qquad Q = \frac{a\,P^{1+1/n}}{r^{n+1}} \\
B_r &= -\frac{P'(\mu)}{r^{n+2}}, \qquad
B_\theta = \frac{n\,P}{r^{n+2}\sin\theta}, \qquad
B_\varphi = \frac{a\,P^{1+1/n}}{r^{n+2}\sin\theta}
\end{aligned}
$$

with

$$
(1-\mu^2)\,P'' + n(n+1)\,P + a^2\,\frac{1+n}{n}\,P^{1+2/n} = 0, \qquad P(\pm1) = 0, \quad P'(-1) = 10 .
$$

The eigenvalue $a^2$ is obtained by shooting ($m$-th non-trivial root).

The source is placed at depth ``l`` below the photosphere with the axis
tilted by an angle ``phi``. Standard case (Schrijver et al. 2006, case I):
$n = 1$, $m = 1$, $l = 0.3$, $\Phi = \pi/4$.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .field import VectorField3D
from .magnetogram import Magnetogram

_EPS = 1e-6
_DP0 = 10.0


def _spow(p, e):
    return np.sign(p) * np.abs(p) ** e


def _shoot(n, a2):
    def rhs(mu, y):
        P, dP = y
        return [dP, -(n * (n + 1) * P + a2 * (1 + n) / n * _spow(P, 1 + 2 / n)) / (1 - mu * mu)]

    return solve_ivp(rhs, [-1 + _EPS, 1 - _EPS], [_DP0 * _EPS, _DP0],
                     rtol=1e-10, atol=1e-12, dense_output=True)


@lru_cache(maxsize=16)
def low_lou_eigenvalue(n: int = 1, m: int = 1) -> float:
    """m-th (non-trivial) eigenvalue a² of the Low & Lou problem."""
    grid = np.linspace(0.01, 10.0, 400)
    vals = np.array([_shoot(n, a2).y[0, -1] for a2 in grid])
    roots = []
    for i in range(len(grid) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]):
            roots.append(brentq(lambda a: _shoot(n, a).y[0, -1], grid[i], grid[i + 1], xtol=1e-12))
            if len(roots) == m:
                return float(roots[-1])
    raise RuntimeError(f"No se encontró el autovalor m={m} para n={n}")


def low_lou_field(nx: int = 64, ny: int = 64, nz: int = 64, n: int = 1, m: int = 1,
                  l: float = 0.3, phi: float = np.pi / 4, length_mm: float = 50.0,
                  bmax: float = 1000.0) -> VectorField3D:
    """Low & Lou field in the box x, y ∈ [-1, 1], z ∈ [0, (nz-1)·h] (units of ``length_mm``).

    The result is scaled so that max|Bz(z=0)| = ``bmax`` G and the box is
    2·``length_mm`` Mm wide.
    """
    a2 = low_lou_eigenvalue(n, m)
    a = np.sqrt(a2)
    sol = _shoot(n, a2).sol

    h = 2.0 / (nx - 1)
    x = -1 + h * np.arange(nx)
    y = -1 + h * np.arange(ny)  # same step; if ny != nx the box is not square
    y = y - (y[-1] + y[0]) / 2
    z = h * np.arange(nz)
    X0, Y0, Z0 = np.meshgrid(x, y, z, indexing="ij")

    # Coordinates in the frame of the source (rotated by Φ about y, shifted by l)
    zs = Z0 + l
    X = X0 * np.cos(phi) - zs * np.sin(phi)
    Y = Y0
    Z = X0 * np.sin(phi) + zs * np.cos(phi)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    rho = np.maximum(np.hypot(X, Y), 1e-12)
    mu = np.clip(Z / r, -1 + _EPS, 1 - _EPS)
    sint = np.sqrt(1 - mu ** 2)
    cphi, sphi = X / rho, Y / rho

    P, dP = sol(mu.ravel())
    P = P.reshape(mu.shape)
    dP = dP.reshape(mu.shape)
    rn2 = r ** (n + 2)
    br = -dP / rn2
    bt = n * P / (rn2 * sint)
    bp = a * _spow(P, 1 + 1 / n) / (rn2 * sint)

    BX = br * sint * cphi + bt * mu * cphi - bp * sphi
    BY = br * sint * sphi + bt * mu * sphi + bp * cphi
    BZ = br * mu - bt * sint

    bx = BX * np.cos(phi) + BZ * np.sin(phi)
    by = BY
    bz = -BX * np.sin(phi) + BZ * np.cos(phi)

    s = bmax / np.abs(bz[:, :, 0]).max()
    return VectorField3D(bx * s, by * s, bz * s, dx=h * length_mm,
                         origin=(x[0] * length_mm, y[0] * length_mm, 0.0),
                         meta={"method": "Low&Lou", "n": n, "m": m, "a2": a2, "l": l, "phi": phi})


def low_lou_magnetogram(noise: float = 0.0, rng: int | None = 0, **kw) -> tuple[Magnetogram, VectorField3D]:
    """Return (z=0 magnetogram, exact 3D field) of Low & Lou.

    ``noise``: standard deviation (G) of Gaussian noise added to the three
    components of the magnetogram (to test the preprocessing).
    """
    f = low_lou_field(**kw)
    bx, by, bz = (c.copy() for c in f.bottom())
    if noise > 0:
        g = np.random.default_rng(rng)
        bx += g.normal(0, noise, bx.shape)
        by += g.normal(0, noise, by.shape)
        bz += g.normal(0, noise, bz.shape)
    mag = Magnetogram(bz=bz, bx=bx, by=by, dx=f.dx, origin=f.origin[:2],
                      meta={"source": "Low&Lou", "noise": noise, **f.meta})
    return mag, f


# ---------------------------------------------------------------------------
# Analytical validation fields for Tw, decay index and Q
# ---------------------------------------------------------------------------
def _box(nx, ny, nz, half_width):
    h = 2 * half_width / (nx - 1)
    x = -half_width + h * np.arange(nx)
    y = -(ny - 1) * h / 2 + h * np.arange(ny)
    z = h * np.arange(nz)
    return h, x, y, z


def gold_hoyle_field(nx: int = 48, ny: int = 48, nz: int = 48, half_width: float = 20.0,
                     b: float = 0.3, b0: float = 1000.0) -> VectorField3D:
    r"""Gold & Hoyle (1960) flux tube, vertical and centred in the box.

    $$
    B_z = \frac{B_0}{1 + b^2r^2}, \qquad B_\varphi = \frac{B_0\,b\,r}{1 + b^2r^2}
    $$

    It is force-free with $\alpha = 2b/(1 + b^2r^2)$ and every line is a helix of
    constant radius. The twist number of a line of radius $r$ between $z = 0$
    and $z = H$ is exactly $T_w = b\,H / \big(2\pi\sqrt{1 + b^2r^2}\big)$
    (see ``gold_hoyle_twist``).
    """
    h, x, y, z = _box(nx, ny, nz, half_width)
    X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
    r2 = X ** 2 + Y ** 2
    den = 1 + b * b * r2
    bz = b0 / den
    bphi_over_r = b0 * b / den
    bx = -bphi_over_r * Y
    by = bphi_over_r * X
    return VectorField3D(bx, by, bz, dx=h, origin=(x[0], y[0], 0.0),
                         meta={"method": "Gold-Hoyle", "b": b, "b0": b0})


def gold_hoyle_twist(r, height, b):
    """Exact Tw of the line of radius r of the Gold-Hoyle tube between z = 0 and ``height``."""
    return b * height / (2 * np.pi * np.sqrt(1 + (b * np.asarray(r)) ** 2))


def dipole_field(nx: int = 64, ny: int = 64, nz: int = 40, half_width: float = 50.0,
                 depth: float = 10.0, b0: float = 1000.0) -> VectorField3D:
    """Potential field of a horizontal dipole (moment along +x) buried at
    depth ``depth`` below the centre of the box.

    At the photosphere it gives a bipolar region with the PIL over x = 0. Right above
    the dipole the field is horizontal and decays as $(z + d)^{-3}$, so the
    decay index is exactly $n(z) = 3z / (z + d)$ and reaches the
    torus-instability threshold $n = 1.5$ at $z = d$.
    """
    h, x, y, z = _box(nx, ny, nz, half_width)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    Zs = Z + depth
    r2 = X ** 2 + Y ** 2 + Zs ** 2
    r5 = r2 ** 2.5
    # B = (3 (m·r) r - m r²) / r^5 with m = x̂
    bx = (3 * X * X - r2) / r5
    by = 3 * X * Y / r5
    bz = 3 * X * Zs / r5
    s = b0 / np.abs(bz[:, :, 0]).max()
    return VectorField3D(bx * s, by * s, bz * s, dx=h, origin=(x[0], y[0], 0.0),
                         meta={"method": "dipolo", "depth": depth})


def dipole_decay_index(z, depth):
    """Exact decay index above the dipole: n = 3z / (z + d)."""
    z = np.asarray(z, dtype=float)
    return 3 * z / (z + depth)


def hyperbolic_field(nx: int = 48, ny: int = 48, nz: int = 24, half_width: float = 20.0,
                     lam: float = 0.05, b0: float = 1.0) -> VectorField3D:
    r"""Hyperbolic field $\mathbf{B} = (\lambda x, -\lambda y, B_0)$: potential and divergence-free.

    The lines satisfy $x = x_0\,e^{\lambda z/B_0}$, $y = y_0\,e^{-\lambda z/B_0}$. The mapping between
    the base and the top ($z = H$) has Jacobian $\mathrm{diag}(e^a, e^{-a})$, $a = \lambda H/B_0$, so
    the squashing factor is exactly $Q = 2\cosh(2a)$ over the whole
    domain (see ``hyperbolic_q``). This is the structure of a "hyperbolic
    flux tube" (Titov et al. 2002).
    """
    h, x, y, z = _box(nx, ny, nz, half_width)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return VectorField3D(lam * X, -lam * Y, np.full_like(X, b0), dx=h,
                         origin=(x[0], y[0], 0.0), meta={"method": "hiperbólico", "lam": lam})


def hyperbolic_q(height, lam, b0=1.0):
    """Exact Q of the hyperbolic field between z = 0 and z = ``height``."""
    return 2 * np.cosh(2 * lam * height / b0)


# ---------------------------------------------------------------------------
# Additional exact force-free fields (to calibrate the confidence)
# ---------------------------------------------------------------------------
def lfff_multipole_field(nx: int = 40, ny: int = 40, nz: int = 28, spots=None,
                         alpha_frac: float = 0.7, length_mm: float = 50.0,
                         bmax: float = 1000.0) -> VectorField3D:
    """*Exact* linear force-free field over a multipolar region.

    ``spots``: list of Gaussian spots ``(x, y, sign·amplitude, width)`` in
    box units (x, y ∈ [-1, 1]). The Fourier solution (Alissandrakis
    1981) is an exact solution of ∇×B = αB in the periodic domain, so
    it serves as a known truth different from Low & Lou: currents distributed
    over the whole region instead of concentrated, and another geometry.
    ``alpha_frac``: α as a fraction of α_max = 2π/L.
    """
    from .lfff import alpha_max, lfff
    if spots is None:  # tilted bipole
        spots = [(-0.35, -0.15, 1.0, 0.18), (0.35, 0.15, -1.0, 0.18)]
    h = 2.0 / (nx - 1)
    x = -1 + h * np.arange(nx)
    y = -(ny - 1) * h / 2 + h * np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    bz = np.zeros_like(X)
    for (x0, y0, amp, w) in spots:
        bz += amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * w * w))
    bz -= bz.mean()  # balanced flux (the k = 0 mode is not force-free with α ≠ 0)
    bz *= bmax / np.abs(bz).max()
    mag = Magnetogram(bz=bz, dx=h * length_mm, origin=(x[0] * length_mm, y[0] * length_mm))
    a = alpha_frac * alpha_max(mag)
    f = lfff(mag, a, nz)
    f.meta.update({"method": "LFFF exacto", "alpha_frac": alpha_frac, "spots": spots})
    return f


def mhs_multipole_field(nx: int = 40, ny: int = 40, nz: int = 28, spots=None, alpha_frac: float = 0.7,
                        a: float = 0.6, scale_mm: float = 2.0, length_mm: float = 50.0,
                        bmax: float = 1000.0) -> VectorField3D:
    """*Exact* magnetohydrostatic field over a multipolar region (Low 1991), with a base that is not force-free.

    Same region and same alpha as ``lfff_multipole_field``, plus a horizontal current perpendicular to the
    field of amplitude ``a`` that decays with the scale height ``scale_mm``: below it the Lorentz force is
    balanced by pressure and gravity, above it the field is linear force-free. It plays the part of a
    photosphere that violates the force-free assumption, with a known solution above it.
    """
    from .lfff import alpha_max, linear_mhs
    if spots is None:  # tilted bipole, as in lfff_multipole_field
        spots = [(-0.35, -0.15, 1.0, 0.18), (0.35, 0.15, -1.0, 0.18)]
    h = 2.0 / (nx - 1)
    x = -1 + h * np.arange(nx)
    y = -(ny - 1) * h / 2 + h * np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    bz = np.zeros_like(X)
    for (x0, y0, amp, w) in spots:
        bz += amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * w * w))
    bz -= bz.mean()
    bz *= bmax / np.abs(bz).max()
    mag = Magnetogram(bz=bz, dx=h * length_mm, origin=(x[0] * length_mm, y[0] * length_mm))
    f = linear_mhs(mag, alpha_frac * alpha_max(mag), nz, a=a, kappa=1.0 / scale_mm)
    f.meta.update({"method": "MHS lineal exacto", "alpha_frac": alpha_frac, "spots": spots, "scale_mm": scale_mm})
    return f


def _td_vector_potential(x, rp, R, a, cI):
    """A_I(x, r⊥) of the Titov & Démoulin (1999) current ring, Eqs. (25)–(31)."""
    from scipy.special import ellipe, ellipk

    def calA(k):  # Eq. (26)
        m = k * k
        return ((2 - m) * ellipk(m) - 2 * ellipe(m)) / k

    def dcalA(k):  # Eq. (29)
        m = k * k
        return ((2 - m) * ellipe(m) - 2 * (1 - m) * ellipk(m)) / (m * (1 - m))

    rho = np.sqrt(x ** 2 + (rp - R) ** 2)  # Eq. (18)
    k = 2 * np.sqrt(rp * R / ((rp + R) ** 2 + x ** 2))  # Eq. (27)
    ka = 2 * np.sqrt(rp * R / (4 * rp * R + a * a))  # Eq. (30)
    inside = rho < a
    pre = cI * np.sqrt(R / rp)
    out = np.empty_like(x)
    ko = np.where(inside, 0.5, k)  # outside the tube k < 1
    out[:] = pre * calA(ko)  # Eq. (25)
    if inside.any():
        ki, kai = k[inside], ka[inside]
        out[inside] = pre[inside] * (calA(kai) + dcalA(kai) * (ki - kai))  # Eq. (28)
    return out  # Eq. (31): A_in inside (ρ < a), A_ex outside


def titov_demoulin_field(nx: int = 40, ny: int = 40, nz: int = 28, R: float = 2.2, a: float = 0.7,
                         d: float = 1.0, L: float = 2.0, n_turns: float = 1.0, handedness: int = -1,
                         half_width: float = 3.0, box_mm: float = 100.0,
                         bmax: float = 1000.0, current_factor: float = 1.0) -> VectorField3D:
    """Titov & Démoulin (1999) flux rope: current torus in equilibrium.

    B = B_I + B_θ + B_q (Eqs. 16–31 of TD99, with μ0 = 1):

    * B_I: ring of major radius ``R`` and minor radius ``a`` with uniform toroidal
      current I, in the plane x = 0 and with its centre at depth ``d``;
    * B_q: charges ±q at (±``L``, 0, −``d``), which hold the ring;
    * B_θ: toroidal field of a line current I0 along the symmetry axis
      (x axis at z = −d), which gives the twist.

    I comes from the force balance (Eq. 6, l_i = 1/2). I0 is set by the
    number of turns ``n_turns`` of the lines in the coronal part of the tube
    (Eqs. 7–9); ``handedness`` = −1 gives left-handed twist, as in Valori et al.
    (2010). The default parameters are those of Wiegelmann et al. (2006) in
    units of d (R = 2.2, a = 0.7, d = 1, L = 2).

    ``n_turns`` is *nominal*: Eqs. (7)–(9) assume a uniform B_θ inside the
    tube, which only holds for weak twist. The actual twist of the core
    (T_w of the lines near the axis) is ≈ −0.46, −0.69 and −0.78 turns with
    n_turns = 0.5, 1 and 1.5; with n_turns = 1, I0/I = −3.46 and T_w ≈ −0.69 turns,
    consistent with Wiegelmann et al. (2006): I0/I = −3.83 and −1.41π (−0.7 turns).

    The model is force-free only to first order in a/R and a/L (the field
    is an approximate equilibrium): ``meta["approx"]`` says so. With 64² points,
    CW_sin = 0.063 (0.056 with 96²; Low & Lou gives 0.017), and it is minimal with the
    equilibrium current of Eq. (6): the residual is the first-order error
    of the model, not of the implementation.

    Grid: x, y ∈ [−``half_width``, ``half_width``] (units of d), z ≥ 0 with the
    same step; the box is ``box_mm`` Mm wide and the maximum |B_z| at the base is
    ``bmax`` G. ``current_factor`` multiplies the equilibrium current I
    (only for tests: with values ≠ 1 the torus is not in equilibrium).
    """
    h = 2 * half_width / (nx - 1)
    xs = -half_width + h * np.arange(nx)
    ys = -(ny - 1) * h / 2 + h * np.arange(ny)
    zs = h * np.arange(nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    q = 1.0
    li = 0.5
    I = 8 * np.pi * q * L * R * (R * R + L * L) ** -1.5 / (np.log(8 * R / a) - 1.5 + li / 2)  # Eq. (6)
    I *= current_factor
    Nt = np.pi * n_turns / np.arccos(d / R)  # Eq. (7)
    I0 = handedness * I * R * R / (a * a * Nt)  # Eqs. (8)-(9): N_t = I R / (2π a² |B_θ|), B_θ = I0/(2πR)

    zd = Z + d
    rp = np.sqrt(Y ** 2 + zd ** 2)  # Eq. (19)
    rho = np.sqrt(X ** 2 + (rp - R) ** 2)  # Eq. (18)
    th = (np.zeros_like(X), -zd / rp, Y / rp)  # Eq. (17)

    # B_q, Eqs. (20)-(21)
    Bq = [np.zeros_like(X) for _ in range(3)]
    for s in (+1, -1):
        r = (X - s * L, Y, zd)
        n3 = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 1.5
        for i in range(3):
            Bq[i] += s * q * r[i] / n3

    # B_θ, Eq. (16)
    chi = (rho < a).astype(float)
    inner = 1 / R ** 2 + 2 * chi * (I * I / (a * a * I0 * I0)) * (1 - rho ** 2 / a ** 2)
    bth = I0 / (2 * np.pi) * (np.sqrt(inner) + 1 / rp - 1 / R)

    # B_I, Eq. (22), with centred derivatives of A_I (Eqs. 25-31)
    cI = I / (2 * np.pi)
    eps = 1e-5
    A = _td_vector_potential(X, rp, R, a, cI)
    dAdx = (_td_vector_potential(X + eps, rp, R, a, cI) - _td_vector_potential(X - eps, rp, R, a, cI)) / (2 * eps)
    dAdr = (_td_vector_potential(X, rp + eps, R, a, cI) - _td_vector_potential(X, rp - eps, R, a, cI)) / (2 * eps)
    bI_x = dAdr + A / rp
    bI_r = -dAdx  # along r̂⊥ = (0, y, z + d)/r⊥

    bx = Bq[0] + bI_x
    by = Bq[1] + bth * th[1] + bI_r * Y / rp
    bz = Bq[2] + bth * th[2] + bI_r * zd / rp
    s = bmax / np.abs(bz[:, :, 0]).max()
    unit = box_mm / (2 * half_width)  # Mm per unit of d
    f = VectorField3D(bx * s, by * s, bz * s, dx=h * unit,
                      origin=(xs[0] * unit, ys[0] * unit, 0.0),
                      meta={"method": "Titov-Démoulin (1999)", "approx": "libre de fuerzas a primer orden en a/R",
                            "R": R, "a": a, "d": d, "L": L, "n_turns": n_turns, "handedness": handedness,
                            "I": I, "I0": I0, "unit_mm": unit})
    return f


def field_magnetogram(field: VectorField3D, noise: float = 0.0, noise_t: float | None = None,
                      rng: int | None = 0) -> Magnetogram:
    """z = 0 magnetogram of a 3D field with Gaussian noise.

    ``noise``: σ (G) of B_z; ``noise_t``: σ of B_x and B_y (default = ``noise``).
    With ``noise_t`` > ``noise`` HMI is mimicked, whose transverse field is much
    noisier than the longitudinal one. The error maps are stored in the
    magnetogram (``*_err``).
    """
    noise_t = noise if noise_t is None else noise_t
    bx, by, bz = (c.copy() for c in field.bottom())
    if noise > 0 or noise_t > 0:
        g = np.random.default_rng(rng)
        bx += g.normal(0, noise_t, bx.shape)
        by += g.normal(0, noise_t, by.shape)
        bz += g.normal(0, noise, bz.shape)
    full = lambda s: np.full(bz.shape, float(s))
    return Magnetogram(bz=bz, bx=bx, by=by, dx=field.dx, origin=field.origin[:2],
                       bx_err=full(noise_t), by_err=full(noise_t), bz_err=full(noise),
                       meta={"source": field.meta.get("method", "analítico"), "noise": noise,
                             "noise_t": noise_t, **field.meta})


# Families of exact fields used to calibrate the per-line confidence.
# Each entry: function (nx, ny, nz) -> VectorField3D.
CALIBRATION_FIELDS = {
    # Low & Lou case I of Schrijver et al. (2006)
    "LL-I": lambda nx, ny, nz: low_lou_field(nx, ny, nz, n=1, m=1, l=0.3, phi=np.pi / 4),
    # Low & Lou deeper and tilted to the other side: less current at the base
    "LL-II": lambda nx, ny, nz: low_lou_field(nx, ny, nz, n=1, m=1, l=0.4, phi=4 * np.pi / 5),
    # Low & Lou with n = 3: more compact and more twisted structure (deeper source,
    # l = 0.6, so that it is resolved on grids of 32-48 points)
    "LL-III": lambda nx, ny, nz: low_lou_field(nx, ny, nz, n=3, m=1, l=0.6, phi=np.pi / 3),
    # exact LFFF, tilted bipole with α = 0.7 α_max
    "LFFF-bip": lambda nx, ny, nz: lfff_multipole_field(nx, ny, nz, alpha_frac=0.7),
    # exact LFFF, quadrupolar region with α = 0.5 α_max
    "LFFF-quad": lambda nx, ny, nz: lfff_multipole_field(
        nx, ny, nz, alpha_frac=0.5,
        spots=[(-0.45, -0.3, 1.0, 0.22), (-0.05, -0.35, -0.8, 0.2),
               (0.1, 0.35, 0.9, 0.22), (0.5, 0.25, -1.0, 0.23)]),
}

_RELAXED_CACHE: dict = {}


def relaxed_titov_demoulin_field(nx: int = 40, ny: int = 40, nz: int = 28, smooth: float = 1e-3,
                                 max_iter: int = 4000, tol: float = 1e-7, **td_kw) -> VectorField3D:
    """Titov & Démoulin rope relaxed to a force-free equilibrium.

    The TD99 model is force-free only to first order in a/R. Here it is used
    as the initial state and relaxed by minimising the force-free functional
    with the **six faces fixed** to the TD field itself (as in case I of
    Wiegelmann et al. 2006b; Valori et al. 2010 do it by magnetofriction).
    This gives a field exactly consistent with its boundaries that
    keeps the rope. With 40² points: CW_sin 0.080 → 0.014 (Low & Lou: 0.017),
    C_vec = 0.9993 with respect to TD, core twist −0.69 → −0.73 turns and
    apex 21.6 → 22.1 Mm. Requires PyTorch. The results are cached in memory
    so that the relaxation is not repeated with the same parameters.
    """
    from .nlfff_torch import nlfff_lbfgs
    key = (nx, ny, nz, smooth, max_iter, tol, tuple(sorted(td_kw.items())))
    if key not in _RELAXED_CACHE:
        td = titov_demoulin_field(nx, ny, nz, **td_kw)
        r = nlfff_lbfgs(field_magnetogram(td, 0.0), nz, initial=td, nd=0, relax_bottom=None,
                        smooth=smooth, max_iter=max_iter, tol=tol, verbose=False,
                        div_weight=1.0)  # fixed test field: does not depend on the w_d of the NLFFF
        meta = {**td.meta, "method": "Titov-Démoulin relajado", "approx": None,
                "relax_iterations": r.meta["iterations"], "relax_message": r.meta["message"]}
        _RELAXED_CACHE[key] = (r.bx, r.by, r.bz, r.dx, r.origin, meta)
    bx, by, bz, dx, origin, meta = _RELAXED_CACHE[key]
    return VectorField3D(bx.copy(), by.copy(), bz.copy(), dx=dx, origin=origin, meta=dict(meta))


# Titov & Démoulin (1999) flux rope, as in Wiegelmann et al. (2006b).
# "TD" (analytical) is only approximately force-free; "TD-relax" is the
# same rope relaxed to an equilibrium. They are used as an **external test** of the
# calibration (they are not in CALIBRATION_FIELDS, with which it is fitted).
EXTERNAL_TEST_FIELDS = {
    "TD": lambda nx, ny, nz: titov_demoulin_field(nx, ny, nz),
    "TD-relax": lambda nx, ny, nz: relaxed_titov_demoulin_field(nx, ny, nz),
    # Second rope, used only as an external test in the revision: another geometry
    # (larger ring, deeper, charges further apart), more turns and opposite twist
    "TD2-relax": lambda nx, ny, nz: relaxed_titov_demoulin_field(nx, ny, nz, R=3.0, a=0.6, d=1.3,
                                                                 L=2.6, n_turns=1.5, handedness=1),
    # Three more families, only as an external test (10-Nov-2026; examples/17_external_families.py):
    # Low & Lou deeper and tilted to the other side than the three of the calibration
    "LL-IV": lambda nx, ny, nz: low_lou_field(nx, ny, nz, n=1, m=1, l=0.5, phi=-np.pi / 6),
    # exact delta-type LFFF: two compact, close polarities, two satellites, α = 0.9 α_max
    "LFFF-delta": lambda nx, ny, nz: lfff_multipole_field(
        nx, ny, nz, alpha_frac=0.9,
        spots=[(-0.15, -0.1, 1.0, 0.15), (0.15, 0.1, -1.0, 0.15), (0.45, -0.35, 0.6, 0.2), (-0.5, 0.4, -0.6, 0.2)]),
    # Third relaxed rope: thick (a/R = 0.35), shallow, close charges, 1.25 turns
    "TD3-relax": lambda nx, ny, nz: relaxed_titov_demoulin_field(nx, ny, nz, R=2.6, a=0.9, d=0.8,
                                                                 L=1.6, n_turns=1.25, handedness=-1),
    # Two magnetohydrostatic families (12-Nov-2026; examples/23_mhs_test.py): the regions and alpha of LFFF-bip and
    # LFFF-quad with a base that is not force-free. MHS-bip: net force of the base like that of an HMI magnetogram
    # (eps_force ~ 0.1), layer thinner than a pixel; MHS-quad: twice that force, layer of one to three pixels.
    "MHS-bip": lambda nx, ny, nz: mhs_multipole_field(nx, ny, nz, alpha_frac=0.7, a=0.6, scale_mm=2.0),
    "MHS-quad": lambda nx, ny, nz: mhs_multipole_field(
        nx, ny, nz, alpha_frac=0.5, a=0.9, scale_mm=4.0,
        spots=[(-0.45, -0.3, 1.0, 0.22), (-0.05, -0.35, -0.8, 0.2),
               (0.1, 0.35, 0.9, 0.22), (0.5, 0.25, -1.0, 0.23)]),
}
