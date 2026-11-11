"""3D vector field container and differential operators on a uniform grid."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


_W = {}


def trapezoid_weights(shape) -> np.ndarray:
    """Trapezoid-rule weights on a grid of nodes (2D or 3D): 1/2 on each edge or face,
    1/4 on edges and 1/8 at corners. Every volume or surface integral over the grid must
    use them: summing the nodes with weight 1 over-weights the edges, and at z = 0, where the field is
    strongest, biases the energy (revision R10)."""
    shape = tuple(shape)
    if shape not in _W:
        w = np.ones(shape)
        for ax in range(len(shape)):
            sl = [slice(None)] * len(shape)
            sl[ax] = 0
            w[tuple(sl)] *= 0.5
            sl[ax] = -1
            w[tuple(sl)] *= 0.5
        _W[shape] = w
    return _W[shape]


def gradient(f: np.ndarray, h: float):
    """Gradient (2nd-order centred differences, also at the edges)."""
    return np.gradient(f, h, edge_order=2)


def curl(bx, by, bz, h):
    """Curl of (bx, by, bz) on a uniform grid of step h."""
    dbx = np.gradient(bx, h, edge_order=2)
    dby = np.gradient(by, h, edge_order=2)
    dbz = np.gradient(bz, h, edge_order=2)
    jx = dbz[1] - dby[2]
    jy = dbx[2] - dbz[0]
    jz = dby[0] - dbx[1]
    return jx, jy, jz


def divergence(bx, by, bz, h):
    return (np.gradient(bx, h, axis=0, edge_order=2)
            + np.gradient(by, h, axis=1, edge_order=2)
            + np.gradient(bz, h, axis=2, edge_order=2))


def cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@dataclass
class VectorField3D:
    """3D magnetic field on a uniform Cartesian grid.

    Attributes
    ----------
    bx, by, bz : ndarray (nx, ny, nz), in Gauss.
    dx : grid step (Mm), the same in the three directions.
    origin : coordinates (x0, y0, z0) of node [0, 0, 0] in Mm.
    meta : free dictionary (method, alpha, convergence history, ...).
    """

    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    dx: float
    origin: tuple = (0.0, 0.0, 0.0)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.bx.shape

    @property
    def x(self):
        return self.origin[0] + self.dx * np.arange(self.shape[0])

    @property
    def y(self):
        return self.origin[1] + self.dx * np.arange(self.shape[1])

    @property
    def z(self):
        return self.origin[2] + self.dx * np.arange(self.shape[2])

    @property
    def extent(self):
        """(xmin, xmax, ymin, ymax, zmin, zmax) in Mm."""
        x, y, z = self.x, self.y, self.z
        return x[0], x[-1], y[0], y[-1], z[0], z[-1]

    @property
    def b2(self):
        return self.bx ** 2 + self.by ** 2 + self.bz ** 2

    @property
    def babs(self):
        return np.sqrt(self.b2)

    def current(self):
        """Return ∇×B (without the 4π/c factor; units G/Mm)."""
        return curl(self.bx, self.by, self.bz, self.dx)

    def divergence(self):
        return divergence(self.bx, self.by, self.bz, self.dx)

    def alpha(self, b_min_frac: float = 0.02) -> np.ndarray:
        r"""Local force-free parameter $\alpha = (\nabla\times\mathbf{B})\cdot\mathbf{B} / B^2$ (Mm⁻¹).

        NaN where |B| < ``b_min_frac``·max|B| (α is not well defined in
        weak field). In an LFFF it equals the constant α; in an NLFFF it is
        (ideally) constant along each field line.
        """
        jx, jy, jz = self.current()
        b2 = self.b2
        with np.errstate(invalid="ignore", divide="ignore"):
            a = (jx * self.bx + jy * self.by + jz * self.bz) / b2
        a[np.sqrt(b2) < b_min_frac * np.sqrt(b2.max())] = np.nan
        return a

    def bottom(self):
        """Components at z = 0 as a 2D tuple (bx, by, bz)."""
        return self.bx[:, :, 0], self.by[:, :, 0], self.bz[:, :, 0]

    def stack(self):
        return np.stack([self.bx, self.by, self.bz])

    def copy(self):
        return VectorField3D(self.bx.copy(), self.by.copy(), self.bz.copy(),
                             self.dx, tuple(self.origin), dict(self.meta))

    def __repr__(self):
        name = self.meta.get("method", "campo")
        return f"VectorField3D({name}, shape={self.shape}, dx={self.dx:.4g} Mm)"
