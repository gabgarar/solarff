"""2D vector magnetogram in local Cartesian coordinates (photosphere, z = 0)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Magnetogram:
    """Photospheric field on a local Cartesian grid.

    Attributes
    ----------
    bx, by, bz : ndarray (nx, ny) in Gauss. ``bz`` is the vertical
        (radial) component. ``bx``/``by`` may be ``None`` if only the vertical field is
        available (enough for the potential field and LFFF, not for NLFFF).
    dx : pixel size in Mm.
    origin : coordinates (x0, y0) in Mm of pixel [0, 0].
    meta : header / origin information.
    bx_err, by_err, bz_err : uncertainty (1σ, G) of each component at each
        pixel, if known (SHARP: segments Bp_err, Bt_err, Br_err). Used
        in the Monte Carlo (``solarff.monte_carlo``).
    """

    bz: np.ndarray
    bx: np.ndarray | None = None
    by: np.ndarray | None = None
    dx: float = 1.0
    meta: dict = field(default_factory=dict)
    origin: tuple = (0.0, 0.0)
    bx_err: np.ndarray | None = None
    by_err: np.ndarray | None = None
    bz_err: np.ndarray | None = None

    def __post_init__(self):
        self.bz = np.asarray(self.bz, dtype=float)
        if self.bx is not None:
            self.bx = np.asarray(self.bx, dtype=float)
            self.by = np.asarray(self.by, dtype=float)

    # ------------------------------------------------------------------ props
    @property
    def shape(self):
        return self.bz.shape

    @property
    def has_vector(self) -> bool:
        return self.bx is not None and self.by is not None

    @property
    def has_errors(self) -> bool:
        return self.bz_err is not None

    @property
    def bt(self):
        """Modulus of the transverse field."""
        self._require_vector()
        return np.hypot(self.bx, self.by)

    @property
    def x(self):
        return self.origin[0] + self.dx * np.arange(self.shape[0])

    @property
    def y(self):
        return self.origin[1] + self.dx * np.arange(self.shape[1])

    def _require_vector(self):
        if not self.has_vector:
            raise ValueError("Este magnetograma no tiene componentes transversales (bx, by).")

    # ------------------------------------------------------------- operations
    def _map(self, fn, dx=None, origin=None, efn=None):
        efn = efn or fn
        return Magnetogram(
            bz=fn(self.bz),
            bx=None if self.bx is None else fn(self.bx),
            by=None if self.by is None else fn(self.by),
            dx=self.dx if dx is None else dx,
            meta=dict(self.meta),
            origin=tuple(self.origin) if origin is None else tuple(origin),
            bx_err=None if self.bx_err is None else efn(self.bx_err),
            by_err=None if self.by_err is None else efn(self.by_err),
            bz_err=None if self.bz_err is None else efn(self.bz_err),
        )

    def crop(self, xslice: slice, yslice: slice) -> "Magnetogram":
        """Crop by pixel indices: ``mag.crop(slice(100, 400), slice(50, 250))``."""
        x0 = self.origin[0] + (xslice.start or 0) * self.dx
        y0 = self.origin[1] + (yslice.start or 0) * self.dx
        return self._map(lambda a: a[xslice, yslice].copy(), origin=(x0, y0))

    def rebin(self, factor: int) -> "Magnetogram":
        """Reduce the resolution by averaging ``factor x factor`` blocks.

        Essential with full-resolution SHARPs (~0.36 Mm/pixel): a 600x300x150
        NLFFF is not tractable in pure numpy.
        """
        if factor == 1:
            return self._map(lambda a: a.copy())
        nx, ny = (s // factor * factor for s in self.shape)

        def _bin(a):
            a = a[:nx, :ny]
            return a.reshape(nx // factor, factor, ny // factor, factor).mean(axis=(1, 3))

        def _bin_err(e):
            # Mean of f² pixels with independent errors: σ = sqrt(Σσ²) / f²
            return np.sqrt(_bin(e ** 2) / factor ** 2)

        off = (factor - 1) / 2 * self.dx
        return self._map(_bin, dx=self.dx * factor,
                         origin=(self.origin[0] + off, self.origin[1] + off), efn=_bin_err)

    def balance_flux(self) -> "Magnetogram":
        """Subtract the net flux of bz (needed for the problem to be well posed)."""
        out = self._map(lambda a: a.copy())
        out.bz = out.bz - out.bz.mean()
        return out

    def pad(self, n: int) -> "Magnetogram":
        """Add ``n`` pixels of zeros around (reduces edge effects)."""
        return self._map(lambda a: np.pad(a, n, mode="constant"), efn=lambda e: np.pad(e, n, mode="edge"),
                         origin=(self.origin[0] - n * self.dx, self.origin[1] - n * self.dx))

    # ----------------------------------------------------------- diagnostics
    def flux_imbalance(self) -> float:
        r"""Flux imbalance $|\sum B_z| / \sum |B_z|$."""
        return float(abs(self.bz.sum()) / np.abs(self.bz).sum())

    def jz(self) -> np.ndarray:
        """Vertical component of ∇×B at the photosphere (G/Mm)."""
        self._require_vector()
        return (np.gradient(self.by, self.dx, axis=0, edge_order=2)
                - np.gradient(self.bx, self.dx, axis=1, edge_order=2))

    def __repr__(self):
        kind = "vectorial" if self.has_vector else "sólo Bz"
        return (f"Magnetogram({kind}, shape={self.shape}, dx={self.dx:.4g} Mm, "
                f"|Bz|max={np.abs(self.bz).max():.0f} G)")


def aly_criteria(mag: Magnetogram) -> dict:
    """Aly (1989) criteria: a photospheric field compatible with a force-free
    field must have zero net force and torque.

    Returns ``eps_force`` and ``eps_torque`` (dimensionless). Values << 1
    (typically < 0.1) indicate data consistent with NLFFF. Real
    photospheric magnetograms usually give high values, hence the
    preprocessing (``solarff.preprocess``).
    """
    mag._require_vector()
    bx, by, bz = mag.bx, mag.by, mag.bz
    nx, ny = mag.shape
    x, y = np.meshgrid(np.arange(nx) - (nx - 1) / 2, np.arange(ny) - (ny - 1) / 2, indexing="ij")
    b2 = bx ** 2 + by ** 2 + bz ** 2
    d = bz ** 2 - bx ** 2 - by ** 2
    eps_f = (abs((bx * bz).sum()) + abs((by * bz).sum()) + abs(d.sum())) / b2.sum()
    r = np.hypot(x, y)
    eps_t = (abs((x * d).sum()) + abs((y * d).sum()) + abs((y * bx * bz - x * by * bz).sum())) / (r * b2).sum()
    return {"eps_force": float(eps_f), "eps_torque": float(eps_t),
            "flux_imbalance": mag.flux_imbalance()}
