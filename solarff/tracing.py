"""Field-line tracing (vectorised RK4, trilinear interpolation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from .field import VectorField3D
from .magnetogram import Magnetogram


@dataclass
class FieldLine:
    """Field line in physical coordinates (Mm).

    ``points`` is ordered in the direction of B: the first point is the
    "outgoing" end (positive polarity if the line is closed) and the last the
    "incoming" end (negative polarity).
    """

    points: np.ndarray       # (M, 3)
    seed: np.ndarray         # (3,)
    closed: bool             # both ends at the base (z ≈ 0)

    @property
    def start(self):
        return self.points[0]

    @property
    def end(self):
        return self.points[-1]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    @property
    def apex(self) -> float:
        return float(self.points[:, 2].max())


def _interp(comp, idx):
    return map_coordinates(comp, idx, order=1, mode="nearest")


def _direction(field_arrays, idx, sign):
    bx, by, bz = (_interp(c, idx) for c in field_arrays)
    b = np.sqrt(bx * bx + by * by + bz * bz)
    bs = np.where(b > 0, b, 1.0)
    return sign * np.stack([bx / bs, by / bs, bz / bs]), b


def _stop_at_boundary(p, newp, upper):
    """If the step p -> newp leaves the domain, return the exact crossing point
    with the face (linear interpolation) instead of clipping coordinate by coordinate."""
    up = upper[:, None]
    out = ((newp < 0) | (newp > up)).any(axis=0)
    if out.any():
        d = newp - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = np.where(newp < 0, p / (p - newp), 1.0)
            t_hi = np.where(newp > up, (up - p) / d, 1.0)
        t = np.clip(np.minimum(t_lo, t_hi).min(axis=0), 0.0, 1.0)
        newp = np.where(out, p + t * d, newp)
        newp = np.clip(newp, 0, up)
    return newp, out


def integrate_lines(field: VectorField3D, start_idx: np.ndarray, sign: float,
                    scalar: np.ndarray | None = None, ds: float = 0.25,
                    max_steps: int = 20000, b_min: float = 1e-6):
    """Integrate lines from ``start_idx`` (3, N) (grid indices) in one direction
    without storing the intermediate points.

    Returns (endpoint (3, N) in indices, length (N,) in cells, integral of
    ``scalar`` along the line (N,) in cell units). Meant for
    thousands of lines (Q and Tw maps).
    """
    fa = (field.bx, field.by, field.bz)
    upper = np.array(field.shape, dtype=float) - 1
    pos = np.clip(np.asarray(start_idx, dtype=float).copy(), 0, upper[:, None])
    n = pos.shape[1]
    length = np.zeros(n)
    integ = np.zeros(n)
    active = np.ones(n, dtype=bool)
    for _ in range(max_steps):
        if not active.any():
            break
        ia = np.flatnonzero(active)
        p = pos[:, ia]
        k1, b = _direction(fa, p, sign)
        k2, _ = _direction(fa, p + 0.5 * ds * k1, sign)
        k3, _ = _direction(fa, p + 0.5 * ds * k2, sign)
        k4, _ = _direction(fa, p + ds * k3, sign)
        newp = p + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        newp, out = _stop_at_boundary(p, newp, upper)
        stop = out | (b < b_min)
        seg = np.linalg.norm(newp - p, axis=0)
        length[ia] += seg
        if scalar is not None:
            integ[ia] += map_coordinates(scalar, 0.5 * (p + newp), order=1, mode="nearest") * seg
        pos[:, ia] = newp
        active[ia[stop]] = False
    return pos, length, integ


def _trace_one_direction(fa, start_idx, sign, ds, max_steps, upper, b_min):
    n = start_idx.shape[1]
    paths = [start_idx.copy()]
    pos = start_idx.copy()
    active = np.ones(n, dtype=bool)
    for _ in range(max_steps):
        if not active.any():
            break
        p = pos[:, active]
        k1, b = _direction(fa, p, sign)
        k2, _ = _direction(fa, p + 0.5 * ds * k1, sign)
        k3, _ = _direction(fa, p + 0.5 * ds * k2, sign)
        k4, _ = _direction(fa, p + ds * k3, sign)
        newp = p + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        newp, out = _stop_at_boundary(p, newp, upper)
        out |= b < b_min
        idx_active = np.flatnonzero(active)
        pos[:, idx_active] = newp
        step = np.full((3, n), np.nan)
        step[:, idx_active] = newp
        paths.append(step)
        active[idx_active[out]] = False
    return np.stack(paths, axis=0)  # (steps, 3, n)


def trace_field_lines(field: VectorField3D, seeds: np.ndarray, ds: float = 0.5,
                      max_steps: int = 4000, b_min: float = 1e-3) -> list[FieldLine]:
    """Trace field lines from ``seeds`` (array (N, 3) in Mm) in both directions.

    ``ds`` is the step in cell units. Using the *same* seeds in
    different models allows their connectivity to be compared.
    """
    seeds = np.atleast_2d(np.asarray(seeds, dtype=float))
    origin = np.asarray(field.origin, dtype=float)
    idx0 = ((seeds - origin) / field.dx).T  # (3, N)
    upper = np.array(field.shape, dtype=float) - 1
    idx0 = np.clip(idx0, 0, upper[:, None])
    fa = (field.bx, field.by, field.bz)

    fwd = _trace_one_direction(fa, idx0, +1.0, ds, max_steps, upper, b_min)
    bwd = _trace_one_direction(fa, idx0, -1.0, ds, max_steps, upper, b_min)

    lines = []
    z_tol = 1.0  # cells
    for j in range(seeds.shape[0]):
        f = fwd[:, :, j]
        b = bwd[:, :, j]
        f = f[~np.isnan(f[:, 0])]
        b = b[~np.isnan(b[:, 0])]
        pts = np.concatenate([b[::-1], f[1:]], axis=0)
        pts_phys = pts * field.dx + origin
        closed = bool(pts[0, 2] <= z_tol and pts[-1, 2] <= z_tol and len(pts) > 3)
        lines.append(FieldLine(points=pts_phys, seed=seeds[j], closed=closed))
    return lines


def seeds_from_magnetogram(source: Magnetogram | VectorField3D, n: int = 80,
                           threshold: float = 0.1, z: float | None = None,
                           polarity: str = "positive", method: str = "grid",
                           rng: int | None = 0) -> np.ndarray:
    """Seeds to trace lines at pixels with |Bz| > ``threshold``·max|Bz|.

    **Parameters**

    - ``n``: approximate number of seeds.
    - ``polarity``: ``"positive"`` (default), ``"negative"`` or ``"both"``. Since each
        line is traced in both directions, seeding on a single polarity avoids
        drawing the same loop twice.
    - ``method``: ``"grid"`` (regular grid over the selected region: also covers
        the weak field and gives cleaner figures) or ``"random"``
        (probability proportional to |Bz|, concentrates in the cores).
    - ``z``: height of the seeds (by default one cell above the base, to
        avoid the z = 0 layer, where the NLFFF is noisier).
    """
    if isinstance(source, VectorField3D):
        bz = source.bz[:, :, 0]
        dx = source.dx
        ox, oy, _ = source.origin
    else:
        bz, dx = source.bz, source.dx
        ox, oy = source.origin
    if z is None:
        z = dx
    signed = {"positive": bz, "negative": -bz, "both": np.abs(bz)}[polarity]
    mask = signed > threshold * np.abs(bz).max()
    gen = np.random.default_rng(rng)

    if method == "grid":
        ix, iy = np.nonzero(mask)
        # Grid step such that ~n seeds remain
        step = max(1.0, np.sqrt(mask.sum() / max(n, 1)))
        while True:
            gx = np.floor(ix / step)
            gy = np.floor(iy / step)
            key = gx * 1e6 + gy
            _, first = np.unique(key, return_index=True)
            if first.size <= n or step > max(bz.shape):
                break
            step *= 1.05
        # One point per grid cell: the one closest to the centre of the cell
        cx, cy = (gx + 0.5) * step, (gy + 0.5) * step
        d = (ix - cx) ** 2 + (iy - cy) ** 2
        order = np.lexsort((d, key))
        _, first = np.unique(key[order], return_index=True)
        sel = order[first]
        px, py = ix[sel].astype(float), iy[sel].astype(float)
    elif method == "random":
        cand = np.flatnonzero(mask.ravel())
        p = np.abs(bz).ravel()[cand]
        pick = gen.choice(cand, size=min(n, cand.size), replace=False, p=p / p.sum())
        px, py = np.unravel_index(pick, bz.shape)
        px = px + gen.uniform(-0.5, 0.5, px.size)
        py = py + gen.uniform(-0.5, 0.5, py.size)
    else:
        raise ValueError("method debe ser 'grid' o 'random'")
    return np.column_stack([ox + px * dx, oy + py * dx, np.full(px.size, z)])
