r"""Field topology: twist number (Tw) and squashing factor (Q).

Twist number (Berger & Prior 2006; Liu et al. 2016):

$$
T_w = \frac{1}{4\pi}\int \alpha\,dl, \qquad \alpha = \frac{(\nabla\times\mathbf{B})\cdot\mathbf{B}}{B^2}
$$

Counts how many turns the field makes around a line. $|T_w| \gtrsim 1$ indicates
a flux rope, a structure that is a candidate to erupt.

Squashing factor (Titov et al. 2002; Pariat & Démoulin 2012): measures how much
a small bundle of lines is deformed between its two ends. For the
endpoint mapping $(x_0, y_0) \to (X, Y)$ with Jacobian $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

$$
Q = \frac{a^2 + b^2 + c^2 + d^2}{|B_{n,0} / B_{n,1}|}
$$

$Q = 2$ is the minimum (no deformation); $Q \gg 2$ marks the quasi-separatrix
layers (QSL), where the connectivity changes abruptly and reconnection is likely.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .field import VectorField3D
from .tracing import FieldLine, integrate_lines

_FACES = ("bottom", "top", "x0", "x1", "y0", "y1")
# coordinates on the face and normal component for each face
_FACE_DIMS = {"bottom": ((0, 1), 2), "top": ((0, 1), 2), "x0": ((1, 2), 0),
              "x1": ((1, 2), 0), "y0": ((0, 2), 1), "y1": ((0, 2), 1)}


def _alpha_full(field: VectorField3D) -> np.ndarray:
    a = field.alpha(b_min_frac=0.0)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def twist_number(field: VectorField3D, lines: list[FieldLine]) -> np.ndarray:
    """Tw of each already traced line (integrates α over its points)."""
    a = _alpha_full(field)
    origin = np.asarray(field.origin)
    tw = np.full(len(lines), np.nan)
    for i, l in enumerate(lines):
        if len(l.points) < 2:
            continue
        mid = 0.5 * (l.points[1:] + l.points[:-1])
        seg = np.linalg.norm(np.diff(l.points, axis=0), axis=1)
        av = map_coordinates(a, ((mid - origin) / field.dx).T, order=1, mode="nearest")
        tw[i] = (av * seg).sum() / (4 * np.pi)
    return tw


def _base_seeds(field, step, threshold):
    bz = field.bz[:, :, 0]
    ix = np.arange(0, field.shape[0], step)
    iy = np.arange(0, field.shape[1], step)
    IX, IY = np.meshgrid(ix, iy, indexing="ij")
    strong = np.abs(bz[IX, IY]) > threshold * np.abs(bz).max()
    return IX, IY, strong


def twist_map(field: VectorField3D, step: int = 1, threshold: float = 0.05,
              ds: float = 0.25) -> dict:
    """Tw of the lines leaving each pixel of the base (as in Liu et al. 2016).

    Each line is integrated from the base in the direction the field leaves
    until it returns to the photosphere or leaves the domain.

    Returns a dict with ``tw`` (2D, NaN where |Bz| is weak), ``closed`` (2D,
    the line returns to the base), ``length`` (Mm) and the coordinates ``x``, ``y``.
    """
    IX, IY, strong = _base_seeds(field, step, threshold)
    bz0 = field.bz[IX, IY, 0]
    start = np.stack([IX[strong], IY[strong], np.zeros(strong.sum())]).astype(float)
    sgn = np.sign(bz0[strong])
    a = _alpha_full(field)
    tw = np.full(IX.shape, np.nan)
    closed = np.zeros(IX.shape, dtype=bool)
    length = np.full(IX.shape, np.nan)
    for s in (1.0, -1.0):
        sel = sgn == s
        if not sel.any():
            continue
        end, L, integ = integrate_lines(field, start[:, sel], s, scalar=a, ds=ds)
        # The sign of the integration does not change α; Tw is intrinsic to the line
        idx = np.flatnonzero(strong.ravel())[sel]
        tw.flat[idx] = integ * field.dx / (4 * np.pi)
        closed.flat[idx] = (end[2] <= 0.5) & (L > 2)
        length.flat[idx] = L * field.dx
    x = field.origin[0] + IX[:, 0] * field.dx
    y = field.origin[1] + IY[0, :] * field.dx
    return {"tw": tw, "closed": closed, "length": length, "x": x, "y": y, "step": step}


def _face_of(pos, upper, eps=1e-6):
    """Face of the domain where each line ends ('' if it stopped inside or on an edge)."""
    conds = [pos[2] <= eps, pos[2] >= upper[2] - eps, pos[0] <= eps,
             pos[0] >= upper[0] - eps, pos[1] <= eps, pos[1] >= upper[1] - eps]
    conds = np.array(conds)
    out = np.full(pos.shape[1], "", dtype=object)
    single = conds.sum(axis=0) == 1
    for k, name in enumerate(_FACES):
        out[single & conds[k]] = name
    return out


def squashing_q(field: VectorField3D, step: int = 1, threshold: float = 0.05,
                delta: float = 0.25, ds: float = 0.2) -> dict:
    """Squashing factor Q at the base (z = 0).

    For each pixel the central line and four neighbours displaced by
    ±``delta`` cells in x and y are traced; the Jacobian of the endpoint mapping is
    obtained by centred differences. Works for lines ending on
    any face of the domain (Pariat & Démoulin 2012). Where the neighbours
    end on different faces, Q is undefined (NaN).

    Returns ``q`` (2D), ``logq`` (log10 Q), ``face`` (arrival face) and ``x``, ``y``.
    """
    IX, IY, strong = _base_seeds(field, step, threshold)
    upper = np.array(field.shape, dtype=float) - 1
    n = int(strong.sum())
    cx, cy = IX[strong].astype(float), IY[strong].astype(float)
    sgn = np.sign(field.bz[IX[strong], IY[strong], 0])
    offsets = [(0, 0), (delta, 0), (-delta, 0), (0, delta), (0, -delta)]
    ends = np.zeros((5, 3, n))
    starts = np.zeros((5, 3, n))
    for k, (ox, oy) in enumerate(offsets):
        start = np.stack([np.clip(cx + ox, 0, upper[0]), np.clip(cy + oy, 0, upper[1]), np.zeros(n)])
        starts[k] = start
        for s in (1.0, -1.0):
            sel = sgn == s
            if sel.any():
                e, _, _ = integrate_lines(field, start[:, sel], s, ds=ds)
                ends[k][:, sel] = e
    faces = [_face_of(ends[k], upper) for k in range(5)]
    same = np.ones(n, dtype=bool)
    for k in range(1, 5):
        same &= faces[k] == faces[0]
    same &= faces[0] != ""

    q = np.full(n, np.nan)
    fa = (field.bx, field.by, field.bz)
    bn0 = np.abs(map_coordinates(field.bz, np.stack([cx, cy, np.zeros(n)]), order=1, mode="nearest"))
    for name in _FACES:
        sel = same & (faces[0] == name)
        if not sel.any():
            continue
        (u, v), nd = _FACE_DIMS[name]
        # Actual separation of the seeds (at the edges the neighbour is clipped to the face)
        hx = starts[1][0, sel] - starts[2][0, sel]
        hy = starts[3][1, sel] - starts[4][1, sel]
        hx = np.where(hx > 0, hx, np.nan)
        hy = np.where(hy > 0, hy, np.nan)
        a = (ends[1][u, sel] - ends[2][u, sel]) / hx
        b = (ends[3][u, sel] - ends[4][u, sel]) / hy
        c = (ends[1][v, sel] - ends[2][v, sel]) / hx
        d = (ends[3][v, sel] - ends[4][v, sel]) / hy
        bn1 = np.abs(map_coordinates(fa[nd], ends[0][:, sel], order=1, mode="nearest"))
        with np.errstate(divide="ignore", invalid="ignore"):
            q[sel] = (a * a + b * b + c * c + d * d) * bn1 / bn0[sel]
    Q = np.full(IX.shape, np.nan)
    Q[strong] = q
    F = np.full(IX.shape, "", dtype=object)
    F[strong] = faces[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        logq = np.log10(np.where(Q > 0, Q, np.nan))
    x = field.origin[0] + IX[:, 0] * field.dx
    y = field.origin[1] + IY[0, :] * field.dx
    return {"q": Q, "logq": logq, "face": F, "x": x, "y": y, "step": step}
