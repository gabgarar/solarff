r"""Quality metrics and comparison between models.

Force-free quality (Wheatland et al. 2000; Schrijver et al. 2006):

$$
\mathrm{CW}_{\sin} = \frac{\sum_i |\mathbf{J}_i|\,\sigma_i}{\sum_i |\mathbf{J}_i|}, \qquad
\sigma_i = \frac{|\mathbf{J}_i\times\mathbf{B}_i|}{|\mathbf{J}_i|\,|\mathbf{B}_i|}, \qquad
\langle|f_i|\rangle = \Big\langle\frac{|\nabla\cdot\mathbf{B}|\,\Delta x}{6\,|\mathbf{B}|}\Big\rangle
$$

$\mathrm{CW}_{\sin}$ is the sine of the J–B angle weighted by the current (0 = perfectly force-free),
$\langle|f_i|\rangle$ the non-solenoidal flux fraction, and $L_f$, $L_d$ the terms of Wiegelmann's functional.

Comparison between two fields $\mathbf{B}$ (reference) and $\mathbf{b}$ (Schrijver et al. 2006):

$$
\begin{aligned}
C_{\rm vec} &= \frac{\sum \mathbf{B}\cdot\mathbf{b}}{\sqrt{\sum B^2\,\sum b^2}}, &
C_{\rm CS} &= \Big\langle\frac{\mathbf{B}\cdot\mathbf{b}}{|\mathbf{B}|\,|\mathbf{b}|}\Big\rangle, \\
E'_n &= 1 - \frac{\sum |\mathbf{b}-\mathbf{B}|}{\sum |\mathbf{B}|}, &
E'_m &= 1 - \Big\langle\frac{|\mathbf{b}-\mathbf{B}|}{|\mathbf{B}|}\Big\rangle, \\
\epsilon &= \frac{\sum |\mathbf{b}|^2}{\sum |\mathbf{B}|^2} && \text{(energy ratio)}
\end{aligned}
$$

All equal 1 for identical fields.
"""

from __future__ import annotations

import numpy as np

from .field import VectorField3D, cross, trapezoid_weights
from .tracing import FieldLine, trace_field_lines

MM3_TO_CM3 = 1e24  # (1 Mm)^3 = (1e8 cm)^3


def _inner(field: VectorField3D, margin: int):
    if margin <= 0:
        return (slice(None),) * 3
    nz = field.shape[2]
    return (slice(margin, -margin), slice(margin, -margin), slice(0, max(nz - margin, 1)))


def energy(field: VectorField3D, margin: int = 0) -> float:
    r"""Magnetic energy $\int B^2/(8\pi)\,dV$ in erg (B in G, dx in Mm), with the trapezoid rule
    over the box (without the layer of ``margin`` points on the sides and top)."""
    s = _inner(field, margin)
    b2 = field.b2[s]
    return float((trapezoid_weights(b2.shape) * b2).sum() / (8 * np.pi) * field.dx ** 3 * MM3_TO_CM3)


def force_free_metrics(field: VectorField3D, margin: int = 0) -> dict:
    """Force-free and solenoidal quality metrics.

    ``margin``: points to exclude from the side and top edges (e.g. the
    buffer layer ``nd`` of the NLFFF).
    """
    h = field.dx
    jx, jy, jz = field.current()
    d = field.divergence()
    s = _inner(field, margin)
    # Interior points only (the derivatives at the edges are less accurate)
    core = (slice(1, -1), slice(1, -1), slice(1, -1))
    m = np.zeros(field.shape, dtype=bool)
    m[core] = True
    mm = np.zeros(field.shape, dtype=bool)
    mm[s] = True
    m &= mm

    bx, by, bz = field.bx[m], field.by[m], field.bz[m]
    jx, jy, jz, d = jx[m], jy[m], jz[m], d[m]
    b = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    j = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2)
    fx, fy, fz = cross(jx, jy, jz, bx, by, bz)
    jxb = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    ok = (b > 0) & (j > 0)
    sigma = np.zeros_like(b)
    sigma[ok] = jxb[ok] / (j[ok] * b[ok])
    cwsin = float((j * sigma).sum() / j.sum()) if j.sum() > 0 else 0.0
    bs = np.where(b > 0, b, np.inf)
    fi = float(np.mean(np.abs(d) * h / (6 * bs)))
    Lf = float(np.mean(np.where(b > 0, jxb ** 2 / bs ** 2, 0)))
    Ld = float(np.mean(d ** 2))
    return {
        "CWsin": cwsin,
        "theta_J_deg": float(np.degrees(np.arcsin(min(cwsin, 1.0)))),
        "<|f_i|>": fi,
        "L_f": Lf,
        "L_d": Ld,
        "E (erg)": energy(field, margin),
    }


def compare_vector_fields(ref: VectorField3D, b: VectorField3D, margin: int = 0) -> dict:
    """Schrijver et al. (2006) metrics of ``b`` with respect to ``ref``."""
    if ref.shape != b.shape:
        raise ValueError("Los campos deben tener la misma forma")
    s = _inner(ref, margin)
    B = np.stack([ref.bx[s], ref.by[s], ref.bz[s]]).reshape(3, -1)
    bb = np.stack([b.bx[s], b.by[s], b.bz[s]]).reshape(3, -1)
    nB = np.linalg.norm(B, axis=0)
    nb = np.linalg.norm(bb, axis=0)
    ok = (nB > 0) & (nb > 0)
    diff = np.linalg.norm(bb - B, axis=0)
    return {
        "C_vec": float((B * bb).sum() / np.sqrt((nB ** 2).sum() * (nb ** 2).sum())),
        "C_CS": float(np.mean((B * bb).sum(axis=0)[ok] / (nB[ok] * nb[ok]))),
        "E'_n": float(1 - diff.sum() / nB.sum()),
        "E'_m": float(1 - np.mean(diff[ok] / nB[ok])),
        "eps": float((nb ** 2).sum() / (nB ** 2).sum()),
    }


def compare_connectivity(lines_a: list[FieldLine], lines_b: list[FieldLine],
                         tol: float = 2.0) -> dict:
    """Compare lines traced from the *same* seeds in two models.

    Returns the fraction of lines with the same topology (open/closed),
    the mean distance between endpoints (Mm) and the fraction of closed lines
    whose two endpoints coincide within ``tol`` Mm.
    """
    if len(lines_a) != len(lines_b):
        raise ValueError("Las listas deben venir de las mismas semillas")
    same_topo, dists, match = [], [], []
    for a, b in zip(lines_a, lines_b):
        same_topo.append(a.closed == b.closed)
        d = 0.5 * (np.linalg.norm(a.start - b.start) + np.linalg.norm(a.end - b.end))
        dists.append(d)
        if a.closed and b.closed:
            match.append(d < tol)
    return {
        "same_topology": float(np.mean(same_topo)),
        "mean_endpoint_dist_Mm": float(np.mean(dists)),
        "closed_match_frac": float(np.mean(match)) if match else float("nan"),
        "frac_closed_a": float(np.mean([l.closed for l in lines_a])),
        "frac_closed_b": float(np.mean([l.closed for l in lines_b])),
        "mean_apex_a_Mm": float(np.mean([l.apex for l in lines_a])),
        "mean_apex_b_Mm": float(np.mean([l.apex for l in lines_b])),
    }


def compare_models(models: dict[str, VectorField3D], reference: str | None = None,
                   potential: str | None = None, margin: int = 0,
                   seeds: np.ndarray | None = None) -> dict:
    """Comparative table of several models.

    **Parameters**

    - ``models``: {"name": field}
    - ``reference``: name of the reference model for the vector metrics
        (e.g. the analytical solution, or "NLFFF"). If None they are not computed.
    - ``potential``: name of the potential field to give E/E_pot and free energy.
        If None a model with meta["method"] == "potencial" is looked for.
    - ``seeds``: if given, the connectivity with the reference is compared too.
    """
    if potential is None:
        potential = next((k for k, v in models.items() if v.meta.get("method") == "potencial"), None)
    e_pot = energy(models[potential], margin) if potential else None
    lines = {}
    if seeds is not None and reference is not None:
        lines = {k: trace_field_lines(v, seeds) for k, v in models.items()}

    table = {}
    for name, f in models.items():
        row = force_free_metrics(f, margin)
        if e_pot:
            row["E/E_pot"] = row["E (erg)"] / e_pot
            row["E_free (erg)"] = row["E (erg)"] - e_pot
        if reference is not None:
            row.update(compare_vector_fields(models[reference], f, margin))
            if lines:
                c = compare_connectivity(lines[reference], lines[name])
                row["same_topology"] = c["same_topology"]
                row["endpoint_dist_Mm"] = c["mean_endpoint_dist_Mm"]
        if "alpha" in f.meta:
            row["alpha (1/Mm)"] = f.meta["alpha"]
        table[name] = row
    return table


def print_table(table: dict, floatfmt: str = ".4g") -> None:
    """Print the table of ``compare_models`` with the models in columns."""
    names = list(table)
    keys = []
    for r in table.values():
        keys += [k for k in r if k not in keys]
    w0 = max(len(k) for k in keys) + 2
    w = max(12, max(len(n) for n in names) + 2)
    print("".ljust(w0) + "".join(n.rjust(w) for n in names))
    for k in keys:
        vals = []
        for n in names:
            v = table[n].get(k)
            vals.append("-" if v is None else format(v, floatfmt))
        print(k.ljust(w0) + "".join(v.rjust(w) for v in vals))
