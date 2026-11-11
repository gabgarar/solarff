"""Visualisation with matplotlib: magnetograms, field lines and comparisons.

Colour conventions (the same in every figure):

* Magnetogram Bz in **grey scale** (negative dark, positive light) with
  polarity contours (solid = +Bz, dashed = -Bz) and the polarity
  inversion line (PIL) in black: the polarity never depends on colour alone.
* Height of the lines: single-hue sequential ramp (blue), **shared**
  between models so that the same colour means the same height.
* Connectivity comparison: a few highlighted seeds with fixed categorical
  colours and numbered (same number = same seed in every
  model); the other lines in grey.
* α and errors: blue-grey-red diverging for α (sign), sequential blue
  for the error angle.
"""

from __future__ import annotations

import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates

from .field import VectorField3D
from .magnetogram import Magnetogram
from .tracing import FieldLine, trace_field_lines

# ----------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#b5b4af"
NODATA = "#d6d5d0"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#6250d6", "#e34948"]

BZ_CMAP = LinearSegmentedColormap.from_list("bz_gray", ["#4f4e4a", "#fbfbf9"])
APEX_CMAP = LinearSegmentedColormap.from_list(
    "apex_blue", ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
).with_extremes(bad=NODATA)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#9b2423"]
).with_extremes(bad=NODATA)

_STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11,
    "font.size": 9.5, "axes.linewidth": 0.6, "legend.frameon": False,
}


# ------------------------------------------------------------- language
_LANG = "es"
_TXT = {
    "magnetogram": ("Magnetograma", "Magnetogram"),
    "apex": ("altura máxima de la línea [Mm]", "field-line apex height [Mm]"),
    "bz0": ("Bz (z = 0) [G]", "Bz (z = 0) [G]"),
    "pos": ("+Bz (20 %, 60 %)", "+Bz (20 %, 60 %)"),
    "neg": ("−Bz (20 %, 60 %)", "−Bz (20 %, 60 %)"),
    "pil": ("línea de inversión (PIL)", "polarity inversion line (PIL)"),
    "rest": ("resto de líneas", "other field lines"),
    "same_seed": ("Mismo número = misma semilla en todos los modelos",
                  "Same number = same seed point in every model"),
    "alpha_sup": ("Parámetro libre de fuerzas α a z ≈ {z:.1f} Mm",
                  "Force-free parameter α at z ≈ {z:.1f} Mm"),
    "alpha_foot": ("Contornos: Bz en la base (continuo +, discontinuo −); línea negra: PIL; "
                   "gris: campo débil (α no definido)",
                   "Contours: Bz at the base (solid +, dashed −); black line: PIL; "
                   "grey: weak field (α undefined)"),
    "alpha_mean": ("α medio de la línea [Mm⁻¹]", "mean α along the line [Mm⁻¹]"),
    "n_lines": ("número de líneas", "number of field lines"),
    "alpha_q1": ("¿Qué α tiene cada línea?", "Which α does each line carry?"),
    "alpha_std": ("desviación típica de α a lo largo de la línea [Mm⁻¹]",
                  "standard deviation of α along the line [Mm⁻¹]"),
    "alpha_q2": ("¿Es α constante en cada línea? (0 = libre de fuerzas)",
                 "Is α constant along each line? (0 = force-free)"),
    "angle_cb": ("ángulo entre B del modelo y B de «{ref}» [°]",
                 "angle between model B and “{ref}” B [°]"),
    "err_sup": ("Error angular por altura (línea negra: PIL; gris: campo débil)",
                "Angular difference by height (black line: PIL; grey: weak field)"),
}


def set_language(lang: str) -> None:
    """Language of the figure texts: ``"es"`` or ``"en"``."""
    global _LANG
    if lang not in ("es", "en"):
        raise ValueError("lang debe ser 'es' o 'en'")
    _LANG = lang


def _t(key, **kw):
    s = _TXT[key][0 if _LANG == "es" else 1]
    return s.format(**kw) if kw else s


def _styled(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with plt.rc_context(_STYLE):
            return fn(*a, **k)
    return wrapper


# ------------------------------------------------------------- utilities
def _bz_vlim(bz):
    return float(np.percentile(np.abs(bz), 99.5)) or 1.0


def _edges(x):
    d = x[1] - x[0] if x.size > 1 else 1.0
    return x[0] - d / 2, x[-1] + d / 2


def _pil_field(bz, frac=0.15):
    """Smoothed Bz, NaN outside the strong-field areas (to contour the PIL)."""
    s = gaussian_filter(bz, 1.5)
    strong = maximum_filter(np.abs(bz), size=5) > frac * np.abs(bz).max()
    return np.where(strong, s, np.nan)


def _draw_bz_2d(ax, x, y, bz, vlim, contours=True, pil=True, image=True):
    if image:
        ax.imshow(bz.T, origin="lower", cmap=BZ_CMAP, vmin=-vlim, vmax=vlim,
                  extent=(*_edges(x), *_edges(y)), interpolation="bilinear")
    X, Y = np.meshgrid(x, y, indexing="ij")
    if contours:
        ax.contour(X, Y, bz, levels=[0.2 * vlim, 0.6 * vlim], colors=INK2,
                   linewidths=0.6, linestyles="solid")
        ax.contour(X, Y, bz, levels=[-0.6 * vlim, -0.2 * vlim], colors=INK2,
                   linewidths=0.6, linestyles="dashed")
    if pil:
        p = _pil_field(bz)
        if np.nanmin(p) < 0 < np.nanmax(p):
            ax.contour(X, Y, p, levels=[0.0], colors=INK, linewidths=1.3)


def _bz_legend_handles():
    return [Line2D([], [], color=INK2, lw=0.8, ls="solid", label=_t("pos")),
            Line2D([], [], color=INK2, lw=0.8, ls="dashed", label=_t("neg")),
            Line2D([], [], color=INK, lw=1.3, label=_t("pil"))]


def spread_indices(points: np.ndarray, k: int) -> np.ndarray:
    """k well-spread indices of ``points`` (farthest-point sampling)."""
    points = np.asarray(points)[:, :2]
    k = min(k, len(points))
    if k == 0:
        return np.array([], dtype=int)
    c = points.mean(axis=0)
    idx = [int(np.argmin(np.linalg.norm(points - c, axis=1)))]
    d = np.linalg.norm(points - points[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    # Stable order (by x) so that the numbering is readable
    idx = sorted(idx, key=lambda i: (points[i, 0], points[i, 1]))
    return np.array(idx)


def _style_for_lines(lines, color_by, norm, highlight, lw):
    """List of (colour, lw, zorder, label) per line."""
    out = []
    hl = {int(i): n for n, i in enumerate(highlight if highlight is not None else [])}
    for j, l in enumerate(lines):
        if color_by == "seed":
            if j in hl:
                out.append((CATEGORICAL[hl[j] % len(CATEGORICAL)], 2.0, 6, str(hl[j] + 1)))
            else:
                out.append((MUTED, 0.5, 4, None))
        elif color_by == "closed":
            out.append((CATEGORICAL[0] if l.closed else CATEGORICAL[1], lw, 5, None))
        else:
            out.append((APEX_CMAP(norm(l.apex)), lw, 5, None))
    return out


# ----------------------------------------------------------- magnetogram
@_styled
def plot_magnetogram(mag: Magnetogram, ax=None, quiver_step: int | None = None,
                     bt_min: float = 100.0, title: str | None = None):
    """Bz in greys with contours and PIL; if vector, transverse field with arrows."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    vlim = _bz_vlim(mag.bz)
    _draw_bz_2d(ax, mag.x, mag.y, mag.bz, vlim)
    plt.colorbar(ScalarMappable(Normalize(-vlim, vlim), BZ_CMAP), ax=ax, label="Bz [G]")
    if mag.has_vector:
        st = quiver_step or max(2, max(mag.shape) // 18)
        X, Y = np.meshgrid(mag.x, mag.y, indexing="ij")
        sl = (slice(None, None, st), slice(None, None, st))
        m = mag.bt[sl] > bt_min
        # Longest arrow = 1.5 times the arrow spacing
        bt_max = max(float(mag.bt[sl][m].max()) if m.any() else 1.0, 1e-9)
        ax.quiver(X[sl][m], Y[sl][m], mag.bx[sl][m], mag.by[sl][m], color=CATEGORICAL[0],
                  angles="xy", scale_units="xy", scale=bt_max / (1.5 * st * mag.dx),
                  width=0.004, alpha=0.9)
    ax.set_xlabel("x [Mm]")
    ax.set_ylabel("y [Mm]")
    ax.set_aspect("equal")
    ax.set_title(title or _t("magnetogram"))
    ax.legend(handles=_bz_legend_handles(), loc="upper left", fontsize=8,
              bbox_to_anchor=(0, -0.12), ncol=3)
    return ax


# --------------------------------------------------------- field lines
@_styled
def plot_field_lines(field: VectorField3D, lines: list[FieldLine] | None = None,
                     seeds: np.ndarray | None = None, ax=None, view: str = "3d",
                     color_by: str = "apex", norm: Normalize | None = None,
                     highlight=None, title: str | None = None,
                     elev: float = 35, azim: float = -65, lw: float = 0.9,
                     colorbar: bool = False):
    """Draw field lines over Bz(z=0).

    view : ``"3d"``, ``"top"`` (xy projection) or ``"side"`` (xz projection).
    color_by : ``"apex"`` (maximum height, blue ramp), ``"seed"`` (highlights the
        numbered seeds of ``highlight``; the rest in grey) or ``"closed"``
        (blue = closed, orange = leaves the domain).
    norm : height normalisation; pass a shared one when comparing models.
    highlight : indices of the seeds to highlight (``color_by="seed"``). By
        default 6 well-spread seeds.
    """
    if lines is None:
        if seeds is None:
            raise ValueError("Da 'lines' o 'seeds'")
        lines = trace_field_lines(field, seeds)
    if norm is None:
        norm = Normalize(0, max(max(l.apex for l in lines), 1e-9))
    if color_by == "seed" and highlight is None:
        highlight = spread_indices(np.array([l.seed for l in lines]), 6)
    x, y, z = field.x, field.y, field.z
    bz0 = field.bz[:, :, 0]
    vlim = _bz_vlim(bz0)
    styles = _style_for_lines(lines, color_by, norm, highlight, lw)
    order = np.argsort([s[2] for s in styles], kind="stable")  # highlighted ones last

    if view == "3d":
        if ax is None:
            fig = plt.figure(figsize=(8, 7), layout="constrained")
            ax = fig.add_subplot(projection="3d")
        ax.computed_zorder = False  # the z=0 plane always below the lines
        X, Y = np.meshgrid(x, y, indexing="ij")
        st = max(1, max(bz0.shape) // 120)
        fc = BZ_CMAP(Normalize(-vlim, vlim)(bz0))
        ax.plot_surface(X, Y, np.full_like(X, z[0]), facecolors=fc, rstride=st, cstride=st,
                        shade=False, linewidth=0, antialiased=False, zorder=0)
        p = _pil_field(bz0)
        if np.nanmin(p) < 0 < np.nanmax(p):
            ax.contour(X, Y, p, levels=[0.0], zdir="z", offset=z[0], colors=INK,
                       linewidths=1.2, zorder=1)
        for j in order:
            l, (c, w, zo, lab) = lines[j], styles[j]
            ax.plot(l.points[:, 0], l.points[:, 1], l.points[:, 2], color=c, lw=w, zorder=zo)
            if lab:
                ax.text(*l.start, lab, color=INK, fontsize=8, fontweight="bold", zorder=10)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_zlim(z[0], z[-1])
        ax.set_box_aspect((x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]))
        ax.view_init(elev=elev, azim=azim)
        ax.set_zlabel("z [Mm]")
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_facecolor(SURFACE)
            a.pane.set_edgecolor(MUTED)
            a._axinfo["grid"]["color"] = "#e6e5e1"
    else:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        if view == "top":
            _draw_bz_2d(ax, x, y, bz0, vlim)
            for j in order:
                l, (c, w, zo, lab) = lines[j], styles[j]
                ax.plot(l.points[:, 0], l.points[:, 1], color=c, lw=w, zorder=zo)
                if lab:
                    ax.annotate(lab, l.start[:2], color=INK, fontsize=8, fontweight="bold",
                                zorder=10, ha="center", va="center",
                                bbox=dict(boxstyle="circle,pad=0.15", fc=SURFACE, ec=c, lw=1.2))
            ax.set_xlim(*_edges(x))
            ax.set_ylim(*_edges(y))
            ax.set_ylabel("y [Mm]")
        elif view == "side":
            ax.axhline(z[0], color=INK2, lw=0.8)
            for j in order:
                l, (c, w, zo, lab) = lines[j], styles[j]
                ax.plot(l.points[:, 0], l.points[:, 2], color=c, lw=w, zorder=zo)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(z[0], z[-1])
            ax.set_ylabel("z [Mm]")
        else:
            raise ValueError("view debe ser '3d', 'top' o 'side'")
        ax.set_aspect("equal")
    ax.set_xlabel("x [Mm]")
    if view == "3d":
        ax.set_ylabel("y [Mm]")
    ax.set_title(field.meta.get("method", "") if title is None else title)
    if colorbar and color_by == "apex":
        plt.colorbar(ScalarMappable(norm, APEX_CMAP), ax=ax, shrink=0.6,
                     label=_t("apex"))
    return ax


@_styled
def plot_comparison(models: dict[str, VectorField3D], seeds: np.ndarray,
                    views=("3d", "top", "side"), color_by: str = "apex",
                    n_highlight: int = 6, save: str | None = None,
                    panel_width: float = 4.6, **kw):
    """Grid of models (columns) × views (rows) with the same seeds.

    The height scale and the Bz scale are common to all the panels.
    ``color_by="seed"`` numbers ``n_highlight`` seeds: the same number is the
    same seed in every model, which shows how its connectivity changes.
    """
    names = list(models)
    lines = {k: trace_field_lines(v, seeds) for k, v in models.items()}
    zmax = max(l.apex for ls in lines.values() for l in ls)
    norm = Normalize(0, max(zmax, 1e-9))
    highlight = spread_indices(seeds, n_highlight) if color_by == "seed" else None

    f0 = models[names[0]]
    xext = f0.x[-1] - f0.x[0]
    yext = f0.y[-1] - f0.y[0]
    zext = f0.z[-1] - f0.z[0]
    ratios = {"3d": 0.8, "top": yext / xext, "side": max(zext / xext, 0.25)}
    hr = [ratios[v] for v in views]
    fig = plt.figure(figsize=(panel_width * len(names) + 1.2, panel_width * sum(hr) + 0.8),
                     layout="constrained")
    gs = fig.add_gridspec(len(views), len(names), height_ratios=hr)
    axes = []
    for i, v in enumerate(views):
        for j, name in enumerate(names):
            f = models[name]
            ax = fig.add_subplot(gs[i, j], projection="3d" if v == "3d" else None)
            title = ""
            if i == 0:
                title = name
                a = f.meta.get("alpha")
                if a:
                    title += f"\nα = {a:.3g} Mm⁻¹"
            plot_field_lines(f, lines=lines[name], ax=ax, view=v, color_by=color_by,
                             norm=norm, highlight=highlight, title=title, **kw)
            if j > 0 and v != "3d":
                ax.set_ylabel("")
            axes.append(ax)

    vlim = _bz_vlim(f0.bz[:, :, 0])
    fig.colorbar(ScalarMappable(Normalize(-vlim, vlim), BZ_CMAP), ax=axes, shrink=0.5,
                 label=_t("bz0"), location="right")
    if color_by == "apex":
        fig.colorbar(ScalarMappable(norm, APEX_CMAP), ax=axes, shrink=0.5,
                     label=_t("apex"), location="right")
    handles = _bz_legend_handles()
    if color_by == "seed":
        handles.append(Line2D([], [], color=MUTED, lw=1, label=_t("rest")))
        fig.suptitle(_t("same_seed"), fontsize=10, color=INK2)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=8.5)
    if save:
        fig.savefig(save, dpi=150)
    return fig


# --------------------------------------------------------------------- α
def _model_colors(names, reference=None):
    cols, k = {}, 0
    for n in names:
        if n == reference:
            cols[n] = INK
        else:
            cols[n] = CATEGORICAL[k % len(CATEGORICAL)]
            k += 1
    return cols


@_styled
def plot_alpha_maps(models: dict[str, VectorField3D], z: float = 5.0,
                    vlim: float | None = None, save: str | None = None):
    """Maps of α = (∇×B)·B/B² at height ``z`` (Mm) for each model.

    Potential: α = 0. LFFF: constant α. NLFFF: α varies over the region, which is
    what the LFFF cannot represent. Dark grey = weak field (α not
    defined).
    """
    names = list(models)
    maps, zs = {}, {}
    for n in names:
        f = models[n]
        iz = int(np.argmin(np.abs(f.z - z)))
        maps[n] = f.alpha()[:, :, iz]
        zs[n] = f.z[iz]
    if vlim is None:
        allv = np.concatenate([np.abs(m[np.isfinite(m)]) for m in maps.values()])
        vlim = float(np.percentile(allv, 98)) if allv.size else 1.0
        vlim = vlim or 1.0
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names) + 1, 5.2),
                             layout="constrained", squeeze=False)
    for ax, n in zip(axes[0], names):
        f = models[n]
        x, y = f.x, f.y
        ax.imshow(maps[n].T, origin="lower", cmap=DIV_CMAP, vmin=-vlim, vmax=vlim,
                  extent=(*_edges(x), *_edges(y)), interpolation="nearest")
        _draw_bz_2d(ax, x, y, f.bz[:, :, 0], _bz_vlim(f.bz[:, :, 0]), image=False)
        m = maps[n][np.isfinite(maps[n])]
        sub = f"⟨α⟩ = {m.mean():.3g},  σ = {m.std():.2g} Mm⁻¹" if m.size else ""
        ax.set_title(f"{n}\n{sub}")
        ax.set_xlabel("x [Mm]")
        ax.set_aspect("equal")
    axes[0, 0].set_ylabel("y [Mm]")
    fig.colorbar(ScalarMappable(Normalize(-vlim, vlim), DIV_CMAP), ax=axes[0].tolist(),
                 shrink=0.85, label="α = (∇×B)·B / B²  [Mm⁻¹]")
    fig.suptitle(_t("alpha_sup", z=zs[names[0]]), fontsize=11)
    fig.text(0.5, 0.0, _t("alpha_foot"), ha="center", va="bottom", fontsize=8.5, color=INK2)
    if save:
        fig.savefig(save, dpi=150)
    return fig


def line_alpha_stats(field: VectorField3D, lines: list[FieldLine],
                     min_height_cells: float = 2.0):
    """Mean and standard deviation of α along each line (excludes the lowest layers).

    In a force-free field α is constant on each line: a small
    deviation from the mean indicates good quality.
    """
    a = field.alpha()
    origin = np.asarray(field.origin)
    means, stds = np.full(len(lines), np.nan), np.full(len(lines), np.nan)
    for i, l in enumerate(lines):
        idx = ((l.points - origin) / field.dx).T
        keep = idx[2] >= min_height_cells
        if keep.sum() < 3:
            continue
        v = map_coordinates(a, idx[:, keep], order=0, mode="nearest", cval=np.nan)
        v = v[np.isfinite(v)]
        if v.size >= 3:
            means[i], stds[i] = v.mean(), v.std()
    return means, stds


@_styled
def plot_alpha_histogram(models: dict[str, VectorField3D], seeds: np.ndarray,
                         reference: str | None = None, save: str | None = None):
    """Distribution of α per field line (left) and its constancy along
    the line (right), with the same seeds for every model."""
    names = list(models)
    cols = _model_colors(names, reference)
    stats = {n: line_alpha_stats(models[n], trace_field_lines(models[n], seeds)) for n in names}
    allm = np.concatenate([s[0][np.isfinite(s[0])] for s in stats.values()])
    lo = min(float(np.percentile(allm, 1)), 0.0) if allm.size else -1.0
    hi = max(float(np.percentile(allm, 99)), 0.0) if allm.size else 1.0
    pad = 0.1 * (hi - lo or 1.0)
    bins = np.linspace(lo - pad, hi + pad, 41)
    alls = np.concatenate([s[1][np.isfinite(s[1])] for s in stats.values()])
    sbins = np.linspace(0, float(np.percentile(alls, 99)) * 1.1 if alls.size else 1.0, 31)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for n in names:
        m, s = stats[n]
        ls = "--" if n == reference else "-"
        ax1.hist(m[np.isfinite(m)], bins=bins, histtype="step", lw=2, color=cols[n], ls=ls, label=n)
        ax2.hist(s[np.isfinite(s)], bins=sbins, histtype="step", lw=2, color=cols[n], ls=ls, label=n)
    ax1.axvline(0, color=MUTED, lw=0.8, zorder=0)
    ax1.set_xlabel(_t("alpha_mean"))
    ax1.set_ylabel(_t("n_lines"))
    ax1.set_title(_t("alpha_q1"))
    ax2.set_xlabel(_t("alpha_std"))
    ax2.set_title(_t("alpha_q2"))
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax1.legend(loc="upper left")
    if save:
        fig.savefig(save, dpi=150)
    return fig


# ---------------------------------------------------------------- errors
@_styled
def plot_error_maps(models: dict[str, VectorField3D], reference: str,
                    heights=(5.0, 15.0, 30.0), b_min_frac: float = 0.01,
                    save: str | None = None):
    """Angle between the model B and the reference B on horizontal slices.

    Rows: models; columns: heights (Mm). In each panel, ⟨θ⟩_B is the mean
    weighted by |B_ref| (dominated by the strong field).
    """
    ref = models[reference]
    others = [n for n in models if n != reference]
    izs = [int(np.argmin(np.abs(ref.z - h))) for h in heights]
    fig, axes = plt.subplots(len(others), len(izs),
                             figsize=(3.6 * len(izs) + 1.2, 3.3 * len(others) + 0.6),
                             layout="constrained", squeeze=False)
    bmax = ref.babs.max()
    for i, n in enumerate(others):
        f = models[n]
        for j, iz in enumerate(izs):
            B = np.stack([ref.bx[:, :, iz], ref.by[:, :, iz], ref.bz[:, :, iz]])
            b = np.stack([f.bx[:, :, iz], f.by[:, :, iz], f.bz[:, :, iz]])
            nB, nb = np.linalg.norm(B, axis=0), np.linalg.norm(b, axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                ang = np.degrees(np.arccos(np.clip((B * b).sum(0) / (nB * nb), -1, 1)))
            ang[nB < b_min_frac * bmax] = np.nan
            ax = axes[i, j]
            ax.imshow(ang.T, origin="lower", cmap=SEQ_CMAP, vmin=0, vmax=90,
                      extent=(*_edges(ref.x), *_edges(ref.y)), interpolation="nearest")
            _draw_bz_2d(ax, ref.x, ref.y, ref.bz[:, :, 0], _bz_vlim(ref.bz[:, :, 0]),
                        image=False, contours=False)
            ok = np.isfinite(ang)
            wmean = (ang[ok] * nB[ok]).sum() / nB[ok].sum() if ok.any() else np.nan
            ax.text(0.03, 0.97, f"⟨θ⟩_B = {wmean:.1f}°", transform=ax.transAxes, va="top",
                    fontsize=8.5, color=INK, bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=2))
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(f"z = {ref.z[iz]:.1f} Mm")
            if j == 0:
                ax.set_ylabel(f"{n}\ny [Mm]")
            if i == len(others) - 1:
                ax.set_xlabel("x [Mm]")
    fig.colorbar(ScalarMappable(Normalize(0, 90), SEQ_CMAP), ax=axes.ravel().tolist(),
                 shrink=0.8, label=_t("angle_cb", ref=reference))
    fig.suptitle(_t("err_sup"),
                 fontsize=10, color=INK2)
    if save:
        fig.savefig(save, dpi=150)
    return fig
