"""Generation of all the figures of the report in one language."""

from __future__ import annotations

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .. import plotting as P

NAMES = {
    "es": {"exact": "Exacto", "potential": "Potencial", "lfff": "LFFF", "nlfff": "NLFFF"},
    "en": {"exact": "Exact", "potential": "Potential", "lfff": "LFFF", "nlfff": "NLFFF"},
}


def _L(lang, es, en):
    return es if lang == "es" else en


@P._styled
def fig_preprocess(mag, pp, lang, path):
    db = np.sqrt((pp.bx - mag.bx) ** 2 + (pp.by - mag.by) ** 2 + (pp.bz - mag.bz) ** 2)
    vl = P._bz_vlim(mag.bz)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    ext = (*P._edges(mag.x), *P._edges(mag.y))
    for ax, data, title in ((axes[0], mag.bz, _L(lang, "Bz original", "Original Bz")),
                            (axes[1], pp.bz, _L(lang, "Bz preprocesado", "Preprocessed Bz"))):
        ax.imshow(data.T, origin="lower", cmap=P.BZ_CMAP, vmin=-vl, vmax=vl, extent=ext,
                  interpolation="bilinear")
        ax.set_title(title)
    fig.colorbar(ScalarMappable(Normalize(-vl, vl), P.BZ_CMAP), ax=axes[:2].tolist(),
                 shrink=0.85, label="Bz [G]")
    im = axes[2].imshow(db.T, origin="lower", cmap=P.SEQ_CMAP, extent=ext, interpolation="nearest")
    P._draw_bz_2d(axes[2], mag.x, mag.y, mag.bz, vl, image=False, contours=False)
    axes[2].set_title(_L(lang, "|ΔB| introducido por el preprocesado",
                         "|ΔB| introduced by preprocessing"))
    fig.colorbar(im, ax=axes[2], shrink=0.85, label="|ΔB| [G]")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x [Mm]")
    axes[0].set_ylabel("y [Mm]")
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_alpha_curve(alpha, lang, path):
    grid, costs = alpha["curve"]
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    ax.plot(grid, costs, color=P.CATEGORICAL[0], lw=2)
    ax.axvline(alpha["fit"], color=P.INK, lw=1.2, ls="-")
    right = alpha["fit"] > 0
    ax.annotate(_L(lang, "α ajustado", "best-fit α") + f" = {alpha['fit']:.3g}",
                (alpha["fit"], costs.max()), xytext=(-5 if right else 5, -12),
                textcoords="offset points", ha="right" if right else "left",
                fontsize=8.5, color=P.INK)
    if alpha.get("jz") is not None and abs(alpha["jz"]) < 1.5 * alpha["max"]:
        ax.axvline(alpha["jz"], color=P.INK2, lw=1, ls="--")
        ax.annotate(f"α_Jz = {alpha['jz']:.3g}", (alpha["jz"], costs.min()),
                    xytext=(5, 8), textcoords="offset points", fontsize=8.5, color=P.INK2)
    for s in (-1, 1):
        ax.axvline(s * alpha["max"], color=P.MUTED, lw=1, ls=":")
    ax.text(alpha["max"], costs.max(), " α_max", fontsize=8, color=P.INK2, va="top")
    ax.set_xlabel("α [Mm⁻¹]")
    ax.set_ylabel(_L(lang, "error relativo en Bt", "relative Bt misfit"))
    ax.set_title(_L(lang, "Ajuste de α: diferencia entre el Bt observado y el del LFFF",
                    "α fit: difference between observed Bt and LFFF Bt"))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_convergence(history, lang, path):
    it, L, mu = (np.array(v) for v in zip(*history))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    a1.semilogy(it, L, color=P.CATEGORICAL[0], lw=2)
    a1.set_xlabel(_L(lang, "iteración", "iteration"))
    a1.set_ylabel("L")
    a1.set_title(_L(lang, "Funcional L (fuerza + divergencia)", "Functional L (force + divergence)"))
    a2.semilogy(it, mu, color=P.CATEGORICAL[0], lw=2)
    a2.set_xlabel(_L(lang, "iteración", "iteration"))
    a2.set_ylabel("μ")
    a2.set_title(_L(lang, "Paso de iteración μ (adaptativo)", "Iteration step μ (adaptive)"))
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    fig.savefig(path, dpi=200)
    plt.close(fig)



def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)


def _map_panels(n):
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n + 1.2, 4.4), layout="constrained", squeeze=False)
    return fig, axes[0]


@P._styled
def fig_twist(study, A, lang, path):
    names = NAMES[lang]
    maps = A["_maps"]["twist"]
    keys = [k for k in ("exact", "lfff", "nlfff") if k in maps]
    allv = np.concatenate([np.abs(maps[k]["tw"][np.isfinite(maps[k]["tw"])]) for k in keys])
    vlim = max(1.5, float(np.percentile(allv, 99))) if allv.size else 1.5
    fig, axes = _map_panels(len(keys))
    bz = study.mag.bz
    for ax, k in zip(axes, keys):
        m = maps[k]
        ax.imshow(m["tw"].T, origin="lower", cmap=P.DIV_CMAP, vmin=-vlim, vmax=vlim,
                  extent=(*P._edges(m["x"]), *P._edges(m["y"])), interpolation="nearest")
        X, Y = np.meshgrid(m["x"], m["y"], indexing="ij")
        tw = np.nan_to_num(m["tw"])
        if np.abs(tw).max() >= 1:
            ax.contour(X, Y, np.abs(tw), levels=[1.0], colors=P.INK, linewidths=1.0, linestyles="dotted")
        P._draw_bz_2d(ax, study.mag.x, study.mag.y, bz, P._bz_vlim(bz), image=False, contours=False)
        mx = A["physics"]["twist"][k]["max_abs"]
        ax.set_title(f"{names[k]}\n" + _L(lang, "máx |Tw| (cerradas)", "max |Tw| (closed)")
                     + (f" = {mx:.2f}" if mx is not None else ""))
        ax.set_aspect("equal")
        ax.set_xlabel("x [Mm]")
    axes[0].set_ylabel("y [Mm]")
    fig.colorbar(ScalarMappable(Normalize(-vlim, vlim), P.DIV_CMAP), ax=axes.tolist(), shrink=0.85,
                 label=_L(lang, "Tw de la línea que sale de cada punto [vueltas]",
                          "Tw of the field line rooted at each point [turns]"))
    fig.suptitle(_L(lang, "Número de torsión Tw (punteado: |Tw| = 1; línea negra: PIL)",
                    "Twist number Tw (dotted: |Tw| = 1; black line: PIL)"), fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_decay(study, A, lang, path):
    pr = A["_maps"]["decay"]
    zrel = A["physics"]["decay"]["z_reliable"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    z, n = pr["z"], pr["n"]
    a1.axhspan(1.1, 1.75, color="#efeee9", zorder=0)
    a1.axhline(1.5, color=P.INK2, lw=1, ls="--")
    a1.text(z[-1], 1.52, _L(lang, " n = 1.5 (umbral teórico)", " n = 1.5 (theoretical threshold)"),
            ha="right", va="bottom", fontsize=8, color=P.INK2)
    a1.text(z[-1], 1.12, _L(lang, "rango crítico citado 1.1–1.75 ", "quoted critical range 1.1–1.75 "),
            ha="right", va="bottom", fontsize=8, color=P.INK2)
    ok = z <= zrel
    a1.plot(z[ok], n[ok], color=P.CATEGORICAL[0], lw=2, label=_L(lang, "fiable", "reliable"))
    if (~ok).any():
        a1.plot(z[~ok], n[~ok], color=P.CATEGORICAL[0], lw=1.2, ls=":",
                label=_L(lang, "afectado por los bordes", "affected by boundaries"))
    hc = pr["h_crit"]
    if np.isfinite(hc):
        a1.axvline(hc, color=P.INK, lw=1)
        a1.annotate(f"h_crit = {hc:.1f} Mm", (hc, 0.2), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=P.INK)
    a1.set_xlabel("z [Mm]")
    a1.set_ylabel(_L(lang, "índice de decaimiento n", "decay index n"))
    a1.set_title(_L(lang, "n(z) del campo potencial sobre la PIL", "n(z) of the potential field above the PIL"))
    top = np.nanmax(n[ok]) * 1.05 if ok.any() and np.isfinite(n[ok]).any() else 3.0
    a1.set_ylim(0, max(3.0, top))
    a1.legend(loc="upper left", fontsize=8.5)
    _clean(a1)
    bz = study.mag.bz
    P._draw_bz_2d(a2, study.mag.x, study.mag.y, bz, P._bz_vlim(bz), pil=False)
    ix, iy = np.nonzero(pr["mask"])
    a2.scatter(study.mag.x[ix], study.mag.y[iy], s=6, color=P.CATEGORICAL[1], zorder=5,
               label=_L(lang, "columnas usadas (PIL)", "columns used (PIL)"))
    a2.set_aspect("equal")
    a2.set_xlabel("x [Mm]")
    a2.set_ylabel("y [Mm]")
    a2.set_title(_L(lang, "Dónde se mide n", "Where n is measured"))
    a2.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, -0.12))
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_q(study, A, lang, path):
    names = NAMES[lang]
    maps = A["_maps"]["q"]
    keys = [k for k in ("potential", "exact", "lfff", "nlfff") if k in maps]
    allv = np.concatenate([maps[k]["logq"][np.isfinite(maps[k]["logq"])] for k in keys])
    vmax = max(3.0, float(np.percentile(allv, 99))) if allv.size else 3.0
    fig, axes = _map_panels(len(keys))
    bz = study.mag.bz
    for ax, k in zip(axes, keys):
        m = maps[k]
        ax.imshow(np.clip(m["logq"], np.log10(2), None).T, origin="lower", cmap=P.SEQ_CMAP,
                  vmin=np.log10(2), vmax=vmax, extent=(*P._edges(m["x"]), *P._edges(m["y"])),
                  interpolation="nearest")
        P._draw_bz_2d(ax, study.mag.x, study.mag.y, bz, P._bz_vlim(bz), image=False, contours=False)
        fq = A["physics"]["q"][k]["frac_logq_gt2"]
        ax.set_title(f"{names[k]}\n" + _L(lang, "área con Q > 100", "area with Q > 100")
                     + (f": {100 * fq:.0f} %" if fq is not None else ""))
        ax.set_aspect("equal")
        ax.set_xlabel("x [Mm]")
    axes[0].set_ylabel("y [Mm]")
    fig.colorbar(ScalarMappable(Normalize(np.log10(2), vmax), P.SEQ_CMAP), ax=axes.tolist(),
                 shrink=0.85, label="log₁₀ Q")
    fig.suptitle(_L(lang, "Factor de squashing Q en la base (QSL = bandas oscuras; gris: no definido)",
                    "Squashing factor Q at the base (QSLs = dark bands; grey: undefined)"), fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_validation(val, lang, path):
    V = {t["key"]: t for t in val}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    ex_lab = _L(lang, "exacto", "exact")
    if "twist" in V:
        c = V["twist"]["curve"]
        ax = axes[0]
        ax.scatter(c["r"], c["tw"], s=6, color=P.CATEGORICAL[0], alpha=0.5, label="solarff")
        ax.plot(c["r_exact"], c["tw_exact"], color=P.INK, lw=1.5, label=ex_lab)
        ax.set_xlabel(_L(lang, "radio r [Mm]", "radius r [Mm]"))
        ax.set_ylabel("Tw")
        ax.set_title(_L(lang, "Tw: tubo de Gold–Hoyle", "Tw: Gold–Hoyle tube"))
        ax.legend(fontsize=8.5)
        _clean(ax)
    if "decay" in V:
        c = V["decay"]["curve"]
        ax = axes[1]
        ax.plot(c["z"], c["n_exact"], color=P.INK, lw=1.5, label=ex_lab + " 3z/(z+d)")
        ax.plot(c["z"], c["n_fft"], color=P.CATEGORICAL[0], lw=2, ls="--",
                label=_L(lang, "potencial FFT", "FFT potential"))
        ax.axhline(1.5, color=P.MUTED, lw=1, ls=":")
        ax.set_xlabel("z [Mm]")
        ax.set_ylabel("n")
        ax.set_title(_L(lang, "Índice de decaimiento: dipolo", "Decay index: dipole"))
        ax.legend(fontsize=8.5)
        _clean(ax)
    if "q" in V:
        ax = axes[2]
        q = V["q"]["q_values"]
        ax.hist(q, bins=25, color=P.CATEGORICAL[0], alpha=0.85)
        ax.axvline(V["q"]["expected"], color=P.INK, lw=1.5, label=ex_lab + f" = {V['q']['expected']:.3f}")
        ax.set_xlabel("Q")
        ax.set_ylabel(_L(lang, "número de puntos", "number of points"))
        ax.set_title(_L(lang, "Q: campo hiperbólico", "Q: hyperbolic field"))
        ax.legend(fontsize=8.5)
        _clean(ax)
    fig.savefig(path, dpi=200)
    plt.close(fig)

@P._styled
def fig_convergence_lbfgs(terms, lang, path):
    it, L, lf, ld = (np.array(v) for v in zip(*terms))
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ax.semilogy(it, L, color=P.INK, lw=2, label=_L(lang, "total", "total"))
    ax.semilogy(it, lf, color=P.CATEGORICAL[0], lw=1.6, ls="--",
                label=_L(lang, "fuerza + divergencia (+ suavidad)", "force + divergence (+ smoothness)"))
    if np.any(ld > 0):
        ax.semilogy(it, np.where(ld > 0, ld, np.nan), color=P.CATEGORICAL[1], lw=1.6, ls=":",
                    label=_L(lang, "datos (base relajada)", "data (relaxed base)"))
    ax.set_xlabel(_L(lang, "iteración L-BFGS", "L-BFGS iteration"))
    ax.set_ylabel("L")
    ax.set_title(_L(lang, "Convergencia del NLFFF (L-BFGS)", "NLFFF convergence (L-BFGS)"))
    ax.legend(fontsize=8.5)
    _clean(ax)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_bottom(study, lang, path):
    src = study.mag_pp if study.mag_pp is not None else study.mag
    f = study.nlfff
    bt_obs = np.hypot(src.bx, src.by)
    bt_mod = np.hypot(f.bx[:, :, 0], f.by[:, :, 0])
    dbt = np.hypot(f.bx[:, :, 0] - src.bx, f.by[:, :, 0] - src.by)
    vmax = float(np.percentile(bt_obs, 99.5))
    ext = (*P._edges(src.x), *P._edges(src.y))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    for ax, data, title in ((axes[0], bt_obs, _L(lang, "|B_t| observado", "observed |B_t|")),
                            (axes[1], bt_mod, _L(lang, "|B_t| del NLFFF en z = 0", "NLFFF |B_t| at z = 0")),
                            (axes[2], dbt, _L(lang, "|ΔB_t| (corrección de la base)", "|ΔB_t| (base correction)"))):
        ax.imshow(data.T, origin="lower", cmap=P.SEQ_CMAP, vmin=0, vmax=vmax, extent=ext,
                  interpolation="nearest")
        P._draw_bz_2d(ax, src.x, src.y, src.bz, P._bz_vlim(src.bz), image=False)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x [Mm]")
    axes[0].set_ylabel("y [Mm]")
    fig.colorbar(ScalarMappable(Normalize(0, vmax), P.SEQ_CMAP), ax=axes.tolist(), shrink=0.85, label="G")
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_confidence(study, A, lang, path):
    cm = A["_maps"]["confidence"]
    thr = A["reliability"]["threshold"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    a1.imshow(cm["conf"].T, origin="lower", cmap=P.SEQ_CMAP, vmin=0, vmax=1,
              extent=(*P._edges(cm["x"]), *P._edges(cm["y"])), interpolation="nearest")
    X, Y = np.meshgrid(cm["x"], cm["y"], indexing="ij")
    c = cm["conf"]  # NaN outside the strong field: the contour does not cross those areas
    fin = np.isfinite(c)
    if (c[fin] >= thr).any() and (c[fin] < thr).any():
        a1.contour(X, Y, c, levels=[thr], colors=P.CATEGORICAL[1], linewidths=1.2)
    bz = study.mag.bz
    P._draw_bz_2d(a1, study.mag.x, study.mag.y, bz, P._bz_vlim(bz), image=False, contours=False)
    a1.set_aspect("equal")
    a1.set_xlabel("x [Mm]")
    a1.set_ylabel("y [Mm]")
    a1.set_title(_L(lang, "Confianza de la línea que sale de cada punto (naranja: umbral)",
                    "Confidence of the field line rooted at each point (orange: threshold)"))
    fig.colorbar(ScalarMappable(Normalize(0, 1), P.SEQ_CMAP), ax=a1, shrink=0.85,
                 label=_L(lang, "confianza", "confidence"))
    v = A["_maps"]["confidence"]["assess"]["confidence"]
    v = v[np.isfinite(v)]
    a2.hist(v, bins=25, range=(0, 1), color=P.CATEGORICAL[0], alpha=0.85)
    a2.axvline(thr, color=P.CATEGORICAL[1], lw=1.5)
    a2.text(thr, a2.get_ylim()[1] * 0.95, _L(lang, " umbral", " threshold"), color=P.INK2, fontsize=8.5, va="top")
    a2.set_xlabel(_L(lang, "confianza", "confidence"))
    a2.set_ylabel(_L(lang, "número de líneas", "number of lines"))
    a2.set_title(_L(lang, f"{100 * (v >= thr).mean():.0f} % de líneas fiables",
                    f"{100 * (v >= thr).mean():.0f} % reliable lines"))
    _clean(a2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_reliable_lines(study, A, lang, path, max_lines=160):
    cm = A["_maps"]["confidence"]
    lines = cm["lines"]
    rel = cm["assess"]["reliable"]
    model = study.nlfff if A["reliability"]["model"] == "nlfff" else study.lfff
    rng = np.random.default_rng(0)
    idx = np.arange(len(lines))
    if len(idx) > max_lines:
        idx = np.sort(rng.choice(idx, max_lines, replace=False))
    sub = [lines[i] for i in idx]
    hl = np.flatnonzero(rel[idx])
    norm = Normalize(0, max(max(l.apex for l in sub), 1e-9))
    fig = plt.figure(figsize=(13, 5.4), layout="constrained")
    for k, view in enumerate(("3d", "top")):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d" if view == "3d" else None)
        P.plot_field_lines(model, lines=sub, ax=ax, view=view, color_by="seed", highlight=[], norm=norm,
                           title="")
        # highlight reliable lines coloured by height
        for j in hl:
            l = sub[j]
            c = P.APEX_CMAP(norm(l.apex))
            if view == "3d":
                ax.plot(l.points[:, 0], l.points[:, 1], l.points[:, 2], color=c, lw=1.1, zorder=6)
            else:
                ax.plot(l.points[:, 0], l.points[:, 1], color=c, lw=1.1, zorder=6)
    fig.colorbar(ScalarMappable(norm, P.APEX_CMAP), ax=fig.axes, shrink=0.6,
                 label=_L(lang, "altura máxima de las líneas fiables [Mm]", "apex height of reliable lines [Mm]"))
    fig.suptitle(_L(lang, "Líneas fiables (color) y no fiables (gris)", "Reliable (colour) and unreliable (grey) field lines"),
                 fontsize=11)
    fig.get_layout_engine().set(wspace=0.12)  # room for the z label of the 3D view, which constrained layout does not see
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_calibration(val, lang, path):
    from ..confidence import CALIBRATION_METRICS
    V = {t["key"]: t for t in val}
    t = V.get("confidence")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    feats = CALIBRATION_METRICS["auc_features"]
    from ..confidence import DEFAULT_CALIBRATION
    used = set(DEFAULT_CALIBRATION["features"])
    names = list(feats)
    vals = [feats[k] for k in names]
    cols = [P.CATEGORICAL[0] if k in used else P.MUTED for k in names]
    a1.barh(range(len(names)), vals, color=cols)
    a1.axvline(0.5, color=P.INK2, lw=1, ls="--")
    a1.axvline(CALIBRATION_METRICS["auc_combined"], color=P.INK, lw=1.2)
    a1.text(CALIBRATION_METRICS["auc_combined"], len(names) - 0.4,
            _L(lang, " combinado", " combined"), fontsize=8.5, color=P.INK)
    a1.set_yticks(range(len(names)), names)
    a1.set_xlim(0.4, 1.0)
    a1.set_xlabel("AUC")
    a1.set_title(_L(lang, "Poder predictivo de cada indicador (azul: usados)",
                    "Predictive power of each indicator (blue: used)"))
    _clean(a1)
    if t is not None:
        c, e, bad = t["curve"]["conf"], t["curve"]["err"], t["curve"]["bad"]
        a2.scatter(c[~bad], e[~bad], s=10, color=P.CATEGORICAL[0], alpha=0.6,
                   label=_L(lang, "buenas", "good"))
        a2.scatter(c[bad], e[bad], s=10, color=P.CATEGORICAL[1], alpha=0.6,
                   label=_L(lang, "malas", "bad"))
        a2.axvline(t["threshold"], color=P.INK2, lw=1, ls="--")
        a2.axhline(0.1, color=P.MUTED, lw=1, ls=":")
        a2.set_yscale("log")
        a2.set_xlabel(_L(lang, "confianza", "confidence"))
        a2.set_ylabel(_L(lang, "error real de la línea (relativo)", "true line error (relative)"))
        a2.set_title(_L(lang, f"Realización nueva: AUC = {1 - t['error']:.2f}",
                        f"New realisation: AUC = {1 - t['error']:.2f}"))
        a2.legend(fontsize=8.5)
        _clean(a2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_energy_helicity(A, lang, path):
    names = NAMES[lang]
    H = A["helicity"]
    keys = list(H)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    x = np.arange(len(keys))
    Ep = np.array([H[k]["energy"]["E_p_s"] for k in keys])
    Ej = np.array([H[k]["energy"]["E_J_s"] for k in keys])
    Ed = np.array([H[k]["energy"]["E_div"] for k in keys])
    a1.bar(x, Ep, color=P.MUTED, label=_L(lang, "potencial (solenoidal)", "potential (solenoidal)"))
    a1.bar(x, Ej, bottom=Ep, color=P.CATEGORICAL[0], label=_L(lang, "con corrientes (solenoidal) = libre", "current-carrying (solenoidal) = free"))
    a1.bar(x, Ed, bottom=Ep + Ej, color=P.CATEGORICAL[1], label=_L(lang, "debida a la divergencia", "due to divergence"))
    for i, k in enumerate(keys):
        a1.text(i, Ep[i] + Ej[i] + Ed[i], f"{100 * H[k]['energy']['E_div_frac']:.1f} %", ha="center",
                va="bottom", fontsize=8.5, color=P.INK2)
    a1.set_xticks(x, [names[k] for k in keys])
    a1.set_ylabel("E [erg]")
    a1.set_title(_L(lang, "Descomposición de la energía (cifra: E_div/E)", "Energy decomposition (label: E_div/E)"))
    a1.legend(fontsize=8, loc="lower right")
    _clean(a1)
    w = 0.38
    hj = np.array([H[k]["helicity"]["H_J"] for k in keys])
    hpj = np.array([H[k]["helicity"]["H_PJ"] for k in keys])
    a2.bar(x - w / 2, hj, w, color=P.CATEGORICAL[0], label="H_J")
    a2.bar(x + w / 2, hpj, w, color=P.MUTED, label="H_PJ")
    for i, k in enumerate(keys):
        r = H[k]["helicity"]["ratio_HJ_HV"]
        top = max(hj[i], hpj[i], 0)
        a2.text(i, top, f"|H_J|/|H_V| = {r:.2f}", ha="center", va="bottom", fontsize=8.5, color=P.INK2)
    a2.axhline(0, color=P.INK2, lw=0.8)
    a2.set_xticks(x, [names[k] for k in keys])
    a2.set_ylabel("H [Mx²]")
    a2.set_title(_L(lang, "Helicidad relativa: H_V = H_J + H_PJ", "Relative helicity: H_V = H_J + H_PJ"))
    a2.legend(fontsize=8.5)
    _clean(a2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_montecarlo(A, lang, path):
    mc = A["_mc"]
    s = mc["summary"]
    panels = [("E_free", _L(lang, "E libre (pot. FFT) [erg]", "free E (FFT pot.) [erg]")),
              ("E_free_vol", _L(lang, "E libre (pot. volumen) [erg]", "free E (volume pot.) [erg]")),
              ("H_V", "H_V [Mx²]"), ("HJ_HV", "|H_J|/|H_V|"),
              ("h_crit", "h_crit [Mm]"), ("tw_p99", _L(lang, "percentil 99 de |Tw|", "99th pct of |Tw|"))]
    panels = [p for p in panels if p[0] in s]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), layout="constrained")
    for ax, (k, lab) in zip(axes.ravel(), panels):
        v = mc["values"][k]
        v = v[np.isfinite(v)]
        ax.hist(v, bins=min(12, max(4, len(v) // 2)), color=P.CATEGORICAL[0], alpha=0.85)
        ax.axvspan(s[k]["p16"], s[k]["p84"], color="#efeee9", zorder=0)
        if s[k]["nominal"] is not None:
            ax.axvline(s[k]["nominal"], color=P.INK, lw=1.5, label=_L(lang, "nominal", "nominal"))
        ax.set_xlabel(lab)
        _clean(ax)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8.5)
    axes.ravel()[0].set_ylabel(_L(lang, "realizaciones", "realisations"))
    fig.suptitle(_L(lang, f"Monte Carlo ({mc['config']['n']} realizaciones): banda gris = intervalo 16–84 %",
                    f"Monte Carlo ({mc['config']['n']} realisations): grey band = 16–84 % interval"), fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_stability(study, A, lang, path):
    mc = A["_mc"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    seeds = mc["seeds"]
    sc = ax.scatter(seeds[:, 0], seeds[:, 1], c=mc["stability"], cmap=P.SEQ_CMAP, vmin=0, vmax=1, s=14,
                    marker="s", linewidths=0)
    bz = study.mag.bz
    P._draw_bz_2d(ax, study.mag.x, study.mag.y, bz, P._bz_vlim(bz), image=False, contours=True)
    ax.set_aspect("equal")
    ax.set_xlabel("x [Mm]")
    ax.set_ylabel("y [Mm]")
    ax.set_title(_L(lang, "Estabilidad de la línea que sale de cada punto frente al ruido de medida",
                    "Stability of the field line rooted at each point against measurement noise"))
    fig.colorbar(sc, ax=ax, shrink=0.85, label=_L(lang, "fracción de realizaciones en que coincide",
                                                     "fraction of realisations in which it matches"))
    fig.savefig(path, dpi=200)
    plt.close(fig)


GR_NAMES = {"es": {"nlfff": "NLFFF (optimización)", "gr_p": "Grad–Rubin P", "gr_n": "Grad–Rubin N"},
            "en": {"nlfff": "NLFFF (optimisation)", "gr_p": "Grad–Rubin P", "gr_n": "Grad–Rubin N"}}


@P._styled
def fig_methods_lines(study, A, lang, path):
    gr = study.grad_rubin
    models = {GR_NAMES[lang]["nlfff"]: study.nlfff}
    for p in ("P", "N"):
        if p in gr:
            models[GR_NAMES[lang][f"gr_{p.lower()}"]] = gr[p]
    f = P.plot_comparison(models, study.seeds, views=("3d", "top"), color_by="seed", save=path)
    plt.close(f)


@P._styled
def fig_methods_bars(A, lang, path):
    M = A["methods"]["models"]
    keys = list(M)
    names = [GR_NAMES[lang][k] for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax, (q, lab) in zip(axes, (("E_free", _L(lang, "E libre (pot. FFT) [erg]", "free E (FFT pot.) [erg]")),
                                   ("E_free_vol", _L(lang, "E libre (pot. volumen) [erg]", "free E (volume pot.) [erg]")),
                                   ("H_V", "H_V [Mx²]"))):
        vals = [M[k][q] for k in keys]
        ax.bar(range(len(keys)), vals, color=[P.CATEGORICAL[0], P.CATEGORICAL[1], P.CATEGORICAL[2]][:len(keys)])
        ax.set_xticks(range(len(keys)), names, fontsize=8.5)
        ax.set_title(lab)
        _clean(ax)
    fig.suptitle(_L(lang, "Mismos datos, distintos métodos", "Same data, different methods"), fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_figures(study, A, lang: str, out_dir: str, validation=None) -> dict:
    """Generate the figures in ``out_dir`` and return {key: relative path}."""
    os.makedirs(out_dir, exist_ok=True)
    P.set_language(lang)
    names = NAMES[lang]
    models = study.models(names)
    ref_key = A["reference"]
    ref_name = names[ref_key] if ref_key else None
    figs = {}

    def path(k):
        p = os.path.join(out_dir, f"{k}.png")
        figs[k] = p
        return p

    try:
        ax = P.plot_magnetogram(study.mag, title=_L(lang, "Magnetograma de entrada", "Input magnetogram"))
        ax.figure.savefig(path("magnetogram"), dpi=200)
        plt.close(ax.figure)
        if study.mag_pp is not None:
            fig_preprocess(study.mag, study.mag_pp, lang, path("preprocess"))
        if study.alpha.get("curve") is not None:
            fig_alpha_curve(study.alpha, lang, path("alpha_fit"))
        if getattr(study, "nlfff_terms", None):
            fig_convergence_lbfgs(study.nlfff_terms, lang, path("convergence"))
            if study.nlfff.meta.get("relax_bottom"):
                fig_bottom(study, lang, path("bottom"))
        elif study.nlfff_history:
            fig_convergence(study.nlfff_history, lang, path("convergence"))
        f = P.plot_comparison(models, study.seeds, save=path("lines"))
        plt.close(f)
        f = P.plot_comparison(models, study.seeds, color_by="seed", views=("3d", "top"),
                              save=path("connectivity"))
        plt.close(f)
        f = P.plot_alpha_maps(models, z=A["heights"][0] if A["heights"] else 3 * study.mag.dx,
                              save=path("alpha_maps"))
        plt.close(f)
        f = P.plot_alpha_histogram(models, study.seeds, reference=ref_name, save=path("alpha_hist"))
        plt.close(f)
        if ref_name:
            f = P.plot_error_maps(models, ref_name, heights=A["heights"], save=path("errors"))
            plt.close(f)
        if A["_maps"].get("twist"):
            fig_twist(study, A, lang, path("twist"))
        if A["_maps"].get("decay") is not None:
            fig_decay(study, A, lang, path("decay"))
        if A["_maps"].get("q"):
            fig_q(study, A, lang, path("qmap"))
        if A.get("helicity"):
            fig_energy_helicity(A, lang, path("energy_helicity"))
        if A.get("methods"):
            fig_methods_lines(study, A, lang, path("methods_lines"))
            fig_methods_bars(A, lang, path("methods_bars"))
        if A.get("_mc") is not None:
            fig_montecarlo(A, lang, path("montecarlo"))
            fig_stability(study, A, lang, path("stability"))
        if A["_maps"].get("confidence") is not None:
            fig_confidence(study, A, lang, path("confidence"))
            fig_reliable_lines(study, A, lang, path("reliable_lines"))
            from ..calibration import load_results
            S = load_results()
            if S:
                fig_calibration_cv(S, lang, path("calibration_cv"))
        if A.get("_aia"):
            fig_aia_steps(A["aia"], A["_aia"], lang, path("aia_steps"))
            fig_aia(A["aia"], A["_aia"], lang, path("aia"))
        if validation:
            fig_validation(validation, lang, path("validation"))
            if any(t["key"] == "confidence" for t in validation):
                fig_calibration(validation, lang, path("calibration"))
    finally:
        P.set_language("es")
    return figs


@P._styled
def fig_calibration_cv(S, lang, path):
    """Calibration with diverse data: reliability diagram, AUC per left-out field and rate of
    wrong lines versus resolution."""
    from ..calibration import CLASSIFIER_NAMES
    cv = S["cv"]["field"]["models"]
    shown = [m for m in dict.fromkeys(["logistic", S.get("chosen", "logistic")]) if m in cv]
    nm = lambda m: CLASSIFIER_NAMES[m][0 if lang == "es" else 1]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    # (a) reliability diagram
    a1.plot([0, 1], [0, 1], color=P.MUTED, lw=1, ls="--")
    for i, m in enumerate(shown):
        r = cv[m]["reliability"]
        a1.plot(r["p"], r["freq"], "-o", color=P.CATEGORICAL[i], lw=2, ms=5,
                label=f"{nm(m)} (ECE {cv[m]['ece']:.2f})")
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.set_aspect("equal")
    a1.set_xlabel(_L(lang, "probabilidad predicha de línea mala", "predicted probability of a bad line"))
    a1.set_ylabel(_L(lang, "fracción observada de líneas malas", "observed fraction of bad lines"))
    a1.set_title(_L(lang, "(a) Fiabilidad (campo nunca visto)", "(a) Reliability (unseen field)"))
    a1.legend(fontsize=8, loc="upper left")
    _clean(a1)
    # (b) AUC per left-out field
    fields = list(cv[shown[0]]["per_group"])
    w = 0.8 / len(shown)
    for i, m in enumerate(shown):
        v = [cv[m]["per_group"][f]["auc"] for f in fields]
        a2.bar(np.arange(len(fields)) + (i - (len(shown) - 1) / 2) * w, v, width=w * 0.92,
               color=P.CATEGORICAL[i], label=nm(m))
    a2.axhline(0.5, color=P.INK2, lw=1, ls="--")
    a2.set_xticks(range(len(fields)), fields, rotation=20)
    a2.set_ylim(0.4, 1.1)
    a2.set_yticks(np.arange(0.4, 1.01, 0.1))
    a2.set_ylabel("AUC")
    a2.set_title(_L(lang, "(b) AUC dejando fuera cada campo", "(b) AUC leaving out each field"))
    a2.legend(fontsize=8, loc="upper center", ncol=len(shown), frameon=False)
    _clean(a2)
    # (c) rate of wrong lines versus resolution
    br = S["bad_rate_field_size"]
    for i, f in enumerate(br):
        ns = [int(k) for k in br[f]]
        a3.plot(ns, [br[f][str(n)] for n in ns], "-o", color=P.CATEGORICAL[i % len(P.CATEGORICAL)],
                lw=2, ms=5, label=f)
    a3.set_xticks(S["sizes"])
    a3.set_xlabel(_L(lang, "puntos por lado de la malla", "grid points per side"))
    a3.set_ylabel(_L(lang, "fracción de líneas malas", "fraction of bad lines"))
    a3.set_ylim(bottom=0)
    a3.set_title(_L(lang, "(c) Líneas malas frente a resolución", "(c) Bad lines versus resolution"))
    a3.legend(fontsize=8)
    _clean(a3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@P._styled
def fig_aia_steps(S, plot, lang, path):
    """The processing of the AIA comparison, step by step, on the 171 Å image."""
    from ..aia import ridge_orientation
    W = plot["wavelengths"]
    w0 = 171 if 171 in W else sorted(W)[0]
    d = W[w0]
    rel = np.asarray(plot["reliable"], bool)
    img = np.asarray(d["cutout"], float)
    sigma = float(S.get("sigma_px", 2.0))
    pct = float(S.get("ridge_pct", 90))
    th, st = ridge_orientation(img, sigma=sigma)
    thr = np.percentile(st, pct)
    ang = np.asarray(d["angles"], float)
    logimg = np.log10(np.clip(img, 1, None))
    # Example line: reliable, with an angle close to the median of the reliable ones and fairly long
    ok = np.isfinite(ang) & rel
    if not ok.any():
        ok = np.isfinite(ang)
    med = np.nanmedian(ang[ok])
    lens = np.array([np.hypot(*np.diff(p, axis=0).T).sum() if len(p) > 1 else 0 for p in d["pix"]])
    score = np.where(ok, np.abs(ang - med) - 0.02 * np.minimum(lens, 300), np.inf)
    k = int(np.argmin(score))
    P_ex = d["pix"][k]
    # points of the example line (as in line_misalignment)
    seg = np.hypot(*np.diff(P_ex, axis=0).T)
    sc = np.concatenate([[0], np.cumsum(seg)])
    si = np.arange(0, sc[-1], 2.0)
    x, y = np.interp(si, sc, P_ex[:, 0]), np.interp(si, sc, P_ex[:, 1])
    tx, ty = np.gradient(x), np.gradient(y)
    la = np.mod(np.arctan2(ty, tx), np.pi)
    ix = np.clip(np.rint(x).astype(int), 0, img.shape[1] - 1)
    iy = np.clip(np.rint(y).astype(int), 0, img.shape[0] - 1)
    good = st[iy, ix] >= thr
    dd = np.abs(la - th[iy, ix])
    dd = np.degrees(np.minimum(dd, np.pi - dd))
    # Zoomed inset with the same aspect ratio as the image, so that panel 5 has
    # the same size as the others
    pad = 25
    H, Wd = img.shape
    asp = Wd / H
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    zw = max(x.max() - x.min() + 2 * pad, (y.max() - y.min() + 2 * pad) * asp)
    zw = min(zw, Wd)
    zh = min(zw / asp, H)
    zx0 = int(np.clip(cx - zw / 2, 0, Wd - zw)); zx1 = int(zx0 + zw)
    zy0 = int(np.clip(cy - zh / 2, 0, H - zh)); zy1 = int(zy0 + zh)

    fig, axs = plt.subplots(2, 3, figsize=(13, 6.4), layout="constrained")
    ext = (0, img.shape[1], 0, img.shape[0])
    a = axs[0, 0]
    a.imshow(logimg, origin="lower", cmap="gray", extent=ext)
    a.set_title(_L(lang, f"1. Imagen AIA {w0} Å (log)", f"1. AIA {w0} Å image (log)"), fontsize=10)
    a = axs[0, 1]
    a.imshow(logimg, origin="lower", cmap="gray", extent=ext)
    for p, r in zip(d["pix"], rel):
        a.plot(p[:, 0], p[:, 1], lw=0.5, color="#2a6fdb" if r else "#e8833a", alpha=0.8 if r else 0.6)
    a.set_title(_L(lang, "2. Líneas NLFFF proyectadas (azul: fiables)", "2. Projected NLFFF lines (blue: reliable)"), fontsize=10)
    a = axs[0, 2]
    a.imshow(np.log10(st + 1e-3), origin="lower", cmap="magma", extent=ext)
    a.set_title(_L(lang, "3. Fuerza de cresta (hessiana)", "3. Ridge strength (Hessian)"), fontsize=10)
    a = axs[1, 0]
    a.imshow(logimg, origin="lower", cmap="gray", extent=ext)
    a.imshow(np.ma.masked_where(st < thr, np.ones_like(st)), origin="lower", cmap="autumn", alpha=0.55, extent=ext)
    a.add_patch(plt.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, color="cyan", lw=1.2))
    a.set_title(_L(lang, f"4. Crestas claras (el {100 - pct:.0f} % más intenso)",
                   f"4. Clear ridges (strongest {100 - pct:.0f}%)"), fontsize=10)
    a = axs[1, 1]
    a.imshow(logimg, origin="lower", cmap="gray", extent=ext)
    yy, xx = np.mgrid[zy0:zy1:4, zx0:zx1:4]
    m = st[yy, xx] >= thr
    u, v = np.cos(th[yy, xx]), np.sin(th[yy, xx])
    a.quiver(xx[m], yy[m], u[m], v[m], color="#ffd23f", headlength=0, headaxislength=0, pivot="middle",
             scale=30, width=0.004)
    a.plot(P_ex[:, 0], P_ex[:, 1], color="#2a6fdb", lw=1.6)
    sc_ = a.scatter(x[good], y[good], c=dd[good], cmap="viridis", vmin=0, vmax=45, s=18, zorder=3,
                    edgecolors="k", linewidths=0.3)
    a.set_xlim(zx0, zx1); a.set_ylim(zy0, zy1)
    # colour bar inside the panel, so as not to misalign it with the others
    a.add_patch(plt.Rectangle((0.02, 0.02), 0.5, 0.25, transform=a.transAxes, fc="white", ec="none", alpha=0.85, zorder=4))
    cax = a.inset_axes([0.05, 0.17, 0.44, 0.05])
    cb = fig.colorbar(sc_, cax=cax, orientation="horizontal")
    cb.set_label(_L(lang, "ángulo local [°]", "local angle [°]"), fontsize=7, color="#333333")
    cb.ax.tick_params(labelsize=7, colors="#333333")
    a.set_title(_L(lang, "5. Dirección de los bucles (amarillo) y ángulo local", "5. Loop direction (yellow) and local angle"), fontsize=10)
    a = axs[1, 2]
    a.hist(dd[good], bins=np.linspace(0, 90, 19), color="#2a6fdb", alpha=0.8)
    a.axvline(np.median(dd[good]) if good.any() else np.nan, color="k", lw=1.5)
    a.axvline(45, color="#888", ls=":", lw=1)
    a.set_xlabel(_L(lang, "ángulo local [°]", "local angle [°]"))
    a.set_ylabel(_L(lang, "puntos", "points"))
    a.set_box_aspect(img.shape[0] / img.shape[1])
    a.set_title(_L(lang, f"6. Mediana = ángulo de la línea ({np.median(dd[good]):.0f}°)",
                   f"6. Median = line angle ({np.median(dd[good]):.0f}°)") if good.any() else "6.", fontsize=10)
    for a in axs.flat[:5]:
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_aia(S, plot, lang, path):
    """Validation with AIA: projected lines, angles and NLFFF versus potential."""
    rel = np.asarray(plot["reliable"], bool)
    W = plot["wavelengths"]
    w0 = 171 if 171 in W else sorted(W)[0]
    fig = plt.figure(figsize=(12, 7.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, :])
    d = W[w0]
    ax.imshow(np.log10(np.clip(d["cutout"], 1, None)), origin="lower", cmap="gray")
    for p, r in zip(d["pix"], rel):
        if not r:
            ax.plot(p[:, 0], p[:, 1], lw=0.5, color="#e8833a", alpha=0.75)
    for p, r in zip(d["pix"], rel):
        if r:
            ax.plot(p[:, 0], p[:, 1], lw=0.6, color="#2a6fdb", alpha=0.9)
    ax.set_xlim(0, d["cutout"].shape[1]); ax.set_ylim(0, d["cutout"].shape[0])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(_L(lang, f"(a) AIA {w0} Å, {d['date']}: líneas NLFFF fiables (azul) y no fiables (naranja)",
                    f"(a) AIA {w0} Å, {d['date']}: reliable (blue) and unreliable (orange) NLFFF lines"), fontsize=10)
    # (b) angle distribution
    axb = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 90, 19)
    for w, ls in zip(sorted(W), ("-", "--")):
        ang = W[w]["angles"]
        for m, col, lab in ((rel, "#2a6fdb", _L(lang, "fiables", "reliable")), (~rel, "#e8833a", _L(lang, "no fiables", "unreliable"))):
            v = ang[m & np.isfinite(ang)]
            axb.hist(v, bins=bins, histtype="step", density=True, color=col, ls=ls, lw=1.6, label=f"{lab}, {w} Å")
    axb.axvline(45, color="#888", lw=0.8, ls=":")
    axb.set_xlabel(_L(lang, "ángulo con los bucles de AIA [°]", "angle to the AIA loops [°]"))
    axb.set_ylabel(_L(lang, "densidad", "density"))
    axb.set_title(_L(lang, "(b) Desalineamiento por línea (45° = sin relación)", "(b) Misalignment per line (45° = unrelated)"), fontsize=10)
    axb.legend(fontsize=8, frameon=False)
    _clean(axb)
    # (c) NLFFF − potential, by group
    axc = fig.add_subplot(gs[1, 1])
    groups = [(_L(lang, "fiables", "reliable"), "reliable"), (_L(lang, "no fiables", "unreliable"), "unreliable")]
    x = np.arange(2)
    for k, w in enumerate(sorted(W)):
        r = S["by_wavelength"][str(w)]
        allv = [r["models"]["potential"][g]["median"] for _, g in groups]
        mid = [r["terciles"][1][f"nlfff_minus_potential_{g}"]["median"] for _, g in groups]
        off = (k - 0.5) * 0.36
        axc.bar(x + off - 0.08, allv, 0.16, color=["#2a6fdb", "#e8833a"], alpha=0.9 - 0.3 * k,
                label=_L(lang, f"todas las alturas, {w} Å", f"all heights, {w} Å"))
        axc.bar(x + off + 0.08, mid, 0.16, color=["#2a6fdb", "#e8833a"], alpha=0.9 - 0.3 * k, hatch="//",
                edgecolor="white", label=_L(lang, f"altura intermedia, {w} Å", f"intermediate height, {w} Å"))
    axc.axhline(0, color="#333", lw=0.8)
    axc.set_xticks(x); axc.set_xticklabels([g[0] for g in groups])
    axc.set_ylabel(_L(lang, "NLFFF − potencial [°] (mediana por línea)", "NLFFF − potential [°] (median per line)"))
    axc.set_title(_L(lang, "(c) ¿Mejora el NLFFF al potencial? (< 0 = sí)", "(c) Does the NLFFF beat the potential? (< 0 = yes)"), fontsize=10)
    axc.legend(fontsize=7, frameon=False, ncol=2)
    _clean(axc)
    fig.savefig(path, dpi=200)
    plt.close(fig)
