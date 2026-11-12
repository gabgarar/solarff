"""Figures of the paper, in Spanish and English: ``docs/paper/figuras/{es,en}/``.

1. Diagram of the method: calibration and application rows sharing the extrapolation and the indicators.
2. Calibration leaving out each family: reliability and risk versus coverage.
3. External flux rope (TD2): exact and NLFFF lines coloured by confidence, and
   fraction of wrong lines by length.
4. Confidence in HARP 377 (from the region report).
5. Comparison with AIA step by step (from the region report).
6. Energy drop by zone in the three flares and its sensitivity.
7. Sensitivity to the definition of a wrong line (r11) and E_div/E versus the fraction of wrong
   lines per case (examples/18): a global quantity does not warn of the optimistic probability.
8. Confidence versus apex height in the three regions, closed and open lines
   (writes docs/resultados/height_confidence.json).

    python examples/16_paper_figures.py
"""
import json
import os
import pickle
import shutil
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from solarff.analytic import EXTERNAL_TEST_FIELDS, field_magnetogram  # noqa: E402
from solarff.calibration import _nz  # noqa: E402
from solarff.confidence import DEFAULT_CALIBRATION, confidence, line_errors, line_indicators  # noqa: E402
from solarff.nlfff_torch import nlfff_lbfgs  # noqa: E402
from solarff.preprocess import preprocess  # noqa: E402
from solarff.tracing import seeds_from_magnetogram, trace_field_lines  # noqa: E402

warnings.filterwarnings("ignore")
BLUE, ORANGE, GREY, INK = "#2a6fdb", "#e8833a", "#9a9a9a", "#333333"
ROOT = os.path.join("docs", "paper", "figuras")


def T(lang, es, en):
    return es if lang == "es" else en


def style(a):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color="#e5e5e5", lw=0.6)
    a.set_axisbelow(True)


# ------------------------------------------------------------------ 1. diagram
def fig_scheme(lang, path):
    """Two rows that share the extrapolation and the indicators: calibration (top) and application (bottom)."""
    fig, ax = plt.subplots(figsize=(13, 4.9))
    ax.set_xlim(0, 5); ax.set_ylim(0, 1); ax.axis("off")
    TOP, BOT, H = 0.56, 0.08, 0.32          # lower edge of each row and box height
    CAL, APP, SHARED = ("#eef3fb", BLUE), ("#fdf1e7", ORANGE), ("#f3f3f3", "#6b6b6b")

    def box(col, y, h, title, sub, colors, step=None):
        x = col + 0.08
        ax.add_patch(FancyBboxPatch((x, y), 0.84, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=colors[0], ec=colors[1], lw=1.4))
        yc = y + h / 2
        ax.text(x + 0.42, yc + 0.065, title, ha="center", va="center", fontsize=11, color=INK, weight="bold")
        ax.text(x + 0.42, yc - 0.075, sub, ha="center", va="center", fontsize=8.5, color=INK)
        if step:
            ax.text(x + 0.01, y + h - 0.012, step, ha="left", va="top", fontsize=8, color=GREY)

    def arrow(x0, y0, x1, y1, color=INK, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, ls=ls))

    st = lambda i: T(lang, f"paso {i}", f"step {i}")  # noqa: E731
    yt, yb = TOP + H / 2, BOT + H / 2
    box(0, TOP, H, T(lang, "Campos exactos\nlibres de fuerzas", "Exact\nforce-free fields"),
        T(lang, "6 familias, 3 ruidos,\n3 mallas; error real por línea", "6 families, 3 noise models,\n3 grids; true error per line"), CAL, st(1))
    box(0, BOT, H, T(lang, "Magnetogramas\nvectoriales de HMI", "HMI vector\nmagnetograms"),
        T(lang, "3 regiones, antes y después\nde una fulguración X", "3 regions, before and after\nan X-class flare"), APP)
    box(1, BOT, TOP + H - BOT, T(lang, "Extrapolación\nNLFFF", "NLFFF\nextrapolation"),
        T(lang, "mismo preprocesado,\ncódigo y parámetros;\nlíneas desde semillas", "same preprocessing,\ncode and parameters;\nlines from seeds"), SHARED, st(2))
    box(2, BOT, TOP + H - BOT, T(lang, "Indicadores\npor línea", "Line\nindicators"),
        T(lang, "variación de α, capa,\nlongitud (+ su cuadrado)", "α variation, buffer,\nlength (+ its square)"), SHARED, st(3))
    box(3, TOP, H, T(lang, "Calibración", "Calibration"),
        T(lang, "regresión logística;\nvalidación por familias", "logistic regression;\nleave-one-family-out"), CAL)
    box(3, BOT, H, T(lang, "Confianza\npor línea", "Per-line\nconfidence"),
        T(lang, "aceptada si ≥ 0.8", "accepted if ≥ 0.8"), APP)
    box(4, TOP, H, T(lang, "Pruebas externas", "External tests"),
        T(lang, "4 familias no vistas;\nerrores sistemáticos\nen la base; sin reajustar", "4 unseen families;\nsystematic base errors;\nno refitting"), CAL)
    box(4, BOT, H, T(lang, "Uso", "Use"),
        T(lang, "bucles de AIA;\nenergía en fulguraciones", "AIA loops;\nenergy in flares"), APP)
    # steps 4 and 5 span two boxes each: one bracket per group instead of a label in every box
    for y0, y1, label in ((TOP + H + 0.030, TOP + H + 0.050, st(4)), (BOT - 0.030, BOT - 0.050, st(5))):
        ax.plot([3.08, 3.08, 4.92, 4.92], [y0, y1, y1, y0], color=GREY, lw=1)
        ax.text(4.0, y1, label, ha="center", va="center", fontsize=8, color=GREY,
                bbox=dict(fc="white", ec="none", pad=1.5))
    for y in (yt, yb):
        for i in (0, 1, 2, 3):
            arrow(i + 0.94, y, i + 1.06, y)
    arrow(3.5, TOP - 0.02, 3.5, BOT + H + 0.02, color=BLUE)
    ax.text(3.53, (TOP + BOT + H) / 2, T(lang, "coeficientes\npublicados", "published\ncoefficients"), ha="left", va="center",
            fontsize=8.5, color=BLUE)
    ax.text(0.06, 0.965, T(lang, "CALIBRACIÓN: la solución exacta se conoce", "CALIBRATION: the exact solution is known"),
            ha="left", fontsize=8.5, color=BLUE, weight="bold")
    ax.text(0.06, 0.012, T(lang, "APLICACIÓN: la solución no se conoce", "APPLICATION: the solution is not known"),
            ha="left", fontsize=8.5, color=ORANGE, weight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------- 2. calibration
def fig_calibration(lang, path):
    S = pickle.load(open("data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl", "rb"))["field"]
    p = np.asarray(S["models"]["logistic"]["p"], float)
    y = np.asarray(S["y"], float)
    fam = np.asarray(S["groups"])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    bins = np.linspace(0, 1, 11)
    for g in list(np.unique(fam)) + ["all"]:
        m = np.ones_like(y, bool) if g == "all" else fam == g
        k = np.digitize(p[m], bins[1:-1])
        xs = [p[m][k == i].mean() for i in range(10) if (k == i).sum() > 20]
        ys = [y[m][k == i].mean() for i in range(10) if (k == i).sum() > 20]
        lab = T(lang, "todas", "all") if g == "all" else str(g)
        ax[0].plot(xs, ys, "-o" if g == "all" else "-", lw=2.4 if g == "all" else 0.9, ms=4,
                   color=INK if g == "all" else None, label=lab)
    # 95 % bands: whole extrapolations are resampled, since the lines of one extrapolation are not independent
    case = np.asarray(S["case"])
    members = [np.flatnonzero(case == c) for c in np.unique(case)]
    rng = np.random.default_rng(0)
    grid = np.linspace(0.02, 1, 99)
    rel_b, rc_b = [], []
    for _ in range(300):
        i = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        pb, yb = p[i], y[i]
        kb = np.digitize(pb, bins[1:-1])
        rel_b.append([yb[kb == q].mean() if (kb == q).sum() > 20 else np.nan for q in range(10)])
        ob = np.argsort(pb)
        rc_b.append(np.interp(grid, np.arange(1, len(pb) + 1) / len(pb), np.cumsum(yb[ob]) / np.arange(1, len(pb) + 1)))
    k_all = np.digitize(p, bins[1:-1])
    used = [q for q in range(10) if (k_all == q).sum() > 20]
    lo, hi = np.nanpercentile(np.array(rel_b)[:, used], [2.5, 97.5], axis=0)
    ax[0].fill_between([p[k_all == q].mean() for q in used], lo, hi, color=INK, alpha=0.15, lw=0)
    ax[0].plot([0, 1], [0, 1], ":", color=GREY)
    ax[0].set_xlabel(T(lang, "probabilidad predicha de línea mala", "predicted probability of a wrong line"))
    ax[0].set_ylabel(T(lang, "fracción observada de líneas malas", "observed fraction of wrong lines"))
    ax[0].set_title(T(lang, "(a) Fiabilidad, dejando fuera cada familia", "(a) Reliability, leaving out each family"), fontsize=10)
    ax[0].legend(fontsize=7.5, frameon=False)
    order = np.argsort(p)
    cov = np.arange(1, len(p) + 1) / len(p)
    risk = np.cumsum(y[order]) / np.arange(1, len(p) + 1)
    ax[1].plot(cov, 100 * risk, color=INK)
    lo, hi = np.percentile(np.array(rc_b), [2.5, 97.5], axis=0)
    ax[1].fill_between(grid, 100 * lo, 100 * hi, color=INK, alpha=0.15, lw=0)
    for t, mk in ((0.7, "s"), (0.8, "o"), (0.9, "^")):
        acc = 1 - p >= t
        ax[1].plot(acc.mean(), 100 * y[acc].mean(), mk, color=BLUE, ms=8,
                   label=T(lang, f"confianza ≥ {t}", f"confidence ≥ {t}"))
    ax[1].axhline(100 * y.mean(), ls=":", color=GREY)
    ax[1].text(0.02, 100 * y.mean() + 1, T(lang, "sin filtrar", "unfiltered"), color=GREY, fontsize=8)
    ax[1].set_xlabel(T(lang, "fracción de líneas conservadas", "fraction of lines kept"))
    ax[1].set_ylabel(T(lang, "líneas malas entre las conservadas [%]", "wrong lines among those kept [%]"))
    ax[1].set_title(T(lang, "(b) Riesgo frente a cobertura", "(b) Risk versus coverage"), fontsize=10)
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        style(a)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- 3. TD2 rope
_ROPE = {}


def rope_case():
    if _ROPE:
        return _ROPE
    n = 48
    exact = EXTERNAL_TEST_FIELDS["TD2-relax"](n, n, _nz(n))
    mag = field_magnetogram(exact, noise=10.0, noise_t=60.0, rng=7)
    pp = preprocess(mag, verbose=False)
    model = nlfff_lbfgs(pp, _nz(n), verbose=False)
    seeds = seeds_from_magnetogram(mag, n=300, method="random", polarity="both", threshold=0.05, rng=11)
    lm, le = trace_field_lines(model, seeds), trace_field_lines(exact, seeds)
    conf = confidence(line_indicators(model, lm, mag=pp, noise_bt=60.0), DEFAULT_CALIBRATION)
    err = line_errors(lm, le)
    bad = (err["shape"] > 0.1) | (err["topology"] > 0)
    _ROPE.update(lm=lm, le=le, conf=conf, bad=bad)
    return _ROPE


def fig_rope(lang, path):
    R = rope_case()
    r7 = json.load(open("docs/revision/resultados/r7_models.json", encoding="utf-8"))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    cmap = plt.get_cmap("RdYlBu")
    # (a) examples of good and wrong lines, chosen with a fixed rule: among the good ones with
    # confidence ≥ 0.8 and among the wrong ones with confidence < 0.5, those at the 25th, 50th and 75th
    # percentiles of length
    lm, le, conf, bad = R["lm"], R["le"], np.asarray(R["conf"]), np.asarray(R["bad"])
    closed = np.array([a.closed and b.closed for a, b in zip(lm, le)])
    L = np.array([np.linalg.norm(np.diff(l.points, axis=0), axis=1).sum() for l in le])
    a = ax[0]
    shades = {"good": ["#1f4e9c", "#2a6fdb", "#7fb0f5"], "bad": ["#8e1b2c", "#d1495b", "#f08f9c"]}
    for name, grp in (("good", (~bad) & (conf >= 0.8) & closed), ("bad", bad & (conf < 0.5) & closed)):
        idx = np.where(grp)[0]
        if not idx.size:
            continue
        for q, col in zip((25, 50, 75), shades[name]):  # each pair with its own hue
            k = idx[np.argmin(np.abs(L[idx] - np.percentile(L[idx], q)))]
            a.plot(le[k].points[:, 1], le[k].points[:, 2], color=col, lw=1.3, ls="--")
            a.plot(lm[k].points[:, 1], lm[k].points[:, 2], color=col, lw=2.0)
    a.plot([], [], color="#555555", ls="--", label=T(lang, "solución exacta (mismo color que su pareja)",
                                                       "exact solution (same colour as its pair)"))
    a.plot([], [], color=BLUE, label=T(lang, "NLFFF: buena (confianza ≥ 0.8)", "NLFFF: good (confidence ≥ 0.8)"))
    a.plot([], [], color="#d1495b", label=T(lang, "NLFFF: mala (confianza < 0.5)", "NLFFF: wrong (confidence < 0.5)"))
    a.legend(fontsize=7.5, frameon=False, loc="upper left")
    a.set_title(T(lang, "(a) Ejemplos: misma semilla, exacta frente a NLFFF", "(a) Examples: same seed, exact versus NLFFF"), fontsize=10)
    a.set_xlabel("y [Mm]"); a.set_ylabel("z [Mm]")
    a = ax[1]
    for k, l in enumerate(lm):
        if l.closed:
            a.plot(l.points[:, 1], l.points[:, 2], lw=0.6, color=cmap(float(np.clip(conf[k], 0, 1))), alpha=0.8)
    a.set_title(T(lang, "(b) Todas las líneas del NLFFF, color = confianza", "(b) All NLFFF lines, colour = confidence"), fontsize=10)
    a.set_xlabel("y [Mm]"); a.set_ylabel("z [Mm]")
    ax[0].set_ylim(ax[1].get_ylim())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax[1], label=T(lang, "confianza", "confidence"), shrink=0.85)
    oc = r7["C4_bad_rate_by_length_octile"]
    x = np.arange(1, 9)
    ax[2].plot(x, 100 * np.array(oc["rel"]), "-o", color=BLUE, label=T(lang, "error relativo (> 10 %)", "relative error (> 10%)"))
    ax[2].plot(x, 100 * np.array(oc["abs"]), "-s", color=ORANGE, label=T(lang, "error absoluto (> 2 Mm)", "absolute error (> 2 Mm)"))
    ax[2].set_xlabel(T(lang, "octil de longitud (1 = más cortas)", "length octile (1 = shortest)"))
    ax[2].set_ylabel(T(lang, "líneas malas [%]", "wrong lines [%]"))
    ext = r7["C3_external_TD2"]
    a0, a1 = 100 * ext["linear_5fam_no_rope"]["bad_accepted"], 100 * ext["reduced_5fam_no_rope"]["bad_accepted"]
    ax[2].set_title(T(lang, f"(c) Malas según la longitud (6 familias)\ncuerda externa: {a0:.0f} % → {a1:.1f} % malas aceptadas",
                      f"(c) Wrong lines by length (6 families)\nexternal rope: {a0:.0f}% → {a1:.1f}% wrong accepted"), fontsize=10)
    ax[2].legend(fontsize=8, frameon=False)
    style(ax[2])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------- 6. flares
def fig_flares(lang, path):
    F = json.load(open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "resultados"), "flare_energy.json"), encoding="utf-8"))["events"]
    R1 = json.load(open("docs/revision/resultados/r1_flare_energy.json", encoding="utf-8"))["events"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    x = np.arange(len(F))
    labels = [f"NOAA {e['noaa']}\n{e['flare'].split(' ')[0]}" for e in F]
    zones = (("reliable", T(lang, "líneas fiables", "reliable lines"), BLUE),
             ("unreliable", T(lang, "líneas no fiables", "unreliable lines"), ORANGE),
             ("rest", T(lang, "resto de la caja (no la cruza ninguna línea trazada)", "rest of the box (crossed by no traced line)"), GREY))
    w = 0.26
    top = max(-v / 1e32 for e in F for v in e["H1"]["dE_by_zone"].values())
    for i, (k, lab, c) in enumerate(zones):
        h = [-e["H1"]["dE_by_zone"][k] / 1e32 for e in F]
        ax[0].bar(x + (i - 1) * w, h, w * 0.92, color=c, label=lab)
        for j, e in enumerate(F):  # number of lines of each zone, above its bar (the rest of the box has none by definition)
            n = e["H1"].get("n_lines", {}).get(k)
            txt = f"{n}" if (n is not None and k != "rest") else ""
            ax[0].text(x[j] + (i - 1) * w, max(h[j], 0) + 0.02 * top, txt, ha="center", va="bottom", fontsize=7.5, color=INK)
    ax[0].set_ylim(0, 1.38 * top)   # headroom for the legend above the tallest bar
    ax[0].text(0.99, 0.97, T(lang, "número sobre las barras: líneas de esa zona", "number above the bars: lines in that zone"),
               transform=ax[0].transAxes, ha="right", va="top", fontsize=8, color=GREY)
    ax[0].set_ylabel(T(lang, r"caída de $E_J$ [$10^{32}$ erg]", r"drop of $E_J$ [$10^{32}$ erg]"))
    ax[0].set_title(T(lang, "(a) Caída por zonas (zonas de las líneas previas)", "(a) Drop by zone (zones of the pre-flare lines)"), fontsize=10)
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    for j, ev in enumerate(R1):
        smax = max(c["step"] for c in ev["configs"])
        for c in ev["configs"]:
            d = c["step"] == smax and c["threshold"] == 0.8 and c["dilate"] == 1
            ax[1].plot(x[j] + (0.12 if c["step"] != smax else -0.12), 100 * c["frac_drop_reliable"],
                       "o" if d else ".", color=BLUE if d else GREY, ms=9 if d else 7)
    ax[1].axhline(50, color=GREY, ls=":", lw=1)
    ax[1].set_ylim(bottom=0)        # room for the note under the lowest point
    ax[1].set_ylabel(T(lang, "parte de la caída en la zona fiable [%]", "share of the drop in the reliable zone [%]"))
    ax[1].set_title(T(lang, "(b) Sensibilidad a semillas, dilatación y umbral (azul: por defecto)",
                      "(b) Sensitivity to seeds, dilation and threshold (blue: default)"), fontsize=10)
    ax[1].text(0.01, 0.02, T(lang, "izquierda de cada región: semillas por defecto; derecha: 4 veces más",
                             "left of each region: default seeds; right: 4 times more"), transform=ax[1].transAxes, fontsize=8, color=INK)
    for a in ax:
        a.set_xticks(x, labels)
        style(a)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------- 7. wrong-line threshold and global quantity per case
def fig_threshold_global(lang, path):
    r11 = json.load(open("docs/revision/resultados/r11_label_nested.json", encoding="utf-8"))["labels"]
    keys = ["rel5", "rel10", "rel20", "abs2mm", "mixed", "rel10_or_end2mm", "rel10_or_end5mm"]
    names = {"rel5": "5 %", "rel10": "10 %", "rel20": "20 %", "abs2mm": "> 2 Mm", "mixed": T(lang, "mixta", "mixed"),
             "rel10_or_end2mm": T(lang, "10 % ó\nextremos\n> 2 Mm", "10% or\nends\n> 2 Mm"),
             "rel10_or_end5mm": T(lang, "10 % ó\nextremos\n> 5 Mm", "10% or\nends\n> 5 Mm")}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    x = np.arange(len(keys))
    ax[0].plot(x, [r11[k]["auc_refit"] for k in keys], "o-", color=INK, label=T(lang, "modelo reajustado a la etiqueta", "model refitted to the label"))
    ax[0].plot(x, [r11[k]["auc_published_model"] for k in keys], "s--", color=BLUE, label=T(lang, "modelo publicado", "published model"))
    ax[0].set_ylabel("AUC")
    ax[0].set_ylim(0.84, 0.94)
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    ax[1].plot(x, [100 * r11[k]["ece_refit"] for k in keys], "o-", color=INK, label="ECE")
    ax[1].plot(x, [100 * r11[k]["bad_accepted"] for k in keys], "^-", color=ORANGE, label=T(lang, "malas entre las aceptadas (≥ 0.8)", "wrong among the accepted (≥ 0.8)"))
    ax[1].plot(x, [100 * r11[k]["bad_rate"] for k in keys], "x", color=GREY, ms=7, ls="none", label=T(lang, "malas sin filtrar", "wrong, unfiltered"))
    ax[1].set_ylabel("%")
    ax[1].legend(fontsize=8, frameon=False)
    for a, t in zip(ax[:2], (T(lang, "(a) Ordenación según la definición de línea mala", "(a) Ranking versus the definition of a wrong line"),
                            T(lang, "(b) Calibración según la definición", "(b) Calibration versus the definition"))):
        a.set_xticks(x)
        a.set_xticklabels([names[k] for k in keys], fontsize=7)
        a.set_xlabel(T(lang, "definición de línea mala (error de forma relativo salvo indicación)", "definition of a wrong line (relative shape error unless stated)"), fontsize=8.5)
        a.set_title(t, fontsize=10)
    ap = json.load(open(os.path.join("docs", "resultados", "applicability.json"), encoding="utf-8"))["rows"]
    fams = sorted({r["field"] for r in ap})
    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(fams):
        rr = [r for r in ap if r["field"] == f]
        opt = np.array([r["bad_accepted"] is not None and r["bad_accepted"] >= 0.10 for r in rr])
        xx = np.array([100 * r["bad_rate"] for r in rr])
        yy = np.array([r["e_div_frac"] for r in rr])
        lab = f.replace("LFFF-delta", "LFFF-δ").replace("-relax", "")
        ax[2].scatter(xx[~opt], yy[~opt], s=22, color=cmap(i), label=lab, zorder=3)
        ax[2].scatter(xx[opt], yy[opt], s=46, facecolors="none", edgecolors=cmap(i), lw=1.3, zorder=3)
    ax[2].scatter([], [], s=46, facecolors="none", edgecolors=INK, lw=1.3, label=T(lang, "optimista (≥ 10 % de malas aceptadas)", "optimistic (≥ 10% wrong accepted)"))
    ax[2].set_xlabel(T(lang, "líneas malas del caso [%]", "wrong lines of the case [%]"))
    ax[2].set_ylabel(r"$E_{\rm div}/E$")
    ax[2].set_title(T(lang, "(c) Magnitud global frente a líneas malas", "(c) Global quantity against wrong lines"), fontsize=10)
    ax[2].legend(fontsize=7, frameon=False, ncol=2)
    for a in ax:
        style(a)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------- 8. confidence versus height in the regions
REGION_PKLS = {"11158": "data/harp377_study_mc10.pkl", "11429": "data/harp1449_20120306_2336_r7_nz32_mc10.pkl",
               "12673": "data/harp7115_20170906_1136_r4_nz56_mc10.pkl"}
_HEIGHT = {}


def height_data():
    """Confidence, apex height and closure (closed/open) of the nominal lines of each region."""
    if _HEIGHT:
        return _HEIGHT
    from scipy.stats import spearmanr
    from solarff.confidence import assess_lines
    summary = {}
    for noaa, p in REGION_PKLS.items():
        st, ex = pickle.load(open(p, "rb"))
        lines = ex["lines_nominal"]
        src = st.mag_pp if st.mag_pp is not None else st.mag
        a = assess_lines(st.nlfff, lines, mag=src if src.has_vector else None, noise_bt=float(st.mag.meta.get("noise") or 100.0))
        conf, rel = a["confidence"], a["reliable"]
        apex = np.array([l.apex for l in lines])
        closed = np.array([bool(l.closed) for l in lines])
        ok = np.isfinite(conf)
        rho, pv = spearmanr(apex[ok], conf[ok])
        med = float(np.median(apex[ok]))
        summary[noaa] = {"n_lines": int(ok.sum()), "frac_reliable": float(rel[ok].mean()), "spearman_conf_apex": float(rho), "spearman_p": float(pv),
                         "apex_median_mm": med, "frac_reliable_below_median": float(rel[ok & (apex <= med)].mean()),
                         "frac_reliable_above_median": float(rel[ok & (apex > med)].mean()),
                         "frac_closed": float(closed[ok].mean()),
                         "frac_reliable_closed": float(rel[ok & closed].mean()), "frac_reliable_open": float(rel[ok & ~closed].mean()),
                         "spearman_conf_apex_closed": float(spearmanr(apex[ok & closed], conf[ok & closed])[0]),
                         "noise_bt": float(st.mag.meta.get("noise") or 100.0)}
        _HEIGHT[noaa] = dict(conf=conf, rel=rel, apex=apex, closed=closed, ok=ok)
    _HEIGHT["summary"] = summary
    json.dump({"regions": summary, "note": "líneas nominales del Monte Carlo (ex['lines_nominal']), confianza con assess_lines; abierta = llega a un contorno"},
              open(os.path.join("docs", "resultados", "height_confidence.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return _HEIGHT


def fig_height(lang, path):
    H = height_data()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained", sharey=True)
    for a, noaa in zip(ax, REGION_PKLS):
        d = H[noaa]
        ok = d["ok"]
        for cl, col, lab in ((True, BLUE, T(lang, "cerrada", "closed")), (False, ORANGE, T(lang, "abierta (llega a un contorno)", "open (reaches a boundary)"))):
            m = ok & (d["closed"] == cl)
            a.scatter(d["apex"][m], d["conf"][m], s=9, color=col, alpha=0.45, lw=0, label=lab)
            q = np.quantile(d["apex"][m], np.linspace(0, 1, 7))
            k = np.clip(np.digitize(d["apex"][m], q[1:-1]), 0, 5)
            xs = [np.median(d["apex"][m][k == i]) for i in range(6)]
            ys = [np.median(d["conf"][m][k == i]) for i in range(6)]
            a.plot(xs, ys, "-o", color=col, ms=4, lw=1.8)
        a.axhline(DEFAULT_CALIBRATION["threshold"], ls=":", color=GREY)
        a.set_xscale("log")
        a.set_xticks([1, 3, 10, 30, 100])
        a.set_xticklabels(["1", "3", "10", "30", "100"])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xlabel(T(lang, "altura del ápice [Mm]", "apex height [Mm]"))
        s = H["summary"][noaa]
        a.set_title(f"NOAA {noaa}: " + T(lang, f"{100 * s['frac_reliable']:.0f} % fiables, ρ = {s['spearman_conf_apex']:+.2f}",
                                       f"{100 * s['frac_reliable']:.0f}% reliable, ρ = {s['spearman_conf_apex']:+.2f}"), fontsize=10)
        style(a)
    ax[0].set_ylabel(T(lang, "confianza de la línea", "line confidence"))
    ax[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


for lang in ("es", "en"):
    out = os.path.join(ROOT, lang)
    os.makedirs(out, exist_ok=True)
    rep = os.path.join("docs", "informes", "harp377", "figures", lang)
    fig_scheme(lang, os.path.join(out, "fig1_esquema.png" if lang == "es" else "fig1_pipeline.png"))
    fig_calibration(lang, os.path.join(out, "fig2_calibracion.png" if lang == "es" else "fig2_calibration.png"))
    fig_rope(lang, os.path.join(out, "fig3_cuerda.png" if lang == "es" else "fig3_flux_rope.png"))
    shutil.copy(os.path.join(rep, "reliable_lines.png"), os.path.join(out, "fig4_harp377.png"))
    from solarff.report.figures import fig_aia_steps
    aia = pickle.load(open("data/aia_validation_harp377.pkl", "rb"))
    fig_aia_steps(aia["summary"], aia["plot"], lang, os.path.join(out, "fig5_aia.png"))
    fig_flares(lang, os.path.join(out, "fig6_fulguraciones.png" if lang == "es" else "fig6_flares.png"))
    fig_threshold_global(lang, os.path.join(out, "fig7_umbral_global.png" if lang == "es" else "fig7_threshold_global.png"))
    fig_height(lang, os.path.join(out, "fig8_altura.png" if lang == "es" else "fig8_height.png"))
    print(lang, sorted(os.listdir(out)))
