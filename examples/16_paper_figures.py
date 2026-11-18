"""Figures of the paper, in Spanish and English: ``docs/paper/figuras/{es,en}/``.

1. Diagram of the method: calibration and application rows sharing the extrapolation and the indicators.
2. Calibration leaving out each family: reliability and risk versus coverage.
3. External flux rope (TD2): exact and NLFFF lines coloured by confidence, and
   fraction of wrong lines by length.
4. Field lines of NOAA 11158, accepted and rejected, in perspective and from above (every traced line).
5. Comparison with AIA step by step (from the region report).
6. Energy drop by zone in the three flares and its sensitivity.
7. Sensitivity to the definition of a wrong line (r11) and E_div/E versus the fraction of wrong
   lines per case (examples/18): a global quantity does not warn of the optimistic probability.
8. Confidence versus apex height in the three regions, closed and open lines
   (writes docs/resultados/height_confidence.json).
9. Lines of the three regions from above, by the model that accepts them: the published one (five terms) and the
   one without the height term of the length (four terms).

The figures are printed 17 cm wide, the two columns of the journal. They are drawn close to that width, with short
panel titles (the captions explain the panels), and the script stops if any text of a figure would be smaller than
6 pt on the page.

    python examples/16_paper_figures.py
    python examples/16_paper_figures.py --only 3 7 --lang en      # some figures only
"""
import argparse
import functools
import json
import os
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from solarff.analytic import EXTERNAL_TEST_FIELDS, field_magnetogram  # noqa: E402
from solarff.calibration import _nz  # noqa: E402
from solarff.confidence import DEFAULT_CALIBRATION, confidence, line_errors, line_indicators  # noqa: E402
from solarff.nlfff_torch import nlfff_lbfgs  # noqa: E402
from solarff.preprocess import preprocess  # noqa: E402
from solarff.tracing import seeds_from_magnetogram, trace_field_lines  # noqa: E402

ap = argparse.ArgumentParser(description="Figures of the paper, in Spanish and English")
ap.add_argument("--only", type=int, nargs="+", default=list(range(1, 10)), choices=range(1, 10), help="figures to draw (default: the nine)")
ap.add_argument("--lang", nargs="+", default=["es", "en"], choices=["es", "en"])
ARGS = ap.parse_args()

warnings.filterwarnings("ignore")
BLUE, ORANGE, GREY, INK = "#2a6fdb", "#e8833a", "#9a9a9a", "#333333"
ROOT = os.path.join("docs", "paper", "figuras")

# Size of the text on the page. A figure drawn W inches wide and printed PRINT_W wide is reduced by PRINT_W / W, and so is
# its text. The plots of this script are drawn FULL_W wide and the maps of the regions 12.5 in wide, with larger fonts.
PRINT_W = 17 / 2.54        # in: width of a two-column figure
FULL_W = 8.5               # in: width of the plots of this script
MIN_PT = 6.0               # smallest text allowed on the page, in points
WHITE = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}
PAPER_STYLE = {**WHITE, "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
               "legend.fontsize": 8, "legend.frameon": False}
SMALL = 8                  # notes inside the plots
PRINTED = {}               # smallest text of each figure on the page, in points


def T(lang, es, en):
    return es if lang == "es" else en


def paper(fn):
    """Draw with the fonts of the plots of the paper."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with plt.rc_context(PAPER_STYLE):
            return fn(*a, **k)
    return wrapper


def text_sizes(fig):
    """Font sizes, in points, of the text that the figure shows. To be called after drawing: the tick labels exist only then."""
    return [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def check_printed(path, dpi, smallest):
    """Size on the page of the smallest text of a saved figure; the script stops if it is below ``MIN_PT``."""
    from PIL import Image
    with Image.open(path) as im:
        pt = smallest * PRINT_W / (im.size[0] / dpi)
    PRINTED[os.path.basename(path)] = round(pt, 1)
    assert pt >= MIN_PT - 0.05, f"{path}: text of {pt:.1f} pt on the page (the minimum is {MIN_PT})"


def save(fig, path, dpi=300, **kw):
    fig.savefig(path, dpi=dpi, **kw)
    smallest = min(text_sizes(fig))
    plt.close(fig)
    check_printed(path, dpi, smallest)


def style(a):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color="#e5e5e5", lw=0.6)
    a.set_axisbelow(True)


# ------------------------------------------------------------------ 1. diagram
@paper
def fig_scheme(lang, path):
    """Two rows that share the extrapolation and the indicators: calibration (top) and application (bottom)."""
    fig = plt.figure(figsize=(FULL_W, 3.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 5); ax.set_ylim(0, 1); ax.axis("off")
    TOP, BOT, H = 0.56, 0.08, 0.32          # lower edge of each row and box height
    CAL, APP, SHARED = ("#eef3fb", BLUE), ("#fdf1e7", ORANGE), ("#f3f3f3", "#6b6b6b")

    def box(col, y, h, title, sub, colors, step=None):
        x = col + 0.08
        ax.add_patch(FancyBboxPatch((x, y), 0.84, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=colors[0], ec=colors[1], lw=1.2))
        yc = y + h / 2
        ax.text(x + 0.42, yc + 0.060, title, ha="center", va="center", fontsize=9, color=INK, weight="bold")
        ax.text(x + 0.42, yc - 0.085, sub, ha="center", va="center", fontsize=SMALL, color=INK)
        if step:
            ax.text(x + 0.42, y + h - 0.008, step, ha="center", va="top", fontsize=SMALL, color=GREY)

    def arrow(x0, y0, x1, y1, color=INK, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, ls=ls))

    st = lambda i: T(lang, f"paso {i}", f"step {i}")  # noqa: E731
    yt, yb = TOP + H / 2, BOT + H / 2
    box(0, TOP, H, T(lang, "Campos exactos\nlibres de fuerzas", "Exact\nforce-free fields"),
        T(lang, "6 familias, 3 ruidos,\n3 mallas; error real\npor línea", "6 families, 3 noise\nmodels, 3 grids;\ntrue error per line"), CAL, st(1))
    box(0, BOT, H, T(lang, "Magnetogramas\nvectoriales de HMI", "HMI vector\nmagnetograms"),
        T(lang, "3 regiones, antes y\ndespués de una\nfulguración X", "3 regions, before\nand after an\nX-class flare"), APP)
    box(1, BOT, TOP + H - BOT, T(lang, "Extrapolación\nNLFFF", "NLFFF\nextrapolation"),
        T(lang, "mismo preprocesado,\ncódigo y parámetros;\nlíneas desde semillas", "same preprocessing,\ncode and parameters;\nlines from seeds"), SHARED, st(2))
    box(2, BOT, TOP + H - BOT, T(lang, "Indicadores\npor línea", "Line\nindicators"),
        T(lang, "variación de α,\ncapa, longitud\n(+ su cuadrado)", "α variation, buffer,\nlength (+ its square)"), SHARED, st(3))
    box(3, TOP, H, T(lang, "Calibración", "Calibration"),
        T(lang, "regresión logística;\nvalidación por\nfamilias", "logistic regression;\nleave-one-family-out"), CAL)
    box(3, BOT, H, T(lang, "Confianza\npor línea", "Per-line\nconfidence"),
        T(lang, "aceptada si ≥ 0.8", "accepted if ≥ 0.8"), APP)
    box(4, TOP, H, T(lang, "Pruebas externas", "External tests"),
        T(lang, "4 familias no vistas;\nerrores en la base;\nsin reajustar", "4 unseen families;\nerrors in the base;\nno refitting"), CAL)
    box(4, BOT, H, T(lang, "Uso", "Use"),
        T(lang, "bucles de AIA;\nenergía en las\nfulguraciones", "AIA loops;\nenergy in flares"), APP)
    # steps 4 and 5 span two boxes each: one bracket per group instead of a label in every box
    for y0, y1, label in ((TOP + H + 0.030, TOP + H + 0.050, st(4)), (BOT - 0.030, BOT - 0.050, st(5))):
        ax.plot([3.08, 3.08, 4.92, 4.92], [y0, y1, y1, y0], color=GREY, lw=1)
        ax.text(4.0, y1, label, ha="center", va="center", fontsize=SMALL, color=GREY,
                bbox=dict(fc="white", ec="none", pad=1.5))
    for y in (yt, yb):
        for i in (0, 1, 2, 3):
            arrow(i + 0.94, y, i + 1.06, y)
    arrow(3.5, TOP - 0.02, 3.5, BOT + H + 0.02, color=BLUE)
    ax.text(3.53, (TOP + BOT + H) / 2, T(lang, "coeficientes\ndel modelo", "model\ncoefficients"), ha="left", va="center",
            fontsize=SMALL, color=BLUE)
    ax.text(0.06, 0.965, T(lang, "CALIBRACIÓN: la solución exacta se conoce", "CALIBRATION: the exact solution is known"),
            ha="left", va="center", fontsize=SMALL, color=BLUE, weight="bold")
    ax.text(0.06, 0.012, T(lang, "APLICACIÓN: la solución no se conoce", "APPLICATION: the solution is not known"),
            ha="left", va="center", fontsize=SMALL, color=ORANGE, weight="bold")
    save(fig, path, bbox_inches="tight", pad_inches=0.03)


# -------------------------------------------------------------- 2. calibration
@paper
def fig_calibration(lang, path):
    S = pickle.load(open("data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax_boxes.pkl", "rb"))["field"]
    p = np.asarray(S["models"]["logistic"]["p"], float)
    y = np.asarray(S["y"], float)
    fam = np.asarray(S["groups"])
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 3.3), layout="constrained")
    bins = np.linspace(0, 1, 11)
    for g in list(np.unique(fam)) + ["all"]:
        m = np.ones_like(y, bool) if g == "all" else fam == g
        k = np.digitize(p[m], bins[1:-1])
        xs = [p[m][k == i].mean() for i in range(10) if (k == i).sum() > 20]
        ys = [y[m][k == i].mean() for i in range(10) if (k == i).sum() > 20]
        lab = T(lang, "todas", "all") if g == "all" else str(g)
        ax[0].plot(xs, ys, "-o" if g == "all" else "-", lw=2.2 if g == "all" else 0.9, ms=3.5,
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
    ax[0].set_title(T(lang, "(a) Fiabilidad, dejando fuera cada familia", "(a) Reliability, leaving out each family"))
    ax[0].legend()
    order = np.argsort(p)
    cov = np.arange(1, len(p) + 1) / len(p)
    risk = np.cumsum(y[order]) / np.arange(1, len(p) + 1)
    ax[1].plot(cov, 100 * risk, color=INK)
    lo, hi = np.percentile(np.array(rc_b), [2.5, 97.5], axis=0)
    ax[1].fill_between(grid, 100 * lo, 100 * hi, color=INK, alpha=0.15, lw=0)
    for t, mk in ((0.7, "s"), (0.8, "o"), (0.9, "^")):
        acc = 1 - p >= t
        ax[1].plot(acc.mean(), 100 * y[acc].mean(), mk, color=BLUE, ms=7,
                   label=T(lang, f"confianza ≥ {t}", f"confidence ≥ {t}"))
    ax[1].axhline(100 * y.mean(), ls=":", color=GREY)
    ax[1].text(0.02, 100 * y.mean() + 1, T(lang, "sin filtrar", "unfiltered"), color=GREY, fontsize=SMALL)
    ax[1].set_xlabel(T(lang, "fracción de líneas conservadas", "fraction of lines kept"))
    ax[1].set_ylabel(T(lang, "líneas malas entre las conservadas [%]", "wrong lines among those kept [%]"))
    ax[1].set_title(T(lang, "(b) Riesgo frente a cobertura", "(b) Risk versus coverage"))
    ax[1].legend()
    for a in ax:
        style(a)
    save(fig, path)


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


@paper
def fig_rope(lang, path):
    R = rope_case()
    r7 = json.load(open("docs/revision/resultados/r7_models.json", encoding="utf-8"))
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 3.0), layout="constrained")
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
            a.plot(le[k].points[:, 1], le[k].points[:, 2], color=col, lw=1.1, ls="--")
            a.plot(lm[k].points[:, 1], lm[k].points[:, 2], color=col, lw=1.7)
    # the caption gives the confidence of each group and says that each exact line has the colour of its pair
    a.plot([], [], color="#555555", ls="--", label=T(lang, "exacta", "exact"))
    a.plot([], [], color=BLUE, label=T(lang, "NLFFF, buena", "NLFFF, good"))
    a.plot([], [], color="#d1495b", label=T(lang, "NLFFF, mala", "NLFFF, wrong"))
    a.legend(loc="upper left")
    a.set_title(T(lang, "(a) Exacta frente a NLFFF", "(a) Exact versus NLFFF"))
    a.set_xlabel("y [Mm]"); a.set_ylabel("z [Mm]")
    a = ax[1]
    for k, l in enumerate(lm):
        if l.closed:
            a.plot(l.points[:, 1], l.points[:, 2], lw=0.5, color=cmap(float(np.clip(conf[k], 0, 1))), alpha=0.8)
    a.set_title(T(lang, "(b) Líneas del NLFFF", "(b) NLFFF lines"))
    a.set_xlabel("y [Mm]"); a.set_ylabel("z [Mm]")
    ax[0].set_ylim(ax[1].get_ylim())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax[1], label=T(lang, "confianza", "confidence"), shrink=0.85)
    oc = r7["C4_bad_rate_by_length_octile"]
    x = np.arange(1, 9)
    ax[2].plot(x, 100 * np.array(oc["rel"]), "-o", ms=4, color=BLUE, label=T(lang, "relativo (> 10 %)", "relative (> 10%)"))
    ax[2].plot(x, 100 * np.array(oc["abs"]), "-s", ms=4, color=ORANGE, label=T(lang, "absoluto (> 2 Mm)", "absolute (> 2 Mm)"))
    ax[2].set_xticks(x)
    ax[2].set_xlabel(T(lang, "octil de longitud (1 = más cortas)", "length octile (1 = shortest)"))
    ax[2].set_ylabel(T(lang, "líneas malas [%]", "wrong lines [%]"))
    ax[2].set_title(T(lang, "(c) Malas según la longitud", "(c) Wrong lines by length"))
    ax[2].legend(title=T(lang, "error", "error"), title_fontsize=SMALL, alignment="left")
    style(ax[2])
    save(fig, path)


# ---------------------------------------------------------- 6. flares
@paper
def fig_flares(lang, path):
    F = json.load(open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "resultados"), "flare_energy.json"), encoding="utf-8"))["events"]
    R1 = json.load(open("docs/revision/resultados/r1_flare_energy.json", encoding="utf-8"))["events"]
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 3.4), layout="constrained")
    x = np.arange(len(F))
    labels = [f"NOAA {e['noaa']}\n{e['flare'].split(' ')[0]}" for e in F]
    # the paper calls the lines with confidence >= 0.8 accepted, and the others rejected; the caption says that the number
    # above each bar is the number of lines of its zone, and that the rest of the box is crossed by no traced line
    zones = (("reliable", T(lang, "líneas aceptadas", "accepted lines"), BLUE),
             ("unreliable", T(lang, "líneas rechazadas", "rejected lines"), ORANGE),
             ("rest", T(lang, "resto de la caja", "rest of the box"), GREY))
    w = 0.26
    top = max(-v / 1e32 for e in F for v in e["H1"]["dE_by_zone"].values())
    for i, (k, lab, c) in enumerate(zones):
        h = [-e["H1"]["dE_by_zone"][k] / 1e32 for e in F]
        ax[0].bar(x + (i - 1) * w, h, w * 0.92, color=c, label=lab)
        for j, e in enumerate(F):  # number of lines of each zone, above its bar (the rest of the box has none by definition)
            n = e["H1"].get("n_lines", {}).get(k)
            txt = f"{n}" if (n is not None and k != "rest") else ""
            ax[0].text(x[j] + (i - 1) * w, max(h[j], 0) + 0.02 * top, txt, ha="center", va="bottom", fontsize=SMALL, color=INK)
    ax[0].set_ylim(0, 1.25 * top)   # headroom for the legend above the tallest bar
    ax[0].set_ylabel(T(lang, r"caída de $E_J$ [$10^{32}$ erg]", r"drop of $E_J$ [$10^{32}$ erg]"))
    ax[0].set_title(T(lang, "(a) Caída por zonas", "(a) Drop by zone"))
    ax[0].legend(loc="upper left")
    for j, ev in enumerate(R1):
        smax = max(c["step"] for c in ev["configs"])
        for c in ev["configs"]:
            d = c["step"] == smax and c["threshold"] == 0.8 and c["dilate"] == 1
            ax[1].plot(x[j] + (0.12 if c["step"] != smax else -0.12), 100 * c["frac_drop_reliable"],
                       "o" if d else ".", color=BLUE if d else GREY, ms=8 if d else 6)
    ax[1].axhline(50, color=GREY, ls=":", lw=1)
    ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel(T(lang, "parte de la caída en la zona aceptada [%]", "share of the drop in the accepted zone [%]"))
    ax[1].set_title(T(lang, "(b) Sensibilidad a semillas, dilatación y umbral", "(b) Sensitivity to seeds, dilation and threshold"))
    for a in ax:
        a.set_xticks(x, labels)
        style(a)
    save(fig, path)


# ------------------------------------- 7. wrong-line threshold and global quantity per case
@paper
def fig_threshold_global(lang, path):
    r11 = json.load(open("docs/revision/resultados/r11_label_nested.json", encoding="utf-8"))["labels"]
    keys = ["rel5", "rel10", "rel20", "abs2mm", "mixed", "rel10_or_end2mm", "rel10_or_end5mm"]
    # short labels: the caption lists the seven definitions in this order
    names = {"rel5": T(lang, "5 %", "5%"), "rel10": T(lang, "10 %", "10%"), "rel20": T(lang, "20 %", "20%"), "abs2mm": "2 Mm",
             "mixed": T(lang, "mixta", "mixed"), "rel10_or_end2mm": T(lang, "extr. 2 Mm", "ends 2 Mm"),
             "rel10_or_end5mm": T(lang, "extr. 5 Mm", "ends 5 Mm")}
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 3.2), layout="constrained")
    x = np.arange(len(keys))
    ax[0].plot(x, [r11[k]["auc_refit"] for k in keys], "o-", ms=4, color=INK, label=T(lang, "reajustado a cada definición", "refitted to each definition"))
    ax[0].plot(x, [r11[k]["auc_published_model"] for k in keys], "s--", ms=4, color=BLUE, label=T(lang, "modelo adoptado", "adopted model"))
    ax[0].set_ylabel("AUC")
    ax[0].set_ylim(0.82, 0.94)
    ax[0].legend(loc="lower right")
    ax[1].plot(x, [100 * r11[k]["ece_refit"] for k in keys], "o-", ms=4, color=INK, label="ECE")
    ax[1].plot(x, [100 * r11[k]["bad_accepted"] for k in keys], "^-", ms=4, color=ORANGE, label=T(lang, "malas aceptadas (≥ 0.8)", "wrong accepted (≥ 0.8)"))
    ax[1].plot(x, [100 * r11[k]["bad_rate"] for k in keys], "x", color=GREY, ms=6, ls="none", label=T(lang, "malas sin filtrar", "wrong, unfiltered"))
    ax[1].set_ylabel("%")
    ax[1].set_ylim(0, 84)            # headroom for the legend
    ax[1].legend(loc="upper left")
    for a, t in zip(ax[:2], (T(lang, "(a) Ordenación", "(a) Ranking"), T(lang, "(b) Calibración", "(b) Calibration"))):
        a.set_xticks(x)
        a.set_xticklabels([names[k] for k in keys], rotation=40, ha="right", rotation_mode="anchor")
        a.set_xlabel(T(lang, "definición de línea mala", "definition of a wrong line"))
        a.set_title(t)
    ap_rows = json.load(open(os.path.join("docs", "resultados", "applicability.json"), encoding="utf-8"))["rows"]
    fams = sorted({r["field"] for r in ap_rows})
    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(fams):
        rr = [r for r in ap_rows if r["field"] == f]
        opt = np.array([r["bad_accepted"] is not None and r["bad_accepted"] >= 0.10 for r in rr])
        xx = np.array([100 * r["bad_rate"] for r in rr])
        yy = np.array([r["e_div_frac"] for r in rr])
        lab = f.replace("LFFF-delta", "LFFF-δ").replace("-relax", "")
        ax[2].scatter(xx[~opt], yy[~opt], s=12, color=cmap(i), label=lab, zorder=3)
        ax[2].scatter(xx[opt], yy[opt], s=26, facecolors="none", edgecolors=cmap(i), lw=1.0, zorder=3)
    ax[2].scatter([], [], s=26, facecolors="none", edgecolors=INK, lw=1.0, label=T(lang, "optimista", "optimistic"))
    ax[2].set_xlabel(T(lang, "líneas malas del caso [%]", "wrong lines of the case [%]"))
    ax[2].set_ylabel(r"$E_{\rm div}/E$")
    ax[2].set_title(T(lang, "(c) Magnitud global", "(c) Global quantity"))
    ax[2].set_ylim(-0.008, 0.31)     # the legend goes above the families with a low divergence, left of the flux ropes
    ax[2].legend(loc="upper left", labelspacing=0.2, handletextpad=0.1, borderaxespad=0.1)
    for a in ax:
        style(a)
    save(fig, path)


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


@paper
def fig_height(lang, path):
    H = height_data()
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 3.1), layout="constrained", sharey=True)
    for a, noaa in zip(ax, REGION_PKLS):
        d = H[noaa]
        ok = d["ok"]
        # the caption says that the open lines are those that reach a boundary of the box
        for cl, col, lab in ((True, BLUE, T(lang, "cerrada", "closed")), (False, ORANGE, T(lang, "abierta", "open"))):
            m = ok & (d["closed"] == cl)
            a.scatter(d["apex"][m], d["conf"][m], s=5, color=col, alpha=0.45, lw=0, label=lab)
            q = np.quantile(d["apex"][m], np.linspace(0, 1, 7))
            k = np.clip(np.digitize(d["apex"][m], q[1:-1]), 0, 5)
            xs = [np.median(d["apex"][m][k == i]) for i in range(6)]
            ys = [np.median(d["conf"][m][k == i]) for i in range(6)]
            a.plot(xs, ys, "-o", color=col, ms=3.5, lw=1.5)
        a.axhline(DEFAULT_CALIBRATION["threshold"], ls=":", color=GREY)
        a.set_xscale("log")
        a.set_xticks([1, 3, 10, 30, 100])
        a.set_xticklabels(["1", "3", "10", "30", "100"])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xlabel(T(lang, "altura del ápice [Mm]", "apex height [Mm]"))
        s = H["summary"][noaa]
        a.set_title(f"NOAA {noaa}\n" + T(lang, f"{100 * s['frac_reliable']:.0f} % aceptadas, ρ = {s['spearman_conf_apex']:+.2f}",
                                       f"{100 * s['frac_reliable']:.0f}% accepted, ρ = {s['spearman_conf_apex']:+.2f}"))
        style(a)
    ax[0].set_ylabel(T(lang, "confianza de la línea", "line confidence"))
    ax[0].legend(loc="lower left", markerscale=1.6)
    save(fig, path)


# ------------------------------------------------------------------ 4 and 9. field lines of the regions
_LINES = {}


def region_lines(noaa):
    """Lines of the paper in a region (seeds at every second pixel of the base with strong field), and which of the two
    models accepts each: the published one and that of version 1.1, without the height term of the length."""
    if noaa not in _LINES:
        from solarff.confidence import CALIBRATION_V1, assess_lines
        from solarff.montecarlo import base_seeds
        st, _ = pickle.load(open(REGION_PKLS[noaa], "rb"))
        f = st.nlfff
        src = st.mag_pp if st.mag_pp is not None else st.mag
        lines = trace_field_lines(f, base_seeds(f, 2))
        a = assess_lines(f, lines, mag=src if src.has_vector else None, noise_bt=float(st.mag.meta.get("noise") or 100.0))
        t = DEFAULT_CALIBRATION["threshold"]
        acc = np.nan_to_num(np.asarray(a["confidence"], float), nan=0.0) >= t
        acc4 = np.nan_to_num(confidence(a["indicators"], CALIBRATION_V1), nan=0.0) >= t
        _LINES[noaa] = (f, lines, acc, acc4)
    return _LINES[noaa]


def _top_view(ax, f, P):
    """B_z at the base in grey, with the polarity inversion line above the field lines."""
    bz0 = f.bz[:, :, 0]
    P._draw_bz_2d(ax, f.x, f.y, bz0, P._bz_vlim(bz0), contours=False, pil=False)
    pil = P._pil_field(bz0)
    if np.nanmin(pil) < 0 < np.nanmax(pil):
        X, Y = np.meshgrid(f.x, f.y, indexing="ij")
        ax.contour(X, Y, pil, levels=[0.0], colors=P.INK, linewidths=1.2, zorder=9)
    ax.set_xlim(*P._edges(f.x))
    ax.set_ylim(*P._edges(f.y))
    ax.set_aspect("equal")


def fig_lines_region(lang, path, noaa="11158"):
    """Every traced line of a region, in perspective and from above: accepted ones coloured by apex height, rejected in grey."""
    import solarff.plotting as P
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    f, lines, acc, _ = region_lines(noaa)
    norm = Normalize(0, max(l.apex for l, a in zip(lines, acc) if a))
    with plt.rc_context({**P._STYLE, **WHITE, "font.size": 13, "axes.labelsize": 13}):
        fig = plt.figure(figsize=(12.5, 4.9), layout="constrained")
        for k, view in enumerate(("3d", "top")):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d" if view == "3d" else None)
            if view == "3d":
                P.plot_field_lines(f, lines=lines, ax=ax, view="3d", color_by="seed", highlight=[], norm=norm, title="", elev=32, azim=-62)
                ax.set_box_aspect((f.x[-1] - f.x[0], f.y[-1] - f.y[0], f.z[-1] - f.z[0]), zoom=1.1)
                ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))      # the box is low: few ticks in height
                ax.tick_params(labelsize=11.5)
                ax.xaxis.labelpad, ax.yaxis.labelpad = 14, 9
            else:
                _top_view(ax, f, P)
                ax.set_xlabel("x [Mm]")
                ax.set_ylabel("y [Mm]")
                for l in lines:
                    ax.plot(l.points[:, 0], l.points[:, 1], color=P.MUTED, lw=0.4, zorder=4)
            for j in np.argsort([l.apex for l in lines])[::-1]:        # the lowest accepted lines on top
                if acc[j]:
                    l, col = lines[j], P.APEX_CMAP(norm(lines[j].apex))
                    if view == "3d":
                        ax.plot(l.points[:, 0], l.points[:, 1], l.points[:, 2], color=col, lw=0.8, zorder=6)
                    else:
                        ax.plot(l.points[:, 0], l.points[:, 1], color=col, lw=0.8, zorder=6)
        fig.colorbar(ScalarMappable(norm, P.APEX_CMAP), ax=fig.axes, shrink=0.62, pad=0.015,
                     label=T(lang, "altura máxima de las líneas aceptadas [Mm]", "apex height of the accepted lines [Mm]"))
        fig.legend(handles=[Line2D([], [], color=P.APEX_CMAP(0.6), lw=2, label=T(lang, "aceptada (confianza ≥ 0.8)", "accepted (confidence ≥ 0.8)")),
                            Line2D([], [], color=P.MUTED, lw=2, label=T(lang, "rechazada", "rejected")),
                            Line2D([], [], color=P.INK, lw=1.4, label=T(lang, "línea de inversión de polaridad", "polarity inversion line"))],
                   loc="lower center", ncols=3, frameon=False, fontsize=12.5)
        fig.get_layout_engine().set(wspace=0.11, rect=(0, 0.07, 1, 1))
        save(fig, path)


def fig_lines_models(lang, path):
    """The lines of the three regions from above, by the model that accepts them (five terms and four terms)."""
    import solarff.plotting as P
    from matplotlib.lines import Line2D
    blue, orange, green = P.CATEGORICAL[:3]
    d = {n: region_lines(n) for n in REGION_PKLS}
    r1, r2, r3 = ((v[0].y[-1] - v[0].y[0]) / (v[0].x[-1] - v[0].x[0]) for v in d.values())
    wr = (r1 + r2) / r3                 # the third region, in the right column, as high as the other two together
    with plt.rc_context({**P._STYLE, **WHITE, "font.size": 12.5, "axes.labelsize": 12.5}):
        fig = plt.figure(figsize=(12.5, 12.5 / (1 + wr) * (r1 + r2) + 1.3), layout="constrained")
        gs = fig.add_gridspec(2, 2, width_ratios=[1, wr], height_ratios=[r1, r2])
        axes = dict(zip(REGION_PKLS, (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1]))))
        for noaa, ax in axes.items():
            f, lines, acc, acc4 = d[noaa]
            _top_view(ax, f, P)
            for sel, col, lw, zo in ((~acc & ~acc4, P.MUTED, 0.35, 4), (acc & acc4, blue, 0.7, 5), (~acc & acc4, orange, 0.95, 6),
                                     (acc & ~acc4, green, 1.5, 7)):
                for j in np.flatnonzero(sel):
                    ax.plot(lines[j].points[:, 0], lines[j].points[:, 1], color=col, lw=lw, zorder=zo)
            ax.text(0.012, 0.972, f"NOAA {noaa}", transform=ax.transAxes, va="top", ha="left", color=P.INK, zorder=12,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88))
            ax.set_ylabel("y [Mm]")
        axes["11429"].set_xlabel("x [Mm]")
        axes["12673"].set_xlabel("x [Mm]")
        fig.legend(handles=[Line2D([], [], color=blue, lw=2.2, label=T(lang, "aceptada con cinco y con cuatro términos", "accepted with five and with four terms")),
                            Line2D([], [], color=orange, lw=2.2, label=T(lang, "aceptada sólo con cuatro términos", "accepted only with four terms")),
                            Line2D([], [], color=green, lw=2.2, label=T(lang, "aceptada sólo con cinco términos", "accepted only with five terms")),
                            Line2D([], [], color=P.MUTED, lw=2.2, label=T(lang, "rechazada con los dos", "rejected with both")),
                            Line2D([], [], color=P.INK, lw=1.4, label=T(lang, "línea de inversión de polaridad", "polarity inversion line"))],
                   loc="upper center", ncols=3, frameon=False, fontsize=12)
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.89), hspace=0.02)
        save(fig, path)
    return {n: {"both": int((v[2] & v[3]).sum()), "only_four_terms": int((~v[2] & v[3]).sum()), "only_five_terms": int((v[2] & ~v[3]).sum()),
                "neither": int((~v[2] & ~v[3]).sum())} for n, v in d.items()}


def fig_aia(lang, path):
    """The AIA comparison step by step: the figure of the region report, drawn for the page of the paper."""
    from solarff.report.figures import fig_aia_steps
    aia = pickle.load(open("data/aia_validation_harp377.pkl", "rb"))
    smallest = fig_aia_steps(aia["summary"], aia["plot"], lang, path, word=("aceptadas", "accepted"), compact=True)
    check_printed(path, 300, smallest)


def fig_models_counts(lang, path):
    counts = fig_lines_models(lang, path)
    # the counts of the caption of figure 9, next to the results they come from
    json.dump({"note": "lines of the paper in each region by the model that accepts them (confidence >= 0.8): five terms (published) and "
                       "four terms (version 1.1)", "regions": counts},
              open(os.path.join("docs", "resultados", "lines_by_model.json"), "w", encoding="utf-8"), indent=1)


# number: (function, file name in Spanish, file name in English)
FIGURES = {1: (fig_scheme, "fig1_esquema.png", "fig1_pipeline.png"), 2: (fig_calibration, "fig2_calibracion.png", "fig2_calibration.png"),
           3: (fig_rope, "fig3_cuerda.png", "fig3_flux_rope.png"), 4: (fig_lines_region, "fig4_harp377.png", "fig4_harp377.png"),
           5: (fig_aia, "fig5_aia.png", "fig5_aia.png"), 6: (fig_flares, "fig6_fulguraciones.png", "fig6_flares.png"),
           7: (fig_threshold_global, "fig7_umbral_global.png", "fig7_threshold_global.png"), 8: (fig_height, "fig8_altura.png", "fig8_height.png"),
           9: (fig_models_counts, "fig9_lineas_modelos.png", "fig9_lines_models.png")}

for lang in ARGS.lang:
    out = os.path.join(ROOT, lang)
    os.makedirs(out, exist_ok=True)
    PRINTED.clear()
    for n in sorted(set(ARGS.only)):
        fn, es_name, en_name = FIGURES[n]
        fn(lang, os.path.join(out, es_name if lang == "es" else en_name))
    print(lang, sorted(os.listdir(out)))
    print("   smallest text on the page [pt]:", PRINTED)
