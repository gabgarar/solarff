"""Figures of the tests against the MURaM simulation, for the documentation.

Figure 1: the same seeds traced in the simulation (the truth) and in the extrapolations made with the default and the
recommended treatment of the base, seen from above and from the side. The case is one of test F
(examples/43): snapshot 280000, pixel 0.77 Mm, HMI-like noise, same noise and same seeds, so the fractions of wrong lines
are those of that test for this case.

Figure 2: wrong lines against the weight of the base term, for each regime of test F (read from
docs/resultados/base_weight_regimes.json; nothing is recomputed). As in that scan, the magnetogram is not preprocessed.

    python examples/46_simulation_figures.py                  # about 10 minutes the first time, then from the cache
    python examples/46_simulation_figures.py --dpi 300        # higher resolution

Writes the figures at full resolution in docs/informes/simulations/ and a copy 1600 pixels wide in docs/web/img/realistic/.
"""
import argparse
import json
import os
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from solarff.analytic import field_magnetogram  # noqa: E402
from solarff.calibration import NOISE_MODELS  # noqa: E402
from solarff.confidence import line_errors  # noqa: E402
from solarff.field import VectorField3D  # noqa: E402
from solarff.nlfff_torch import nlfff_lbfgs  # noqa: E402
from solarff.preprocess import preprocess  # noqa: E402
from solarff.simulations import SIMULATIONS, reduced_field  # noqa: E402
from solarff.tracing import seeds_from_magnetogram, trace_field_lines  # noqa: E402
from solarff.utils import keep_awake  # noqa: E402

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", default="data")
ap.add_argument("--raw", default=os.path.join(os.path.expanduser("~"), "solarff-data", "muram_ar098192"))
ap.add_argument("--dpi", type=int, default=300)
ap.add_argument("--shown", type=int, default=90, help="lines drawn in figure 1 (all 300 are counted)")
a = ap.parse_args()

SIM, SNAP, R, J = "ar098192", 280000, 4, 2                     # the case of test F; J = 2 is the HMI-like noise
ORDER = [290000, 310000, 270000, 280000, 300000, 296000]           # USED + FRESH of examples/43, for the seed
K = 7919 * (ORDER.index(SNAP) + 1) + 100 * R + J + 100000 * list(SIMULATIONS).index(SIM)
CONFIGS = {"Default treatment": (1e-2, 1e-2, 1e-1), "Recommended treatment": (1e-1, 1e-2, 1e-2)}
OUT_FULL, OUT_WEB = os.path.join("docs", "informes", "simulations"), os.path.join("docs", "web", "img", "realistic")
os.makedirs(OUT_FULL, exist_ok=True)
os.makedirs(OUT_WEB, exist_ok=True)
INK, MUTED, RIGHT, WRONG, TRUE = "#1f2328", "#6e7781", "#2f6fdf", "#e8710a", "#3a3f44"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name):
    full = os.path.join(OUT_FULL, name)
    fig.savefig(full, dpi=a.dpi, bbox_inches="tight")
    from PIL import Image
    im = Image.open(full).convert("RGB")
    w = min(1600, im.width)
    im.resize((w, round(im.height * w / im.width)), Image.LANCZOS).save(os.path.join(OUT_WEB, name), optimize=True)
    print("written", full, im.size)


def truth():
    path = os.path.join(a.data, f"truth_{SIM}_{SNAP}_r{R}.npz")
    if not os.path.exists(path):
        f = reduced_field(SIM, SNAP, R, a.raw)
        np.savez_compressed(path, bx=f.bx.astype(np.float32), by=f.by.astype(np.float32), bz=f.bz.astype(np.float32), dx=f.dx, origin=np.array(f.origin))
    d = np.load(path)
    return VectorField3D(d["bx"].astype(float), d["by"].astype(float), d["bz"].astype(float), float(d["dx"]), (float(d["origin"][0]), float(d["origin"][1]), 0.0))


# ------------------------------------------------------------------ figure 1: the lines
exact = truth()
nz = exact.shape[2]
sz, st = list(NOISE_MODELS.values())[J]
mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=K)
seeds = seeds_from_magnetogram(mag, n=300, method="random", polarity="both", threshold=0.05, rng=K + 7)
true_lines = trace_field_lines(exact, seeds)
cache = os.path.join(a.data, f"simulation_figures_{SNAP}_r{R}.pkl")
models = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
with keep_awake():
    for name, (mu3, mu4, nu) in CONFIGS.items():
        if name not in models:
            models[name] = nlfff_lbfgs(preprocess(mag, mu3=mu3, mu4=mu4, verbose=False), nz, relax_bottom=nu, max_iter=4000, verbose=False)
            pickle.dump(models, open(cache, "wb"))
lines = {n: trace_field_lines(m, seeds) for n, m in models.items()}
wrong = {}
for n, ls in lines.items():
    e = line_errors(ls, true_lines)
    wrong[n] = (np.asarray(e["shape"], float) > 0.10) | (np.asarray(e["topology"]) > 0)
    print(f"{n}: {100 * wrong[n].mean():.1f} % wrong of {len(ls)}")

# the drawn lines: closed true lines, the same for both models, spread over the region
rng = np.random.default_rng(3)
cand = [i for i, l in enumerate(true_lines) if l.closed and len(l.points) > 8]
shown = np.sort(rng.choice(cand, size=min(a.shown, len(cand)), replace=False))
x0, y0 = exact.origin[0], exact.origin[1]
ext = (x0, x0 + exact.shape[0] * exact.dx, y0, y0 + exact.shape[1] * exact.dx)
bz = exact.bz[:, :, 0].T
vmax = np.percentile(np.abs(bz), 99)
zmax = max(max(l.points[:, 2].max() for l in true_lines if len(l.points)), 1) * 1.05

fig, ax = plt.subplots(2, 2, figsize=(12, 7.2), gridspec_kw={"height_ratios": [1.0, 0.55]}, layout="constrained")
for c, (n, ls) in enumerate(lines.items()):
    top, side = ax[0, c], ax[1, c]
    top.imshow(bz, origin="lower", extent=ext, cmap="gray", vmin=-vmax, vmax=vmax, alpha=0.55)
    for i in shown:
        t, m = true_lines[i].points, ls[i].points
        col = WRONG if wrong[n][i] else RIGHT
        top.plot(t[:, 0], t[:, 1], color=TRUE, lw=0.6, alpha=0.55)
        side.plot(t[:, 0], t[:, 2], color=TRUE, lw=0.6, alpha=0.55)
        top.plot(m[:, 0], m[:, 1], color=col, lw=1.0)
        side.plot(m[:, 0], m[:, 2], color=col, lw=1.0)
    top.set_title(f"{n}: {100 * wrong[n].mean():.0f} % of the lines wrong", color=INK, fontsize=11)
    top.set_xlim(ext[:2]), top.set_ylim(ext[2:]), top.set_aspect("equal")
    side.set_xlim(ext[:2]), side.set_ylim(0, zmax)
    top.set_ylabel("y [Mm]"), side.set_ylabel("z [Mm]"), side.set_xlabel("x [Mm]")
handles = [Line2D([], [], color=TRUE, lw=1, alpha=0.7, label="true line (simulation)"),
           Line2D([], [], color=RIGHT, lw=1.5, label="model line, right"),
           Line2D([], [], color=WRONG, lw=1.5, label="model line, wrong (error > 10 % of its length)")]
fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
fig.suptitle(f"MURaM active region, snapshot {SNAP}: the same seeds in the truth and in two extrapolations "
             f"(pixel {exact.dx:.2f} Mm, HMI-like noise; {len(shown)} of {len(seeds)} lines drawn)", color=INK, fontsize=11)
save(fig, "lines_truth_default_recommended.png")
plt.close(fig)

# ------------------------------------------------------------------ figure 2: the weight of the base
F = json.load(open(os.path.join("docs", "resultados", "base_weight_regimes.json"), encoding="utf-8"))
REGIMES = [("photosphere, 1.54 Mm", "Simulated photosphere, 1.54 Mm", "#2f6fdf", "-"),
           ("photosphere, 0.77 Mm", "Simulated photosphere, 0.77 Mm", "#2f6fdf", "--"),
           ("base at 1.5 Mm, 1.54 Mm", "Simulation, base at 1.5 Mm", "#7a4fd6", "-"),
           ("exact families, uniforme-10G", "Exact force-free fields, 10 G noise", "#1a9e77", "-"),
           ("exact families, HMI (10/60 G)", "Exact force-free fields, HMI-like noise", "#e8710a", "-")]
fig, axw = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
for key, label, col, ls in REGIMES:
    scan = F["groups"][key]["scan"]
    w = np.array(sorted(map(float, scan)))
    y = np.array([100 * scan[str(v) if str(v) in scan else repr(v)] for v in w])
    axw.plot(w, y, color=col, ls=ls, lw=2, marker="o", ms=4, label=label)
    b = w[np.argmin(y)]
    axw.plot([b], [y.min()], marker="o", ms=9, mfc="white", mec=col, mew=2)
axw.set_xscale("log")
axw.set_xlabel(r"weight of the base term, relax_bottom ($\nu$)")
axw.set_ylabel("wrong lines [%]")
axw.axvline(0.1, color=MUTED, lw=1, ls=":")
axw.text(0.1, axw.get_ylim()[1] * 0.98, " default", color=MUTED, va="top")
axw.grid(alpha=0.25)
axw.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
axw.set_title("The best weight of the base depends on how force-free the base is\n"
              "(no preprocessing; open circle: best weight of each regime)", color=INK, fontsize=10.5)
save(fig, "base_weight_by_regime.png")
plt.close(fig)
