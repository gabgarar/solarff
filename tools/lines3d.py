"""Interactive 3D view of the field lines of one or more extrapolated regions, in a single HTML file.

Every line traced from the base is drawn in 3D over the magnetogram, coloured by its apex height, by its confidence
or by the confidence model that accepts it (the current one, with five terms, or that of version 1.1, with four).
Hovering over a line shows its confidence. The horizontal field can be drawn as arrows: that of the magnetogram,
the preprocessed one, or the extrapolated one at any height, over a plane with B_z at that height. The interface is in
English or Spanish and the page works offline: three.js r147 (MIT licence) is put inside it.

    python tools/lines3d.py                                   # the three regions of the paper
    python tools/lines3d.py my_study.pkl -o my_region.html    # any study
    python tools/lines3d.py a.pkl b.pkl --lang es --every 1

Each input is a pickle with a ``solarff.Study`` that has an NLFFF extrapolation, alone (``pickle.dump(study, fh)``)
or as the first element of a tuple, as ``examples/02_hmi_sharp.py`` writes it. Only open pickles you trust.
The lines are traced as in the paper: seeds every ``--every`` pixels in x and in y where |B_z| at the base exceeds 5 %
of its maximum, and the confidence is that of ``solarff.assess_lines``.

The first run downloads three.js from jsDelivr into ``data/three-0.147.0/`` and checks its SHA-256;
``--three DIR`` uses a local copy of those two files instead.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import pickle
import re
import sys
import urllib.request
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import solarff  # noqa: E402
import solarff.plotting as P  # noqa: E402
from solarff.confidence import CALIBRATION_V1, DEFAULT_CALIBRATION, assess_lines, confidence  # noqa: E402
from solarff.montecarlo import base_seeds  # noqa: E402
from solarff.study import Study  # noqa: E402
from solarff.tracing import trace_field_lines  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "tools", "lines3d.html")
PAPER_REGIONS = [os.path.join(ROOT, "data", p) for p in (
    "harp377_study_mc10.pkl", "harp1449_20120306_2336_r7_nz32_mc10.pkl", "harp7115_20170906_1136_r4_nz56_mc10.pkl")]
SEED_THRESHOLD = 0.05          # seeds where |B_z| > 5 % of its maximum at the base, as in the paper
MAX_ARROWS = 56                # arrows along the longer side of the box (block means of k x k pixels)
MAX_POINTS = 160               # points per line in the page, evenly spaced along it
THREE_VERSION = "0.147.0"
THREE_FILES = {                # file -> (path in the npm package, SHA-256)
    "three.min.js": ("build/three.min.js", "f34446bf875b5fb0dcd93819ffe1d9e182d46634ee855f5d904c6c4ac7cdbc95"),
    "OrbitControls.js": ("examples/js/controls/OrbitControls.js",
                         "b4c6e53f98538535b11fd6655627d47fb5877b3ad972568bff0a7026c4b6d5c4"),
}
MARKERS = ("/*THREE*/", "/*ORBIT*/", "/*DATA*/")


def load_study(path):
    """The ``Study`` inside a pickle: the object itself or the first ``Study`` of a tuple or list."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    for o in (obj, *(obj if isinstance(obj, (tuple, list)) else ())):
        if isinstance(o, Study):
            if o.nlfff is None:
                raise ValueError(f"{path}: the study has no NLFFF extrapolation")
            return o
    raise ValueError(f"{path}: there is no solarff.Study in this pickle")


def region_label(st, path):
    """Short key (the NOAA number, else the name of the study or of the file) and a longer label for the page."""
    m = getattr(st.mag, "meta", None) or {}
    noaa = m.get("NOAA_AR")
    key = str(noaa) if noaa else (st.name or os.path.splitext(os.path.basename(path))[0])
    parts = [f"NOAA {noaa}"] if noaa else [st.name or key]
    if m.get("HARPNUM"):
        parts.append(f"HARP {m['HARPNUM']}")
    t = re.match(r"(\d{4})\.(\d\d)\.(\d\d)_(\d\d):(\d\d)", str(m.get("T_REC") or ""))
    if t:
        parts.append("{}-{}-{} {}:{} TAI".format(*t.groups()))
    return key, " · ".join(parts)


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _png(img, vlim):
    """Grey image of B_z (black negative, white positive), x to the right and y up."""
    buf = io.BytesIO()
    plt.imsave(buf, np.nan_to_num(img).T, cmap="gray", vmin=-vlim, vmax=vlim, origin="lower", format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resample(p, step):
    """Points evenly spaced along the line, between 8 and ``MAX_POINTS``."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    n = int(np.clip(s[-1] / step, 8, MAX_POINTS))
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1)


def export_region(st, every=2):
    """Everything the page needs from one study: lines, confidence, magnetogram, inversion line and vector fields."""
    f = st.nlfff
    src = st.mag_pp if st.mag_pp is not None else st.mag
    lines = trace_field_lines(f, base_seeds(f, every, SEED_THRESHOLD))
    a = assess_lines(f, lines, mag=src if src.has_vector else None,
                     noise_bt=float(st.mag.meta.get("noise") or 100.0))
    thr = DEFAULT_CALIBRATION["threshold"]
    c5 = np.nan_to_num(np.asarray(a["confidence"], float), nan=0.0)
    c4 = np.nan_to_num(np.asarray(confidence(a["indicators"], CALIBRATION_V1), float), nan=0.0)
    dx = float(f.x[1] - f.x[0])
    ex, ey = P._edges(f.x), P._edges(f.y)
    box = [float(ex[0]), float(ex[1]), float(ey[0]), float(ey[1]), 0.0, float(f.z[-1])]

    # lines, as int16 in units of 1/q Mm, q as fine as the largest coordinate allows (at most 0.02 Mm)
    res = [_resample(np.asarray(l.points, float), 0.5 * dx) for l in lines]
    big = max([np.abs(box).max()] + [np.abs(r).max() for r in res])
    q = min(50.0, 32000.0 / big)
    pts = np.round(np.concatenate(res) * q).astype(np.int16) if res else np.zeros((0, 3), np.int16)

    # magnetogram at the base and polarity inversion line, as in the paper figures
    bz0 = f.bz[:, :, 0]
    pil, segs = P._pil_field(bz0), []
    if np.nanmin(pil) < 0 < np.nanmax(pil):
        X, Y = np.meshgrid(f.x, f.y, indexing="ij")
        cs = plt.contour(X, Y, pil, levels=[0.0])
        segs = [np.round(s, 2).tolist() for s in cs.allsegs[0] if len(s) > 4]
    plt.close("all")

    # vector fields, as block means of k x k pixels: the magnetogram and the preprocessed one if they are on the grid
    # of the extrapolation, and the extrapolation at every height with a grey image of B_z at that height
    nx, ny, nz = f.bx.shape
    k = int(np.ceil(max(nx, ny) / MAX_ARROWS))
    nxd, nyd = nx // k, ny // k

    def block(c):
        c = np.nan_to_num(np.asarray(c, float))
        return c[:nxd * k, :nyd * k].reshape(nxd, k, nyd, k).mean(axis=(1, 3))

    def pack(bx, by, bz):
        v = np.stack([block(bx), block(by), block(bz)], axis=-1)
        s = float(np.abs(v).max() / 30000.0) or 1.0
        return s, np.round(v / s).astype(np.int16)

    ext = [pack(f.bx[:, :, j], f.by[:, :, j], f.bz[:, :, j]) for j in range(nz)]
    vec = {"k": k, "nx": nxd, "ny": nyd, "d": k * dx, "x0": float(f.x[:k].mean()), "y0": float(f.y[:k].mean()),
           "z": [round(float(z), 3) for z in f.z],
           "ext": {"scale": [s for s, _ in ext], "data": _b64(np.stack([v for _, v in ext])),
                   "png": [_png(f.bz[:, :, j], P._bz_vlim(f.bz[:, :, j]) or 1.0) for j in range(nz)]}}
    for name, m in (("obs", st.mag), ("pp", st.mag_pp)):
        if m is not None and m.has_vector and np.shape(m.bz) == (nx, ny):
            s, v = pack(m.bx, m.by, m.bz)
            vec[name] = {"scale": s, "data": _b64(v)}

    return {
        "box": box, "pixel_mm": dx, "q": q, "vec": vec,
        "bz_png": _png(bz0, P._bz_vlim(bz0) or 1.0), "pil": segs,
        "points": _b64(pts), "counts": [len(r) for r in res],
        "conf5": np.round(c5, 4).tolist(), "conf4": np.round(c4, 4).tolist(),
        "acc5": (c5 >= thr).astype(int).tolist(),        # decided before rounding
        "acc4": (c4 >= thr).astype(int).tolist(),
        "apex": [round(float(np.nan_to_num(l.apex)), 2) for l in lines],
        "length": [round(float(np.nan_to_num(l.length)), 1) for l in lines],
        "closed": [bool(l.closed) for l in lines],
    }


def three_js(folder=None):
    """Text of three.min.js and OrbitControls.js of three.js r147, downloaded once and checked against SHA-256."""
    given = folder is not None
    folder = folder or os.path.join(ROOT, "data", f"three-{THREE_VERSION}")
    out = []
    for name, (path, sha) in THREE_FILES.items():
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            if given:
                raise FileNotFoundError(f"{p} does not exist")
            url = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/{path}"
            print("downloading", url, flush=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "solarff-lines3d"}),
                                        timeout=60) as r:
                data = r.read()
            if hashlib.sha256(data).hexdigest() != sha:
                raise RuntimeError(f"{url}: unexpected SHA-256, the file was not saved")
            os.makedirs(folder, exist_ok=True)
            with open(p + ".tmp", "wb") as fh:
                fh.write(data)
            os.replace(p + ".tmp", p)
        data = open(p, "rb").read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"{p}: unexpected SHA-256 (not three.js {THREE_VERSION}); delete it to download it again")
        out.append(data.decode("utf-8"))
    return out


def build_html(regions, three, orbit, lang="en", every=2):
    """The page: the template with three.js, OrbitControls and the regions ({key: export_region(...) + label})."""
    data = {"threshold": DEFAULT_CALIBRATION["threshold"], "version": solarff.__version__, "every": every,
            "seed_threshold": SEED_THRESHOLD, "lang": lang, "regions": regions}
    js = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    tpl = open(TEMPLATE, encoding="utf-8").read()
    parts = dict(zip(MARKERS, (three, orbit, js)))
    for marker, text in parts.items():
        assert tpl.count(marker) == 1, marker
        if re.search(r"</script", text, re.I):
            raise ValueError(f"the text for {marker} would close its <script>")
    # split on the markers of the template only, so that no inserted text is searched again
    return "".join(parts.get(p, p) for p in re.split("(" + "|".join(map(re.escape, MARKERS)) + ")", tpl))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pickles", nargs="*", help="pickles with a solarff.Study (default: the three regions of the paper)")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "docs", "informes", "lines3d.html"),
                    help="HTML file to write (default: docs/informes/lines3d.html)")
    ap.add_argument("--lang", choices=("en", "es"), default="en", help="language of the page when it opens")
    ap.add_argument("--every", type=int, default=2, help="seed spacing in pixels, in x and in y (default 2, as in the paper)")
    ap.add_argument("--three", metavar="DIR", help=f"folder with three.min.js and OrbitControls.js of three.js "
                                                   f"{THREE_VERSION} (default: downloaded once into data/)")
    args = ap.parse_args(argv)
    if args.every < 1:
        ap.error("--every must be 1 or more")
    warnings.filterwarnings("ignore")
    three, orbit = three_js(args.three)          # first, so that a failed download does not waste the tracing
    regions = {}
    for path in args.pickles or PAPER_REGIONS:
        st = load_study(path)
        key, label = region_label(st, path)
        base, n = key, 2
        while key in regions:
            key, n = f"{base} ({n})", n + 1
        R = export_region(st, args.every)
        R["label"] = label
        regions[key] = R
        print(f"{label}: {len(R['counts'])} lines, {sum(R['acc5'])} accepted (five terms), "
              f"{sum(R['acc4'])} (four terms)", flush=True)
    html = build_html(regions, three, orbit, args.lang, args.every)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    print(f"{args.out}: {len(html.encode()) / 1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
