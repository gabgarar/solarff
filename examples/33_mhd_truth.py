"""A radiative MHD simulation as the truth: does the confidence hold against a realistic active region?

The reference fields of the paper are exact force-free solutions, magnetohydrostatic solutions, fields built on the
observed flux and the models of the regions themselves. None has a photosphere with convection, out of force balance,
under a dynamic corona. A published simulation that has them is taken here as the truth:

- run ``ar098192`` of the MURaM code (Cheung et al. 2019, Nature Astronomy 3, 160), an active region with a flare,
  from the public catalogue of the Hinode Science Data Centre Europe (https://sdc.uio.no/search/simulations). The
  cubes of the magnetic field (403 MB each) are downloaded to ``--raw`` if they are not there;
- the truth is the field above the photosphere, the plane z = 0 of the run. It is averaged in blocks of r x r pixels
  horizontally, as a magnetogram is rebinned, and sampled every 3r planes in height, so that the cells are cubic and
  the base is the photospheric plane itself and not an average with higher layers;
- each case as in the suite: the base with noise (three noise models), the same preprocessing and code, 300 random
  seeds, lines traced in the model and in the truth, and the published coefficients applied without refitting;
- the energies of the model and of the truth, each with its own volume potential;
- the same labels with the truth at the pixel of the simulation (r = 1) instead of averaged to the working pixel.

What this test is not: one simulation is one configuration, not a family, and its corona is not in equilibrium.

The criteria (M1 to M6) were written down in the plan of the paper before running (18-Nov-2026). After seeing the
result, an exploratory block was added (``exploratory`` in the JSON): how the result changes with the definition of a
wrong line, what each indicator ranks on its own, and what the indicators reach when they are refitted on the
simulation itself, leaving one snapshot out.

Saves ``data/mhd_truth.pkl`` (checkpoint, with the per-line records), the reduced truths
``data/mhd_truth_<snapshot>_r<r>.npz`` and ``docs/resultados/mhd_truth.json``.

    python -u examples/33_mhd_truth.py                    # downloads 3.6 GB the first time
    python -u examples/33_mhd_truth.py --raw D:/muram     # the cubes kept somewhere else
    python examples/33_mhd_truth.py --analyze-only
"""
import argparse
import json
import os
import pickle
import time
import urllib.request
import warnings

import numpy as np

from solarff.analytic import field_magnetogram
from solarff.calibration import NOISE_MODELS, ece
from solarff.confidence import (ALL_FEATURES, CALIBRATION_V1, DEFAULT_CALIBRATION, _design, _fit_logistic, auc, calibration_case, confidence,
                                line_errors)
from solarff.field import VectorField3D
from solarff.helicity import energy_decomposition
from solarff.magnetogram import aly_criteria
from solarff.metrics import force_free_metrics
from solarff.potential import potential_field
from solarff.tracing import seeds_from_magnetogram, trace_field_lines
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=os.path.join("data", "muram_ar098192"), help="folder of the cubes of the simulation")
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "mhd_truth.json"))
ap.add_argument("--snapshots", nargs="*", type=int, default=[290000, 310000, 270000],
                help="before the flare, after it and an earlier time")
ap.add_argument("--r", nargs="*", type=int, default=[8, 4], help="pixels of the simulation averaged into one of the model")
ap.add_argument("--no-native", action="store_true", help="skip the labels with the truth at the pixel of the simulation")
ap.add_argument("--analyze-only", action="store_true")
a = ap.parse_args()

URL = "https://sdc.uio.no/vol/simulations/ar098192/atmos/MURaM_ar098192_{var}_{snap}.fits"
SIZE = 402670080                       # bytes of each cube
T = DEFAULT_CALIBRATION["threshold"]
CKPT = os.path.join(a.data, "mhd_truth.pkl")
N_BOOT = 1000
PRE, POST = 290000, 310000            # the two times between which the change of the energy is measured


# ------------------------------------------------------------------ the cubes of the simulation
def cube(var, snap):
    """Path of a cube of the simulation, downloaded if it is not in ``--raw``."""
    path = os.path.join(a.raw, f"MURaM_ar098192_{var}_{snap}.fits")
    if not (os.path.exists(path) and os.path.getsize(path) == SIZE):
        os.makedirs(a.raw, exist_ok=True)
        print(f"  downloading {os.path.basename(path)} (403 MB)", flush=True)
        urllib.request.urlretrieve(URL.format(var=var, snap=snap), path + ".part")
        assert os.path.getsize(path + ".part") == SIZE, f"{path}: incomplete download"
        os.replace(path + ".part", path)
    return path


def reduce(snap, r):
    """The field above the photosphere on cubic cells of r pixels: block mean in x and y, one plane of every 3r in z.

    Returns the field (G, Mm) and the time of the snapshot (s).
    """
    from astropy.io import fits
    comp = {}
    for var in ("bx", "by", "bz"):
        with fits.open(cube(var, snap), memmap=True) as h:
            hd, z = h[0].header, np.asarray(h[1].data, float)
            assert hd["BUNIT"].strip() == "T" and abs(hd["CDELT1"] - hd["CDELT2"]) < 1e-6 and np.allclose(np.diff(z), hd["DZ"], rtol=1e-3)
            assert abs(hd["CDELT1"] / hd["DZ"] - 3) < 1e-3, "the cells are not three planes high"
            k0 = int(np.argmin(np.abs(z)))                      # the photosphere
            planes = np.arange(k0, len(z), 3 * r)
            d = np.stack([np.asarray(h[0].data[k], np.float32) for k in planes])        # (z, y, x), tesla
            nz, ny, nx = d.shape
            d = d[:, :ny - ny % r, :nx - nx % r].reshape(nz, ny // r, r, nx // r, r).mean(axis=(2, 4))
            comp[var] = 1e4 * d.transpose(2, 1, 0)             # (x, y, z), gauss
            dx = float(hd["CDELT1"]) * r
            origin = (float(hd["CRVAL1"]) + 0.5 * (r - 1) * hd["CDELT1"], float(hd["CRVAL2"]) + 0.5 * (r - 1) * hd["CDELT2"], float(z[k0]))
            elapsed = float(hd["ELAPSED"])
    f = VectorField3D(comp["bx"].astype(float), comp["by"].astype(float), comp["bz"].astype(float), dx, origin,
                      {"method": f"MURaM ar098192, snapshot {snap}, r = {r}"})
    return f, elapsed


def truth(snap, r):
    """Reduced truth, kept in ``data`` so that the cubes are read once."""
    path = os.path.join(a.data, f"mhd_truth_{snap}_r{r}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return VectorField3D(d["bx"].astype(float), d["by"].astype(float), d["bz"].astype(float), float(d["dx"]), tuple(d["origin"]),
                             {"method": f"MURaM ar098192, snapshot {snap}, r = {r}"}), float(d["elapsed"])
    f, elapsed = reduce(snap, r)
    np.savez_compressed(path, bx=f.bx.astype(np.float32), by=f.by.astype(np.float32), bz=f.bz.astype(np.float32), dx=f.dx,
                        origin=np.array(f.origin), elapsed=elapsed)
    return f, elapsed


def ratio(nx, ny, nz):
    return (nz - 1) / np.hypot(nx - 1, ny - 1)


def global_metrics(mag, pp, model, nz):
    """The quantities of the applicability check of the paper, as in ``examples/18_applicability.py``."""
    ff = force_free_metrics(model, margin=int(model.meta.get("nd", 0)))
    P = potential_field(pp, nz)
    B = np.sqrt(P.bx ** 2 + P.by ** 2 + P.bz ** 2)
    base = B[:, :, 0].mean()
    side = np.mean([B[0].mean(), B[-1].mean(), B[:, 0].mean(), B[:, -1].mean()])
    top = B[:, :, -1].mean()
    b0, b1 = np.stack([mag.bx, mag.by, mag.bz]), np.stack([pp.bx, pp.by, pp.bz])
    return {"cwsin": float(ff["CWsin"]), "boundary_weight": float((4 * side + top) / 5 / base),
            "pp_change": float(np.linalg.norm(b1 - b0) / np.linalg.norm(b0))}


def energies(f):
    e = energy_decomposition(f)
    return {k: float(e[k]) for k in ("E", "E_p", "E_J", "E_free", "E_div_frac")}


# ------------------------------------------------------------------ the cases
state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {"cases": {}, "truth": {}}
if not a.analyze_only:
    with keep_awake():
        t0 = time.time()
        for si, snap in enumerate(a.snapshots):
            native = None
            for r in a.r:
                group = f"{snap} | r{r}"
                todo = [nm for nm in NOISE_MODELS if f"{group} | {nm}" not in state["cases"]]
                if group in state["truth"] and not todo:
                    continue
                tr, elapsed = truth(snap, r)
                nz = tr.shape[2]
                if group not in state["truth"]:
                    state["truth"][group] = {"snapshot": int(snap), "r": int(r), "elapsed_s": elapsed, "grid": [int(n) for n in tr.shape],
                                             "dx_Mm": float(tr.dx), "height_over_diagonal": float(ratio(*tr.shape)),
                                             "cwsin": float(force_free_metrics(tr)["CWsin"]),
                                             "eps_force": float(aly_criteria(field_magnetogram(tr, 0.0, 0.0))["eps_force"]),
                                             "bz_max_G": float(np.abs(tr.bz[:, :, 0]).max()), "energy": energies(tr)}
                    pickle.dump(state, open(CKPT, "wb"))
                for nm in todo:
                    sz, st = NOISE_MODELS[nm]
                    tc = time.time()
                    k = 7919 * (si + 1) + 100 * r + list(NOISE_MODELS).index(nm)
                    mag = field_magnetogram(tr, noise=sz, noise_t=st, rng=k)
                    rec = calibration_case(tr, mag, nz, n_seeds=300, seed=k + 7, noise_bt=st, nlfff_kw={"max_iter": 4000}, keep_model=True)
                    model, pp = rec.pop("model"), rec.pop("pp")
                    rec.update(snapshot=int(snap), r=int(r), noise_model=nm, metrics=global_metrics(mag, pp, model, nz), energy=energies(model),
                               iterations=int(model.meta.get("iterations", 0)))
                    rec["metrics"]["e_div_frac"] = rec["energy"]["E_div_frac"]
                    if not a.no_native:
                        # the same model lines against the truth at the pixel of the simulation, cropped to the same box
                        if native is None:
                            native, _ = reduce(snap, 1)
                        seeds = seeds_from_magnetogram(mag, n=300, method="random", polarity="both", threshold=0.05, rng=k + 7)
                        en = line_errors(trace_field_lines(model, seeds), trace_field_lines(native, seeds))
                        rec["err_native"] = en
                        rec["bad_native"] = (en["shape"] > 0.1) | (en["topology"] > 0)
                    rec["seconds"] = time.time() - tc
                    state["cases"][f"{group} | {nm}"] = rec
                    pickle.dump(state, open(CKPT, "wb"))
                    print(f"  {group} | {nm}: {rec['bad'].mean():.0%} wrong lines"
                          + (f" ({rec['bad_native'].mean():.0%} against the truth at r = 1)" if "bad_native" in rec else "")
                          + f", CWsin {rec['metrics']['cwsin']:.2f} ({rec['seconds']:.0f} s; total {(time.time() - t0) / 60:.0f} min)", flush=True)


# ------------------------------------------------------------------ analysis
def evaluate(d, rule="default", label="bad"):
    """``default``: the published model (five terms). ``v1``: that of version 1.1 (four terms)."""
    cal = DEFAULT_CALIBRATION if rule == "default" else CALIBRATION_V1
    cs = [confidence(x["ind"], cal) for x in d]
    c = np.concatenate(cs)
    y = np.concatenate([np.asarray(x[label]) for x in d]).astype(int)
    ok = np.isfinite(c)
    acc = ok & (np.nan_to_num(c, nan=0.0) >= T)
    p = 1 - c[ok]
    out = {"n_cases": len(d), "n_lines": int(ok.sum()), "bad_rate": float(y[ok].mean()), "auc": auc(p, y[ok]), "ece": ece(p, y[ok]),
           "predicted_bad": float(p.mean()), "accepted": float(acc[ok].mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None}
    if acc.any():
        rng = np.random.default_rng(0)
        if len(d) > 1:      # whole extrapolations are resampled: the lines of one extrapolation are not independent
            ya = [np.asarray(x[label]).astype(int)[np.nan_to_num(ci, nan=0.0) >= T] for x, ci in zip(d, cs)]
            b = []
            for _ in range(N_BOOT):
                pick = np.concatenate([ya[i] for i in rng.integers(0, len(ya), len(ya))])
                if pick.size:
                    b.append(pick.mean())
        else:
            ya = y[acc]
            b = [ya[rng.integers(0, ya.size, ya.size)].mean() for _ in range(N_BOOT)]
        out["bad_accepted_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    m = ok & (np.nan_to_num(c, nan=0.0) >= 0.9)
    out["accepted_0.9"], out["bad_accepted_0.9"] = float(m[ok].mean()), (float(y[m].mean()) if m.any() else None)
    return out


def block(d):
    o = {"default": evaluate(d), "v1": evaluate(d, "v1")}
    if all("bad_native" in x for x in d):
        o["default_native_truth"] = evaluate(d, label="bad_native")
    return o


cases = state["cases"]
out = {"code_version": code_version()["label"], "calibration": "DEFAULT_CALIBRATION (five terms) and CALIBRATION_V1 (four terms, version 1.1), not refitted",
       "source": {"simulation": "MURaM ar098192 (Cheung et al. 2019, Nature Astronomy 3, 160)", "catalogue": "https://sdc.uio.no/search/simulations",
                  "variables": ["bx", "by", "bz"]},
       "noise_models": list(NOISE_MODELS), "truth": state["truth"], "groups": {}, "by_r": {}, "by_snapshot": {}, "pooled": {}, "cases": {}, "energy_change": {}}
for key, x in cases.items():
    out["cases"][key] = {"bad_rate": float(np.mean(x["bad"])), "metrics": x["metrics"], "energy": x["energy"], "iterations": x.get("iterations"),
                         "seconds": x["seconds"], **({"bad_rate_native_truth": float(np.mean(x["bad_native"]))} if "bad_native" in x else {})}
for group in state["truth"]:
    d = [x for k, x in cases.items() if k.startswith(group + " |")]
    if d:
        out["groups"][group] = block(d)
        out["groups"][group]["cwsin"] = [min(x["metrics"]["cwsin"] for x in d), max(x["metrics"]["cwsin"] for x in d)]
for r in sorted({x["r"] for x in cases.values()}):
    out["by_r"][str(r)] = block([x for x in cases.values() if x["r"] == r])
for s in sorted({x["snapshot"] for x in cases.values()}):
    out["by_snapshot"][str(s)] = block([x for x in cases.values() if x["snapshot"] == s])
if cases:
    out["pooled"] = block(list(cases.values()))
    out["pooled"]["metrics_range"] = {q: [min(x["metrics"][q] for x in cases.values()), max(x["metrics"][q] for x in cases.values())]
                                      for q in ("cwsin", "e_div_frac", "boundary_weight")}
# change of the energy between the two times, in the truth and in the model (per noise model)
for r in sorted({x["r"] for x in cases.values()}):
    g0, g1 = f"{PRE} | r{r}", f"{POST} | r{r}"
    if g0 in state["truth"] and g1 in state["truth"]:
        e0, e1 = state["truth"][g0]["energy"], state["truth"][g1]["energy"]
        ch = {"truth": {q: {"pre": e0[q], "post": e1[q], "change": e1[q] - e0[q], "relative": e1[q] / e0[q] - 1} for q in ("E_free", "E_J")}, "model": {}}
        for nm in NOISE_MODELS:
            if f"{g0} | {nm}" in cases and f"{g1} | {nm}" in cases:
                m0, m1 = cases[f"{g0} | {nm}"]["energy"], cases[f"{g1} | {nm}"]["energy"]
                ch["model"][nm] = {q: {"pre": m0[q], "post": m1[q], "change": m1[q] - m0[q], "relative": m1[q] / m0[q] - 1,
                                       "pre_over_truth": m0[q] / e0[q], "post_over_truth": m1[q] / e1[q]} for q in ("E_free", "E_J")}
        out["energy_change"][str(r)] = ch


# ------------------------------------------------------------------ exploratory: added after seeing the registered result
def explore(d):
    """Not part of the registered test. The registered definition of a wrong line is the one of 10 %."""
    cat = lambda f: np.concatenate([np.asarray(f(x), float) for x in d])  # noqa: E731
    c = cat(lambda x: confidence(x["ind"], DEFAULT_CALIBRATION))
    sh, ab, tp = cat(lambda x: x["err"]["shape"]), cat(lambda x: x["err"]["shape_abs"]), cat(lambda x: x["err"]["topology"]) > 0
    ok = np.isfinite(c)
    acc = ok & (np.nan_to_num(c, nan=0.0) >= T)
    defs = {"rel5": (sh > 0.05) | tp, "rel10": (sh > 0.10) | tp, "rel20": (sh > 0.20) | tp, "rel30": (sh > 0.30) | tp,
            "abs2mm": (ab > 2.0) | tp, "abs5mm": (ab > 5.0) | tp, "open_closed": tp}
    o = {"definitions": {n: {"bad_rate": float(y[ok].mean()), "auc": auc(1 - c[ok], y[ok]), "bad_accepted": float(y[acc].mean())} for n, y in defs.items()},
         "shape_error_median_same_topology": {"all": float(np.nanmedian(sh[ok & ~tp])), "accepted": float(np.nanmedian(sh[acc & ~tp])),
                                              "rejected": float(np.nanmedian(sh[ok & ~acc & ~tp]))}}
    feats = list(ALL_FEATURES) + [f for f in DEFAULT_CALIBRATION["features"] if f not in ALL_FEATURES]
    X = _design({f: cat(lambda x, f=f: x["ind"][f]) for f in feats if f != "length_sq"}, feats)
    o["single_indicator_auc"] = {f: auc(X[np.isfinite(X[:, j]), j], defs["rel10"][np.isfinite(X[:, j])]) for j, f in enumerate(feats)}
    snap = np.concatenate([np.full(len(x["bad"]), x["snapshot"]) for x in d])
    o["refit_leave_one_snapshot_out"] = {}
    if len(np.unique(snap)) > 1:
        for tag, use in (("five_terms", DEFAULT_CALIBRATION["features"]), ("all_indicators", feats)):
            Xs = X[:, [feats.index(f) for f in use]]
            good = np.isfinite(Xs).all(axis=1)
            o["refit_leave_one_snapshot_out"][tag] = {}
            for n in ("rel10", "rel20"):
                y = defs[n].astype(float)
                pr = np.full(len(y), np.nan)
                for sn in np.unique(snap):
                    tr, te = good & (snap != sn), good & (snap == sn)
                    fit = _fit_logistic(Xs[tr], y[tr])
                    pr[te] = 1 / (1 + np.exp(-((Xs[te] - np.array(fit["mu"])) / np.array(fit["sd"]) @ np.array(fit["coef"]) + fit["intercept"])))
                half = good & (pr <= np.nanmedian(pr[good]))
                o["refit_leave_one_snapshot_out"][tag][n] = {"auc": auc(pr[good], y[good] > 0), "bad_rate": float(y[good].mean()),
                                                             "bad_in_better_half": float(y[half].mean())}
    return o


if cases:
    out["exploratory"] = {"note": "added after seeing the registered result (18-Nov-2026); not part of the registered test",
                          "pooled": explore(list(cases.values())),
                          "by_r": {str(r): explore([x for x in cases.values() if x["r"] == r]) for r in sorted({x["r"] for x in cases.values()})}}
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def fmt(v, f=".3f"):
    return "-" if v is None else format(v, f)


def line(name, q):
    print(f"  {name:28s} wrong {q['bad_rate']:.2f} (predicted {q['predicted_bad']:.2f}) | AUC {q['auc']:.3f} | ECE {q['ece']:.3f} | accepted {q['accepted']:.2f}, "
          f"wrong among them {fmt(q['bad_accepted'])} {[round(v, 3) for v in q.get('bad_accepted_ci', [])]} (0.9: {fmt(q['bad_accepted_0.9'])}, accepted {q['accepted_0.9']:.2f})")


for group, g in out["groups"].items():
    e = out["truth"][group]
    print(f"{group}: t = {e['elapsed_s']:.0f} s, grid {e['grid']}, pixel {e['dx_Mm']:.2f} Mm, height / diagonal {e['height_over_diagonal']:.2f}, "
          f"CWsin of the truth {e['cwsin']:.2f}, force residual of its base {e['eps_force']:.2f}, CWsin of the models {g['cwsin'][0]:.2f}-{g['cwsin'][1]:.2f}")
    for name in ("default", "v1", "default_native_truth"):
        if name in g:
            line(name, g[name])
if out["pooled"]:
    print("all the cases:")
    for name in ("default", "v1", "default_native_truth"):
        if name in out["pooled"]:
            line(name, out["pooled"][name])
    print("  ranges of the models:", {k: [round(x, 3) for x in v] for k, v in out["pooled"]["metrics_range"].items()})
for r, ch in out["energy_change"].items():
    for q in ("E_free", "E_J"):
        t = ch["truth"][q]
        print(f"r = {r}, {q}: truth {t['pre']:.3e} -> {t['post']:.3e} erg ({t['relative']:+.1%}) | model " +
              "; ".join(f"{nm}: {m[q]['relative']:+.1%}, {m[q]['pre_over_truth']:.2f} and {m[q]['post_over_truth']:.2f} of the truth" for nm, m in ch["model"].items()))
if cases:
    ex = out["exploratory"]["pooled"]
    print("exploratory (after the result): definition of a wrong line -> wrong, AUC, wrong among the accepted: " +
          "; ".join(f"{n}: {q['bad_rate']:.2f}, {q['auc']:.3f}, {q['bad_accepted']:.3f}" for n, q in ex["definitions"].items()))
    print("  each indicator alone (AUC): " + ", ".join(f"{f} {v:.2f}" for f, v in ex["single_indicator_auc"].items()))
    for tag, q in ex["refit_leave_one_snapshot_out"].items():
        print(f"  refitted on the simulation, one snapshot left out, {tag}: " +
              "; ".join(f"{n}: AUC {v['auc']:.3f}, wrong in the better half {v['bad_in_better_half']:.2f} of {v['bad_rate']:.2f}" for n, v in q.items()))
print("written", a.json)
