r"""Reliability of each field line of a model.

For each traced line, indicators are computed that, without knowing the true
solution, flag whether that part of the model can be trusted:

| Indicator | Meaning |
|---|---|
| ``sin_jb`` | mean sine of the J–B angle along the line, weighted by the current (0 = force-free) |
| ``alpha_spread`` | relative variation of α along the line (in a force-free field α is constant on each line) |
| ``div`` | mean relative divergence $\langle \lvert\nabla\cdot\mathbf{B}\rvert\,\Delta x / 6\lvert\mathbf{B}\rvert \rangle$ over the line |
| ``buffer`` | fraction of the line inside the buffer layer, where the (unobserved) boundary rules |
| ``open`` | 1 if the line leaves through a side or the top of the domain |
| ``foot_snr`` | observed transverse field at the weakest footpoint divided by its noise (the α carried by the line is measured at the footpoint) |
| ``base_change`` | relative correction of the transverse field that the model applied at the footpoints (relaxed base): how much it ignored the data there |
| ``apex`` | maximum height of the line / height of the box (high lines depend more on the unobserved top) |
| ``length`` | length of the line / horizontal diagonal of the box |
| ``length_h`` | length of the line / height of the box (a line that is long compared with the height depends on the unobserved top, whatever the horizontal size of the box) |

The indicators are combined into a **confidence** ∈ [0, 1] (1 − probability
that the line is wrong) by a logistic regression calibrated with exact
force-free fields with noise, where the true error of each line is
known. The default calibration (``DEFAULT_CALIBRATION``) was fitted
with the diverse suite of ``solarff.calibration`` (6 field families,
3 noise models, 3 resolutions and boxes of four proportions) and validated
leaving out each family.

Limitation: these are *necessary* conditions, not sufficient ones. A line can be
perfectly force-free and still be wrong if the data at its footpoint are
consistently erroneous; that is only detected by noise ensembles,
independent methods or coronal observations.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from .field import VectorField3D, cross
from .magnetogram import Magnetogram
from .nlfff import weight_function
from .tracing import FieldLine
from .utils import keeps_awake, load_checkpoint, save_checkpoint

ALL_FEATURES = ("sin_jb", "alpha_spread", "div", "buffer", "open", "foot_snr", "base_change",
                "apex", "length")
FEATURES = ALL_FEATURES


def _interp(arr, idx):
    return map_coordinates(arr, idx, order=1, mode="nearest")


def _interp2(arr, x, y):
    return map_coordinates(arr, np.stack([x, y]), order=1, mode="nearest")


def line_indicators(field: VectorField3D, lines: list[FieldLine], mag: Magnetogram | None = None,
                    noise_bt: float = 100.0, nd: int | None = None) -> dict:
    """Reliability indicators of each line (see the table of the module).

    **Parameters**

    - ``field``: 3D model (normally the NLFFF).
    - ``lines``: lines traced in ``field``.
    - ``mag``: magnetogram *used as boundary condition* (the preprocessed one if
        there was one). Needed for ``foot_snr`` and ``base_change``.
    - ``noise_bt``: noise of the transverse field in G (~100 G in HMI).
    - ``nd``: buffer layer (by default that of the model, ``meta["nd"]``).
    """
    h = field.dx
    origin = np.asarray(field.origin)
    jx, jy, jz = field.current()
    bx, by, bz = field.bx, field.by, field.bz
    fx, fy, fz = cross(jx, jy, jz, bx, by, bz)
    jxb = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    babs = field.babs
    jb = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2) * babs
    alpha = np.nan_to_num(field.alpha(b_min_frac=0.0), nan=0.0, posinf=0.0, neginf=0.0)
    div = np.abs(field.divergence()) * h / (6 * np.maximum(babs, 1e-12 * babs.max()))
    nd = nd if nd is not None else field.meta.get("nd", max(4, min(field.shape) // 10))
    w = weight_function(field.shape, nd)
    upper = np.array(field.shape) - 1
    # reference α for normalisation (scale of the domain)
    alpha0 = 2 * np.pi / (max(field.shape[:2]) * h) * 0.1

    out = {k: np.full(len(lines), np.nan) for k in ALL_FEATURES + ("length_h",)}
    ztop = max(upper[2] * h, 1e-9)
    diag = np.hypot(*(upper[:2] * h))
    for i, l in enumerate(lines):
        p = l.points
        if len(p) < 3:
            continue
        idx = ((p - origin) / h).T
        seg = np.r_[np.linalg.norm(np.diff(p, axis=0), axis=1), 0.0]
        if seg.sum() <= 0:
            continue
        a_jxb = _interp(jxb, idx)
        a_jb = _interp(jb, idx)
        den = (a_jb * seg).sum()
        out["sin_jb"][i] = (a_jxb * seg).sum() / den if den > 0 else 0.0
        al = _interp(alpha, idx)
        wt = seg / seg.sum()
        m = (al * wt).sum()
        sd = np.sqrt(((al - m) ** 2 * wt).sum())
        out["alpha_spread"][i] = sd / (abs(m) + alpha0)
        out["div"][i] = (_interp(div, idx) * wt).sum()
        ww = _interp(w, idx)
        out["buffer"][i] = (wt * (ww < 0.99)).sum()
        ends = np.array([idx[:, 0], idx[:, -1]])
        on_side = ((ends[:, 0] <= 1e-6) | (ends[:, 0] >= upper[0] - 1e-6) |
                   (ends[:, 1] <= 1e-6) | (ends[:, 1] >= upper[1] - 1e-6) |
                   (ends[:, 2] >= upper[2] - 1e-6))
        out["open"][i] = float(on_side.any() or not l.closed)
        out["length"][i] = seg.sum() / diag
        out["length_h"][i] = seg.sum() / ztop
        out["apex"][i] = (p[:, 2].max() - origin[2]) / ztop

    if mag is not None and mag.has_vector:
        # Footpoints: endpoints at the base (z < 1 cell)
        bt_obs = np.hypot(mag.bx, mag.by)
        dbt = np.hypot(field.bx[:, :, 0] - mag.bx, field.by[:, :, 0] - mag.by)
        for i, l in enumerate(lines):
            if len(l.points) < 3:
                continue
            feet = [q for q in (l.points[0], l.points[-1]) if q[2] - origin[2] < h]
            if not feet:
                feet = [l.points[0]]
            fi = np.array(feet)
            gx = (fi[:, 0] - origin[0]) / h
            gy = (fi[:, 1] - origin[1]) / h
            bt = _interp2(bt_obs, gx, gy)
            out["foot_snr"][i] = float(bt.min() / noise_bt)
            out["base_change"][i] = float((_interp2(dbt, gx, gy) / np.maximum(bt, noise_bt)).max())
    else:
        out["foot_snr"][:] = 1.0
        out["base_change"][:] = 0.0
    return out


# ------------------------------------------------------------- true error
def line_errors(lines_model: list[FieldLine], lines_ref: list[FieldLine], n: int = 40) -> dict:
    """True error of each line with respect to the reference traced from the same seed.

    ``shape``: mean distance between the two lines sampled at the same
    fractions of length, relative to the length of the reference line.
    ``endpoint``: mean distance between endpoints (Mm).
    ``topology``: 1 if one is closed and the other open.
    ``shape_abs``: the same mean distance in Mm (not divided by the length);
    ``length_ref`` / ``length_model``: lengths (Mm). They allow "wrong line" to be
    redefined with an absolute threshold without tracing again.
    """
    shape = np.full(len(lines_ref), np.nan)
    shape_abs = np.full(len(lines_ref), np.nan)
    lref = np.full(len(lines_ref), np.nan)
    lmod = np.full(len(lines_ref), np.nan)
    endp = np.full(len(lines_ref), np.nan)
    topo = np.zeros(len(lines_ref))
    for i, (a, b) in enumerate(zip(lines_model, lines_ref)):
        if len(a.points) < 3 or len(b.points) < 3:
            continue
        def res(p):
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            t = np.linspace(0, s[-1], n)
            return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1), s[-1]
        pa, la = res(a.points)
        pb, lb = res(b.points)
        shape_abs[i] = np.linalg.norm(pa - pb, axis=1).mean()
        shape[i] = shape_abs[i] / max(lb, 1e-9)
        lref[i], lmod[i] = lb, la
        endp[i] = 0.5 * (np.linalg.norm(a.start - b.start) + np.linalg.norm(a.end - b.end))
        topo[i] = float(a.closed != b.closed)
    return {"shape": shape, "endpoint": endp, "topology": topo, "shape_abs": shape_abs,
            "length_ref": lref, "length_model": lmod}


def length_in_heights(length, shape) -> np.ndarray:
    """Length of the lines in box heights (``length_h``) from ``length`` and the grid ``shape`` of the model.

    ``length`` is relative to the horizontal diagonal of the box. For per-line records saved before
    ``length_h`` existed: both indicators are the same length divided by two sizes of the same box.
    """
    nx, ny, nz = shape
    return np.asarray(length, dtype=float) * np.hypot(nx - 1, ny - 1) / max(nz - 1, 1)


# --------------------------------------------------------------- confidence
LENGTH_LOG_CENTER = -0.54  # log10 of the typical relative length (median of the suite)


def _design(ind: dict, feats=ALL_FEATURES):
    cols = []
    for k in feats:
        if k == "length_sq":
            # quadratic term in log(length): allows both the very short and the very
            # long lines to be suspicious (U shape), something the linear
            # regression cannot represent and that appears with flux ropes
            lg = np.log10(np.maximum(np.asarray(ind["length"], dtype=float), 1e-4))
            cols.append((lg - LENGTH_LOG_CENTER) ** 2)
            continue
        x = np.asarray(ind[k], dtype=float)
        if k in ("sin_jb", "alpha_spread", "div", "base_change", "length", "length_h"):
            x = np.log10(np.maximum(x, 1e-4))
        elif k == "foot_snr":
            x = np.log10(np.maximum(x, 1e-2))
        cols.append(x)
    return np.column_stack(cols)


def _fit_logistic(X, y, l2=1e-2):
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-12
    Z = (X - mu) / sd

    def f(beta):
        z = Z @ beta[1:] + beta[0]
        p = 1 / (1 + np.exp(-z))
        eps = 1e-12
        nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        g = Z.T @ (p - y) / len(y)
        return nll + l2 * (beta[1:] ** 2).sum(), np.r_[np.mean(p - y), g + 2 * l2 * beta[1:]]

    r = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B")
    return {"intercept": float(r.x[0]), "coef": r.x[1:].tolist(), "mu": mu.tolist(), "sd": sd.tolist()}


def auc(score, bad):
    """Area under the ROC curve: probability that a wrong line has a higher ``score`` than a good one."""
    score, bad = np.asarray(score, float), np.asarray(bad, bool)
    ok = np.isfinite(score)
    s, b = score[ok], bad[ok]
    if b.all() or (~b).all():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # ties: mean rank
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = b.sum()
    return float((ranks[b].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(s) - n1)))


def confidence(ind: dict, calibration: dict | None = None) -> np.ndarray:
    """Confidence ∈ [0, 1] of each line (1 − estimated probability of being wrong)."""
    cal = calibration or DEFAULT_CALIBRATION
    if "length_h" in cal["features"] and "length_h" not in ind:
        raise KeyError("this calibration needs the indicator 'length_h' (length of the line in box heights); for indicators "
                       "saved without it, add ind['length_h'] = length_in_heights(ind['length'], shape of the model)")
    X = _design(ind, cal["features"])
    Z = (X - np.array(cal["mu"])) / np.array(cal["sd"])
    z = Z @ np.array(cal["coef"]) + cal["intercept"]
    p_bad = 1 / (1 + np.exp(-z))
    if cal.get("iso_x"):
        # isotonic recalibration (monotone table fitted out of sample)
        p_bad = np.interp(p_bad, cal["iso_x"], cal["iso_y"])
    c = 1 - p_bad
    c[~np.isfinite(c)] = np.nan
    return c


def calibration_case(exact: VectorField3D, mag: Magnetogram, nz: int, n_seeds: int = 300,
                     bad_threshold: float = 0.1, method: str = "lbfgs", seed: int = 0,
                     noise_bt: float | None = None, mc: int = 0, gr: bool = False,
                     mc_seed: int = 1000, nlfff_kw: dict | None = None,
                     preprocess_kw: dict | None = None, keep_model: bool = False,
                     ensemble: tuple = ()) -> dict:
    """One calibration case: NLFFF model of ``mag``, indicators and true error per line.

    ``ensemble``: variants of the extrapolation in which something that is not known is changed; for each
    one, the displacement of every line with respect to the nominal model is stored as ``ens_<variant>``
    (mean distance relative to the length of the nominal line, as the shape error) and ``ens_<variant>_topo``
    (1 if the line changes between open and closed). Variants: ``"coarse"`` (magnetogram rebinned by 2 and
    half the levels in height), ``"lfff"`` (side and top faces fixed to the LFFF of best-fit alpha instead of
    the potential field), ``"smooth"`` (three times the smoothing weight of the preprocessing). Unlike the
    Monte Carlo over the noise, they probe systematic errors.

    ``exact``: true field; ``mag``: its base with noise (with ``*_err`` if
    ``mc`` is wanted). Returns ``{"ind", "err", "bad", "noise", "bad_threshold"}``
    (and ``model``, ``pp`` if ``keep_model``). ``nlfff_kw`` and ``preprocess_kw`` are
    passed to the NLFFF and to the preprocessing (to study the robustness to their parameters).
    """
    from .preprocess import preprocess
    from .tracing import seeds_from_magnetogram, trace_field_lines
    from .nlfff import nlfff

    def model_of(m):
        if method == "lbfgs":
            from .nlfff_torch import nlfff_lbfgs
            return nlfff_lbfgs(m, nz, verbose=False, **(nlfff_kw or {}))
        return nlfff(m, nz, verbose=False, **(nlfff_kw or {}))

    noise = float(mag.meta.get("noise", 0.0))
    noise_bt = noise_bt if noise_bt is not None else float(mag.meta.get("noise_t", noise)) or 100.0
    pp = preprocess(mag, verbose=False, **(preprocess_kw or {}))
    model = model_of(pp)
    seeds = seeds_from_magnetogram(mag, n=n_seeds, method="random", polarity="both",
                                   threshold=0.05, rng=seed)
    lm = trace_field_lines(model, seeds)
    le = trace_field_lines(exact, seeds)
    ind = line_indicators(model, lm, mag=pp, noise_bt=noise_bt)
    if mc:
        from .montecarlo import perturb
        rng = np.random.default_rng(mc_seed)
        good = np.zeros((mc, len(lm)), dtype=bool)
        for k in range(mc):
            if mag.has_errors:
                mk = perturb(mag, rng)
            else:
                mk = perturb(mag, rng, noise=noise)
            ek = line_errors(trace_field_lines(model_of(preprocess(mk, verbose=False, **(preprocess_kw or {}))),
                                               seeds), lm)
            good[k] = (ek["shape"] <= bad_threshold) & (ek["topology"] == 0)
        ind["mc_instability"] = 1 - good.mean(axis=0)
    if gr:
        from .grad_rubin import nlfff_grad_rubin
        dis = np.zeros((2, len(lm)))
        for j, pol in enumerate(("P", "N")):
            g = nlfff_grad_rubin(pp, nz, polarity=pol, verbose=False)
            eg = line_errors(trace_field_lines(g, seeds), lm)
            dis[j] = (eg["shape"] > bad_threshold) | (eg["topology"] > 0)
        ind["method_disagreement"] = dis.mean(axis=0)
    for v in ensemble:
        from .nlfff_torch import nlfff_lbfgs
        kw = nlfff_kw or {}
        if v == "coarse":
            pc = preprocess(mag.rebin(2), verbose=False, **(preprocess_kw or {}))
            mv = nlfff_lbfgs(pc, max(8, int(round(nz / 2))), verbose=False, **kw)
        elif v == "lfff":
            from .alpha import alpha_best_fit
            from .lfff import lfff
            mv = nlfff_lbfgs(pp, nz, initial=lfff(pp, alpha_best_fit(pp), nz), verbose=False, **kw)
        elif v == "smooth":
            kp = dict(preprocess_kw or {})
            kp["mu4"] = 3 * kp.get("mu4", 1e-2)
            mv = nlfff_lbfgs(preprocess(mag, verbose=False, **kp), nz, verbose=False, **kw)
        else:
            raise ValueError(f"variante desconocida: {v}")
        ev = line_errors(trace_field_lines(mv, seeds), lm)
        ind["ens_" + v] = ev["shape"]
        ind["ens_" + v + "_topo"] = ev["topology"]
    err = line_errors(lm, le)
    bad = (err["shape"] > bad_threshold) | (err["topology"] > 0)
    rec = {"ind": ind, "err": err, "bad": bad, "noise": noise, "bad_threshold": bad_threshold}
    if keep_model:
        rec.update(model=model, pp=pp)
    return rec


@keeps_awake
def calibration_data(realizations: int = 3, n: int = 40, nz: int = 28, noise: float = 30.0,
                     n_seeds: int = 300, bad_threshold: float = 0.1, method: str = "lbfgs",
                     seed: int = 0, mc: int = 0, gr: bool = False, checkpoint: str | None = None,
                     verbose: bool = True) -> list:
    """Generate the calibration data: Low & Lou + noise, NLFFF model and true error per line.

    A line is considered *wrong* if its relative shape error exceeds
    ``bad_threshold`` (10 % of its length) or if it changes topology.
    With ``mc > 0`` the indicator ``mc_instability`` is added: fraction of the ``mc``
    realizations (observation + noise of the same level) in which the line
    no longer matches that of the nominal model.
    With ``gr=True`` ``method_disagreement`` is added: fraction of the Grad–Rubin
    solutions (P and N) in which the line does not match that of the model.
    ``checkpoint``: file where each finished realization is saved; if it exists
    with the same configuration, it resumes from there.
    """
    from . import analytic as an
    say = print if verbose else (lambda *a, **k: None)
    signature = {"realizations": realizations, "n": n, "nz": nz, "noise": noise, "n_seeds": n_seeds,
                 "bad_threshold": bad_threshold, "method": method, "seed": seed, "mc": mc, "gr": gr}
    state = load_checkpoint(checkpoint, signature)
    data = state["data"] if state is not None else []
    if data:
        say(f"  calibración: reanudando desde {checkpoint} ({len(data)}/{realizations} realizaciones hechas)")
    for r in range(len(data), realizations):
        mag, exact = an.low_lou_magnetogram(noise=noise, rng=seed + r, nx=n, ny=n, nz=nz)
        rec = calibration_case(exact, mag, nz, n_seeds=n_seeds, bad_threshold=bad_threshold,
                               method=method, seed=seed + 100 + r, noise_bt=noise, mc=mc, gr=gr,
                               mc_seed=seed + 1000 + r)
        data.append(rec)
        say(f"  realización {r}: {len(rec['bad'])} líneas, {rec['bad'].mean():.0%} malas")
        if checkpoint:
            save_checkpoint(checkpoint, {"signature": signature, "data": data})
    return data


def fit_calibration(data: list, features=("sin_jb", "alpha_spread", "buffer", "apex"),
                    max_bad_accepted: float = 0.2) -> dict:
    """Fit the confidence with all the realizations but the last and evaluate it on the last.

    Returns ``{"calibration": ..., "metrics": ...}``. The confidence threshold
    is the lowest one that leaves at most ``max_bad_accepted`` wrong lines among
    the accepted ones (in the test realization).
    """
    features = list(features)
    train, test = data[:-1], data[-1]
    Xtr = np.vstack([_design(d["ind"], features) for d in train])
    ytr = np.concatenate([d["bad"] for d in train]).astype(float)
    ok = np.all(np.isfinite(Xtr), axis=1)
    cal = _fit_logistic(Xtr[ok], ytr[ok])
    cal.update(features=features, bad_threshold=test["bad_threshold"], noise=test["noise"])
    ind, err, bad = test["ind"], test["err"], test["bad"]
    conf = confidence(ind, cal)
    per_feature = {k: auc(-np.asarray(ind[k], float) if k == "foot_snr" else np.asarray(ind[k], float), bad)
                   for k in ALL_FEATURES + tuple(x for x in ("mc_instability", "method_disagreement") if x in ind)}
    thr = 0.5
    for t in np.linspace(0.05, 0.95, 91):
        acc = conf >= t
        if acc.sum() >= 10 and bad[acc].mean() <= max_bad_accepted:
            thr = float(t)
            break
    acc = conf >= thr
    cal["threshold"] = thr
    metrics = {
        "auc_combined": auc(-conf, bad), "auc_features": per_feature, "threshold": thr,
        "bad_fraction_all": float(bad.mean()),
        "bad_fraction_accepted": float(bad[acc].mean()) if acc.any() else float("nan"),
        "accepted_fraction": float(acc.mean()),
        "error_median_accepted": float(np.nanmedian(err["shape"][acc])) if acc.any() else float("nan"),
        "error_median_rejected": float(np.nanmedian(err["shape"][~acc])) if (~acc).any() else float("nan"),
        "test_conf": conf, "test_err": err["shape"], "test_bad": bad,
    }
    return {"calibration": cal, "metrics": metrics}


def calibrate(features=("sin_jb", "alpha_spread", "buffer", "apex"), verbose: bool = True, **kw) -> dict:
    """Shortcut: ``calibration_data`` + ``fit_calibration``."""
    return fit_calibration(calibration_data(verbose=verbose, **kw), features=features)


# LINEAR calibration with 5 families (version before the flux rope):
# logistic regression fitted with the suite of solarff.calibration (5 families
# of exact force-free fields x 3 noise models x 3 resolutions x 2
# realizations = 90 cases, 26 000 lines). Leave-one-family-out cross-validation:
# AUC 0.876, ECE 0.021. On the Titov–Démoulin rope, which was not in
# the suite, it fails (31 % wrong lines among the reliable ones): it cannot represent
# that very short lines are also suspicious. Kept as a reference.
LINEAR5_CALIBRATION = {
    "intercept": -1.7884794661295411,
    "coef": [0.03162067114773997, 0.6858185419933508, 0.5785254152941334, 0.385234354400301, 0.9462196567065165],
    "mu": [-1.1143200396207724, -0.8750426377164509, 0.12785320018871946, 0.2629589419363598, -0.5417204796103525],
    "sd": [0.4055083579355104, 0.6018340390532804, 0.22061094791824434, 0.23403672682595736, 0.28154688141563394],
    "features": ['sin_jb', 'alpha_spread', 'buffer', 'apex', 'length'],
    "bad_threshold": 0.1,
    "noise": 'varios',
    "threshold": 0.8,
}

# Metrics of that calibration (out of sample, leaving out each family)
LINEAR5_METRICS = {'auc_combined': 0.8757, 'threshold': 0.8, 'bad_fraction_all': 0.2367, 'bad_fraction_accepted': 0.0517, 'accepted_fraction': 0.5772, 'error_median_accepted': 0.0229, 'error_median_rejected': 0.0963, 'brier': 0.1162, 'ece': 0.0213, 'bad_accepted_worst_field': 0.1065, 'auc_features': {'sin_jb': 0.647, 'alpha_spread': 0.805, 'div': 0.638, 'buffer': 0.698, 'open': 0.624, 'foot_snr': 0.531, 'base_change': 0.5, 'apex': 0.786, 'length': 0.774}, 'per_field': {'LFFF-bip': {'accepted': 0.544, 'bad_accepted': 0.098, 'bad_all': 0.319}, 'LFFF-quad': {'accepted': 0.488, 'bad_accepted': 0.002, 'bad_all': 0.131}, 'LL-I': {'accepted': 0.588, 'bad_accepted': 0.106, 'bad_all': 0.344}, 'LL-II': {'accepted': 0.533, 'bad_accepted': 0.039, 'bad_all': 0.23}, 'LL-III': {'accepted': 0.734, 'bad_accepted': 0.02, 'bad_all': 0.169}}, 'grid': '32², 48², 64²', 'noise_G': '10-60', 'realizations': '90 casos, 5 familias de campos; métricas dejando fuera cada familia'}


# Earlier calibration (Low & Lou case I only, 40x40x28, 30 G; 3 training
# realizations + 1 test). Kept as a reference: it ranks the lines well
# in new fields (AUC 0.87 over the suite) but its probability is
# badly calibrated outside Low & Lou (ECE 0.17).
# calibration_data(realizations=4, n=40, nz=28, noise=30 G,
# lbfgs method) + fit_calibration(features=sin_jb, alpha_spread, buffer, apex).
# Trained with 3 noise realizations and evaluated on an independent 4th one.
LEGACY_LOWLOU_CALIBRATION = {
    "features": ["sin_jb", "alpha_spread", "buffer", "apex"],
    "intercept": -0.3220784354016901,
    "coef": [0.0800962413151637, 0.31233024051795927, 0.6646544632875264, 1.1938300584448183],
    "mu": [-1.1421962232584932, -0.8501699633130311, 0.14326122514987102, 0.2706159510411101],
    "sd": [0.43685114350785875, 0.64021432328661, 0.24627152879587536, 0.24263815016856943],
    "threshold": 0.44, "bad_threshold": 0.1, "noise": 30.0,
}

LEGACY_LOWLOU_METRICS = {
    "auc_combined": 0.919,
    "auc_features": {"sin_jb": 0.819, "alpha_spread": 0.855, "div": 0.601, "buffer": 0.719,
                     "open": 0.606, "foot_snr": 0.525, "base_change": 0.498, "apex": 0.687,
                     "length": 0.714, "mc_instability": 0.754, "method_disagreement": 0.683},
    "bad_fraction_all": 0.397, "bad_fraction_accepted": 0.188, "accepted_fraction": 0.673,
    "error_median_accepted": 0.050, "error_median_rejected": 0.296,
    "grid": "40x40x28", "noise_G": 30, "realizations": "3 entrenamiento + 1 prueba",
}


# Calibration with the Monte Carlo indicator (Low & Lou only, old indicators;
# not selected automatically until it is recalibrated with the diverse suite):
# calibration_data(realizations=4, mc=5)
# + fit_calibration(features=sin_jb, alpha_spread, buffer, apex, mc_instability).
LEGACY_LOWLOU_CALIBRATION_MC: dict = {'features': ['sin_jb', 'alpha_spread', 'buffer', 'apex', 'mc_instability'], 'intercept': -0.2868481262984232, 'coef': [0.05740133967627646, 0.15505276676884602, 0.4039347590636497, 1.2304060578470906, 0.6344888200963819], 'mu': [-1.1421962232584932, -0.8501699633130311, 0.14326122514987102, 0.2706159510411101, 0.16685205784204685], 'sd': [0.43685114350785875, 0.64021432328661, 0.24627152879587536, 0.24263815016856943, 0.27505119621369467], 'threshold': 0.41, 'bad_threshold': 0.1, 'noise': 30.0}
LEGACY_LOWLOU_METRICS_MC: dict = {'auc_combined': 0.942, 'auc_features': {'sin_jb': 0.819, 'alpha_spread': 0.855, 'div': 0.601, 'buffer': 0.719, 'open': 0.606, 'foot_snr': 0.525, 'base_change': 0.498, 'apex': 0.687, 'length': 0.714, 'mc_instability': 0.754, 'method_disagreement': 0.683}, 'bad_fraction_all': 0.397, 'bad_fraction_accepted': 0.188, 'accepted_fraction': 0.693, 'error_median_accepted': 0.049, 'error_median_rejected': 0.344, 'grid': '40x40x28', 'noise_G': 30, 'realizations': '3 entrenamiento + 1 prueba', 'mc_realizations': 5}


# Calibration with the Monte Carlo stability (``mc_instability``), fitted with the
# diverse suite with 4 Monte Carlo realizations per case (5 families x 3 noises x 32², 48² x 2
# realizations = 60 cases; examples/04_calibration_suite.py --sizes 32 48 --mc 4). sin_jb is
# dropped (coefficient ~0, slightly negative). Improves the AUC over the base on the same
# cases by +0.012 [0.008, 0.017] (bootstrap paired by cases). Used when there is a Monte Carlo.
LINEAR5_CALIBRATION_MC: dict = {
    "intercept": -1.6589438775573975,
    "coef": [0.4795691705962766, 0.6267453053042168, 0.5099021058118862, 0.8934224411428404, 0.42124193140548594],
    "mu": [-0.9048305413886825, 0.1553119788103595, 0.2662856958751346, -0.53206570896377, 0.07063074901445467],
    "sd": [0.5983771377687831, 0.24847859578687953, 0.23324321455667849, 0.27912410252413805, 0.1919463773783996],
    "features": ['alpha_spread', 'buffer', 'apex', 'length', 'mc_instability'],
    "bad_threshold": 0.1,
    "noise": 'varios',
    "threshold": 0.8,
}
LINEAR5_METRICS_MC: dict = {'auc_combined': 0.8891, 'threshold': 0.8, 'bad_fraction_all': 0.2516, 'bad_fraction_accepted': 0.0545, 'accepted_fraction': 0.5832, 'error_median_accepted': 0.0275, 'error_median_rejected': 0.1044, 'brier': 0.1103, 'ece': 0.0146, 'bad_accepted_worst_field': 0.1291, 'auc_features': {'sin_jb': 0.646, 'alpha_spread': 0.802, 'div': 0.629, 'buffer': 0.698, 'open': 0.612, 'foot_snr': 0.54, 'base_change': 0.513, 'apex': 0.796, 'length': 0.782, 'mc_instability': 0.678}, 'per_field': {'LFFF-bip': {'accepted': 0.599, 'bad_accepted': 0.096, 'bad_all': 0.309}, 'LFFF-quad': {'accepted': 0.501, 'bad_accepted': 0.003, 'bad_all': 0.13}, 'LL-I': {'accepted': 0.561, 'bad_accepted': 0.129, 'bad_all': 0.391}, 'LL-II': {'accepted': 0.52, 'bad_accepted': 0.031, 'bad_all': 0.258}, 'LL-III': {'accepted': 0.734, 'bad_accepted': 0.022, 'bad_all': 0.187}}, 'auc_gain_mc': 0.0121, 'auc_gain_mc_ci': [0.0077, 0.0166], 'mc_realizations': 4, 'grid': '32², 48²', 'noise_G': '10-60', 'realizations': '60 casos, 5 familias; métricas dejando fuera cada familia'}


# Calibration of version 1.1 (revision of 9-Nov-2026): four terms (three indicators, the length with its
# square), fitted with 6 families of exact force-free fields (the 5 earlier ones + the relaxed
# Titov–Démoulin flux rope) x 3 noises x 3 resolutions x 2 realizations = 108 cases, 31 862 lines
# (examples/04_calibration_suite.py --extra-fields TD-relax). Leave-one-family-out: AUC 0.896, ECE 0.041.
# All its boxes are n x n x 0.7 n. In flatter boxes, like those of real regions, it accepts 8 to 12 % of
# wrong lines instead of 4 % (examples/27_box_proportion.py). Kept because it is the model of the per-line
# table deposited with version 1.1.0.
CALIBRATION_V1 = {'intercept': -1.3337206920976723, 'coef': [0.7928596961397087, 0.7870382847633223, 1.1210321254417894, 0.779454514249511], 'mu': [-0.7884217644911373, 0.166054561849928, -0.5426093136241851, 0.09803603048440751], 'sd': [0.6062190705654161, 0.2773708374632872, 0.313096186446664, 0.1621739780759091], 'features': ['alpha_spread', 'buffer', 'length', 'length_sq'], 'bad_threshold': 0.1, 'noise': 'varios', 'threshold': 0.8}
CALIBRATION_METRICS_V1 = {'auc_combined': 0.8964, 'auc_features': {'sin_jb': 0.71, 'alpha_spread': 0.8087, 'div': 0.688, 'buffer': 0.7409, 'open': 0.6571, 'foot_snr': 0.5732, 'base_change': 0.616, 'apex': 0.7453, 'length': 0.6878}, 'threshold': 0.8, 'bad_fraction_all': 0.295, 'bad_fraction_accepted': 0.04, 'accepted_fraction': 0.5071, 'error_median_accepted': 0.0248, 'error_median_rejected': 0.1126, 'brier': 0.1193, 'ece': 0.0412, 'per_field': {'LFFF-bip': {'accepted': 0.5162, 'bad_accepted': 0.074, 'bad_all': 0.3187}, 'LFFF-quad': {'accepted': 0.4994, 'bad_accepted': 0.0044, 'bad_all': 0.1311}, 'LL-I': {'accepted': 0.5709, 'bad_accepted': 0.0853, 'bad_all': 0.3443}, 'LL-II': {'accepted': 0.5251, 'bad_accepted': 0.0356, 'bad_all': 0.2304}, 'LL-III': {'accepted': 0.7378, 'bad_accepted': 0.0138, 'bad_all': 0.1693}, 'TD-relax': {'accepted': 0.1986, 'bad_accepted': 0.0326, 'bad_all': 0.5805}}, 'bad_accepted_worst_field': 0.0853, 'n_cases': 108, 'realizations': '108 casos, 6 familias (5 + cuerda de Titov–Démoulin relajada); 3 indicadores; métricas dejando fuera cada familia', 'grid': '32², 48², 64²', 'noise_G': '10-60'}

# Default calibration (revision of 13-Nov-2026): the four terms of version 1.1 plus the logarithm of the
# length of the line in box heights (length_h), fitted with the same 6 families in boxes of four
# proportions: the 108 cases of the suite plus 54 in flat, very flat and tall boxes
# (solarff.calibration.proportion_suite) = 162 cases, 48 062 lines
# (examples/04_calibration_suite.py --extra-fields TD-relax --boxes --write-package).
# Leave-one-family-out: AUC 0.890 [0.881, 0.900], ECE 0.037. Against version 1.1 it leaves the boxes of the
# suite as they were (4.2 % of wrong lines among the accepted ones, 4.0 % before), corrects the flat ones
# (4.1 and 5.9 % instead of 8.5 and 12.1 %) and improves every test it has not seen: exact fields on
# the observed flux and region models taken as the truth (examples/30_recalibration_boxes.py).
# It needs the indicator length_h; for records saved without it, see length_in_heights.
DEFAULT_CALIBRATION = {'intercept': -1.4104432685159631, 'coef': [0.7353764882722877, 0.8980991135185304, 0.8239068286442317, 0.7395858055228461, 0.4764990992259401], 'mu': [-0.7784923383337429, 0.16304309561625155, -0.5664134470543077, 0.09382846811896481, -0.18660488449550888], 'sd': [0.5954801278584424, 0.26446108237441224, 0.30517338995116183, 0.164667581592362, 0.33018654150419], 'features': ['alpha_spread', 'buffer', 'length', 'length_sq', 'length_h'], 'bad_threshold': 0.1, 'noise': 'varios', 'threshold': 0.8}
CALIBRATION_METRICS = {'auc_combined': 0.8905, 'auc_features': {'sin_jb': 0.7129, 'alpha_spread': 0.8021, 'div': 0.6969, 'buffer': 0.7491, 'open': 0.6549, 'foot_snr': 0.5755, 'base_change': 0.6229, 'apex': 0.7123, 'length': 0.6748, 'length_h': 0.6599}, 'threshold': 0.8, 'bad_fraction_all': 0.2873, 'bad_fraction_accepted': 0.0441, 'accepted_fraction': 0.511, 'error_median_accepted': 0.0233, 'error_median_rejected': 0.107, 'brier': 0.1215, 'ece': 0.0367, 'per_field': {'LFFF-bip': {'accepted': 0.5146, 'bad_accepted': 0.0873, 'bad_all': 0.3107}, 'LFFF-quad': {'accepted': 0.4989, 'bad_accepted': 0.004, 'bad_all': 0.1242}, 'LL-I': {'accepted': 0.5797, 'bad_accepted': 0.0753, 'bad_all': 0.308}, 'LL-II': {'accepted': 0.5236, 'bad_accepted': 0.0337, 'bad_all': 0.223}, 'LL-III': {'accepted': 0.7352, 'bad_accepted': 0.0141, 'bad_all': 0.1599}, 'TD-relax': {'accepted': 0.2176, 'bad_accepted': 0.0829, 'bad_all': 0.5992}}, 'bad_accepted_worst_field': 0.0873, 'n_cases': 162, 'realizations': '162 casos: 6 familias (5 + cuerda de Titov–Démoulin relajada) en cajas de cuatro proporciones; 3 indicadores, la longitud en dos escalas; métricas dejando fuera cada familia', 'grid': '32², 48², 64² (n x n x 0.7 n) y 64 x 64 x 22, 64 x 64 x 16, 48 x 48 x 48', 'noise_G': '10-60'}
# Calibration of paper v5 (5 indicators), kept for comparison.
FULL5_CALIBRATION = {'intercept': -1.372280991117355, 'coef': [0.1203071169340021, 0.6938562434597622, 0.7539955423114819, 0.43653203474441826, 0.8334484891664526, 0.6602212208475255], 'mu': [-0.9941074038849534, -0.7884217644911373, 0.166054561849928, 0.2669435421299315, -0.5426093136241851, 0.09803603048440751], 'sd': [0.4829876647613927, 0.6062190705654161, 0.2773708374632872, 0.23584851134034693, 0.313096186446664, 0.1621739780759091], 'features': ['sin_jb', 'alpha_spread', 'buffer', 'apex', 'length', 'length_sq'], 'bad_threshold': 0.1, 'noise': 'varios', 'threshold': 0.8}
FULL5_METRICS = {'n_cases': 108, 'auc_combined': 0.8903, 'auc_features': {'sin_jb': 0.7100200786941462, 'alpha_spread': 0.8086721606159839, 'div': 0.6879711242193953, 'buffer': 0.740885646693186, 'open': 0.6571105072779196, 'foot_snr': 0.5731891713865332, 'base_change': 0.6160040663043684, 'apex': 0.7452627511325046, 'length': 0.6877619432217741}, 'threshold': 0.8, 'bad_fraction_all': 0.295, 'bad_fraction_accepted': 0.0447, 'accepted_fraction': 0.5243, 'error_median_accepted': 0.0258, 'error_median_rejected': 0.1166, 'brier': 0.1224, 'ece': 0.0421, 'per_field': {'LFFF-bip': {'accepted': 0.5431865431865431, 'bad_accepted': 0.08663883089770355, 'bad_all': 0.31865431865431865}, 'LFFF-quad': {'accepted': 0.4925925925925926, 'bad_accepted': 0.005263157894736842, 'bad_all': 0.13111111111111112}, 'LL-I': {'accepted': 0.5884483799557255, 'bad_accepted': 0.09439124487004104, 'bad_all': 0.34433487623264236}, 'LL-II': {'accepted': 0.5449157251342841, 'bad_accepted': 0.03772943575798776, 'bad_all': 0.23041303945175032}, 'LL-III': {'accepted': 0.7462962962962963, 'bad_accepted': 0.014640198511166254, 'bad_all': 0.16925925925925925}, 'TD-relax': {'accepted': 0.23596962400444527, 'bad_accepted': 0.029827315541601257, 'bad_all': 0.5804778662715318}}, 'bad_accepted_worst_field': 0.0944, 'realizations': '108 casos, 6 familias (5 + cuerda de Titov–Démoulin relajada); métricas dejando fuera cada familia', 'grid': '32², 48², 64²', 'noise_G': '10-60'}
# With the Monte Carlo stability (mc_instability): same 6-family suite with 4
# Monte Carlo realizations per case (32², 48²; examples/04_calibration_suite.py --extra-fields
# TD-relax --sizes 32 48 --mc 4). AUC gain over the base on the same cases with
# bootstrap paired by cases in auc_gain_mc(_ci). Used when there is a Monte Carlo.
DEFAULT_CALIBRATION_MC = {'intercept': -1.2727456044599927, 'coef': [0.09124752720469306, 0.5437918552913845, 0.7787572608875536, 0.5222898122632077, 0.7891950256679422, 0.7037116775670935, 0.3166445499606509], 'mu': [-1.0165106458446047, -0.8164746924449097, 0.19625416667099746, 0.270443062513526, -0.533303539399419, 0.09574007901905243, 0.06149599582997678], 'sd': [0.47694214552971914, 0.6050737305972986, 0.30093986625983055, 0.23524462992224138, 0.309346466660112, 0.15345784692265968, 0.18001893672989205], 'features': ['sin_jb', 'alpha_spread', 'buffer', 'apex', 'length', 'length_sq', 'mc_instability'], 'bad_threshold': 0.1, 'noise': 'varios', 'threshold': 0.8}
CALIBRATION_METRICS_MC = {'auc_combined': 0.8937, 'auc_features': {'sin_jb': 0.7073899519782973, 'alpha_spread': 0.8084518428564303, 'div': 0.6816776163487416, 'buffer': 0.7426623528008759, 'open': 0.6489763847691604, 'foot_snr': 0.5790946606286489, 'base_change': 0.6214824682951722, 'apex': 0.7542870991983519, 'length': 0.6947036732073738}, 'threshold': 0.8, 'bad_fraction_all': 0.3075, 'bad_fraction_accepted': 0.0489, 'accepted_fraction': 0.5276, 'error_median_accepted': 0.0302, 'error_median_rejected': 0.1239, 'brier': 0.1212, 'ece': 0.0408, 'per_field': {'LFFF-bip': {'accepted': 0.5783475783475783, 'bad_accepted': 0.08472906403940887, 'bad_all': 0.3091168091168091}, 'LFFF-quad': {'accepted': 0.4866666666666667, 'bad_accepted': 0.003424657534246575, 'bad_all': 0.1297222222222222}, 'LL-I': {'accepted': 0.5706232383338553, 'bad_accepted': 0.1300768386388584, 'bad_all': 0.3914813654870028}, 'LL-II': {'accepted': 0.5316666666666666, 'bad_accepted': 0.03082549634273772, 'bad_all': 0.2575}, 'LL-III': {'accepted': 0.7416666666666667, 'bad_accepted': 0.016853932584269662, 'bad_all': 0.18722222222222223}, 'TD-relax': {'accepted': 0.2627777777777778, 'bad_accepted': 0.02748414376321353, 'bad_all': 0.5797222222222222}}, 'bad_accepted_worst_field': 0.1301, 'auc_gain_mc': 0.0006, 'auc_gain_mc_ci': [-0.0025, 0.0039], 'auc_gain_mc_5fam': 0.0094, 'mc_realizations': 4, 'n_cases': 72, 'grid': '32², 48²', 'noise_G': '10-60', 'realizations': '72 casos, 6 familias (con la cuerda relajada); métricas dejando fuera cada familia'}


# Calibrations with the Grad–Rubin indicator. They were tried (calibration_data(..., mc=5,
# gr=True)): on its own it predicts the error (AUC 0.683) but does not improve the combination
# (base 0.919 -> 0.915; base+MC 0.942 -> 0.939): it is redundant with the other
# indicators, so they are left empty and select_calibration uses base or base+MC.
DEFAULT_CALIBRATION_GR: dict = {}
CALIBRATION_METRICS_GR: dict = {}
DEFAULT_CALIBRATION_MC_GR: dict = {}
CALIBRATION_METRICS_MC_GR: dict = {}


def select_calibration(extras: tuple = ()) -> dict:
    """Calibration suited to the additional indicators available."""
    has_mc = "mc_instability" in extras
    has_gr = "method_disagreement" in extras
    if has_mc and has_gr and DEFAULT_CALIBRATION_MC_GR:
        return DEFAULT_CALIBRATION_MC_GR
    if has_gr and DEFAULT_CALIBRATION_GR:
        return DEFAULT_CALIBRATION_GR
    if has_mc and DEFAULT_CALIBRATION_MC and mc_gain_significant():
        return DEFAULT_CALIBRATION_MC
    return DEFAULT_CALIBRATION


def mc_gain_significant() -> bool:
    """Does the Monte Carlo stability improve the confidence significantly?

    The calibration with ``mc_instability`` is used only if the 95 % interval of
    its AUC gain (bootstrap paired by cases) excludes zero. With the
    first five families the gain was +0.009–0.011, but when the flux rope is
    added it vanishes (+0.001 [−0.003, +0.004]): on the rope the NLFFF
    is wrong in a way that is *stable* against noise (systematic error), so
    the stability does not tell the good lines from the wrong ones.
    """
    ci = CALIBRATION_METRICS_MC.get("auc_gain_mc_ci") if CALIBRATION_METRICS_MC else None
    return bool(ci) and ci[0] > 0


def calibration_metrics(cal: dict) -> dict:
    """Metrics of the calibration ``cal`` (for the report)."""
    for c, m in ((DEFAULT_CALIBRATION_MC_GR, CALIBRATION_METRICS_MC_GR), (DEFAULT_CALIBRATION_GR, CALIBRATION_METRICS_GR),
                 (DEFAULT_CALIBRATION_MC, CALIBRATION_METRICS_MC), (CALIBRATION_V1, CALIBRATION_METRICS_V1),
                 (DEFAULT_CALIBRATION, CALIBRATION_METRICS)):
        if c and c is cal:
            return m
    return CALIBRATION_METRICS


# ------------------------------------------------------------ high level
def assess_lines(field: VectorField3D, lines: list[FieldLine], mag: Magnetogram | None = None,
                 noise_bt: float = 100.0, calibration: dict | None = None,
                 extra: dict | None = None) -> dict:
    """Indicators, confidence and mask of reliable lines (confidence ≥ calibrated threshold).

    ``extra``: additional per-line indicators, e.g. ``{"mc_instability": ...}``
    from the Monte Carlo. If it includes ``mc_instability`` and no calibration is given,
    ``DEFAULT_CALIBRATION_MC`` is used.
    """
    if calibration is None:
        calibration = select_calibration(tuple(sorted(extra)) if extra else ())
    cal = calibration
    ind = line_indicators(field, lines, mag=mag, noise_bt=noise_bt)
    if extra:
        ind.update({k: np.asarray(v, float) for k, v in extra.items()})
    conf = confidence(ind, cal)
    return {"indicators": ind, "confidence": conf,
            "reliable": np.nan_to_num(conf, nan=0.0) >= cal["threshold"],
            "threshold": cal["threshold"],
            "calibration": "+".join(["base"] + [x for x in ("mc_instability", "method_disagreement") if x in cal["features"]])}


def confidence_map(field: VectorField3D, mag: Magnetogram | None = None, step: int = 1,
                   threshold: float = 0.05, noise_bt: float = 100.0,
                   calibration: dict | None = None, extra: dict | None = None) -> dict:
    """Confidence of the line leaving each pixel of the base (|Bz| > ``threshold``·max).

    Returns ``conf`` (2D), ``reliable`` (2D), ``x``, ``y``, the lines and their
    indicators (``lines``, ``assess``) for reuse.
    """
    from .tracing import trace_field_lines
    bz = field.bz[:, :, 0]
    ix = np.arange(0, field.shape[0], step)
    iy = np.arange(0, field.shape[1], step)
    IX, IY = np.meshgrid(ix, iy, indexing="ij")
    strong = np.abs(bz[IX, IY]) > threshold * np.abs(bz).max()
    x0, y0 = field.origin[0], field.origin[1]
    seeds = np.column_stack([x0 + IX[strong] * field.dx, y0 + IY[strong] * field.dx,
                             np.full(strong.sum(), field.origin[2] + 0.5 * field.dx)])
    lines = trace_field_lines(field, seeds)
    a = assess_lines(field, lines, mag=mag, noise_bt=noise_bt, calibration=calibration, extra=extra)
    conf = np.full(IX.shape, np.nan)
    conf[strong] = a["confidence"]
    rel = np.zeros(IX.shape, dtype=bool)
    rel[strong] = a["reliable"]
    return {"conf": conf, "reliable": rel, "x": x0 + ix * field.dx, "y": y0 + iy * field.dx,
            "lines": lines, "assess": a, "strong": strong, "step": step}


def reliable_volume(field: VectorField3D, lines: list[FieldLine], reliable: np.ndarray,
                    dilate: int = 1) -> np.ndarray:
    """3D mask of the cells threaded by reliable lines (dilated by ``dilate`` cells)."""
    from scipy.ndimage import binary_dilation
    m = np.zeros(field.shape, dtype=bool)
    origin = np.asarray(field.origin)
    up = np.array(field.shape) - 1
    for l, ok in zip(lines, reliable):
        if not ok or len(l.points) < 2:
            continue
        idx = np.clip(np.rint((l.points - origin) / field.dx).astype(int), 0, up)
        m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if dilate:
        m = binary_dilation(m, iterations=dilate)
    return m
