"""Rigorous calibration of the per-line confidence with diverse data.

Generates a suite of cases with known truth (several families of exact
force-free fields, several noise levels and types and several
resolutions; see ``solarff.analytic.CALIBRATION_FIELDS`` and
``NOISE_MODELS``), and evaluates classifiers with **cross-validation
leaving out a whole field type** (the classifier has never seen the
geometry it is evaluated on), with bootstrap confidence intervals.

Metrics (see Niculescu-Mizil & Caruana 2005; Guo et al. 2017):

* **AUC**: ranking ability (do the wrong lines get less confidence?).
* **Brier**: mean squared error of the predicted probability.
* **ECE** (expected calibration error): mean difference between the
  predicted probability and the observed frequency of wrong lines.

The comparison of classifiers uses scikit-learn (optional dependency,
``pip install scikit-learn``); the rest does not need it.
"""

from __future__ import annotations

import time

import numpy as np

from .confidence import ALL_FEATURES, _design, _fit_logistic, auc, calibration_case
from .utils import keeps_awake, load_checkpoint, save_checkpoint

#: Noise models: name -> (σ of B_z, σ of B_x and B_y) in G, for |B_z|max = 1000 G.
#: "HMI" mimics the real HMI ratio (transverse ~5-10 times noisier).
NOISE_MODELS = {
    "uniforme-10G": (10.0, 10.0),
    "uniforme-30G": (30.0, 30.0),
    "HMI (10/60 G)": (10.0, 60.0),
}

#: Indicators used by the confidence: those with the right physical sign that improve the
#: leave-one-field-out cross-validation (``length`` lowers the ECE from 0.06 to 0.02).
# Revision of 9-Nov-2026: the ablation showed that ``sin_jb`` and ``apex`` add nothing (leave-one-family-out
# AUC 0.892 and 0.894 without them, against 0.890) and the three-indicator model
# met the rule registered in advance (docs/revision/): it is the default model.
USED_FEATURES = ("alpha_spread", "buffer", "length", "length_sq")
LINEAR_FEATURES = ("alpha_spread", "buffer", "length")  # the same without the quadratic term
FULL5_FEATURES = ("sin_jb", "alpha_spread", "buffer", "apex", "length", "length_sq")  # paper v5
LINEAR5_FEATURES = ("sin_jb", "alpha_spread", "buffer", "apex", "length")  # earlier linear calibration
#: +1: a larger value indicates a worse line (for monotone models).
FEATURE_SIGN = {"sin_jb": 1, "alpha_spread": 1, "div": 1, "buffer": 1, "open": 1, "foot_snr": -1,
                "base_change": 1, "apex": 1, "length": 1, "mc_instability": 1, "method_disagreement": 1,
                "length_sq": 0}  # 0 = no monotonicity constraint (quadratic term)


def _nz(n):
    return int(round(0.7 * n))


# ------------------------------------------------------------------ data
@keeps_awake
def calibration_suite(fields=None, noises=None, sizes=(32, 48), realizations: int = 2,
                      n_seeds: int = 300, bad_threshold: float = 0.1, method: str = "lbfgs",
                      mc: int = 0, seed: int = 0, checkpoint: str | None = None,
                      verbose: bool = True, ensemble: tuple = ()) -> list:
    """Suite of calibration cases: fields × noises × resolutions × realizations.

    Each element is the result of ``solarff.confidence.calibration_case``
    plus ``field``, ``noise_model``, ``size``, ``realization`` and ``seconds``.
    The true error of each line (``err``) is stored, so the "wrong line"
    threshold can be changed afterwards without recomputing (see ``relabel``).
    ``checkpoint``: saved after each case and resumed if interrupted.
    """
    from .analytic import CALIBRATION_FIELDS, EXTERNAL_TEST_FIELDS, field_magnetogram
    known = {**CALIBRATION_FIELDS, **EXTERNAL_TEST_FIELDS}
    fields = list(fields or CALIBRATION_FIELDS)
    noises = list(noises or NOISE_MODELS)
    say = print if verbose else (lambda *a, **k: None)
    cases = [(f, nm, s, r) for f in fields for nm in noises for s in sizes for r in range(realizations)]
    signature = {"cases": cases, "n_seeds": n_seeds, "bad_threshold": bad_threshold,
                 "method": method, "mc": mc, "seed": seed}
    if ensemble:
        signature["ensemble"] = list(ensemble)
    state = load_checkpoint(checkpoint, signature)
    data = state["data"] if state is not None else []
    if data:
        say(f"  batería: reanudando desde {checkpoint} ({len(data)}/{len(cases)} casos hechos)")
    exact_cache = {}
    t0 = time.time()
    for i in range(len(data), len(cases)):
        fname, nm, size, r = cases[i]
        tc = time.time()
        if (fname, size) not in exact_cache:
            exact_cache = {(fname, size): known[fname](size, size, _nz(size))}
        exact = exact_cache[(fname, size)]
        sz, st = NOISE_MODELS[nm]
        k = seed * 100003 + i
        mag = field_magnetogram(exact, noise=sz, noise_t=st, rng=k)
        rec = calibration_case(exact, mag, _nz(size), n_seeds=n_seeds, bad_threshold=bad_threshold,
                               method=method, seed=k + 7, noise_bt=st, mc=mc, mc_seed=k + 13, ensemble=tuple(ensemble))
        rec.update(field=fname, noise_model=nm, size=size, realization=r, seconds=time.time() - tc)
        data.append(rec)
        say(f"  caso {i + 1}/{len(cases)}: {fname}, {nm}, {size}², real. {r}: "
            f"{rec['bad'].mean():.0%} líneas malas ({rec['seconds']:.0f}s; "
            f"total {time.time() - t0:.0f}s)", flush=True)
        if checkpoint:
            save_checkpoint(checkpoint, {"signature": signature, "data": data})
    return data


def relabel(data: list, bad_threshold: float) -> list:
    """Copy of ``data`` with another "wrong line" threshold (relative shape error)."""
    out = []
    for d in data:
        e = dict(d)
        e["bad"] = (d["err"]["shape"] > bad_threshold) | (d["err"]["topology"] > 0)
        e["bad_threshold"] = bad_threshold
        out.append(e)
    return out


def stack(data: list, features) -> tuple:
    """Design matrix, labels and groups (field, noise, size) of the valid lines."""
    X = np.vstack([_design(d["ind"], features) for d in data])
    y = np.concatenate([d["bad"] for d in data]).astype(int)
    g = {k: np.concatenate([[d.get(k)] * len(d["bad"]) for d in data])
         for k in ("field", "noise_model", "size")}
    ok = np.all(np.isfinite(X), axis=1)
    return X[ok], y[ok], {k: v[ok] for k, v in g.items()}


# ---------------------------------------------------------------- metrics
def brier(p, y) -> float:
    """Mean squared error of the probability (0 = perfect)."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error with ``bins`` equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def reliability_curve(p, y, bins: int = 10) -> dict:
    """Mean predicted probability and observed frequency per bin (reliability diagram)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    pm, fo, n = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            pm.append(p[m].mean())
            fo.append(y[m].mean())
            n.append(int(m.sum()))
    return {"p": np.array(pm), "freq": np.array(fo), "n": np.array(n)}


def bootstrap_ci(p, y, metric, groups=None, n_boot: int = 500, seed: int = 0, level: float = 0.95):
    """Bootstrap confidence interval of ``metric(p, y)``.

    If ``groups`` are given (e.g. the case each line belongs to), whole
    groups are resampled: the lines of the same case are not
    independent, and resampling single lines would give intervals that are too
    narrow.
    """
    rng = np.random.default_rng(seed)
    p, y = np.asarray(p), np.asarray(y)
    vals = []
    if groups is None:
        for _ in range(n_boot):
            i = rng.integers(0, len(p), len(p))
            vals.append(metric(p[i], y[i]))
    else:
        ug = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in ug}
        for _ in range(n_boot):
            i = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
            vals.append(metric(p[i], y[i]))
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


# ------------------------------------------------------------ classifiers
class _Logistic:
    """solarff's logistic regression (no dependencies): the one of the default confidence."""

    def __init__(self, l2=1e-2):
        self.l2 = l2

    def fit(self, X, y):
        self.cal = _fit_logistic(X, y.astype(float), l2=self.l2)
        return self

    def predict_proba(self, X):
        Z = (X - np.array(self.cal["mu"])) / np.array(self.cal["sd"])
        p = 1 / (1 + np.exp(-(Z @ np.array(self.cal["coef"]) + self.cal["intercept"])))
        return np.column_stack([1 - p, p])


class _Isotonic:
    """Wrap a classifier and recalibrate its probability with isotonic regression.

    The recalibration is fitted with *out-of-sample* predictions (internal
    group cross-validation), as recommended by Niculescu-Mizil & Caruana (2005).
    """

    def __init__(self, base_factory, inner_groups=None):
        self.base_factory = base_factory

    def fit(self, X, y, groups=None):
        from sklearn.isotonic import IsotonicRegression
        oof = np.full(len(y), np.nan)
        if groups is None or len(np.unique(groups)) < 2:
            groups = np.random.default_rng(0).integers(0, 4, len(y))
        for g in np.unique(groups):
            m = groups == g
            oof[m] = self.base_factory().fit(X[~m], y[~m]).predict_proba(X[m])[:, 1]
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y)
        self.base = self.base_factory().fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.iso.predict(self.base.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


def classifiers(features) -> dict:
    """Candidate classifiers: name -> (factory, needs groups in fit)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    mono = [FEATURE_SIGN[k] for k in features]
    gb = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                monotonic_cst=mono, random_state=0)
    return {
        "logistic": (lambda: _Logistic(), False),
        "logistic+iso": (lambda: _Isotonic(lambda: _Logistic()), True),
        "gam": (lambda: make_pipeline(SplineTransformer(n_knots=5, degree=3), StandardScaler(),
                                      LogisticRegression(C=1.0, max_iter=3000)), False),
        "gb_monotone": (gb, False),
        "gb_monotone+iso": (lambda: _Isotonic(gb), True),
        "random_forest": (lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                         random_state=0, n_jobs=-1), False),
    }


CLASSIFIER_NAMES = {
    "logistic": ("Regresión logística", "Logistic regression"),
    "logistic+iso": ("Logística + isotónica", "Logistic + isotonic"),
    "gam": ("GAM (splines)", "GAM (splines)"),
    "gb_monotone": ("Gradient boosting monótono", "Monotone gradient boosting"),
    "gb_monotone+iso": ("GB monótono + isotónica", "Monotone GB + isotonic"),
    "random_forest": ("Random forest", "Random forest"),
}


def cross_validate(data: list, features=USED_FEATURES, models=None, group: str = "field",
                   n_boot: int = 500, verbose: bool = True, l2: float = 1e-2) -> dict:
    """Cross-validation leaving out one value of ``group`` ("field", "noise_model" or "size").

    For each model returns the out-of-sample probabilities of all the
    lines, global AUC/Brier/ECE with bootstrap 95 % CI (resampling
    whole cases) and the metrics of each left-out group. ``l2`` is the penalty
    of the logistic regression (1e-2 in the published model; 0 = maximum likelihood).
    """
    say = print if verbose else (lambda *a, **k: None)
    features = list(features)
    X, y, G = stack(data, features)
    case_id = np.concatenate([[i] * len(d["bad"]) for i, d in enumerate(data)])
    case_id = case_id[np.all(np.isfinite(np.vstack([_design(d["ind"], features) for d in data])), axis=1)]
    groups = G[group]
    cands = classifiers(features)
    cands["logistic"] = (lambda: _Logistic(l2=l2), False)
    models = list(models or cands)
    out = {"features": features, "group": group, "y": y, "groups": groups, "case": case_id,
           "base_rate": float(y.mean()), "models": {}}
    for name in models:
        factory, needs_groups = cands[name]
        p = np.full(len(y), np.nan)
        per = {}
        for g in np.unique(groups):
            te = groups == g
            m = factory()
            if needs_groups:
                inner = G["field"][~te] if group != "field" else groups[~te]
                m.fit(X[~te], y[~te], groups=inner)
            else:
                m.fit(X[~te], y[~te])
            p[te] = m.predict_proba(X[te])[:, 1]
            per[str(g)] = {"auc": auc(p[te], y[te]), "brier": brier(p[te], y[te]),
                           "ece": ece(p[te], y[te]), "bad_rate": float(y[te].mean()), "n": int(te.sum())}
        res = {"p": p, "per_group": per}
        for mname, fn in (("auc", auc), ("brier", brier), ("ece", ece)):
            res[mname] = float(fn(p, y))
            res[mname + "_ci"] = bootstrap_ci(p, y, fn, groups=case_id, n_boot=n_boot)
        aucs = [v["auc"] for v in per.values()]
        res["auc_groups_mean"], res["auc_groups_std"] = float(np.mean(aucs)), float(np.std(aucs))
        res["reliability"] = reliability_curve(p, y)
        out["models"][name] = res
        say(f"  {name:18s} AUC {res['auc']:.3f} [{res['auc_ci'][0]:.3f}, {res['auc_ci'][1]:.3f}]  "
            f"Brier {res['brier']:.3f}  ECE {res['ece']:.3f}  "
            f"(AUC por {group}: {res['auc_groups_mean']:.3f} ± {res['auc_groups_std']:.3f})")
    return out


def single_feature_auc(data: list, features=ALL_FEATURES) -> dict:
    """AUC of each indicator separately (over all the cases)."""
    res = {}
    for k in features:
        if k not in data[0]["ind"]:
            continue
        v = np.concatenate([np.asarray(d["ind"][k], float) for d in data])
        y = np.concatenate([d["bad"] for d in data])
        res[k] = auc(FEATURE_SIGN.get(k, 1) * v, y)
    return res


def threshold_for(p_bad, y, max_bad_accepted: float = 0.2) -> float:
    """Lowest confidence threshold (1 − p) that leaves ≤ ``max_bad_accepted`` wrong lines accepted."""
    conf = 1 - np.asarray(p_bad)
    for t in np.linspace(0.05, 0.95, 91):
        acc = conf >= t
        if acc.sum() >= 10 and y[acc].mean() <= max_bad_accepted:
            return float(t)
    return 0.5


# ------------------------------------------------------------ summary
RESULTS_FILE = __import__("os").path.join(__import__("os").path.dirname(__file__), "calibration_cv.json")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def summarize(data: list, cv: dict, single: dict | None = None) -> dict:
    """Serialisable (JSON) summary of the suite and of the cross-validations.

    ``cv``: ``{group: result of cross_validate}``. This is what the report reads.
    """
    from .utils import code_version
    bad_rate = {}
    for key in ("field", "noise_model", "size"):
        vals = sorted({d[key] for d in data}, key=str)
        bad_rate[key] = {str(v): float(np.mean(np.concatenate([d["bad"] for d in data if d[key] == v])))
                         for v in vals}
    out = {
        "n_cases": len(data), "n_lines": int(sum(len(d["bad"]) for d in data)),
        "fields": sorted({d["field"] for d in data}), "noise_models": sorted({d["noise_model"] for d in data}),
        "sizes": sorted({int(d["size"]) for d in data}),
        "realizations": int(max(d["realization"] for d in data) + 1),
        "bad_threshold": data[0]["bad_threshold"], "bad_rate": bad_rate,
        "bad_rate_field_size": {f: {str(n): float(np.mean(np.concatenate(
            [d["bad"] for d in data if d["field"] == f and d["size"] == n] or [np.array([np.nan])])))
            for n in sorted({d["size"] for d in data})} for f in sorted({d["field"] for d in data})},
        "single_auc": single or single_feature_auc(data),
        "code_version": code_version()["label"], "cv": {},
    }
    for group, res in cv.items():
        out["cv"][group] = {"features": res["features"], "base_rate": res["base_rate"], "models": {}}
        for name, r in res["models"].items():
            out["cv"][group]["models"][name] = {k: r[k] for k in (
                "auc", "auc_ci", "brier", "brier_ci", "ece", "ece_ci", "auc_groups_mean", "auc_groups_std",
                "per_group", "reliability")}
    return _jsonable(out)


def load_results(path: str | None = None) -> dict | None:
    """Saved summary of the last calibration with diverse data (or None)."""
    import json
    import os
    path = path or RESULTS_FILE
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fit_default(data: list, cv_field: dict, features=USED_FEATURES, isotonic: bool = False,
                threshold: float = 0.8) -> dict:
    """Default calibration fitted with the *whole* suite, and its out-of-sample metrics.

    The logistic regression is fitted with all the lines. The metrics, the
    isotonic recalibration (if ``isotonic``) and the threshold are obtained from the
    predictions of the leave-one-field-out cross-validation
    (``cv_field = cross_validate(data, group="field")``), that is, always
    on geometries the model has not seen.
    ``threshold``: minimum confidence of a "reliable" line. Since the probability
    is calibrated (ECE ≈ 0.02 out of sample), it is read literally: 0.8 =
    at most a 20 % probability that the line is wrong.
    Returns ``{"calibration": ..., "metrics": ...}`` with the same format as
    ``solarff.confidence.fit_calibration``.
    """
    features = list(features)
    X, y, _ = stack(data, features)
    cal = _fit_logistic(X, y.astype(float))
    cal.update(features=features, bad_threshold=data[0]["bad_threshold"], noise="varios")
    p_oof = np.asarray(cv_field["models"]["logistic"]["p"], float)
    if isotonic:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_oof, y)
        grid = np.linspace(0, 1, 101)
        cal["iso_x"], cal["iso_y"] = grid.round(4).tolist(), np.round(iso.predict(grid), 4).tolist()
        p_oof = np.interp(p_oof, cal["iso_x"], cal["iso_y"])
    thr = float(threshold)
    cal["threshold"] = thr
    conf = 1 - p_oof
    acc = conf >= thr
    shape = np.concatenate([d["err"]["shape"] for d in data])
    ok = np.all(np.isfinite(np.vstack([_design(d["ind"], features) for d in data])), axis=1)
    shape = shape[ok]
    metrics = {
        "auc_combined": auc(p_oof, y), "auc_features": single_feature_auc(data),
        "threshold": thr, "bad_fraction_all": float(y.mean()),
        "bad_fraction_accepted": float(y[acc].mean()) if acc.any() else float("nan"),
        "accepted_fraction": float(acc.mean()),
        "error_median_accepted": float(np.nanmedian(shape[acc])) if acc.any() else float("nan"),
        "error_median_rejected": float(np.nanmedian(shape[~acc])) if (~acc).any() else float("nan"),
        "brier": brier(p_oof, y), "ece": ece(p_oof, y),
    }
    groups = np.asarray(cv_field["groups"])
    per = {}
    for g in np.unique(groups):
        m = groups == g
        per[str(g)] = {"accepted": float(acc[m].mean()),
                       "bad_accepted": float(y[m][acc[m]].mean()) if acc[m].any() else float("nan"),
                       "bad_all": float(y[m].mean())}
    metrics["per_field"] = per
    metrics["bad_accepted_worst_field"] = float(np.nanmax([v["bad_accepted"] for v in per.values()]))
    return {"calibration": cal, "metrics": metrics}


def load_robustness(path: str | None = None) -> dict | None:
    """Summary of the robustness study (``examples/06_robustness.py``) or None."""
    import os
    return load_results(path or os.path.join(os.path.dirname(__file__), "robustness.json"))


def load_flux_rope_test(path: str | None = None, relaxed: bool = False) -> dict | None:
    """Summary of the external test with the Titov–Démoulin rope (``examples/07_flux_rope_test.py``).

    ``relaxed``: the version relaxed to a force-free equilibrium (``--field TD-relax``).
    """
    import os
    name = "flux_rope_test_relax.json" if relaxed else "flux_rope_test.json"
    return load_results(path or os.path.join(os.path.dirname(__file__), name))
