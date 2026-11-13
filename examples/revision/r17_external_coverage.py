"""Risk against coverage of the published model in the four external families (analysis of 12-Nov-2026, on data
already computed).

The paper quotes the wrong lines among those accepted at the pre-set confidence threshold of 0.8. This script
gives the same quantity for stricter thresholds (0.9, 0.95) and for a rule that does not use the level of the
probability at all: keeping, in each extrapolation, the fraction of lines with the highest confidence (10, 20,
30 and 50 %). It tells apart a failure of the probability level (the ranking is right, the threshold is not) from
a failure of the ranking. It also gives the size of the errors of the accepted lines: the wrong lines among them
when the shape-error criterion is 10 (the adopted one), 15, 20 or 30 % of the length, the lines that change between
open and closed, and the percentiles of the shape error, against the same quantities without filtering. It was
done after seeing the results at 0.8 and is reported as such.

Intervals: bootstrap that resamples whole extrapolations (18 per family). Writes
``docs/revision/resultados/r17_external_coverage.json``.

    python examples/revision/r17_external_coverage.py      # ~1 min
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.confidence import DEFAULT_CALIBRATION, auc, confidence  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join("docs", "revision", "resultados", "r17_external_coverage.json")
THRESHOLDS = (0.7, 0.8, 0.9, 0.95)
FRACTIONS = (0.1, 0.2, 0.3, 0.5)
CRITERIA = (0.10, 0.15, 0.20, 0.30)
N_BOOT = 2000

ext = pickle.load(open(os.path.join("data", "external_families.pkl"), "rb"))
r0 = pickle.load(open(os.path.join("data", "revision", "suite_r0.pkl"), "rb"))
families = {"TD2-relax": [x for x in r0 if x["field"] == "TD2-relax"],
            **{name: ext[name] for name in ("LL-IV", "LFFF-delta", "TD3-relax")}}


def per_case(d):
    out = []
    for x in d:
        c = confidence(x["ind"], DEFAULT_CALIBRATION)
        ok = np.isfinite(c)
        out.append((c[ok], x["bad"][ok].astype(int)))
    return out


def per_case_errors(d):
    out = []
    for x in d:
        c = confidence(x["ind"], DEFAULT_CALIBRATION)
        ok = np.isfinite(c) & np.isfinite(x["err"]["shape"])
        out.append((c[ok], x["err"]["shape"][ok], x["err"]["topology"][ok] > 0))
    return out


def error_size(cases, accepted):
    """Errors of the lines with confidence >= 0.8 (``accepted``) or of all the lines."""
    sel = [(sh[c >= 0.8], tp[c >= 0.8]) if accepted else (sh, tp) for c, sh, tp in cases]
    sh = np.concatenate([a for a, _ in sel])
    tp = np.concatenate([b for _, b in sel])
    r = {f"wrong_{int(100 * t)}": float(np.mean((sh > t) | tp)) for t in CRITERIA}
    r.update(open_closed=float(tp.mean()), shape_p50=float(np.percentile(sh, 50)), shape_p90=float(np.percentile(sh, 90)),
             shape_p95=float(np.percentile(sh, 95)))
    return r


def at_threshold(cases, t):
    acc = sum(int((c >= t).sum()) for c, _ in cases)
    wrong = sum(int(y[c >= t].sum()) for c, y in cases)
    n = sum(len(c) for c, _ in cases)
    return acc / n, (wrong / acc if acc else float("nan"))


def at_fraction(cases, q):
    wrong = n = 0
    for c, y in cases:
        k = max(1, int(round(q * len(c))))
        wrong += int(y[np.argsort(-c)[:k]].sum())
        n += k
    return wrong / n


def interval(cases, fn, rng):
    v = [fn([cases[i] for i in rng.integers(0, len(cases), len(cases))]) for _ in range(N_BOOT)]
    v = np.array([x for x in v if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else None


out = {"model": "DEFAULT_CALIBRATION (3 indicators, 6 families), not refitted", "thresholds": list(THRESHOLDS),
       "fractions": list(FRACTIONS), "n_boot": N_BOOT, "families": {}}
for name, d in families.items():
    cases = per_case(d)
    rng = np.random.default_rng(0)
    c_all, y_all = np.concatenate([c for c, _ in cases]), np.concatenate([y for _, y in cases])
    s = {"n_cases": len(cases), "n_lines": int(y_all.size), "bad_rate": float(y_all.mean()), "auc": auc(1 - c_all, y_all),
         "by_threshold": {}, "by_fraction": {}}
    for t in THRESHOLDS:
        acc, risk = at_threshold(cases, t)
        s["by_threshold"][str(t)] = {"accepted": acc, "bad_accepted": risk,
                                     "bad_accepted_ci": interval(cases, lambda cs, t=t: at_threshold(cs, t)[1], rng)}
    for q in FRACTIONS:
        s["by_fraction"][str(q)] = {"bad": at_fraction(cases, q), "bad_ci": interval(cases, lambda cs, q=q: at_fraction(cs, q), rng),
                                    "worst_case": max(at_fraction([cs], q) for cs in cases)}
    ecases = per_case_errors(d)
    s["error_size"] = {"accepted": error_size(ecases, True), "all": error_size(ecases, False)}
    for key in ("wrong_20", "open_closed"):
        s["error_size"]["accepted"][key + "_ci"] = interval(ecases, lambda cs, key=key: error_size(cs, True)[key], rng)
    out["families"][name] = s
    e = s["error_size"]
    print(f"   errors of the accepted lines: wrong at 10/15/20/30 % " + "/".join(f"{100 * e['accepted'][f'wrong_{int(100 * t)}']:.1f}" for t in CRITERIA) +
          f" (all lines " + "/".join(f"{100 * e['all'][f'wrong_{int(100 * t)}']:.1f}" for t in CRITERIA) + f"); open-closed "
          f"{100 * e['accepted']['open_closed']:.1f} (all {100 * e['all']['open_closed']:.1f}); p95 of the shape error "
          f"{e['accepted']['shape_p95']:.3f} (all {e['all']['shape_p95']:.3f})")
    print(f"{name:11s} wrong {s['bad_rate']:.0%} AUC {s['auc']:.3f} | threshold: " +
          "  ".join(f"{t}: {100 * s['by_threshold'][str(t)]['accepted']:.0f}% kept, {100 * s['by_threshold'][str(t)]['bad_accepted']:.1f}% wrong"
                    for t in THRESHOLDS) +
          " | top: " + "  ".join(f"{int(100 * q)}%: {100 * s['by_fraction'][str(q)]['bad']:.1f}%" for q in FRACTIONS))
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("written", OUT)
