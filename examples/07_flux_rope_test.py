"""External test of the per-line confidence with a flux rope (Titov & Démoulin 1999).

The default calibration was fitted with five families of fields (Low & Lou and
LFFF) without any flux rope. Here it is applied, **without refitting**, to the
Titov & Démoulin flux rope with the parameters of Wiegelmann et al. (2006):
three noise models × three resolutions × 2 realizations. It measures whether it
still ranks the lines well (AUC), whether its probability remains literal (ECE)
and how many wrong lines remain among the reliable ones.

The TD field is force-free only to first order in a/R, so part of the line error
is due to the "truth" not being exactly force-free; its CW_sin is saved to keep
that in mind.

With ``--field TD-relax`` the same rope relaxed to a force-free equilibrium
(``relaxed_titov_demoulin_field``) is used, which separates the error of the
method from the error of the TD model itself.

Saves ``data/flux_rope_test[_relax].pkl`` and the summary for the report in
``solarff/flux_rope_test[_relax].json``.

    python examples/07_flux_rope_test.py                   # analytical TD, ~10-20 min
    python examples/07_flux_rope_test.py --field TD-relax  # relaxed TD
"""
import argparse
import json
import os
import pickle
import warnings

import numpy as np

import solarff as sff
from solarff.analytic import EXTERNAL_TEST_FIELDS
from solarff.calibration import (NOISE_MODELS, USED_FEATURES, bootstrap_ci, calibration_suite,
                                 cross_validate, ece, fit_default, load_results)
from solarff.confidence import LINEAR5_CALIBRATION, LINEAR5_METRICS, auc, confidence
from solarff.metrics import force_free_metrics
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
OUT = "data"
SIZES = (32, 48, 64)
ap = argparse.ArgumentParser()
ap.add_argument("--field", default="TD", choices=["TD", "TD-relax"])
FIELD = ap.parse_args().field
SUF = "" if FIELD == "TD" else "_relax"
with keep_awake():
    data = calibration_suite(fields=[FIELD], sizes=SIZES, realizations=2,
                             checkpoint=os.path.join(OUT, f"flux_rope_test{SUF}.ckpt"))


# Two calibrations, neither has seen the rope: the earlier linear one (5 families) and the
# same regression with the quadratic length term (length_sq), also with 5 families.
SUITE5 = os.path.join(OUT, "calibration_suite_s32-48-64_r2_mc0.pkl")
D5 = pickle.load(open(SUITE5, "rb"))["data"]
cv5 = cross_validate(D5, features=list(USED_FEATURES), models=["logistic"], group="field", verbose=False)
QUAD5 = fit_default(D5, cv5, features=list(USED_FEATURES))
CAL = {"linear5": LINEAR5_CALIBRATION, "quadratic5": QUAD5["calibration"]}
DEFAULT_CALIBRATION = LINEAR5_CALIBRATION  # the one of the short-line diagnostic


def evaluate(d, cal=LINEAR5_CALIBRATION):
    c = np.concatenate([confidence(x["ind"], cal) for x in d])
    y = np.concatenate([x["bad"] for x in d]).astype(int)
    g = np.concatenate([np.full(len(x["bad"]), i) for i, x in enumerate(d)])
    ok = np.isfinite(c)
    c, y, g = c[ok], y[ok], g[ok]
    p = 1 - c
    acc = c >= cal["threshold"]
    return {"n_lines": int(y.size), "bad_rate": float(y.mean()), "auc": auc(p, y),
            "auc_ci": list(bootstrap_ci(p, y, auc, groups=g, n_boot=500)),
            "ece": ece(p, y), "ece_ci": list(bootstrap_ci(p, y, ece, groups=g, n_boot=500)),
            "accepted": float(acc.mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None,
            "mean_predicted_bad": float(p.mean())}


summary = {"code_version": code_version()["label"], "field": FIELD, "sizes": list(SIZES),
           "noise_models": list(NOISE_MODELS), "realizations": 2,
           "params": {k: v for k, v in EXTERNAL_TEST_FIELDS[FIELD](32, 32, 22).meta.items()
                      if k in ("R", "a", "d", "L", "n_turns", "handedness")},
           "all": evaluate(data), "by_noise": {}, "by_size": {}, "exact_cwsin": {}}
for nm in NOISE_MODELS:
    summary["by_noise"][nm] = evaluate([x for x in data if x["noise_model"] == nm])
for s in SIZES:
    summary["by_size"][str(s)] = evaluate([x for x in data if x["size"] == s])
    summary["exact_cwsin"][str(s)] = float(force_free_metrics(
        EXTERNAL_TEST_FIELDS[FIELD](s, s, int(round(0.7 * s))))["CWsin"])
summary["reference"] = {"auc": LINEAR5_METRICS["auc_combined"], "ece": LINEAR5_METRICS["ece"],
                        "bad_accepted": LINEAR5_METRICS["bad_fraction_accepted"],
                        "accepted": LINEAR5_METRICS["accepted_fraction"],
                        "bad_rate": LINEAR5_METRICS["bad_fraction_all"]}
qm = QUAD5["metrics"]
summary["quadratic"] = {"all": evaluate(data, CAL["quadratic5"]), "by_noise": {}, "by_size": {},
                        "reference": {"auc": qm["auc_combined"], "ece": qm["ece"],
                                      "bad_accepted": qm["bad_fraction_accepted"],
                                      "accepted": qm["accepted_fraction"], "bad_rate": qm["bad_fraction_all"]},
                        "coef": dict(zip(CAL["quadratic5"]["features"], CAL["quadratic5"]["coef"]))}
for nm in NOISE_MODELS:
    summary["quadratic"]["by_noise"][nm] = evaluate([x for x in data if x["noise_model"] == nm], CAL["quadratic5"])
for s in SIZES:
    summary["quadratic"]["by_size"][str(s)] = evaluate([x for x in data if x["size"] == s], CAL["quadratic5"])
S = load_results()
if S:
    summary["reference"]["bad_rate_by_field"] = S["bad_rate"]["field"]


# --- Diagnostic: where do the "reliable" lines fail? (by length quartiles)
def cat(d, f):
    return np.concatenate([np.asarray(f(x), float) for x in d])


conf = cat(data, lambda x: confidence(x["ind"], DEFAULT_CALIBRATION))
bad = cat(data, lambda x: x["bad"]).astype(bool)
shape = cat(data, lambda x: x["err"]["shape"])
topo = cat(data, lambda x: x["err"]["topology"])
length = cat(data, lambda x: x["ind"]["length"])
acc = conf >= DEFAULT_CALIBRATION["threshold"]
q25 = float(np.nanpercentile(length, 25))
short = length < q25
m = short & acc & bad
suite = os.path.join(OUT, "calibration_suite_s32-48-64_r2_mc0.pkl")
train_short = None
if os.path.exists(suite):
    D = pickle.load(open(suite, "rb"))["data"]
    Lt, bt = cat(D, lambda x: x["ind"]["length"]), cat(D, lambda x: x["bad"]).astype(bool)
    train_short = float(bt[Lt < np.nanpercentile(Lt, 25)].mean())
summary["short_lines"] = {
    "length_q25": q25, "bad_rate_short": float(bad[short].mean()), "bad_rate_short_train": train_short,
    "frac_of_reliable_bad": float(m.sum() / max(1, (acc & bad).sum())),
    "bad_accepted_without_short": float(bad[acc & ~short].mean()),
    "accepted_without_short": float((acc & ~short).mean()),
    "topology_frac": float((topo[m] > 0).mean()), "shape_median": float(np.nanmedian(shape[m])),
    "bad_accepted_short_50pct": float(((shape > 0.5) | (topo > 0))[short & acc].mean())}
print("líneas cortas:", {k: round(v, 3) if isinstance(v, float) else v for k, v in summary["short_lines"].items()})

with open(os.path.join(OUT, f"flux_rope_test{SUF}.pkl"), "wb") as fh:
    pickle.dump({"data": data, "summary": summary}, fh)
with open(os.path.join(os.path.dirname(sff.__file__), f"flux_rope_test{SUF}.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=1, ensure_ascii=False)
a = summary["all"]
print(f"{FIELD}: {a['n_lines']} líneas, malas {a['bad_rate']:.0%}; AUC {a['auc']:.3f} "
      f"[{a['auc_ci'][0]:.3f}, {a['auc_ci'][1]:.3f}], ECE {a['ece']:.3f}; fiables {a['accepted']:.0%}, "
      f"malas entre ellas {a['bad_accepted']:.1%}")
for k in ("by_noise", "by_size"):
    for n, v in summary[k].items():
        print(f"  {n:14s} AUC {v['auc']:.3f} ECE {v['ece']:.3f} malas {v['bad_rate']:.0%} "
              f"fiables {v['accepted']:.0%} malas entre fiables {v['bad_accepted']:.1%}")
print("CW_sin del campo exacto:", summary["exact_cwsin"])
q = summary["quadratic"]["all"]
print(f"con length_sq (5 familias, sin la cuerda): AUC {q['auc']:.3f} ECE {q['ece']:.3f} fiables {q['accepted']:.0%} "
      f"malas entre ellas {q['bad_accepted']:.1%}; predice de media {q['mean_predicted_bad']:.0%} de malas")
print("guardado")
