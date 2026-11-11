"""Calibration of the per-line confidence with diverse data (~40 min).

1. Generates the suite of cases with known truth (5 families of exact force-free
   fields × 3 noise models × 3 resolutions × 2 realizations) and saves it in
   ``data/`` (with checkpoints: if interrupted, it resumes on the next run).
2. Cross-validation leaving out one field type, one noise model or one
   resolution, comparing six classifiers.
3. Fits the default calibration (logistic regression with all the data) and
   prints the dictionary for ``solarff/confidence.py``.
4. With ``--write-package`` saves the summary in ``solarff/calibration_cv.json``,
   which is what the report shows.

With ``--extra-fields TD-relax`` the relaxed Titov–Démoulin flux rope
(``solarff.analytic.EXTERNAL_TEST_FIELDS``) is added as a sixth family; its cases
are saved with their own checkpoint and the results under another name.

    python examples/04_calibration_suite.py --write-package
    python examples/04_calibration_suite.py --extra-fields TD-relax
"""
import argparse
import json
import os
import pickle
import warnings

from solarff.calibration import (LINEAR_FEATURES, RESULTS_FILE, USED_FEATURES, calibration_suite,
                                 cross_validate, fit_default, single_feature_auc, summarize)

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[32, 48, 64])
ap.add_argument("--realizations", type=int, default=2)
ap.add_argument("--mc", type=int, default=0, help="realizaciones Monte Carlo por caso (indicador mc_instability)")
ap.add_argument("--extra-fields", nargs="*", default=[], help="familias adicionales (p. ej. TD-relax)")
ap.add_argument("--drop", nargs="*", default=[], help="indicadores que no se usan (p. ej. sin_jb con --mc)")
ap.add_argument("--out", default="data")
ap.add_argument("--write-package", action="store_true", help="guardar el resumen en solarff/calibration_cv.json")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
tag = f"s{'-'.join(map(str, a.sizes))}_r{a.realizations}_mc{a.mc}"
data = calibration_suite(sizes=tuple(a.sizes), realizations=a.realizations, mc=a.mc,
                         checkpoint=os.path.join(a.out, f"calibration_suite_{tag}.ckpt"))
if a.extra_fields:
    # The suite without the extra families is saved as well: 07_flux_rope_test.py uses it as
    # the training set that has never seen the rope.
    with open(os.path.join(a.out, f"calibration_suite_{tag}.pkl"), "wb") as fh:
        pickle.dump({"data": data, "fields": sorted({d["field"] for d in data})}, fh)
    xtag = "-".join(a.extra_fields)
    data = data + calibration_suite(fields=a.extra_fields, sizes=tuple(a.sizes), realizations=a.realizations,
                                    mc=a.mc, checkpoint=os.path.join(a.out, f"calibration_suite_extra_{xtag}_{tag}.ckpt"))
    tag = f"{tag}_plus_{xtag}"
features = [f for f in list(USED_FEATURES) + (["mc_instability"] if a.mc else []) if f not in a.drop]
res = {"data": data, "single": single_feature_auc(data), "features": features}
for group in ("field", "noise_model", "size"):
    print(f"\nValidación cruzada dejando fuera: {group}")
    res[group] = cross_validate(data, features=features, group=group)
fit = fit_default(data, res["field"], features=features)
res["default"] = fit
print("\nCalibración por defecto:\n", fit["calibration"])
print("Métricas fuera de muestra:\n", {k: v for k, v in fit["metrics"].items()})
with open(os.path.join(a.out, f"calibration_suite_{tag}.pkl"), "wb") as fh:
    pickle.dump(res, fh)
if a.write_package:
    S = summarize(data, {g: res[g] for g in ("field", "noise_model", "size")}, res["single"])
    S["chosen"] = "logistic"
    S["features"] = features
    # Same regression without the quadratic length term (earlier linear calibration)
    lin = [f for f in LINEAR_FEATURES if f not in a.drop] + (["mc_instability"] if a.mc else [])
    cv_lin = cross_validate(data, features=lin, models=["logistic"], group="field", verbose=False)
    ml = cv_lin["models"]["logistic"]
    S["linear_cv"] = {k: ml[k] for k in ("auc", "auc_ci", "brier", "ece", "ece_ci", "auc_groups_mean", "auc_groups_std")}
    S["default_metrics"] = {k: v for k, v in fit["metrics"].items() if k != "auc_features"}
    # The earlier calibration (Low & Lou only) evaluated on the whole suite
    import numpy as np
    from solarff.calibration import brier, ece
    from solarff.confidence import LEGACY_LOWLOU_CALIBRATION, auc, confidence
    c = np.concatenate([confidence(d["ind"], LEGACY_LOWLOU_CALIBRATION) for d in data])
    yy = np.concatenate([d["bad"] for d in data])
    ok = np.isfinite(c)
    S["legacy_lowlou"] = {"auc": auc(1 - c[ok], yy[ok]), "brier": brier(1 - c[ok], yy[ok]),
                          "ece": ece(1 - c[ok], yy[ok])}
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(S, fh, indent=1, ensure_ascii=False)
    print("resumen guardado en", RESULTS_FILE)
print("guardado")
