"""Revision R13: does the stability against noise (Monte Carlo) add to the 3-indicator model?

With the Monte Carlo suite (``04_calibration_suite.py --extra-fields TD-relax --sizes 32 48
--mc 4``) the leave-one-family-out AUC of the published model is compared with and without the
``mc_instability`` indicator, with the six families and without the rope. The rule of the paper requires an
indicator to improve the AUC by more than its uncertainty (±0.012).

    python examples/revision/r13_mc_indicator.py
"""
import json
import os
import pickle
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.calibration import USED_FEATURES, cross_validate  # noqa: E402

warnings.filterwarnings("ignore")
SUITE = "data/calibration_suite_s32-48_r2_mc4_plus_TD-relax.pkl"
d = pickle.load(open(SUITE, "rb"))
data = d["data"] if isinstance(d, dict) else d


def auc(dd, feats):
    return cross_validate(dd, feats, models=["logistic"], group="field", n_boot=10, verbose=False)["models"]["logistic"]["auc"]


out = {"suite": SUITE, "n_cases": len(data), "features": list(USED_FEATURES)}
for tag, dd in (("6_familias", data), ("sin_cuerda", [x for x in data if x["field"] != "TD-relax"])):
    a0, a1 = auc(dd, list(USED_FEATURES)), auc(dd, list(USED_FEATURES) + ["mc_instability"])
    out[tag] = {"auc": a0, "auc_with_mc": a1, "delta_auc": a1 - a0}
    print(f"{tag}: AUC {a0:.4f} → {a1:.4f} con mc_instability (ΔAUC {a1 - a0:+.4f})", flush=True)
json.dump(out, open("docs/revision/resultados/r13_mc_indicator.json", "w", encoding="utf-8"), indent=1)
