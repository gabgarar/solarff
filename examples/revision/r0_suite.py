"""Revision R0: calibration suite redone with absolute errors and a new external rope.

Same 6 families, noises, grids (32, 48, 64) and realizations as the calibration of the
paper, plus ``TD2-relax`` (Titov–Démoulin rope with another geometry and opposite
twist, not used to design the model). ``line_errors`` now saves ``shape_abs``
(Mm) and ``length_ref``, so that "wrong line" can be redefined. Output in ``data/revision/``;
the original results are not touched.

    python examples/revision/r0_suite.py
"""
import os
import pickle
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.analytic import CALIBRATION_FIELDS  # noqa: E402
from solarff.calibration import calibration_suite  # noqa: E402

warnings.filterwarnings("ignore")
OUT = "data/revision"
os.makedirs(OUT, exist_ok=True)
fields = list(CALIBRATION_FIELDS) + ["TD-relax", "TD2-relax"]
data = calibration_suite(fields=fields, sizes=(32, 48, 64), realizations=2,
                         checkpoint=os.path.join(OUT, "suite_r0.ckpt"))
pickle.dump(data, open(os.path.join(OUT, "suite_r0.pkl"), "wb"))
print("guardado", len(data), "casos")
