"""Per-line table of the calibration: indicators, true error, label and confidence of every traced line.

One row per field line of the three sets behind the paper:

* ``calibration``: the suite of six families (108 extrapolations) on which the published model is fitted
  (``examples/04_calibration_suite.py --extra-fields TD-relax``);
* ``rerun``: the rerun of the same suite plus the held-out rope ``TD2-relax`` (``examples/revision/r0_suite.py``),
  used for the baselines, the label definitions and the external test;
* ``external``: the three stress-test families ``LL-IV``, ``LFFF-delta`` and ``TD3-relax``
  (``examples/17_external_families.py``).

The table lets the model be refitted, or another one tried, without repeating the 180 extrapolations. It is
written as a gzip-compressed CSV; the columns are described in ``docs/resultados/line_table.md``.

    python examples/21_line_table.py        # writes docs/resultados/line_table.csv.gz (a few seconds)
"""
import argparse
import gzip
import io
import os
import pickle

import numpy as np

from solarff.confidence import ALL_FEATURES, DEFAULT_CALIBRATION, confidence

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default=os.path.join("docs", "resultados", "line_table.csv.gz"))
a = ap.parse_args()

ERRORS = ("shape", "shape_abs", "endpoint", "topology", "length_ref", "length_model")
NOISE = {"uniforme-10G": "uniform-10G", "uniforme-30G": "uniform-30G", "HMI (10/60 G)": "HMI-10G-60G"}
HEADER = (["set", "family", "noise_model", "size", "realization", "case", "line"] + list(ALL_FEATURES)
          + ["shape_error", "shape_error_abs", "endpoint_error", "topology_change", "length_ref", "length_model", "wrong", "confidence"])


def fmt(x):
    return "nan" if not np.isfinite(x) else f"{x:.6g}"


def rows(name, cases):
    for k, c in enumerate(cases):
        conf = confidence(c["ind"], DEFAULT_CALIBRATION)
        for i in range(len(c["bad"])):
            yield ([name, c["field"], NOISE.get(c["noise_model"], c["noise_model"]), str(c["size"]), str(c["realization"]), str(k), str(i)]
                   + [fmt(c["ind"][f][i]) for f in ALL_FEATURES] + [fmt(c["err"][e][i]) for e in ERRORS]
                   + [str(int(c["bad"][i])), fmt(conf[i])])


sets = [("calibration", pickle.load(open(os.path.join(a.data, "calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl"), "rb"))["data"]),
        ("rerun", pickle.load(open(os.path.join(a.data, "revision", "suite_r0.pkl"), "rb")))]
ext = pickle.load(open(os.path.join(a.data, "external_families.pkl"), "rb"))
sets.append(("external", [c for name in ("LL-IV", "LFFF-delta", "TD3-relax") for c in ext[name]]))

buf = io.StringIO()
buf.write(",".join(HEADER) + "\n")
count = {}
for name, cases in sets:
    n = 0
    for r in rows(name, cases):
        buf.write(",".join(r) + "\n")
        n += 1
    count[name] = (len(cases), n)
# mtime=0 keeps the file identical between runs
with open(a.out, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as g:
    g.write(buf.getvalue().encode("ascii"))
for name, (nc, n) in count.items():
    print(f"{name:12s} {nc:4d} extrapolations {n:6d} lines")
print("written", a.out, f"({os.path.getsize(a.out) / 1e6:.1f} MB, {sum(n for _, n in count.values())} lines, {len(HEADER)} columns)")
