"""Do the lines get worse with more weight on the divergence? The 108 standard boxes of the suite with w_d = 4 (~50 min).

The suite of ``examples/04_calibration_suite.py`` (five families plus the relaxed flux rope, three noise models,
32/48/64 pixels, two realisations; boxes of the standard aspect ratio) is extrapolated again with the divergence
term weighted by ``--div-weight`` and the force at weight 1. The cases are enumerated with the same two calls as in
04, so each one has the same magnetogram, noise and seeds as with w_d = 1 and the two can be compared case by case.
Compared quantity: the fraction of wrong lines (the label of the calibration: shape error above 10 % or a change of
connectivity), over all lines, by mesh size and in the flux rope, and the paired difference with its 95 % interval
from a bootstrap of whole cases.

Needs ``data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl`` (04 with ``--extra-fields TD-relax``). With
``--package`` the comparison is written in the ``calibration`` block of ``solarff/divergence_weight.json``, which the
report reads (``examples/09`` and ``11`` write the other blocks of the same file).

    python examples/32_divergence_weight_lines.py --package
"""
import argparse
import json
import os
import pickle

import numpy as np

from solarff.calibration import calibration_suite
from solarff.utils import code_version

ap = argparse.ArgumentParser()
ap.add_argument("--div-weight", type=float, default=4.0)
ap.add_argument("--data", default="data")
ap.add_argument("--package", action="store_true", help="escribir la comparación en solarff/divergence_weight.json")
a = ap.parse_args()

SIZES, REAL, ROPE = (32, 48, 64), 2, "TD-relax"
tag = f"s{'-'.join(map(str, SIZES))}_r{REAL}_mc0"
wtag = f"wd{a.div_weight:g}"
kw = {"div_weight": a.div_weight}
# the same two calls as 04 (the case index sets the noise and the seeds)
heavy = calibration_suite(sizes=SIZES, realizations=REAL, nlfff_kw=kw,
                          checkpoint=os.path.join(a.data, f"calibration_suite_{wtag}_{tag}.ckpt"))
heavy = heavy + calibration_suite(fields=[ROPE], sizes=SIZES, realizations=REAL, nlfff_kw=kw,
                                  checkpoint=os.path.join(a.data, f"calibration_suite_{wtag}_extra_{ROPE}_{tag}.ckpt"))
with open(os.path.join(a.data, f"calibration_suite_{wtag}_{tag}_plus_{ROPE}.pkl"), "wb") as fh:
    pickle.dump({"data": heavy, "nlfff_kw": kw}, fh)
base = pickle.load(open(os.path.join(a.data, f"calibration_suite_{tag}_plus_{ROPE}.pkl"), "rb"))["data"]

key = lambda c: (c["field"], c["noise_model"], c["size"], c["realization"])  # noqa: E731
assert [key(c) for c in base] == [key(c) for c in heavy], "the cases of the two suites are not the same"


def bad(cases):
    return np.concatenate([np.asarray(c["bad"], float) for c in cases])


def summary(cases):
    return {"bad_all": float(bad(cases).mean()),
            "bad_by_size": {str(s): float(bad([c for c in cases if c["size"] == s]).mean()) for s in SIZES},
            "rope_bad_all": float(bad([c for c in cases if c["field"] == ROPE]).mean()),
            "n_cases": len(cases), "n_lines": int(bad(cases).size)}


# paired difference: the same resampled cases in both suites
rng = np.random.default_rng(0)
nb = [np.asarray(c["bad"], float) for c in base]
nh = [np.asarray(c["bad"], float) for c in heavy]
diff = float(bad(heavy).mean() - bad(base).mean())
boot = []
for _ in range(2000):
    i = rng.integers(0, len(base), len(base))
    boot.append(np.concatenate([nh[j] for j in i]).mean() - np.concatenate([nb[j] for j in i]).mean())
worse = sum(bad([h]).mean() > bad([b]).mean() for b, h in zip(base, heavy))
better = sum(bad([h]).mean() < bad([b]).mean() for b, h in zip(base, heavy))
out = {"1.0": summary(base), f"{a.div_weight:.1f}": summary(heavy),
       "difference": {"bad_all": diff, "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                      "n_boot": len(boot), "cases_worse": int(worse), "cases_better": int(better)},
       "cases": "108 standard boxes of the suite (examples/04 --extra-fields TD-relax), same magnetograms, noise and seeds",
       "script": "examples/32_divergence_weight_lines.py", "code_version": code_version()["label"]}
r1, r4, d = out["1.0"], out[f"{a.div_weight:.1f}"], out["difference"]
print(f"wrong lines: w_d = 1 {100 * r1['bad_all']:.1f} % | w_d = {a.div_weight:g} {100 * r4['bad_all']:.1f} % | "
      f"difference {100 * d['bad_all']:+.1f} points [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}] | "
      f"cases worse {worse}, better {better} of {len(base)}")
print("by size:", {s: f"{100 * r1['bad_by_size'][s]:.1f} -> {100 * r4['bad_by_size'][s]:.1f} %" for s in r1["bad_by_size"]},
      f"| rope {100 * r1['rope_bad_all']:.1f} -> {100 * r4['rope_bad_all']:.1f} %")
json.dump(out, open(os.path.join(a.data, f"divergence_weight_lines_{wtag}.json"), "w", encoding="utf-8"), indent=1)
if a.package:
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "solarff", "divergence_weight.json")
    D = json.load(open(pkg, encoding="utf-8"))
    D["calibration"] = out
    json.dump(D, open(pkg, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("escrito en", pkg)
