"""How much weight does the divergence need for the helicity to be reliable?

Repeats the NLFFF extrapolation of a real region with several weights w_d of the
divergence term (w_f = 1) and measures the solenoidal and force-free quality that
Thalmann et al. (2019) use to decide whether energy and helicity are reliable:
E_div/E, <|f_i|>, CW_sin, the difference of H_V between two gauges and the free
energy with respect to the volume potential. Saves ``data/divergence_weight_<tag>.json``; with
``--package``, also the sweep in ``solarff/divergence_weight.json`` (used by the report).

    python examples/09_divergence_weight.py --cache data/harp377_study_mc10.pkl --tag harp377
"""
import argparse
import json
import os
import pickle
import time
import warnings

from solarff.helicity import helicity_report
from solarff.metrics import force_free_metrics
from solarff.nlfff_torch import nlfff_lbfgs

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="data/harp377_study_mc10.pkl")
ap.add_argument("--tag", default="harp377")
ap.add_argument("--weights", type=float, nargs="+", default=[1, 2, 4, 8, 16])
ap.add_argument("--package", action="store_true", help="escribir el barrido en solarff/divergence_weight.json")
a = ap.parse_args()

study, _ = pickle.load(open(a.cache, "rb"))
kw = dict(study.config.get("nlfff_kw") or {})
out = []
for wd in a.weights:
    t0 = time.time()
    if study.nlfff is not None and study.nlfff.meta.get("div_weight", 1.0) == wd:
        f, iters = study.nlfff, study.nlfff.meta.get("iterations")
    else:
        r = nlfff_lbfgs(study.mag_pp, study.nz, return_result=True, verbose=False, div_weight=wd, **kw)
        f, iters = r.field, r.field.meta.get("iterations")
    nd = f.meta.get("nd") or max(4, min(f.shape) // 10)
    q = force_free_metrics(f, margin=nd)
    h = helicity_report(f)
    e = h["energy"]
    row = dict(div_weight=wd, iterations=iters, CWsin=q["CWsin"], fi=q["<|f_i|>"],
               E_div_frac=e["E_div_frac"], E_free=e["E_free"], E_div_over_free=e["E_div"] / e["E_free"],
               gauge=h["gauge_diff"], H_V=h["helicity"]["H_V"], seconds=time.time() - t0)
    out.append(row)
    print(json.dumps(row), flush=True)
    json.dump(out, open(f"data/divergence_weight_{a.tag}.json", "w"), indent=1)
if a.package:
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "solarff", "divergence_weight.json")
    D = json.load(open(pkg, encoding="utf-8"))
    D["scan"] = out
    json.dump(D, open(pkg, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
