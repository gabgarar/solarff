"""Do the field lines change when the tracer step (RK4) is reduced?

Traces the same seeds in the NLFFF of a region with ds = 0.5 (default), 0.25 and
0.125 cells and compares each line with the finest-step one: distance between
endpoints, length, maximum height and fraction of lines whose connectivity
changes (endpoint displaced by more than 2 pixels). Saves ``data/tracing_step_<tag>.json``, or with
``--package`` ``solarff/tracing_step.json`` (used by the report).

    python examples/10_tracing_step.py --cache data/harp377_study_mc10.pkl --tag harp377
"""
import argparse
import json
import os
import pickle
import warnings

import numpy as np

from solarff.tracing import seeds_from_magnetogram, trace_field_lines

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="data/harp377_study_mc10.pkl")
ap.add_argument("--tag", default="harp377")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--package", action="store_true", help="escribir solarff/tracing_step.json")
a = ap.parse_args()

study, _ = pickle.load(open(a.cache, "rb"))
f = study.nlfff
seeds = seeds_from_magnetogram(study.mag, n=a.n)
steps = [0.5, 0.25, 0.125]
lines = {ds: trace_field_lines(f, seeds, ds=ds, max_steps=int(4000 * 0.5 / ds)) for ds in steps}
ref = lines[steps[-1]]
dx = f.dx


def ends(l):
    return l.start, l.end


out = {"tag": a.tag, "n_lines": len(ref), "dx_Mm": dx, "steps": {}}
for ds in steps[:-1]:
    d_end, rel_len, d_apex, closed_same = [], [], [], []
    for la, lb in zip(lines[ds], ref):
        e = max(np.linalg.norm(la.start - lb.start), np.linalg.norm(la.end - lb.end))
        d_end.append(e)
        if lb.length > 0:
            rel_len.append(abs(la.length - lb.length) / lb.length)
        d_apex.append(abs(la.apex - lb.apex))
        closed_same.append(la.closed == lb.closed)
    d_end = np.array(d_end) / dx
    r = {"end_shift_px_median": float(np.median(d_end)), "end_shift_px_p95": float(np.percentile(d_end, 95)),
         "frac_end_shift_gt_2px": float(np.mean(d_end > 2)),
         "length_rel_median": float(np.median(rel_len)), "length_rel_p95": float(np.percentile(rel_len, 95)),
         "apex_shift_Mm_median": float(np.median(d_apex)), "apex_shift_Mm_p95": float(np.percentile(d_apex, 95)),
         "frac_closed_changed": float(1 - np.mean(closed_same))}
    out["steps"][str(ds)] = r
    print(f"ds={ds} vs {steps[-1]}: " + ", ".join(f"{k}={v:.3g}" for k, v in r.items()), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "solarff", "tracing_step.json") if a.package else f"data/tracing_step_{a.tag}.json", "w"),
          indent=1)
