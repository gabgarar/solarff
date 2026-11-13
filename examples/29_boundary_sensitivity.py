"""How much do the lines of the three regions depend on the side and top faces, which are not observed?

The extrapolation fixes the side and top faces of the box to the potential field. If the real field is still sheared
far from the base, those faces are wrong and the long, high lines inherit the error: that is what happens in the
exact linear fields built on the observed flux (``examples/25_observed_flux_fields.py``), and much less in the
fields of the suite put in flat boxes (``examples/27_box_proportion.py``). Which of the two cases a real region
resembles cannot be known from the magnetogram, but the dependence can be measured: the published extrapolation of
NOAA 11158, 11429 and 12673 is repeated with the same magnetogram and settings, with the side and top faces fixed
to the linear force-free field of best-fit alpha instead of the potential field (the ``lfff`` variant of
``examples/24_ensemble_indicator.py``), and the lines of the paper are traced in both.

A line "moves" if its mean distance to the published one exceeds 10 % of its length or if it changes between open
and closed. As a control, the published extrapolation is repeated unchanged.

The criteria (B0 to B3) were registered in the plan of the paper before running (13-Nov-2026).

Saves ``data/boundary_sensitivity_<NOAA>.pkl`` (the two fields) and ``docs/resultados/boundary_sensitivity.json``.

    python -u examples/29_boundary_sensitivity.py                 # ~15 min
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

from solarff.calibration import _nz
from solarff.confidence import CALIBRATION_V1, DEFAULT_CALIBRATION, assess_lines, auc, confidence, line_errors
from solarff.helicity import energy_decomposition
from solarff.metrics import force_free_metrics
from solarff.montecarlo import base_seeds
from solarff.nlfff_torch import nlfff_lbfgs
from solarff.tracing import trace_field_lines
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--regions", nargs="*", default=["11429", "11158", "12673"])
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "boundary_sensitivity.json"))
a = ap.parse_args()

REGIONS = {"11158": "harp377_study_mc10.pkl", "11429": "harp1449_20120306_2336_r7_nz32_mc10.pkl",
           "12673": "harp7115_20170906_1136_r4_nz56_mc10.pkl"}
T = DEFAULT_CALIBRATION["threshold"]


def ratio(nx, ny, nz):
    return (nz - 1) / np.hypot(nx - 1, ny - 1)


C0 = float(np.mean([ratio(n, n, _nz(n)) for n in (32, 48, 64)]))


def frac(m, sel):
    return float(np.mean(m[sel])) if sel.any() else None


def equal_height(y, reliable, apex, rng, n_boot=2000):
    """As in ``examples/22_resolution_check.py``: coefficient of ``reliable`` with log(1 + apex height) as control."""
    def coef(i):
        X = np.column_stack([np.ones(len(i)), reliable[i], np.log(1 + apex[i])])
        return float(np.linalg.lstsq(X, y[i], rcond=None)[0][1])
    idx = np.arange(len(y))
    b = [coef(rng.choice(idx, len(idx))) for _ in range(n_boot)]
    return {"value": coef(idx), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "sigma": float(np.std(b))}


out = json.load(open(a.json, encoding="utf-8")) if os.path.exists(a.json) else {}
out.update(code_version=code_version()["label"], calibration="DEFAULT_CALIBRATION (five terms) and CALIBRATION_V1 (four terms, version 1.1), not refitted", c0=C0)
out.setdefault("regions", {})
with keep_awake():
    for noaa in a.regions:
        study, _ = pickle.load(open(os.path.join(a.data, REGIONS[noaa]), "rb"))
        nz, kw = study.config["nz"], study.config["nlfff_kw"]
        src = study.mag_pp if study.mag_pp is not None else study.mag
        cache = os.path.join(a.data, f"boundary_sensitivity_{noaa}.pkl")
        if os.path.exists(cache):
            fields = pickle.load(open(cache, "rb"))
        else:
            t0 = time.time()
            fields = {"control": nlfff_lbfgs(src, nz, verbose=False, **kw),                         # the published run, repeated
                      "lfff": nlfff_lbfgs(src, nz, initial=study.lfff, verbose=False, **kw)}        # side and top faces: LFFF of best-fit alpha
            fields["seconds"] = time.time() - t0
            pickle.dump(fields, open(cache, "wb"))
        pub = study.nlfff
        seeds = base_seeds(pub, 2)                                   # the seeds of the paper
        lines = trace_field_lines(pub, seeds)
        res = assess_lines(pub, lines, mag=src if src.has_vector else None, noise_bt=float(study.mag.meta.get("noise") or 100.0))
        c = np.asarray(res["confidence"], float)                      # the published model (five terms)
        ind = {k: np.asarray(res["indicators"][k], float) for k in ("alpha_spread", "buffer", "length", "length_h")}
        assert np.allclose(confidence(ind, DEFAULT_CALIBRATION), c, equal_nan=True)
        c1 = confidence(ind, CALIBRATION_V1)                         # the model of version 1.1 (four terms)
        ind["length"] = ind["length"] * C0 / ratio(*pub.shape)
        ch = confidence(ind, CALIBRATION_V1)                         # conservative variant of version 1.1: length relative to the height of the box
        acc, acc_1, acc_h = np.nan_to_num(c, nan=0.0) >= T, np.nan_to_num(c1, nan=0.0) >= T, np.nan_to_num(ch, nan=0.0) >= T
        apex = np.array([l.points[:, 2].max() for l in lines])
        e0 = energy_decomposition(pub)
        r = {"grid": [int(n) for n in pub.shape], "alpha_best_fit": float(study.lfff.meta["alpha"]), "alpha_over_alpha_max": float(study.alpha["fit"] / study.alpha["max"]),
             "n_lines": int(len(lines)), "frac_reliable": float(acc.mean()), "frac_reliable_v1": float(acc_1.mean()),
             "frac_reliable_height": float(acc_h.mean()),
             "seconds": float(fields.get("seconds", float("nan")))}
        for name in ("control", "lfff"):
            f = fields[name]
            err = line_errors(trace_field_lines(f, seeds), lines)
            val = np.isfinite(err["shape"])
            topo = err["topology"] > 0
            moved, moved20 = (err["shape"] > 0.1) | topo, (err["shape"] > 0.2) | topo
            e = energy_decomposition(f)
            q = {"n_valid": int(val.sum()), "moved_10": {}, "moved_20": {}, "open_closed": {},
                 "E_J_change": float(e["E_J"] / e0["E_J"] - 1), "E_free_change": float(e["E_free"] / e0["E_free"] - 1),
                 "CWsin": float(force_free_metrics(f, margin=int(f.meta.get("nd", 0)))["CWsin"]),
                 "message": f.meta.get("message"), "iterations": int(f.meta.get("iterations", -1))}
            groups = {"all": val, "accepted": val & acc, "rejected": val & ~acc, "accepted_v1": val & acc_1, "rejected_v1": val & ~acc_1,
                      "accepted_height": val & acc_h, "accepted_only_v1": val & acc_1 & ~acc, "accepted_both": val & acc & acc_1,
                      "accepted_only_default": val & acc & ~acc_1}
            for g, sel in groups.items():
                q["moved_10"][g], q["moved_20"][g], q["open_closed"][g] = frac(moved, sel), frac(moved20, sel), frac(topo, sel)
            q["n_groups"] = {g: int(sel.sum()) for g, sel in groups.items()}
            use = val & np.isfinite(c) & np.isfinite(c1) & np.isfinite(ch)
            if moved[use].any() and not moved[use].all():
                q["auc_default_vs_moved"] = auc(1 - c[use], moved[use].astype(int))
                q["auc_v1_vs_moved"] = auc(1 - c1[use], moved[use].astype(int))
                q["auc_height_vs_moved"] = auc(1 - ch[use], moved[use].astype(int))
                q["auc_apex_vs_moved"] = auc(apex[use], moved[use].astype(int))
            q["median_shift"] = {g: (float(np.nanmedian(err["shape"][sel])) if sel.any() else None) for g, sel in groups.items()}
            if name == "lfff":      # added after seeing the result: is it only the height of the line?
                q["equal_height_moved_10"] = equal_height(moved[use].astype(float), acc[use].astype(float), apex[use], np.random.default_rng(0))
                q["equal_height_moved_10_v1"] = equal_height(moved[use].astype(float), acc_1[use].astype(float), apex[use], np.random.default_rng(0))
                q["equal_height_moved_10_height_variant"] = equal_height(moved[use].astype(float), acc_h[use].astype(float), apex[use], np.random.default_rng(0))
            r[name] = q
        out["regions"][noaa] = r
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        v, k = r["lfff"], r["control"]
        m = v["moved_10"]
        print(f"NOAA {noaa}: alpha / alpha_max {r['alpha_over_alpha_max']:+.2f} | control: moved {k['moved_10']['all']:.3f}, E_J {100 * k['E_J_change']:+.2f} % | "
              f"faces changed: moved all {m['all']:.2f}, accepted {m['accepted']:.2f}, rejected {m['rejected']:.2f} "
              f"(version 1.1: accepted {m['accepted_v1']:.2f}, rejected {m['rejected_v1']:.2f}) | "
              f"accepted only by version 1.1 {m['accepted_only_v1']} (n {v['n_groups']['accepted_only_v1']}), "
              f"by both {m['accepted_both']} (n {v['n_groups']['accepted_both']}) | open/closed among accepted {v['open_closed']['accepted']:.3f} | "
              f"AUC for the lines that move: default {v.get('auc_default_vs_moved', float('nan')):.3f}, version 1.1 {v.get('auc_v1_vs_moved', float('nan')):.3f}, "
              f"height {v.get('auc_height_vs_moved', float('nan')):.3f}, "
              f"apex {v.get('auc_apex_vs_moved', float('nan')):.3f} | E_J {100 * v['E_J_change']:+.1f} %, E - E_p {100 * v['E_free_change']:+.1f} % | "
              f"CWsin {v['CWsin']:.3f} | {v['message']}", flush=True)
        e = v["equal_height_moved_10"]
        print(f"   at equal apex height, accepted minus rejected: {e['value']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]", flush=True)
for suffix in ("", "_v1"):
    eh = [out["regions"][n]["lfff"]["equal_height_moved_10" + suffix] for n in out["regions"]]
    w = 1 / np.array([e["sigma"] for e in eh]) ** 2
    m, sd = float((w * np.array([e["value"] for e in eh])).sum() / w.sum()), float(w.sum() ** -0.5)
    out["pooled_equal_height_moved_10" + suffix] = {"value": m, "ci": [m - 1.96 * sd, m + 1.96 * sd], "sigma": sd, "regions": list(out["regions"])}
    print(f"three regions, at equal apex height{' (version 1.1)' if suffix else ''}: {m:+.3f} [{m - 1.96 * sd:+.3f}, {m + 1.96 * sd:+.3f}]")
json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("written", a.json)
