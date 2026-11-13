"""Resolution check of the per-line confidence, without refitting the published model.

1. Regions: each region before its flare is extrapolated again at about twice the resolution (NOAA 11158
   and 12673: rebin 2 instead of 4, 0.73 Mm instead of 1.46 Mm; NOAA 11429: rebin 4 instead of 7, 1.46 Mm
   instead of 2.55 Mm), with the same box height. The lines are traced from the same seeds, in Mm, in the
   two fields; their confidence, the fraction of reliable lines, the displacement of each line and the
   energies are compared. Since the truth is not known, the displacement is a consistency test: two lines
   that differ by more than 20 % of their length cannot both be within 10 % of the true line. The test at
   equal height (H6 of the plan of the paper) asks whether the lines accepted at the published resolution
   move less than the rejected ones once the apex height is accounted for.
2. Suite: the ten families of exact fields at 96 and 128 pixels per side (the calibration used 32,
   48 and 64), with the three noise models and one realisation. The published model is applied as is.

Nothing existing is overwritten: each region is cached in ``data/resolution_check_harp<N>_r<rebin>_nz<nz>.pkl``
and the suite in ``data/resolution_check_suite_s<size>.pkl`` (with a checkpoint, so it resumes if interrupted).
Summary in ``docs/resultados/resolution_check.json``.

    python examples/22_resolution_check.py                    # region and suite, ~6 h
    python examples/22_resolution_check.py --part region      # NOAA 11158, ~1 h
    python examples/22_resolution_check.py --part region --region 12673
    python examples/22_resolution_check.py --part region --region 12673 --coarser   # half the published resolution
    python examples/22_resolution_check.py --part suite --sizes 96
    python examples/22_resolution_check.py --dry              # quick self-test with what is already computed
"""
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np
from scipy.stats import spearmanr

import solarff as sff
from solarff.analytic import CALIBRATION_FIELDS
from solarff.calibration import NOISE_MODELS, bootstrap_ci, calibration_suite, ece
from solarff.confidence import DEFAULT_CALIBRATION, assess_lines, auc, confidence, line_errors
from solarff.helicity import energy_decomposition
from solarff.metrics import force_free_metrics
from solarff.montecarlo import base_seeds
from solarff.tracing import trace_field_lines
from solarff.utils import code_version, keep_awake

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=("region", "suite", "all"), default="all")
ap.add_argument("--region", choices=("11158", "11429", "12673"), default="11158")
ap.add_argument("--coarser", action="store_true", help="compare the published model with half its resolution instead of twice")
ap.add_argument("--sizes", type=int, nargs="+", default=[96, 128])
ap.add_argument("--data", default="data")
ap.add_argument("--json", default=os.path.join("docs", "resultados", "resolution_check.json"))
ap.add_argument("--dry", action="store_true", help="self-test: region at the published resolution (from its cache), suite with one family at 32")
a = ap.parse_args()

CALIBRATION = list(CALIBRATION_FIELDS) + ["TD-relax"]            # the six families of the published model
EXTERNAL = ["TD2-relax", "LL-IV", "LFFF-delta", "TD3-relax"]     # the four external families
THRESHOLD = DEFAULT_CALIBRATION["threshold"]
# region: (folder of the SHARP, cache of the published study, harp number, rebin and levels of the finer extrapolation)
REGIONS = {"11158": ("harp377", "harp377_study_mc10.pkl", 377, 2, 96),
           "11429": ("harp1449_20120306_2336", "harp1449_20120306_2336_r7_nz32_mc10.pkl", 1449, 4, 56),
           "12673": ("harp7115_20170906_1136", "harp7115_20170906_1136_r4_nz56_mc10.pkl", 7115, 2, 112)}
COARSER = {"11158": (8, 24), "11429": (14, 16), "12673": (8, 28)}      # rebin and levels at half the published resolution
N_BOOT = 2000


def save(out):
    json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def evaluate(d):
    """Metrics of the published model on the cases ``d`` (interval by bootstrap grouped by case)."""
    c = np.concatenate([confidence(x["ind"], DEFAULT_CALIBRATION) for x in d])
    y = np.concatenate([x["bad"] for x in d]).astype(int)
    g = np.concatenate([np.full(len(x["bad"]), i) for i, x in enumerate(d)])
    ok = np.isfinite(c)
    c, y, g = c[ok], y[ok], g[ok]
    p = 1 - c
    acc = c >= THRESHOLD
    return {"n_cases": int(len(d)), "n_lines": int(y.size), "bad_rate": float(y.mean()), "auc": auc(p, y),
            "auc_ci": list(bootstrap_ci(p, y, auc, groups=g, n_boot=500)), "ece": ece(p, y),
            "accepted": float(acc.mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None,
            "mean_predicted_bad": float(p.mean()), "seconds_median": float(np.median([x["seconds"] for x in d]))}


def region_fields(study, seeds):
    f = study.nlfff
    src = study.mag_pp if study.mag_pp is not None else study.mag
    lines = trace_field_lines(f, seeds)
    res = assess_lines(f, lines, mag=src if src.has_vector else None, noise_bt=float(study.mag.meta.get("noise") or 100.0))
    conf = res["confidence"]
    ff = force_free_metrics(f, margin=int(f.meta.get("nd", 0)))
    e = energy_decomposition(f)
    summary = {"grid": [int(n) for n in f.shape], "dx_Mm": float(f.dx), "n_lines": int(len(lines)),
               "frac_reliable": float(np.mean(np.nan_to_num(conf, nan=0.0) >= THRESHOLD)),
               "conf_median": float(np.nanmedian(conf)), "CWsin": float(ff["CWsin"]), "E_div_frac": float(e["E_div_frac"]),
               "E_free": float(e["E_free"]), "E_J": float(e["E_J"]), "converged": bool(study.nlfff_converged),
               "message": study.nlfff_message, "seconds_nlfff": float(study.timings.get("nlfff", float("nan")))}
    return lines, conf, summary, res["indicators"]


def equal_height(y, reliable, apex, rng):
    """Coefficient of ``reliable`` in the regression of ``y`` on ``reliable`` and log(1 + apex height), with its 95 % interval."""
    def coef(i):
        X = np.column_stack([np.ones(len(i)), reliable[i], np.log(1 + apex[i])])
        return float(np.linalg.lstsq(X, y[i], rcond=None)[0][1])
    idx = np.arange(len(y))
    b = [coef(rng.choice(idx, len(idx))) for _ in range(N_BOOT)]
    return {"value": coef(idx), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "sigma": float(np.std(b))}


def region(out):
    folder, published, harp, rebin, nz = REGIONS[a.region]
    if a.coarser:
        rebin, nz = COARSER[a.region]
    key, tag = ("regions_coarser", "_coarser") if a.coarser else ("regions", "")
    lo_study, _ = pickle.load(open(os.path.join(a.data, published), "rb"))
    if a.dry:
        rebin, nz = None, lo_study.nlfff.shape[2]
    cache = os.path.join(a.data, f"resolution_check_harp{harp}_r{rebin}_nz{nz}.pkl")
    if a.dry:
        hi_study = lo_study
    elif os.path.exists(cache):
        hi_study = pickle.load(open(cache, "rb"))
        print("loaded", cache)
    else:
        mag = sff.load_sharp_cea(os.path.join(a.data, folder)).rebin(rebin).balance_flux()
        print(mag, f"| rebin {rebin}, nz {nz}", flush=True)
        t0 = time.time()
        hi_study = sff.run_study(mag, nz=nz, nlfff_kw={"max_iter": 4000}, n_seeds=100, name=f"HARP {harp}, rebin {rebin}")
        print(f"extrapolation: {time.time() - t0:.0f} s", flush=True)
        pickle.dump(hi_study, open(cache, "wb"))
    seeds = base_seeds(lo_study.nlfff, 2)     # the seeds of the paper (every second pixel of the base where |Bz| > 5 % of its maximum)
    lines_lo, c_lo, s_lo, ind_lo = region_fields(lo_study, seeds)
    lines_hi, c_hi, s_hi, ind_hi = region_fields(hi_study, seeds)
    ok = np.isfinite(c_lo) & np.isfinite(c_hi)
    acc_lo, acc_hi = c_lo >= THRESHOLD, c_hi >= THRESHOLD
    err = line_errors(lines_hi, lines_lo)     # how far each line moves when the resolution doubles
    moved = (err["shape"] > 0.1) | (err["topology"] > 0)
    val = np.isfinite(err["shape"])
    apex = np.array([l.points[:, 2].max() for l in lines_lo])                  # apex height in the published field, Mm
    if not a.dry:                                                              # per-line arrays, for later analyses
        np.savez(os.path.join(a.data, f"resolution_check_lines_{a.region}{tag}.npz"), conf_published=c_lo, conf_double=c_hi, apex=apex,
                 closed_published=np.array([l.closed for l in lines_lo]), seeds=np.asarray(seeds),
                 **{"err_" + k: np.asarray(v) for k, v in err.items()},
                 **{"published_" + k: np.asarray(v) for k, v in ind_lo.items()},
                 **{"double_" + k: np.asarray(v) for k, v in ind_hi.items()})
    rng = np.random.default_rng(0)
    use = val & np.isfinite(c_lo)
    rel = acc_lo[use].astype(float)
    moved20 = ((err["shape"] > 0.2) | (err["topology"] > 0))
    both = ok & acc_lo & acc_hi
    # Lower bound of the error of the accepted lines. If two lines are within 10 % of the true one, they differ by at
    # most 20 % of its length. The true length is not known: the longer of the two is used, which gives the smaller figure.
    rel_long = err["shape_abs"] / np.maximum(err["length_ref"], err["length_model"])
    differ, rng_d = {}, np.random.default_rng(1)   # its own generator, so that the intervals of the equal-height test do not change
    for t in (0.20, 0.25, 0.30):
        d = ((rel_long > t) | (err["topology"] > 0))[both & val].astype(float)
        b = [float(np.mean(rng_d.choice(d, d.size))) for _ in range(N_BOOT)]
        differ[f"{t:.2f}"] = {"value": float(d.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n": int(d.size)}
    out.setdefault(key, {})[a.region] = {
        "accepted_both_differ_longer_line": differ,
        "not_reproduced_10": float(np.mean(moved[val])), "not_reproduced_10_accepted": float(np.mean(moved[val & acc_lo])),
        "published": s_lo, "double": s_hi, "n_seeds": int(len(seeds)), "n_valid": int(ok.sum()),
        "equal_height": {"moved_20": equal_height(moved20[use].astype(float), rel, apex[use], rng),
                         "open_closed": equal_height((err["topology"][use] > 0).astype(float), rel, apex[use], rng),
                         "displacement": equal_height(err["shape"][use], rel, apex[use], rng)},
        "accepted_both_differ_20": float(np.mean(moved20[both])), "accepted_both_open_closed": float(np.mean(err["topology"][both] > 0)),
        "rejected_both_open_closed": float(np.mean(err["topology"][ok & ~acc_lo & ~acc_hi] > 0)),
        "auc_confidence_vs_moved_20": auc(1 - c_lo[use], moved20[use]),
        "spearman_confidence": float(spearmanr(c_lo[ok], c_hi[ok])[0]),
        "same_decision": float(np.mean(acc_lo[ok] == acc_hi[ok])),
        "accepted_both": float(np.mean(acc_lo[ok] & acc_hi[ok])), "accepted_only_published": float(np.mean(acc_lo[ok] & ~acc_hi[ok])),
        "accepted_only_double": float(np.mean(~acc_lo[ok] & acc_hi[ok])),
        "E_J_change": float(s_hi["E_J"] / s_lo["E_J"] - 1), "E_free_change": float(s_hi["E_free"] / s_lo["E_free"] - 1),
        "line_shift": {"reliable_median": float(np.nanmedian(err["shape"][val & acc_lo])),
                       "unreliable_median": float(np.nanmedian(err["shape"][val & ~acc_lo])),
                       "reliable_moved": float(np.mean(moved[val & acc_lo])), "unreliable_moved": float(np.mean(moved[val & ~acc_lo]))},
    }
    r = out[key][a.region]
    e = r["equal_height"]["moved_20"]
    print(f"NOAA {a.region}, at equal height, reliable - unreliable lines that move by more than 20 %: {e['value']:+.3f} "
          f"[{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]", flush=True)
    print(f"region: reliable {s_lo['frac_reliable']:.3f} -> {s_hi['frac_reliable']:.3f} | Spearman {r['spearman_confidence']:.3f} | "
          f"same decision {r['same_decision']:.3f} | E_J {100 * r['E_J_change']:+.1f} % | E - E_p {100 * r['E_free_change']:+.1f} % | "
          f"lines that move: reliable {r['line_shift']['reliable_moved']:.3f}, unreliable {r['line_shift']['unreliable_moved']:.3f}", flush=True)
    print("accepted at both resolutions that differ by more than 20, 25 and 30 % of the longer line: " +
          ", ".join(f"{v['value']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" for v in differ.values()) + f" (n = {differ['0.20']['n']})", flush=True)
    save(out)


def suite(out):
    out.setdefault("suite", {})
    fields = ["LL-I"] if a.dry else CALIBRATION + EXTERNAL
    for size in ([32] if a.dry else a.sizes):
        tag = f"resolution_check_suite_s{size}" + ("_dry" if a.dry else "")
        d = calibration_suite(fields=fields, sizes=(size,), realizations=1, checkpoint=os.path.join(a.data, tag + ".ckpt"))
        pickle.dump(d, open(os.path.join(a.data, tag + ".pkl"), "wb"))
        cal = [x for x in d if x["field"] in CALIBRATION]
        s = {"calibration_families": evaluate(cal) if cal else None,
             "by_family": {name: evaluate([x for x in d if x["field"] == name]) for name in fields}}
        out["suite"][str(size)] = s
        if cal:
            print(f"suite {size}: six families AUC {s['calibration_families']['auc']:.3f}, wrong among accepted "
                  f"{s['calibration_families']['bad_accepted']:.3f}", flush=True)
        for name in fields:
            f = s["by_family"][name]
            print(f"  {name:11s} wrong {f['bad_rate']:.2f} | AUC {f['auc']:.3f} | accepted {f['accepted']:.2f} | wrong among accepted "
                  f"{f['bad_accepted'] if f['bad_accepted'] is None else round(f['bad_accepted'], 3)}", flush=True)
        save(out)


out = {"code_version": code_version()["label"], "calibration": "DEFAULT_CALIBRATION (3 indicators, 6 families), not refitted",
       "noise_models": list(NOISE_MODELS), "realizations": 1, "dry_run": bool(a.dry)}
if os.path.exists(a.json) and not a.dry:
    old = json.load(open(a.json, encoding="utf-8"))
    out.update({k: old[k] for k in ("regions", "regions_coarser", "suite") if k in old})
if a.dry:
    a.json = a.json.replace(".json", "_dry.json")
with keep_awake():
    if a.part in ("region", "all"):
        region(out)
    if a.part in ("suite", "all"):
        suite(out)
print("written", a.json)
