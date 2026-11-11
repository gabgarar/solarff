"""Computation of every number of the report and its automatic assessment.

The assessments use INDICATIVE thresholds taken from the literature
(Schrijver et al. 2006; Metcalf et al. 2008; DeRosa et al. 2009, 2015;
Wiegelmann & Sakurai 2021). They do not replace the judgement of the researcher.
"""

from __future__ import annotations

import numpy as np

from ..field import trapezoid_weights as _tw

from ..magnetogram import aly_criteria
from ..metrics import compare_vector_fields, compare_connectivity, energy, force_free_metrics
from ..plotting import line_alpha_stats
from ..potential import potential_field
from ..study import reference_potential
from ..stability import decay_index_profile, pil_mask
from ..topology import squashing_q, twist_map
from ..tracing import trace_field_lines

GOOD, CAUTION, POOR, INFO = "good", "caution", "poor", "info"
KEYS = ("exact", "potential", "lfff", "nlfff")


def _grade(v, good, caution, higher_is_better=False):
    if v is None or not np.isfinite(v):
        return INFO
    if higher_is_better:
        return GOOD if v >= good else (CAUTION if v >= caution else POOR)
    return GOOD if v < good else (CAUTION if v < caution else POOR)


def _models(study):
    return {k: getattr(study, k) for k in KEYS if getattr(study, k) is not None}


def analyze(study, monte_carlo: dict | None = None) -> dict:
    """Return a dictionary with data, metrics, assessments, warnings and findings."""
    A: dict = {}
    mag = study.mag
    dx = mag.dx
    dA = (dx * 1e8) ** 2  # cm²
    bz = mag.bz
    models = _models(study)
    nd = study.nlfff.meta.get("nd") if study.nlfff is not None else max(4, min(*mag.shape, study.nz) // 10)
    nd = int(nd)

    # ------------------------------------------------------------------ data
    meta = mag.meta
    d = {
        "nx": mag.shape[0], "ny": mag.shape[1], "nz": study.nz, "dx": dx,
        "fov_x": mag.shape[0] * dx, "fov_y": mag.shape[1] * dx, "height": (study.nz - 1) * dx,
        "bz_max": float(bz.max()), "bz_min": float(bz.min()),
        "flux_pos": float((_tw(bz.shape) * np.clip(bz, 0, None)).sum() * dA),
        "flux_neg": float((_tw(bz.shape) * np.clip(bz, None, 0)).sum() * dA),
        "imbalance": mag.flux_imbalance(), "has_vector": mag.has_vector,
        "bt_max": float(mag.bt.max()) if mag.has_vector else None,
        "source": meta.get("source", ""), "t_rec": meta.get("T_REC"),
        "harpnum": meta.get("HARPNUM"), "noaa": meta.get("NOAA_ARS") or meta.get("NOAA_AR"),
        "lon": meta.get("LON_FWT"), "lat": meta.get("LAT_FWT"),
        "benchmark": study.exact is not None,
    }
    A["data"] = d
    A["aly"] = aly_criteria(mag) if mag.has_vector else None
    A["aly_pp"] = aly_criteria(study.mag_pp) if study.mag_pp is not None else None
    if study.mag_pp is not None:
        p = study.mag_pp
        db = np.sqrt((p.bx - mag.bx) ** 2 + (p.by - mag.by) ** 2 + (p.bz - mag.bz) ** 2)
        bb = np.sqrt(mag.bx ** 2 + mag.by ** 2 + mag.bz ** 2)
        A["pp_change"] = float(np.sqrt((db ** 2).mean() / (bb ** 2).mean()))
        A["pp_change_bz"] = float(np.sqrt(((p.bz - mag.bz) ** 2).mean() / (mag.bz ** 2).mean()))
    else:
        A["pp_change"] = A["pp_change_bz"] = None

    # ---------------------------------------------------------------- alpha
    al = {k: v for k, v in study.alpha.items() if k != "curve"}
    if al.get("fit") is not None and al.get("jz") is not None:
        al["sign_agree"] = bool(np.sign(al["fit"]) == np.sign(al["jz"]))
    A["alpha"] = al

    # -------------------------------------------------------------- quality
    A["nd"] = nd
    A["quality"] = {k: force_free_metrics(f, margin=nd) for k, f in models.items()}
    e_pot = energy(reference_potential(study), nd)
    A["energy"] = {}
    for k, f in models.items():
        e = energy(f, nd)
        A["energy"][k] = {"E": e, "ratio": e / e_pot, "free": e - e_pot}

    # ---------------------------------------------------------- reference
    ref = "exact" if "exact" in models else ("nlfff" if "nlfff" in models else None)
    A["reference"] = ref
    A["vector"] = {}
    if ref:
        for k, f in models.items():
            if k != ref:
                A["vector"][k] = compare_vector_fields(models[ref], f, margin=nd)

    # -------------------------------------------------------------- lines
    lines = {k: trace_field_lines(f, study.seeds) for k, f in models.items()}
    A["lines"] = {}
    for k, ls in lines.items():
        means, stds = line_alpha_stats(models[k], ls)
        ok = np.isfinite(means) & np.isfinite(stds) & (np.abs(means) > 1e-3 * (al.get("max") or 1))
        rel = float(np.median(stds[ok] / np.abs(means[ok]))) if ok.sum() >= 5 else None
        A["lines"][k] = {
            "n": len(ls),
            "closed": float(np.mean([l.closed for l in ls])),
            "apex_mean": float(np.mean([l.apex for l in ls])),
            "apex_max": float(np.max([l.apex for l in ls])),
            "alpha_line_median": float(np.nanmedian(means)) if np.isfinite(means).any() else None,
            "alpha_rel_spread": rel,
        }
        if ref and k != ref:
            A["lines"][k].update(compare_connectivity(lines[ref], ls))

    # ------------------------------------------------- error by height
    A["heights"] = []
    if ref:
        H = d["height"]
        A["heights"] = [round(h * H, 1) for h in (0.08, 0.25, 0.5)]
        A["angle_by_height"] = {}
        R = models[ref]
        bmax = R.babs.max()
        for k, f in models.items():
            if k == ref:
                continue
            row = []
            for h in A["heights"]:
                iz = int(np.argmin(np.abs(R.z - h)))
                B = np.stack([R.bx[..., iz], R.by[..., iz], R.bz[..., iz]])
                b = np.stack([f.bx[..., iz], f.by[..., iz], f.bz[..., iz]])
                nB, nb = np.linalg.norm(B, axis=0), np.linalg.norm(b, axis=0)
                m = (nB > 0.01 * bmax) & (nb > 0)
                c = np.clip((B * b).sum(0)[m] / (nB[m] * nb[m]), -1, 1)
                row.append(float((np.degrees(np.arccos(c)) * nB[m]).sum() / nB[m].sum()))
            A["angle_by_height"][k] = row

    # --------------------------------------------------------- convergence
    if study.nlfff is not None:
        h = study.nlfff_history
        A["convergence"] = {
            "iterations": int(study.nlfff.meta.get("iterations", len(h) - 1)),
            "L0": float(h[0][1]), "Lf": float(h[-1][1]),
            "reduction": float(h[0][1] / h[-1][1]) if h[-1][1] > 0 else None,
            "converged": "max_iter" not in study.nlfff_message,
            "message": study.nlfff_message,
            "method": study.nlfff.meta.get("backend", "classic"),
            "evaluations": study.nlfff.meta.get("evaluations"),
        }
    A["timings"] = dict(study.timings)

    # --- Relaxed base (NLFFF L-BFGS)
    A["relax"] = None
    if study.nlfff is not None and study.nlfff.meta.get("relax_bottom"):
        src = study.mag_pp if study.mag_pp is not None else mag
        dbx = study.nlfff.bx[:, :, 0] - src.bx
        dby = study.nlfff.by[:, :, 0] - src.by
        bt = np.hypot(src.bx, src.by)
        strong = bt > np.percentile(bt, 80)
        d2 = dbx ** 2 + dby ** 2

        def rel(m):
            return float(np.sqrt(d2[m].mean() / (bt[m] ** 2).mean())) if m.any() else None
        A["relax"] = {"nu": study.nlfff.meta["relax_bottom"], "change": rel(np.ones_like(strong)),
                      "change_strong": rel(strong), "change_weak": rel(~strong)}

    A["physics"], A["_maps"] = _physics(study, A, models)
    A["helicity"] = _helicity(models)
    A["methods"] = _methods(study, A)
    A["mc"] = _mc_summary(monte_carlo) if monte_carlo else None
    A["_mc"] = monte_carlo
    A["reliability"] = _reliability(study, A, models)
    from ..budget import uncertainty_budget
    A["budget"] = uncertainty_budget(A)

    A["assessments"] = _assessments(A, study)
    A["warnings"] = _warnings(A, study)
    A["findings"] = _findings(A, study)
    return A


# ------------------------------------------------------------- assessments
def _assessments(A, study):
    out = []
    d = A["data"]

    def add(key, value, status, fmt="{:.3g}"):
        out.append({"key": key, "value": value, "status": status, "fmt": fmt})

    add("imbalance", d["imbalance"], _grade(d["imbalance"], 0.1, 0.2), "{:.1%}")
    if d["lon"] is not None and np.isfinite(d["lon"]):
        add("position", abs(d["lon"]), _grade(abs(d["lon"]), 30, 45), "{:.0f}°")
    if A["aly"]:
        add("eps_force_raw", A["aly"]["eps_force"], _grade(A["aly"]["eps_force"], 0.1, 0.3))
        add("eps_torque_raw", A["aly"]["eps_torque"], _grade(A["aly"]["eps_torque"], 0.1, 0.3))
    if A["aly_pp"]:
        add("eps_force_pp", A["aly_pp"]["eps_force"], _grade(A["aly_pp"]["eps_force"], 0.05, 0.15))
        add("pp_change", A["pp_change"], _grade(A["pp_change"], 0.2, 0.4), "{:.1%}")
    al = A["alpha"]
    if al.get("used") is not None:
        add("alpha_clipped", abs(al.get("raw", al["used"])) / al["max"],
            POOR if al.get("clipped") else (CAUTION if abs(al["used"]) > 0.9 * al["max"] else GOOD),
            "{:.0%}")
    add("height", d["height"] / min(d["fov_x"], d["fov_y"]),
        _grade(d["height"] / min(d["fov_x"], d["fov_y"]), 0.5, 0.3, higher_is_better=True), "{:.2f}")
    if "nlfff" in A["quality"]:
        q = A["quality"]["nlfff"]
        c = A["convergence"]
        add("converged", c["iterations"], GOOD if c["converged"] else CAUTION, "{:d}")
        if A.get("relax"):
            add("base_change", A["relax"]["change_strong"],
                _grade(A["relax"]["change_strong"], 0.15, 0.3), "{:.1%}")
        add("cwsin", q["CWsin"], _grade(q["CWsin"], 0.1, 0.3))
        add("fi", q["<|f_i|>"], _grade(q["<|f_i|>"], 1e-3, 1e-2), "{:.2e}")
        r = A["energy"]["nlfff"]["ratio"]
        add("e_ratio", r, GOOD if 1.0 <= r < 2.0 else (CAUTION if 0.97 <= r < 1.0 or r >= 2 else POOR))
        add("open_lines", 1 - A["lines"]["nlfff"]["closed"],
            _grade(1 - A["lines"]["nlfff"]["closed"], 0.5, 0.75), "{:.0%}")
        rs = A["lines"]["nlfff"]["alpha_rel_spread"]
        if rs is not None:
            add("alpha_const", rs, _grade(rs, 0.3, 0.6), "{:.2f}")
    hel = A.get("helicity", {}).get("nlfff")
    if hel:
        add("e_div", hel["energy"]["E_div_frac"], _grade(hel["energy"]["E_div_frac"], 0.05, 0.1), "{:.1%}")
        add("gauge", hel["gauge_diff"], _grade(hel["gauge_diff"], 0.05, 0.15), "{:.1%}")
    if A["reference"] == "exact" and "nlfff" in A["vector"]:
        cv = A["vector"]["nlfff"]["C_vec"]
        add("cvec_exact", cv, _grade(cv, 0.99, 0.95, higher_is_better=True), "{:.3f}")
    return out


def _warnings(A, study):
    w = []
    for a in A["assessments"]:
        if a["status"] in (POOR, CAUTION):
            w.append({"key": a["key"], "status": a["status"], "value": a["value"], "fmt": a["fmt"]})
    al = A["alpha"]
    if al.get("sign_agree") is False:
        w.append({"key": "alpha_sign", "status": CAUTION, "value": None, "fmt": ""})
    if not A["data"]["has_vector"]:
        w.append({"key": "no_vector", "status": POOR, "value": None, "fmt": ""})
    return w


def _findings(A, study):
    f = []
    E = A["energy"]
    if "nlfff" in E:
        f.append({"key": "free_energy", "free": E["nlfff"]["free"], "ratio": E["nlfff"]["ratio"]})
    if "lfff" in E and "nlfff" in E:
        f.append({"key": "lfff_vs_nlfff", "r_l": E["lfff"]["ratio"], "r_n": E["nlfff"]["ratio"]})
    al = A["alpha"]
    if al.get("used") is not None:
        f.append({"key": "helicity", "alpha": al["used"], "lat": A["data"]["lat"]})
    if A["reference"] and A["vector"]:
        best = max(A["vector"], key=lambda k: A["vector"][k]["C_vec"])
        f.append({"key": "closest", "ref": A["reference"], "best": best,
                  "cvec": {k: v["C_vec"] for k, v in A["vector"].items()}})
    if A.get("angle_by_height") and "nlfff" in A["angle_by_height"]:
        f.append({"key": "height_trend", "heights": A["heights"],
                  "angles": A["angle_by_height"]})
    ph = A.get("physics", {})
    main = ph.get("main")
    if main and main in ph.get("twist", {}):
        f.append({"key": "twist", "model": main, **ph["twist"][main]})
    if ph.get("decay"):
        f.append({"key": "decay", **ph["decay"]})
    if main and ph.get("q", {}).get(main):
        f.append({"key": "qsl", "model": main, **ph["q"][main]})
    if A.get("methods"):
        f.append({"key": "methods", **{k: v for k, v in A["methods"].items() if k in ("spread", "ref", "excluded")}})
    if A.get("mc"):
        f.append({"key": "mc", **A["mc"]})
    hel = A.get("helicity", {}).get(main) if main else None
    if hel:
        f.append({"key": "helicity_rel", "model": main, "H_V": hel["helicity"]["H_V"],
                  "ratio": hel["helicity"]["ratio_HJ_HV"], "e_div": hel["energy"]["E_div_frac"],
                  "free_vol": hel["energy"]["E_free"], "Ep_vol": hel["energy"]["E_p"]})
    if A.get("reliability"):
        f.append({"key": "reliability", **{k: v for k, v in A["reliability"].items()
                                           if not isinstance(v, dict)}})
    if ph.get("combined"):
        f.append({"key": "combined", **ph["combined"]})
    if A.get("reliability") and A["reliability"].get("combined"):
        f.append({"key": "combined_reliable", **A["reliability"]["combined"]})
    if A["reference"]:
        topo = {k: v.get("same_topology") for k, v in A["lines"].items() if "same_topology" in v}
        if topo:
            f.append({"key": "connectivity", "ref": A["reference"], "topo": topo})
    return f


# ------------------------------------------------------- physical quantities
def _physics(study, A, models):
    """Tw, Q and decay index; returns (numerical summary, 2D maps)."""
    mag = study.mag
    step = max(1, int(round(max(mag.shape) / 80)))
    ph = {"step": step, "twist": {}, "q": {}}
    maps = {"twist": {}, "q": {}}

    # Main model: the NLFFF if it exists (it is the "result"), otherwise the LFFF
    main = "nlfff" if "nlfff" in models else ("lfff" if "lfff" in models else None)
    ph["main"] = main

    # --- Tw: only in models with currents
    for k in ("exact", "lfff", "nlfff"):
        if k not in models:
            continue
        tm = twist_map(models[k], step=step)
        tw, cl = tm["tw"], tm["closed"]
        valid = np.isfinite(tw) & cl
        atw = np.abs(tw[valid])
        ph["twist"][k] = {
            "max_abs": float(atw.max()) if atw.size else None,
            "p99_abs": float(np.percentile(atw, 99)) if atw.size else None,
            "frac_gt1": float((atw >= 1).mean()) if atw.size else 0.0,
            "n_gt1": int((atw >= 1).sum()),
            "median": float(np.median(tw[valid])) if atw.size else None,
            "sign": float(np.sign(np.nanmedian(tw[valid]))) if atw.size else 0.0,
        }
        maps["twist"][k] = tm

    # --- Decay index with a padded potential (pad=2): the periodic copies
    #     of the FFT distort the decay with height (see validation).
    ph["decay"] = None
    if pil_mask(mag.bz).any():
        pot_ext = potential_field(mag, nz=study.nz, pad=2)
        pr = decay_index_profile(pot_ext)
        zmax_ok = 0.25 * max(mag.shape) * mag.dx * 2  # reliable up to ~L_pad/4
        ph["decay"] = {"h_crit": pr["h_crit"], "n_pil_pixels": pr["n_pil_pixels"],
                       "z_reliable": float(min(zmax_ok, pr["z"][-1])),
                       "n_at": {f"{h:.0f}": float(np.interp(h, pr["z"], np.nan_to_num(pr["n"])))
                                for h in (5.0, 10.0, 20.0, 40.0) if h <= pr["z"][-1]}}
        maps["decay"] = pr

    # --- Q: potential (current-free reference) and main / exact model
    for k in ("potential", "exact", main):
        if k is None or k not in models or k in maps["q"]:
            continue
        qm = squashing_q(models[k], step=step)
        lq = qm["logq"]
        ok = np.isfinite(lq)
        ph["q"][k] = {"frac_logq_gt2": float((lq[ok] >= 2).mean()) if ok.any() else None,
                      "max_logq": float(np.nanmax(lq)) if ok.any() else None,
                      "frac_defined": float(ok.mean())}
        maps["q"][k] = qm

    # --- Joint reading: height of the most twisted lines versus h_crit
    ph["combined"] = None
    if main in maps["twist"] and ph["decay"] is not None:
        tm = maps["twist"][main]
        tw = np.where(tm["closed"], np.abs(tm["tw"]), np.nan)
        flat = np.flatnonzero(np.isfinite(tw.ravel()))
        if flat.size:
            top = flat[np.argsort(tw.ravel()[flat])[::-1][:30]]
            ix, iy = np.unravel_index(top, tw.shape)
            seeds = np.column_stack([tm["x"][ix], tm["y"][iy], np.full(ix.size, mag.dx * 0.5)])
            lines = trace_field_lines(models[main], seeds)
            apex = np.array([l.apex for l in lines])
            ph["combined"] = {"model": main, "tw_top_median": float(np.median(tw.ravel()[top])),
                              "apex_median": float(np.median(apex)),
                              "h_crit": ph["decay"]["h_crit"],
                              "is_rope": bool(np.median(tw.ravel()[top]) >= 1.0)}
    return ph, maps


# ----------------------------------------------------- per-line reliability
def _reliability(study, A, models):
    """Confidence of each line of the main model and physics restricted to the reliable ones."""
    from ..confidence import confidence_map, reliable_volume
    from ..topology import twist_number
    main = A["physics"].get("main")
    if main is None or main not in models:
        return None
    f = models[main]
    mag = study.mag
    src = study.mag_pp if study.mag_pp is not None else mag
    noise_bt = float(mag.meta.get("noise") or 100.0)
    step = A["physics"]["step"]
    extra = {}
    mc = A.get("_mc")
    if mc is not None and main == "nlfff" and mc.get("step") == step:
        extra["mc_instability"] = 1 - mc["stability"]
    gr = getattr(study, "grad_rubin", None) or {}
    if main == "nlfff" and gr:
        from ..confidence import line_errors
        from ..montecarlo import base_seeds
        seeds = base_seeds(f, step)
        lines_main = trace_field_lines(f, seeds)
        dis = []
        for g in gr.values():
            eg = line_errors(trace_field_lines(g, seeds), lines_main)
            dis.append((eg["shape"] > 0.1) | (eg["topology"] > 0))
        extra["method_disagreement"] = np.mean(dis, axis=0)
    cm = confidence_map(f, src if src.has_vector else None, step=step, noise_bt=noise_bt, extra=extra or None)
    A["_maps"]["confidence"] = cm
    lines = cm["lines"]
    rel = cm["assess"]["reliable"]
    conf = cm["assess"]["confidence"]
    ind = cm["assess"]["indicators"]
    R = {"model": main, "n_lines": len(lines), "frac_reliable": float(rel.mean()) if len(rel) else 0.0,
         "calibration": cm["assess"]["calibration"],
         "extras": sorted(extra) if extra else [],
         "conf_median": float(np.nanmedian(conf)), "threshold": cm["assess"]["threshold"],
         "noise_bt": noise_bt}
    R["indicators_median"] = {k: {"all": float(np.nanmedian(ind[k])),
                                  "reliable": float(np.nanmedian(ind[k][rel])) if rel.any() else None}
                              for k in ("sin_jb", "alpha_spread", "buffer", "apex")}
    # Energy in the well-determined volume
    vol = reliable_volume(f, lines, rel)
    if vol.any():
        def e(fld):
            return float((_tw(fld.shape) * fld.b2)[vol].sum() / (8 * np.pi) * fld.dx ** 3 * 1e24)
        e_m, e_p = e(f), e(reference_potential(study))
        # Current-carrying energy |B − B_p|²/8π (volume potential): it is the only well-defined
        # local density; B² − B_p² only gives the free energy integrated over the whole volume.
        from ..helicity import volume_potential
        pv = volume_potential(f)
        dJ = (((f.bx - pv.bx) ** 2 + (f.by - pv.by) ** 2 + (f.bz - pv.bz) ** 2) / (8 * np.pi) * f.dx ** 3 * 1e24
              * _tw(f.shape))
        R["volume_frac"] = float(vol.mean())
        R["energy"] = {"E": e_m, "E_pot": e_p, "ratio": e_m / e_p, "free": float(dJ[vol].sum()),
                       "free_frac_of_total": float(dJ[vol].sum() / dJ.sum()) if dJ.sum() else None}
    # Twist with reliable lines only
    tw = twist_number(f, lines)
    closed = np.array([l.closed for l in lines])
    atw = np.abs(tw)
    for tag, m in (("all", closed), ("reliable", closed & rel)):
        v = atw[m & np.isfinite(atw)]
        R[f"twist_{tag}"] = {"p99_abs": float(np.percentile(v, 99)) if v.size else None,
                             "frac_gt1": float((v >= 1).mean()) if v.size else 0.0, "n": int(v.size)}
    # Joint reading with reliable lines only
    R["combined"] = None
    dec = A["physics"].get("decay")
    m = closed & rel & np.isfinite(atw)
    if m.any() and dec is not None:
        idx = np.flatnonzero(m)
        top = idx[np.argsort(atw[idx])[::-1][:30]]
        apex = np.array([lines[i].apex for i in top])
        R["combined"] = {"model": main, "tw_top_median": float(np.median(atw[top])),
                         "apex_median": float(np.median(apex)), "h_crit": dec["h_crit"],
                         "is_rope": bool(np.median(atw[top]) >= 1.0), "n_used": int(top.size)}
    return R


# ------------------------------------------------- helicity and energy
def _helicity(models):
    """Energy decomposition and relative helicity of each model with currents."""
    from ..helicity import helicity_report
    out = {}
    for k in ("exact", "lfff", "nlfff"):
        if k not in models:
            continue
        r = helicity_report(models[k])
        f = models[k]
        b0 = np.abs(f.bz[:, :, 0])
        phi = float((_tw(b0.shape) * b0).sum() * (f.dx * 1e8) ** 2 / 2)  # unsigned flux / 2
        out[k] = {"energy": r["energy"], "helicity": r["helicity"],
                  "helicity_bottom": r["helicity_bottom_gauge"], "gauge_diff": r["gauge_diff"],
                  "phi": phi, "H_norm": r["helicity"]["H_V"] / phi ** 2 if phi else None}
    return out


# ------------------------------------------------------- Monte Carlo
def _mc_summary(mc):
    s = mc["summary"]
    out = {"n": mc["config"]["n"], "error_scale": mc["config"]["error_scale"],
           "source": mc["config"]["source"], "summary": s,
           "stable_frac": float((mc["stability"] >= 0.8).mean()),
           "stability_median": float(np.median(mc["stability"]))}
    for k in ("E_free", "E_free_vol", "H_V"):
        if k in s and s[k]["p50"]:
            out[f"{k}_rel"] = float((s[k]["p84"] - s[k]["p16"]) / 2 / abs(s[k]["p50"]))
    return out


# ------------------------------------------------- comparison of methods
def _methods(study, A):
    """NLFFF by optimisation versus Grad–Rubin P and N: metrics, energy, helicity, connectivity."""
    gr = getattr(study, "grad_rubin", None) or {}
    if not gr or study.nlfff is None:
        return None
    from ..helicity import helicity_report
    from ..topology import twist_number
    fields = {"nlfff": study.nlfff, **{f"gr_{p.lower()}": f for p, f in gr.items()}}
    nd = A["nd"]
    e_pot = energy(reference_potential(study), nd)
    seeds = study.seeds
    lines = {k: trace_field_lines(f, seeds) for k, f in fields.items()}
    out = {"ref": "exact" if study.exact is not None else "nlfff", "models": {}}
    for k, f in fields.items():
        q = force_free_metrics(f, nd)
        hr = helicity_report(f)
        e = energy(f, nd)
        tw = np.abs(twist_number(f, lines[k]))
        cl = np.array([l.closed for l in lines[k]])
        v = tw[cl & np.isfinite(tw)]
        row = {"CWsin": q["CWsin"], "fi": q["<|f_i|>"], "E": e, "E_ratio": e / e_pot, "E_free": e - e_pot,
               "E_free_vol": hr["energy"]["E_free"], "E_div_frac": hr["energy"]["E_div_frac"],
               "H_V": hr["helicity"]["H_V"], "HJ_HV": hr["helicity"]["ratio_HJ_HV"],
               "gauge_diff": hr["gauge_diff"], "tw_p99": float(np.percentile(v, 99)) if v.size else None,
               "iterations": f.meta.get("iterations"), "message": f.meta.get("message", ""),
               "converged": "max_iter" not in f.meta.get("message", "")}
        if study.exact is not None:
            row.update({"vs_exact": compare_vector_fields(study.exact, f, margin=nd)})
        if k != "nlfff":
            row["vs_nlfff"] = compare_vector_fields(study.nlfff, f, margin=nd)
            c = compare_connectivity(lines["nlfff"], lines[k])
            row["same_topology_nlfff"] = c["same_topology"]
            row["endpoint_dist_nlfff"] = c["mean_endpoint_dist_Mm"]
        out["models"][k] = row
    if "gr_p" in fields and "gr_n" in fields:
        out["p_vs_n"] = compare_vector_fields(fields["gr_p"], fields["gr_n"], margin=nd)
        out["p_vs_n"]["same_topology"] = compare_connectivity(lines["gr_p"], lines["gr_n"])["same_topology"]
    spread = {}
    conv = [k for k, r in out["models"].items() if r["converged"]]
    out["excluded"] = [k for k in out["models"] if k not in conv]
    for q in ("E_free", "E_free_vol", "H_V", "E_ratio", "HJ_HV"):
        vals = np.array([out["models"][k][q] for k in conv if out["models"][k].get(q) is not None])
        if vals.size >= 2:
            spread[q] = {"min": float(vals.min()), "max": float(vals.max()), "median": float(np.median(vals)),
                         "rel_halfrange": float((vals.max() - vals.min()) / 2 / abs(np.median(vals)))
                         if np.median(vals) else None}
    out["spread"] = spread
    out["_lines"] = lines
    return out
