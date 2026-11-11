"""Before and after the flares: where is the energy released? (hypotheses H1–H4)

For each region three times (earlier reference, before and after the flare) are
compared, extrapolated with the same method and the same grid. The hypotheses were
registered before seeing the results (Appendix A of the paper):

* **H1** — the released current-carrying energy (|B − B_p|², volume potential) drops in the
  volume threaded by the reliable lines *before* the flare: fraction
  > 0.5 and total drop larger than the noise uncertainty (Monte Carlo) and than the
  natural variation (reference → before).
* **H2** — before the flare, the unreliable lines concentrate over the polarity
  inversion line (PIL): ratio P(unreliable | footpoint < 5 Mm from the
  PIL) / P(unreliable | rest) > 1 with a bootstrap 95 % CI excluding 1.
* **H3** — after the flare the fraction of reliable lines increases.
* **H4** — sign of the free-energy change with the base potential and with the
  volume potential.

Saves ``docs/resultados/flare_energy.json`` (used by the regions table, the figures and the report).

    python examples/12_flare_energy.py
"""
import glob
import json
import os
import pickle
import warnings

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

import solarff as sff
from solarff.confidence import confidence_map, reliable_volume
from solarff.helicity import trapezoid_weights, volume_potential
from solarff.metrics import energy

warnings.filterwarnings("ignore")

#: region: (NOAA, flare, TAI times of the reference, before and after)
EVENTS = [
    ("11158", "X2.2 2011-02-15 01:44 UTC", "20110214_2200", "20110215_0000", "20110215_0300", 377),
    ("11429", "X5.4 2012-03-07 00:02 UTC", "20120306_2136", "20120306_2336", "20120307_0200", 1449),
    ("12673", "X9.3 2017-09-06 11:53 UTC", "20170906_1024", "20170906_1136", "20170906_1300", 7115),
]
PIL_DIST_MM = 5.0
PIL_BZ_G = 150.0


def find_study(harp, stamp):
    if harp == 377 and stamp == "20110215_0000":
        cands = ["data/harp377_study_mc10.pkl"]
    else:
        cands = sorted(glob.glob(f"data/harp{harp}_{stamp}_r*_mc*.pkl"), key=lambda p: "mc10" not in p)
    return next((c for c in cands if os.path.exists(c)), None)


def instant(path):
    study, mc = pickle.load(open(path, "rb"))
    f = study.nlfff
    src = study.mag_pp if study.mag_pp is not None else study.mag
    step = max(1, int(round(max(study.mag.shape) / 80)))  # as in the report
    cm = confidence_map(f, src, step=step, noise_bt=float(study.mag.meta.get("noise") or 100.0))
    lines, rel = cm["lines"], np.asarray(cm["assess"]["reliable"], dtype=bool)
    pv = volume_potential(f)
    # Density of the current-carrying energy |B − B_p|²/8π (erg per cell). It is positive and its
    # integral is E_J (Valori et al. 2013). The earlier version used (B² − B_p²)/8π, which only
    # gives the free energy integrated over the whole volume: in a sub-volume it includes the
    # cross term 2 B_p·B_J, which does not vanish (revision of 9-Nov-2026, docs/revision/).
    dens = (((f.bx - pv.bx) ** 2 + (f.by - pv.by) ** 2 + (f.bz - pv.bz) ** 2) / (8 * np.pi) * f.dx ** 3 * 1e24
            * trapezoid_weights(f.shape))  # trapezoid weights, as in energy_decomposition
    e_base = energy(f) - energy(sff.reference_potential(study))
    sig = None
    if mc and "summary" in mc and "E_free_vol" in mc["summary"]:
        sig = float(mc["summary"]["E_free_vol"]["std"])
    return {"study": study, "f": f, "lines": lines, "rel": rel, "dens": dens, "E_vol": float(dens.sum()),
            "E_base": float(e_base), "sigma_vol": sig, "frac_rel": float(rel.mean()) if len(rel) else 0.0,
            "n_lines": len(lines)}


def pil_ratio(I, n_boot=2000, seed=0):
    """H2: P(unreliable | footpoint near the PIL) / P(unreliable | rest), with bootstrap CI."""
    mag = I["study"].mag
    bz = gaussian_filter(np.asarray(mag.bz, dtype=float), 1.0)
    pos, neg = bz > PIL_BZ_G, bz < -PIL_BZ_G
    pil = binary_dilation(pos) & binary_dilation(neg)
    if not pil.any():
        return None
    dist = distance_transform_edt(~pil) * mag.dx  # Mm
    origin = np.asarray(I["f"].origin)[:2]
    up = np.array(bz.shape) - 1
    d = []
    for l in I["lines"]:
        ends = np.array([l.start[:2], l.end[:2]])
        idx = np.clip(np.rint((ends - origin) / mag.dx).astype(int), 0, up)
        d.append(dist[idx[:, 0], idx[:, 1]].min())
    d = np.array(d)
    unrel = ~I["rel"]
    near = d < PIL_DIST_MM
    if near.sum() < 5 or (~near).sum() < 5:
        return None

    def ratio(u, n):
        a, b = u[n].mean(), u[~n].mean()
        return a / b if b > 0 else np.nan
    r = ratio(unrel, near)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(d), len(d))
        if near[k].sum() and (~near[k]).sum():
            boots.append(ratio(unrel[k], near[k]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"ratio": float(r), "ci": [float(lo), float(hi)], "n_near": int(near.sum()), "n_far": int((~near).sum()),
            "unrel_near": float(unrel[near].mean()), "unrel_far": float(unrel[~near].mean()),
            "pil_dist_mm": PIL_DIST_MM, "pil_bz_G": PIL_BZ_G}


out = {"events": [], "criteria": "registered hypotheses H1-H4 (Appendix A of the paper)", "energy": "E_J = |B - B_p,vol|^2/8pi por vóxel; E_vol = su integral"}
for noaa, flare, t_ref, t_pre, t_post, harp in EVENTS:
    paths = {k: find_study(harp, t) for k, t in (("ref", t_ref), ("pre", t_pre), ("post", t_post))}
    print(f"\nNOAA {noaa} ({flare}): " + ", ".join(f"{k}={v}" for k, v in paths.items()), flush=True)
    if not paths["pre"] or not paths["post"]:
        print("  faltan instantes; se omite")
        continue
    I = {k: instant(p) for k, p in paths.items() if p}
    pre, post = I["pre"], I["post"]
    ev = {"noaa": noaa, "harp": harp, "flare": flare, "times": {"ref": t_ref, "pre": t_pre, "post": t_post},
          "frac_reliable": {k: v["frac_rel"] for k, v in I.items()},
          "E_free_vol": {k: v["E_vol"] for k, v in I.items()}, "E_free_base": {k: v["E_base"] for k, v in I.items()},
          "sigma_noise": {k: v["sigma_vol"] for k, v in I.items()}}
    dE = post["E_vol"] - pre["E_vol"]
    ev["dE_vol"], ev["dE_vol_rel"] = dE, dE / pre["E_vol"]
    ev["dE_base"] = post["E_base"] - pre["E_base"]
    sig = [s for s in (pre["sigma_vol"], post["sigma_vol"]) if s]
    ev["sigma_dE_noise"] = float(np.sqrt(sum(s ** 2 for s in sig))) if sig else None
    if "ref" in I:
        ev["dE_natural"] = pre["E_vol"] - I["ref"]["E_vol"]
        ev["dE_natural_rel"] = ev["dE_natural"] / I["ref"]["E_vol"]
    # H1: same grid -> split of the change by zones defined by the lines BEFORE
    if pre["dens"].shape == post["dens"].shape:
        v_rel = reliable_volume(pre["f"], pre["lines"], pre["rel"])
        v_unr = reliable_volume(pre["f"], pre["lines"], ~pre["rel"]) & ~v_rel
        v_rest = ~(v_rel | v_unr)
        dd = post["dens"] - pre["dens"]
        parts = {"reliable": float(dd[v_rel].sum()), "unreliable": float(dd[v_unr].sum()), "rest": float(dd[v_rest].sum())}
        ev["H1"] = {"dE_by_zone": parts, "volume_frac": {"reliable": float(v_rel.mean()), "unreliable": float(v_unr.mean()),
                                                         "rest": float(v_rest.mean())},
                    "E_pre_by_zone": {"reliable": float(pre["dens"][v_rel].sum()),
                                      "unreliable": float(pre["dens"][v_unr].sum()),
                                      "rest": float(pre["dens"][v_rest].sum())},
                    "frac_reliable_of_drop": parts["reliable"] / dE if dE else None,
                    "n_lines": {"reliable": int(pre["rel"].sum()), "unreliable": int((~pre["rel"]).sum())}}
        thr = max(abs(ev.get("dE_natural") or 0.0), 2 * (ev["sigma_dE_noise"] or 0.0))
        ev["H1"]["drop_significant"] = bool(dE < 0 and abs(dE) > thr)
        ev["H1"]["threshold"] = thr
        ev["H1"]["pass"] = bool(ev["H1"]["drop_significant"] and (ev["H1"]["frac_reliable_of_drop"] or 0) > 0.5)
    else:
        ev["H1"] = {"error": f"mallas distintas {pre['dens'].shape} / {post['dens'].shape}"}
    ev["H2"] = pil_ratio(pre)
    if ev["H2"]:
        ev["H2"]["pass"] = bool(ev["H2"]["ci"][0] > 1)
    ev["H3"] = {"pre": pre["frac_rel"], "post": post["frac_rel"], "pass": bool(post["frac_rel"] > pre["frac_rel"])}
    ev["H4"] = {"sign_base": int(np.sign(ev["dE_base"])), "sign_vol": int(np.sign(dE)),
                "agree": bool(np.sign(ev["dE_base"]) == np.sign(dE))}
    out["events"].append(ev)
    h1 = ev["H1"]
    print(f"  E_free (vol): ref {I['ref']['E_vol'] if 'ref' in I else float('nan'):.3e}, antes {pre['E_vol']:.3e}, "
          f"después {post['E_vol']:.3e} -> ΔE {dE:+.3e} ({ev['dE_vol_rel']:+.1%}); "
          f"natural {ev.get('dE_natural', float('nan')):+.3e}; ruido σ {ev['sigma_dE_noise']}")
    if "dE_by_zone" in h1:
        print(f"  H1: ΔE por zona {h1['dE_by_zone']}; fracción en fiables {h1['frac_reliable_of_drop']}; "
              f"significativa {h1['drop_significant']} -> {'SÍ' if h1['pass'] else 'NO'}")
    else:
        print("  H1:", h1)
    print(f"  H2: {ev['H2']}")
    print(f"  H3: fiables {pre['frac_rel']:.1%} -> {post['frac_rel']:.1%}")
    print(f"  H4: ΔE base {ev['dE_base']:+.3e}, vol {dE:+.3e}, mismo signo {ev['H4']['agree']}")

n = len(out["events"])
summ = {}
for h, key in (("H1", "pass"), ("H2", "pass"), ("H3", "pass")):
    vals = [e[h].get(key) for e in out["events"] if isinstance(e.get(h), dict) and key in e[h]]
    summ[h] = {"n_pass": int(sum(bool(v) for v in vals)), "n": len(vals), "pass": bool(sum(bool(v) for v in vals) >= 2)}
summ["H4"] = {"n_agree": sum(e["H4"]["agree"] for e in out["events"]), "n": n}
out["summary"] = summ
print("\nResumen:", json.dumps(summ))
with open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "resultados"), "flare_energy.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
