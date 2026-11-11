"""Revision R1: split by zones of the free-energy change, with a well-defined density.

The original analysis (``examples/12_flare_energy.py``) split (B² − B_p²)/8π per voxel.
That density only sums to the free energy integrated over the whole volume: locally it contains the
cross term 2 B_p·B_J, which does not vanish in a sub-volume. Here the density of the
current-carrying energy, |B − B_p|²/8π (B_p: volume potential of each time), is used, which is
positive and whose integral is E_J (Valori et al. 2013).

For each flare and each configuration (confidence threshold 0.7/0.8/0.9, seed density
``step`` and ``step/2``, dilation 0/1/2) we compute: energy stored before by zone,
change by zone, fraction of the drop in the reliable zone, relative loss of each zone and the
balance (sum of zones = total). It is compared with the original method. E, E_p,
E − E_p, E_J and its solenoidal part E_J,s are also given for each time.

    python examples/revision/r1_flare_energy.py
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solarff.confidence import confidence_map, reliable_volume  # noqa: E402
from solarff.helicity import energy_decomposition, trapezoid_weights, volume_potential  # noqa: E402

warnings.filterwarnings("ignore")
OUT = "docs/revision/resultados"
os.makedirs(OUT, exist_ok=True)
MM3 = 1e24  # Mm³ -> cm³
EVENTS = [
    ("11158", "X2.2", "data/harp377_study_mc10.pkl", "data/harp377_20110215_0300_r4_nz48_mc10.pkl"),
    ("11429", "X5.4 (+X1.3 de 11430)", "data/harp1449_20120306_2336_r7_nz32_mc10.pkl", "data/harp1449_20120307_0200_r7_nz32_mc10.pkl"),
    ("12673", "X9.3", "data/harp7115_20170906_1136_r4_nz56_mc10.pkl", "data/harp7115_20170906_1300_r4_nz56_mc10.pkl"),
]


def load(path):
    st, mc = pickle.load(open(path, "rb"))
    f = st.nlfff
    pv = volume_potential(f)
    cell = f.dx ** 3 * MM3 / (8 * np.pi)
    w = trapezoid_weights(f.shape)
    dJ = ((f.bx - pv.bx) ** 2 + (f.by - pv.by) ** 2 + (f.bz - pv.bz) ** 2) * cell * w
    dOld = (f.b2 - pv.b2) * cell * w
    ed = energy_decomposition(f, pv)
    sig = None
    try:
        sig = float(mc["summary"]["E_free_vol"]["std"])
    except (TypeError, KeyError):
        pass
    return {"st": st, "f": f, "dJ": dJ, "dOld": dOld, "sig": sig,
            "E": {k: float(ed[k]) for k in ("E", "E_p", "E_J", "E_J_s", "E_div", "E_div_frac", "E_free")}}


def zones(I, step, thr, dil):
    st = I["st"]
    src = st.mag_pp if st.mag_pp is not None else st.mag
    cm = confidence_map(I["f"], src, step=step, noise_bt=float(st.mag.meta.get("noise") or 100.0))
    conf = np.asarray(cm["assess"]["confidence"], float)
    rel = conf >= thr
    lines = cm["lines"]
    vr = reliable_volume(I["f"], lines, rel, dilate=dil)
    vu = reliable_volume(I["f"], lines, ~rel, dilate=dil) & ~vr
    return {"reliable": vr, "unreliable": vu, "rest": ~(vr | vu)}, len(lines), float(rel.mean())


out = {"density": "|B - B_p,vol|^2 / 8pi (energía de las corrientes)", "events": []}
for noaa, name, p_pre, p_post in EVENTS:
    pre, post = load(p_pre), load(p_post)
    step0 = max(1, int(round(max(pre["st"].mag.shape) / 80)))
    ev = {"noaa": noaa, "flare": name, "energies_pre": pre["E"], "energies_post": post["E"],
          "dE_J": post["E"]["E_J"] - pre["E"]["E_J"], "dE_free": post["E"]["E_free"] - pre["E"]["E_free"],
          "dE_J_s": post["E"]["E_J_s"] - pre["E"]["E_J_s"], "sigma_noise_free": [pre["sig"], post["sig"]],
          "configs": []}
    print(f"\nNOAA {noaa} ({name}): E_J {pre['E']['E_J']:.3e} -> {post['E']['E_J']:.3e} ({ev['dE_J'] / pre['E']['E_J']:+.1%}); "
          f"E-E_p {pre['E']['E_free']:.3e} -> {post['E']['E_free']:.3e} ({ev['dE_free'] / pre['E']['E_free']:+.1%}); "
          f"E_J,s {ev['dE_J_s'] / pre['E']['E_J_s']:+.1%}", flush=True)
    for step in sorted({step0, max(1, step0 // 2)}, reverse=True):
        for thr in (0.7, 0.8, 0.9):
            for dil in (0, 1, 2):
                if (thr != 0.8 or step != step0) and dil != 1:
                    continue  # the dilation is varied only in the default configuration
                Z, nl, fr = zones(pre, step, thr, dil)
                c = {"step": step, "threshold": thr, "dilate": dil, "n_lines": nl, "frac_reliable": fr, "zones": {}}
                tot_new = float((post["dJ"] - pre["dJ"]).sum())
                tot_old = float((post["dOld"] - pre["dOld"]).sum())
                for k, m in Z.items():
                    e_pre = float(pre["dJ"][m].sum())
                    d_new = float((post["dJ"][m] - pre["dJ"][m]).sum())
                    d_old = float((post["dOld"][m] - pre["dOld"][m]).sum())
                    c["zones"][k] = {"volume_frac": float(m.mean()), "E_J_pre": e_pre, "dE_J": d_new,
                                     "rel_loss": d_new / e_pre if e_pre else None, "dE_old_method": d_old}
                c["frac_drop_reliable"] = c["zones"]["reliable"]["dE_J"] / tot_new if tot_new else None
                c["frac_drop_reliable_old_method"] = c["zones"]["reliable"]["dE_old_method"] / tot_old if tot_old else None
                c["balance_error"] = (sum(z["dE_J"] for z in c["zones"].values()) - tot_new) / abs(tot_new)
                ev["configs"].append(c)
                z = c["zones"]
                print(f"  step {step} umbral {thr} dil {dil}: fiables {fr:.0%} de {nl} líneas | vol fiable {z['reliable']['volume_frac']:.1%} | "
                      f"E_J antes en fiable {z['reliable']['E_J_pre'] / pre['E']['E_J']:.0%} | caída en fiable {c['frac_drop_reliable']:.0%} "
                      f"(método original {c['frac_drop_reliable_old_method']:.0%}) | pérdida relativa fiable/no fiable/resto "
                      f"{z['reliable']['rel_loss']:+.0%}/{z['unreliable']['rel_loss']:+.0%}/{z['rest']['rel_loss']:+.0%}", flush=True)
    out["events"].append(ev)
json.dump(out, open(os.path.join(OUT, "r1_flare_energy.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
