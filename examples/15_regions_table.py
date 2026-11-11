"""Comparative table of the three regions (task C).

Gathers, for the time before each flare, the main figures of each report
(``docs/informes/*/paper/results.json``), of the AIA validation (with the null
test, ``examples/13``) and of the before/after flare analysis (``examples/12``).
Writes ``docs/resultados/regions_summary.json`` (used by the report and by the regions table of the draft),
with the aggregated H5 (reliable − unreliable at equal height, three regions together).

    python examples/15_regions_table.py
"""
import json
import os
import pickle

REGIONS = [  # (NOAA, HARP, report, AIA validation)
    ("11158", 377, "docs/informes/harp377", "data/aia_validation_harp377.pkl"),
    ("11429", 1449, "docs/informes/harp1449_20120306_2336", "data/harp1449_20120306_2336_aia.pkl"),
    ("12673", 7115, "docs/informes/harp7115_20170906_1136", "data/harp7115_20170906_1136_aia.pkl"),
]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "resultados")
flare = {e["noaa"]: e for e in json.load(open(os.path.join(RES, "flare_energy.json"), encoding="utf-8"))["events"]}

rows = []
for noaa, harp, rep, aia_pkl in REGIONS:
    R = json.load(open(os.path.join(rep, "paper", "results.json"), encoding="utf-8"))
    d, q, h = R["data"], R["quality"]["nlfff"], R["helicity"]["nlfff"]
    rel = R["reliability"]
    mc = R.get("mc") or {}
    row = {"noaa": noaa, "harp": harp, "t_rec": d["t_rec"], "lon": d["lon"], "lat": d["lat"],
           "grid": [d["nx"], d["ny"], d["nz"]], "dx_Mm": d["dx"], "bz_max": max(d["bz_max"], -d["bz_min"]),
           "flux": d["flux_pos"], "CWsin": q["CWsin"], "fi": q["<|f_i|>"],
           "E_div_frac": h["energy"]["E_div_frac"], "E_free_vol": h["energy"]["E_free"],
           "E_free_base": R["energy"]["nlfff"]["free"], "gauge": h.get("gauge_diff"),
           "H_V": h["helicity"]["H_V"], "frac_reliable": rel["frac_reliable"],
           "volume_frac": rel.get("volume_frac"),
           "free_frac_reliable": (rel.get("energy") or {}).get("free_frac_of_total")}
    try:
        row["noise_E_free_vol"] = mc["summary"]["E_free_vol"]["std"] / abs(mc["summary"]["E_free_vol"]["nominal"])
    except (KeyError, TypeError, ZeroDivisionError):
        row["noise_E_free_vol"] = None
    S = pickle.load(open(aia_pkl, "rb"))["summary"]
    NB = (S.get("null") or {}).get("by_wavelength") or {}
    aia = {}
    for w, r in S["by_wavelength"].items():
        n = NB.get(w) or NB.get(int(w)) or NB.get(str(w))
        aia[str(w)] = {"reliable": r["median_reliable"], "unreliable": r["median_unreliable"],
                       "equal_height": r["ols_reliable"], "equal_height_ci": r["ols_reliable_ci"],
                       "nlfff_minus_pot_reliable": r["models"]["potential"]["reliable"]["median"],
                       "nlfff_minus_pot_unreliable": r["models"]["potential"]["unreliable"]["median"],
                       "chance": n["null"]["shift"]["all"]["median"] if n else None}
    row["aia"] = aia
    f = flare.get(noaa)
    if f:
        row["flare"] = {"name": f["flare"], "dE_vol": f["dE_vol"], "dE_vol_rel": f["dE_vol_rel"],
                        "dE_natural": f.get("dE_natural"), "sigma_noise": f.get("sigma_dE_noise"),
                        "frac_from_reliable": (f.get("H1") or {}).get("frac_reliable_of_drop"),
                        "pil_ratio": (f.get("H2") or {}).get("ratio"), "pil_ci": (f.get("H2") or {}).get("ci"),
                        "frac_rel_pre_post": [f["H3"]["pre"], f["H3"]["post"]]}
    rows.append(row)

# Aggregated H5: mean of the OLS coefficients (reliable − unreliable at equal height) of the three
# regions weighted by 1/σ², with σ from the 95 % CI of each region; 95 % CI = mean ± 1.96/√Σw.
pooled = {}
for w in ("171", "193"):
    x = [r["aia"][w]["equal_height"] for r in rows]
    sig = [(r["aia"][w]["equal_height_ci"][1] - r["aia"][w]["equal_height_ci"][0]) / (2 * 1.959964) for r in rows]
    wt = [1 / v ** 2 for v in sig]
    m = sum(wi * xi for wi, xi in zip(wt, x)) / sum(wt)
    half = 1.959964 / sum(wt) ** 0.5
    pooled[w] = {"value": m, "ci": [m - half, m + half], "n_regions": len(rows),
                 "method": "media de los coeficientes OLS por región ponderada por 1/sigma^2"}
out = {"regions": rows, "aia_equal_height_pooled": pooled}
json.dump(out, open(os.path.join(RES, "regions_summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(json.dumps(rows, indent=1, ensure_ascii=False)[:2000])
