"""Uncertainty budget of the free energy and the helicity.

Gathers in one table every source of uncertainty the package knows how to measure,
to see at a glance which one dominates:

* **Random noise** (this case): half-width of the 16–84 % interval of the Monte
  Carlo with the HMI error maps, relative to the median.
* **Noise bias** (this case): median of the Monte Carlo with respect to the nominal
  value. It is not corrected (see ``solarff/bias_validation.json``).
* **Method versus the exact solution** (synthetic): error of the NLFFF, without
  noise, with respect to the exact field, on the fields of ``05_bias_validation.py``.
* **Preprocessing**, **NLFFF parameters**, **buffer layer** and
  **resolution** (synthetic): change of the error with respect to the exact field when each
  choice is varied around the default value (``06_robustness.py``, mean of
  four cases).
* **NLFFF method** (this case): spread between the optimisation and Grad–Rubin,
  if it converged.

The values are signed **relative shifts** ``(lo, hi)`` (``lo ≤ 0 ≤ hi``
for the ranges). They are not added up: the sources are neither independent nor of the
same statistical meaning (one is a standard deviation, others are ranges when
a choice is varied), so the total would be misleading. The table serves to
identify the dominant source and to quote each one separately.
"""
from __future__ import annotations

import json
import os

import numpy as np

QUANTITIES = ("E_free", "E_free_vol", "H_V")
_ROB_KEY = {"E_free": "efree_err", "E_free_vol": "efree_vol_err", "H_V": "hv_err"}
_GROUPS = {"preprocessing": ("mu3=", "mu4="), "nlfff_params": ("relax_bottom=", "smooth=", "eps="),
           "buffer": ("nd=",), "resolution": ("size=",)}


def _load(name):
    p = os.path.join(os.path.dirname(__file__), name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _robustness_range(par: dict, prefixes, q):
    key = _ROB_KEY[q]
    base = par.get("default", {}).get(key)
    if base is None:
        return None, []
    d, names = [], []
    for name, v in par.items():
        if name.startswith(prefixes) and v.get(key) is not None and np.isfinite(v[key]):
            d.append(v[key] - base)
            names.append(name)
    if not d:
        return None, []
    return (min(0.0, min(d)), max(0.0, max(d))), names


def uncertainty_budget(A: dict | None = None, robustness: dict | None = None,
                       bias: dict | None = None) -> dict:
    """Uncertainty budget.

    ``A`` is the analysis of the report (``analysis.analyze``); if missing, only the
    sources measured with synthetic fields are included. ``robustness`` and
    ``bias`` are the summaries of ``06_robustness.py`` and
    ``05_bias_validation.py`` (by default, those shipped with the package).

    Returns ``{"rows": [...], "dominant": {q: key}, "definition": {...}}``.
    Each row: ``key``, ``kind`` (``random``/``systematic``), ``scope``
    (``case``/``synthetic``), ``values`` {q: (lo, hi) or None}, ``detail``.
    """
    A = A or {}
    robustness = robustness if robustness is not None else _load("robustness.json")
    bias = bias if bias is not None else _load("bias_validation.json")
    rows = []

    mc = A.get("mc")
    if mc:
        s = mc["summary"]
        rnd, bia = {}, {}
        for q in QUANTITIES:
            d = s.get(q)
            if d and d.get("p50"):
                h = (d["p84"] - d["p16"]) / 2 / abs(d["p50"])
                rnd[q] = (-h, h)
                b = d["p50"] / d["nominal"] - 1 if d.get("nominal") else None
                bia[q] = (min(0.0, b), max(0.0, b)) if b is not None else None
            else:
                rnd[q] = bia[q] = None
        rows.append({"key": "noise_random", "kind": "random", "scope": "case", "values": rnd,
                     "detail": {"n": mc["n"], "error_scale": mc["error_scale"]}})
        rows.append({"key": "noise_bias", "kind": "systematic", "scope": "case", "values": bia,
                     "detail": {"n": mc["n"]}})

    if bias and bias.get("refs"):
        vals = {}
        for q in QUANTITIES:
            e = [r["model_clean"][q] / r["exact"][q] - 1 for r in bias["refs"].values()
                 if r["exact"].get(q)]
            vals[q] = (min(0.0, min(e)), max(0.0, max(e))) if e else None
        rows.append({"key": "method_exact", "kind": "systematic", "scope": "synthetic", "values": vals,
                     "detail": {"fields": list(bias["refs"]), "size": bias.get("size")}})

    par = (robustness or {}).get("parameters") or {}
    if par:
        for key, prefixes in _GROUPS.items():
            vals, names = {}, set()
            for q in QUANTITIES:
                vals[q], nm = _robustness_range(par, prefixes, q)
                names.update(nm)
            if any(v is not None for v in vals.values()):
                rows.append({"key": key, "kind": "systematic", "scope": "synthetic", "values": vals,
                             "detail": {"variants": sorted(names), "size": robustness.get("size"),
                                        "cases": robustness.get("cases")}})

    M = A.get("methods")
    if M:
        sp = M.get("spread") or {}
        vals = {}
        for q in QUANTITIES:
            d = sp.get(q)
            vals[q] = (-d["rel_halfrange"], d["rel_halfrange"]) if d and d.get("rel_halfrange") is not None else None
        rows.append({"key": "method_gr", "kind": "systematic", "scope": "case", "values": vals,
                     "detail": {"excluded": M.get("excluded") or [],
                                "estimated": any(v is not None for v in vals.values())}})

    dominant = {}
    for q in QUANTITIES:
        best = max(((max(abs(r["values"][q][0]), abs(r["values"][q][1])), r["key"]) for r in rows
                    if r["values"].get(q) is not None), default=None)
        dominant[q] = best[1] if best else None

    definition = None
    nl = (M or {}).get("models", {}).get("nlfff") if M else None
    if nl and nl.get("E_free") and nl.get("E_free_vol"):
        definition = {"E_free": nl["E_free"], "E_free_vol": nl["E_free_vol"],
                      "ratio": nl["E_free_vol"] / nl["E_free"]}
    return {"rows": rows, "dominant": dominant, "definition": definition}


def magnitude(v) -> float | None:
    """Largest absolute shift of a range ``(lo, hi)``."""
    return None if v is None else max(abs(v[0]), abs(v[1]))
