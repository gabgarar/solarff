"""Automatic PDF report (Spanish and English) of an extrapolation.

Usage::

    study = sff.run_study(mag, nz=48)
    sff.generate_report(study, "informe")

Generates in ``informe/``:

* ``informe_es.pdf`` / ``report_en.pdf`` (and their ``.tex``)
* ``figures/es/*.png``, ``figures/en/*.png``
* ``paper/``: ``methods_es.tex``, ``methods_en.tex``, ``table_quality.tex``,
  ``table_comparison.tex``, ``references.bib``, ``results.json``
"""

from __future__ import annotations

import json
import os
import platform

import numpy as np

from .analysis import analyze
from .figures import make_figures, NAMES
from .latex import build_tex, compile_pdf, find_pdflatex, methods_paragraph, num
from .texts import BIBTEX
from ..utils import keeps_awake

__all__ = ["generate_report", "analyze"]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _table_quality(A) -> str:
    names = NAMES["en"]
    keys = list(A["quality"])
    out = [r"\begin{table}", r"\centering",
           r"\caption{Force-free quality and energy of the extrapolated fields.}",
           r"\label{tab:quality}", r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\hline\hline",
           " & " + " & ".join(names[k] for k in keys) + r" \\", r"\hline"]
    out.append(r"$\mathrm{CW}_{\sin}$ & " + " & ".join(num(A["quality"][k]["CWsin"]) for k in keys) + r" \\")
    out.append(r"$\langle|f_i|\rangle$ & " + " & ".join(num(A["quality"][k]["<|f_i|>"], 2) for k in keys) + r" \\")
    out.append(r"$E$ (erg) & " + " & ".join(num(A["energy"][k]["E"]) for k in keys) + r" \\")
    out.append(r"$E/E_{\rm pot}$ & " + " & ".join(num(A["energy"][k]["ratio"], 4) for k in keys) + r" \\")
    out.append(r"$E_{\rm free}$ (erg) & " + " & ".join(num(A["energy"][k]["free"]) for k in keys) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def _table_comparison(A) -> str:
    if not A["vector"]:
        return "% No reference field available\n"
    names = NAMES["en"]
    keys = list(A["vector"])
    ref = names[A["reference"]]
    out = [r"\begin{table}", r"\centering",
           rf"\caption{{Comparison metrics (Schrijver et al. 2006) with respect to the {ref} field.}}",
           r"\label{tab:comparison}", r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\hline\hline",
           " & " + " & ".join(names[k] for k in keys) + r" \\", r"\hline"]
    for lab, key in ((r"$C_{\rm vec}$", "C_vec"), (r"$C_{\rm CS}$", "C_CS"), (r"$E'_n$", "E'_n"),
                     (r"$E'_m$", "E'_m"), (r"$\epsilon$", "eps")):
        out.append(lab + " & " + " & ".join(num(A["vector"][k][key]) for k in keys) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def _table_budget(A) -> str:
    """Uncertainty budget for the paper (English)."""
    from .latex import ReportBuilder
    B = A.get("budget")
    if not B or not B["rows"]:
        return "% No uncertainty budget available\n"
    out = [r"\begin{table}", r"\centering",
           r"\caption{Uncertainty budget of the free energy and relative helicity (relative shifts). "
           r"$\pm$: random $1\sigma$; signed ranges: effect of a modelling choice or bias. "
           r"Synthetic rows: exact force-free fields, one choice varied at a time. Sources are not added.}",
           r"\label{tab:budget}", r"\begin{tabular}{lllccc}", r"\hline\hline",
           r"Source & Type & Measured on & $E_{\rm free}$ (base pot.) & $E_{\rm free}$ (volume pot.) & $H_V$ \\",
           r"\hline"]
    for r in B["rows"]:
        lab = ReportBuilder.BUDGET_LABELS.get(r["key"], (r["key"], r["key"]))[1]
        cells = [ReportBuilder.budget_cell(r["values"].get(q)) for q in ("E_free", "E_free_vol", "H_V")]
        if r["key"] == "method_gr" and not r["detail"].get("estimated"):
            cells = ["not conv."] * 3
        out.append(f"{lab} & {r['kind']} & {'this region' if r['scope'] == 'case' else 'synthetic'} & "
                   + " & ".join(cells) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


@keeps_awake
def generate_report(study, out_dir: str = "report", langs=("es", "en"),
                    title: str | None = None, compile: bool = True, validation: bool = True,
                    monte_carlo: dict | None = None, aia: dict | None = None,
                    regions: dict | None = None, verbose: bool = True) -> dict:
    """Generate the full report of a :class:`solarff.Study`.

    Parameters
    ----------
    study : result of :func:`solarff.run_study`.
    out_dir : output folder (created if it does not exist).
    langs : languages to generate (``"es"``, ``"en"``).
    title : title of the report (optional).
    compile : compile to PDF with pdflatex (if not installed the ``.tex`` are left).
    validation : run the analytical validation suite (a few seconds) and
        include it as an appendix.
    monte_carlo : result of :func:`solarff.monte_carlo` (optional). Adds the
        uncertainties section and the stability of each line to the confidence.
    aia : result of :func:`solarff.aia.aia_validation` (optional). Adds the
        comparison of the lines with the loops observed by AIA.
    regions : results of a multi-region study (optional):
        ``{"summary": <regions_summary.json>, "flares": <flare_energy.json>}``, as those of
        ``examples/15_regions_table.py`` and ``examples/12_flare_energy.py``. Adds the section
        on several regions and flares. The library does not store analysis results.

    Returns a dictionary with the generated paths.
    """
    say = print if verbose else (lambda *a, **k: None)
    os.makedirs(out_dir, exist_ok=True)
    paper = os.path.join(out_dir, "paper")
    os.makedirs(paper, exist_ok=True)

    say("Analizando resultados...")
    A = analyze(study, monte_carlo=monte_carlo)
    if aia:
        A["aia"], A["_aia"] = aia["summary"], aia["plot"]
    if regions:
        A["regions"] = regions
    val = None
    if validation:
        from ..validation import run_validation
        say("Validación analítica...")
        val = run_validation()

    out = {"pdf": {}, "tex": {}, "paper": paper}
    can_compile = compile and find_pdflatex() is not None
    if compile and not can_compile:
        say("AVISO: no se encontró pdflatex; se generan sólo los .tex (compílalos en Overleaf).")

    for lang in langs:
        say(f"Figuras ({lang})...")
        figs = make_figures(study, A, lang, os.path.join(out_dir, "figures", lang), validation=val)
        tex = build_tex(study, A, figs, lang, fig_rel=f"figures/{lang}", title=title, validation=val)
        base = "informe_es" if lang == "es" else "report_en"
        tex_path = os.path.join(out_dir, base + ".tex")
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(tex)
        out["tex"][lang] = tex_path
        if can_compile:
            say(f"Compilando {base}.pdf...")
            out["pdf"][lang] = compile_pdf(tex_path)
        with open(os.path.join(paper, f"methods_{lang}.tex"), "w", encoding="utf-8") as fh:
            fh.write(methods_paragraph(study, A, lang) + "\n")

    with open(os.path.join(paper, "table_quality.tex"), "w", encoding="utf-8") as fh:
        fh.write(_table_quality(A))
    with open(os.path.join(paper, "table_comparison.tex"), "w", encoding="utf-8") as fh:
        fh.write(_table_comparison(A))
    with open(os.path.join(paper, "table_budget.tex"), "w", encoding="utf-8") as fh:
        fh.write(_table_budget(A))
    with open(os.path.join(paper, "references.bib"), "w", encoding="utf-8") as fh:
        fh.write(BIBTEX)
    from .. import __version__
    from ..utils import code_version
    results = {"solarff_version": __version__, "code_version_report": code_version(),
               "code_version_computation": study.config.get("code_version"),
               "python": platform.python_version(),
               "config": study.config, "timings_s": study.timings,
               **{k: v for k, v in A.items() if not k.startswith("_")}}
    if results.get("methods"):
        results["methods"] = {k: v for k, v in results["methods"].items() if not k.startswith("_")}
    if val:
        results["validation"] = [{k: v for k, v in t.items() if k not in ("curve", "q_values")} for t in val]
    from ..confidence import calibration_metrics, select_calibration
    used = (A.get("reliability") or {}).get("calibration") or ""
    cal = select_calibration(tuple(x for x in ("mc_instability", "method_disagreement") if x in used))
    results["confidence_calibration"] = {"parameters": cal, "metrics": calibration_metrics(cal)}
    with open(os.path.join(paper, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=2, ensure_ascii=False)

    say("Informe generado en " + os.path.abspath(out_dir))
    for lang, p in out["pdf"].items():
        say(f"  {p}")
    return out
