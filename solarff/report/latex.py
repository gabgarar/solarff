"""Construction of the LaTeX document of the report (es / en) and compilation to PDF."""

from __future__ import annotations

import datetime as _dt
import glob
import re
import os
import shutil
import subprocess

import numpy as np

from .texts import ASSESS, GLOSSARY, REFERENCES, STATUS, fill
from .figures import NAMES

LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                 "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}",
                 "\\": r"\textbackslash{}"}


def esc(s) -> str:
    return "".join(LATEX_SPECIAL.get(c, c) for c in str(s))


def num(v, sig: int = 3) -> str:
    """Number in LaTeX (math mode) with scientific notation if needed."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    v = float(v)
    if v == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(v))))
    if -3 <= e < sig:
        return f"${v:.{sig}g}$".replace("e", r"\mathrm{e}")
    if sig <= e < 5:
        return f"${v:.0f}$"
    m = round(v / 10 ** e, sig - 1)
    if abs(m) >= 10:  # 9.97e-6 with 2 digits -> 1.0e-5, not 10.0e-6
        m, e = m / 10, e + 1
    return rf"${m:.{sig - 1}f}\times10^{{{e}}}$"


def fx(v, d: int = 2) -> str:
    """Number with ``d`` fixed decimals (for metric tables: 0.90, not 0.9)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"${float(v):.{d}f}$"


def pct(v, d: int = 1) -> str:
    if v is None:
        return "--"
    # No "\," before "\%": Spanish babel already adds the space and "\,\%" breaks the compilation
    return f"{100 * v:.{d}f}\\%"


_ITEM_RE = re.compile(r"(?:(?<=\s)|^)\((\d{1,2})\)\s")


def _as_list(text: str) -> str:
    """Convert a paragraph with items "(1) … (2) …" into a numbered list.

    What comes before "(1)" is kept as an introductory paragraph. Only acts if the
    markers run consecutively from 1 (at least two) and outside formulas.
    """
    marks = []
    for m in _ITEM_RE.finditer(text):
        if text.count("$", 0, m.start()) % 2:  # inside a formula
            continue
        if int(m.group(1)) == len(marks) + 1:
            marks.append(m)
    if len(marks) < 2:
        return text
    head = text[:marks[0].start()].rstrip()
    items = [text[a.end():(b.start() if b is not None else len(text))].strip()
             for a, b in zip(marks, marks[1:] + [None])]
    body = "\n".join(r"\item " + it for it in items)
    return ((head + "\n") if head else "") + "\\begin{enumerate}[label=(\\arabic*)]\n" + body + "\n\\end{enumerate}"


def find_pdflatex() -> str | None:
    p = shutil.which("pdflatex")
    if p:
        return p
    cands = [os.path.expandvars(r"%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"),
             r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"]
    cands += glob.glob(r"C:\texlive\*\bin\windows\pdflatex.exe")
    cands += glob.glob(r"C:\texlive\*\bin\win64\pdflatex.exe")
    return next((c for c in cands if os.path.exists(c)), None)


def compile_pdf(tex_path: str, runs: int = 2) -> str:
    exe = find_pdflatex()
    if exe is None:
        raise FileNotFoundError("No se encontró pdflatex. Instala MiKTeX o TeX Live, o compila el .tex en Overleaf.")
    d, name = os.path.split(os.path.abspath(tex_path))
    for _ in range(runs):
        r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", name],
                           cwd=d, capture_output=True, text=True, errors="replace", timeout=900)
        if r.returncode != 0:
            log = r.stdout[-3000:]
            raise RuntimeError(f"pdflatex falló al compilar {name}:\n{log}")
    for ext in (".aux", ".log", ".out", ".toc"):
        try:
            os.remove(os.path.join(d, name[:-4] + ext))
        except OSError:
            pass
    return os.path.join(d, name[:-4] + ".pdf")


class ReportBuilder:
    def __init__(self, study, A, figs: dict, lang: str, fig_rel: str, title: str | None,
                 validation=None):
        self.s, self.A, self.lang = study, A, lang
        self.figs = {k: f"{fig_rel}/{os.path.basename(v)}" for k, v in figs.items()}
        self.names = NAMES[lang]
        self.title = title
        self.validation = validation
        self.out: list[str] = []

    # -------------------------------------------------------------- helpers
    def L(self, es: str, en: str) -> str:
        return es if self.lang == "es" else en

    def w(self, *chunks):
        self.out.extend(chunks)

    def p(self, es, en, **kw):
        self.w(_as_list(fill(self.L(es, en), **kw)) + "\n\n")

    def box(self, kind, es, en, **kw):
        titles = {"explain": ("¿Qué significa?", "What does it mean?"),
                  "read": ("Cómo leer esta figura", "How to read this figure"),
                  "key": ("Idea clave", "Key idea"),
                  "warn": ("Atención", "Caution")}
        t = titles[kind][0 if self.lang == "es" else 1]
        self.w(f"\\begin{{{kind}box}}{{{t}}}\n" + fill(self.L(es, en), **kw) + f"\n\\end{{{kind}box}}\n\n")

    def figure(self, key, cap_es, cap_en, width=r"\textwidth", **kw):
        if key not in self.figs:
            return False
        self.w("\\begin{figure}[H]\\centering\n"
               f"\\includegraphics[width={width}]{{{self.figs[key]}}}\n"
               f"\\caption{{{fill(self.L(cap_es, cap_en), **kw)}}}\\label{{fig:{key}}}\n"
               "\\end{figure}\n\n")
        return True

    def status(self, st):
        lab = STATUS[st][0 if self.lang == "es" else 1]
        return f"\\st{st}{{{lab}}}"

    def conv_status(self, c):
        m = c["message"]
        if "max_iter" in m:
            return self.L("se alcanzó el máximo de iteraciones sin converger",
                          "maximum number of iterations reached without convergence")
        if "mu" in m and "pequeno" in m:
            return self.L(r"el paso $\mu$ se hizo despreciable (mínimo local alcanzado)",
                          r"the step $\mu$ became negligible (local minimum reached)")
        import re
        r = re.search(r"=\s*([0-9.eE+-]+)", m)
        rel = num(float(r.group(1)), 2) if r else ""
        return self.L(f"convergido (variación relativa de $L$ por iteración {rel})",
                      f"converged (relative change of $L$ per iteration {rel})")

    def mname(self, k):
        return self.names[k]

    # ------------------------------------------------------------ document
    def build(self) -> str:
        self.preamble()
        self.titlepage()
        self.howto()
        self.summary()
        self.background()
        self.data()
        self.methods()
        self.quality()
        self.comparison()
        self.montecarlo()
        self.methods_comparison()
        self.uncertainty_budget()
        self.reliability()
        self.aia_section()
        self.regions_section()
        self.physics()
        self.warnings_section()
        self.paper()
        self.glossary()
        self.validation_section()
        self.references()
        self.w("\\end{document}\n")
        from .texts import resolve_citations
        return resolve_citations("".join(self.out))

    def preamble(self):
        babel = "spanish,es-noshorthands,es-nodecimaldot,es-tabla" if self.lang == "es" else "english"
        self.w(r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[""" + babel + r"""]{babel}
\usepackage{amsmath,amssymb,bm,booktabs,graphicx,longtable,caption,float,enumitem,microtype,array}
\usepackage[dvipsnames]{xcolor}
\usepackage[margin=2.3cm]{geometry}
\usepackage[most]{tcolorbox}
\definecolor{accent}{HTML}{2A78D6}
\definecolor{ink2}{HTML}{52514E}
\definecolor{stg}{HTML}{0CA30C}
\definecolor{stc}{HTML}{C98500}
\definecolor{stp}{HTML}{D03B3B}
\usepackage[colorlinks=true,linkcolor=accent,urlcolor=accent,citecolor=accent]{hyperref}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.45em}\setlength{\parindent}{0pt}
\newcommand{\stgood}[1]{\textcolor{stg}{$\checkmark$}~#1}
\newcommand{\stcaution}[1]{\textcolor{stc}{\textbf{!}}~#1}
\newcommand{\stpoor}[1]{\textcolor{stp}{$\bm{\times}$}~#1}
\newcommand{\stinfo}[1]{\textcolor{ink2}{$\bullet$}~#1}
\newtcolorbox{explainbox}[1]{enhanced,breakable,colback=accent!4,colframe=accent!70,
  fonttitle=\bfseries\small,title=#1,boxrule=0.5pt,arc=2pt}
\newtcolorbox{readbox}[1]{enhanced,breakable,colback=black!2,colframe=ink2!60,
  fonttitle=\bfseries\small,title=#1,boxrule=0.5pt,arc=2pt}
\newtcolorbox{keybox}[1]{enhanced,breakable,colback=stg!5,colframe=stg!70,
  fonttitle=\bfseries\small,title=#1,boxrule=0.5pt,arc=2pt}
\newtcolorbox{warnbox}[1]{enhanced,breakable,colback=stp!4,colframe=stp!70,
  fonttitle=\bfseries\small,title=#1,boxrule=0.5pt,arc=2pt}
\begin{document}
""")

    def region_label(self):
        d = self.A["data"]
        if d["benchmark"]:
            return self.L("Solución analítica de Low \\& Lou (1990)", "Low \\& Lou (1990) analytic solution")
        parts = []
        if d.get("noaa"):
            parts.append(f"NOAA {esc(d['noaa'])}")
        if d.get("harpnum"):
            parts.append(f"HARP {esc(d['harpnum'])}")
        if d.get("t_rec"):
            parts.append(esc(d["t_rec"]))
        return ", ".join(parts) or esc(self.s.name or self.L("Región", "Region"))

    def titlepage(self):
        from ..utils import code_version
        cv = code_version()
        cc = (self.s.config.get("code_version") or {})
        title = self.title or self.L("Informe de extrapolación del campo magnético coronal",
                                     "Coronal magnetic field extrapolation report")
        today = _dt.date.today().isoformat()
        self.w(r"\begin{titlepage}\centering\vspace*{3cm}" "\n"
               r"{\Huge\bfseries " + esc(title) + r"\par}\vspace{1.2cm}" "\n"
               r"{\Large " + self.region_label() + r"\par}\vspace{0.8cm}" "\n"
               r"{\large " + self.L("Potencial, lineal (LFFF) y no lineal (NLFFF) libres de fuerzas",
                                    "Potential, linear (LFFF) and nonlinear (NLFFF) force-free models")
               + r"\par}\vfill" "\n"
               r"{\small\color{ink2} " + self.L("Generado automáticamente con", "Automatically generated with")
               + f" \\texttt{{solarff}} {esc(cv['label'])} --- {today}\\par}}" "\n"
               + ((r"{\small\color{ink2} " + self.L("Cálculos realizados con", "Computations performed with")
                   + f" \\texttt{{solarff}} {esc(cc['label'])}\\par}}" "\n")
                  if cc.get("label") and cc.get("label") != cv["label"] else "")
               + r"\end{titlepage}" "\n" r"\tableofcontents\newpage" "\n")

    # ------------------------------------------------------------ sections
    def howto(self):
        self.w(r"\section*{" + self.L("Cómo usar este informe", "How to use this report") + "}\n")
        self.w(r"\addcontentsline{toc}{section}{" + self.L("Cómo usar este informe", "How to use this report") + "}\n")
        self.p(
            r"""Este informe resume una extrapolación del campo magnético de la corona solar y explica, paso a paso, qué se ha hecho y cómo interpretar cada resultado. Está pensado para leerse de dos formas:
\begin{itemize}[nosep]
\item \textbf{Lectura rápida:} el \hyperref[sec:summary]{resumen} (sección~\ref{sec:summary}) da los resultados principales y un semáforo de calidad; la sección~\ref{sec:warnings} lista los problemas detectados y qué hacer con ellos.
\item \textbf{Lectura completa:} las secciones~\ref{sec:background}--\ref{sec:comparison} explican la física, los datos, los métodos y cada figura. Los recuadros azules (\emph{¿Qué significa?}) explican conceptos; los grises (\emph{Cómo leer esta figura}) guían la lectura de las figuras.
\end{itemize}
La sección~\ref{sec:physics} valora si la región es eruptiva (torsión, inestabilidad de toro y QSL), y el Apéndice~\ref{sec:validation} demuestra que cada herramienta reproduce soluciones analíticas exactas. La sección~\ref{sec:paper} contiene material listo para un artículo: un párrafo de métodos con los valores de este caso, las tablas en LaTeX y las referencias en BibTeX (también exportadas como ficheros sueltos).""",
            r"""This report summarises an extrapolation of the solar coronal magnetic field and explains, step by step, what was done and how to interpret each result. It is meant to be read in two ways:
\begin{itemize}[nosep]
\item \textbf{Quick read:} the \hyperref[sec:summary]{summary} (Section~\ref{sec:summary}) gives the main results and a quality traffic light; Section~\ref{sec:warnings} lists the problems detected and what to do about them.
\item \textbf{Full read:} Sections~\ref{sec:background}--\ref{sec:comparison} explain the physics, the data, the methods and every figure. Blue boxes (\emph{What does it mean?}) explain concepts; grey boxes (\emph{How to read this figure}) guide the reading of figures.
\end{itemize}
Section~\ref{sec:physics} assesses whether the region is eruptive (twist, torus instability and QSLs), and Appendix~\ref{sec:validation} shows that every tool reproduces exact analytic solutions. Section~\ref{sec:paper} contains material ready for a paper: a methods paragraph with the values of this case, the tables in LaTeX and the references in BibTeX (also exported as separate files).""")
        self.box("warn",
                 r"Las valoraciones automáticas (\stgood{}, \stcaution{}, \stpoor{}) usan umbrales \emph{orientativos} de la literatura. Sirven para detectar problemas y orientar la interpretación, no sustituyen el criterio del investigador. Antes de publicar, comprueba las referencias en ADS.",
                 r"The automatic assessments (\stgood{}, \stcaution{}, \stpoor{}) use \emph{indicative} thresholds from the literature. They help detect problems and guide the interpretation; they do not replace the researcher's judgement. Check the references on ADS before publishing.")

    def summary(self):
        A = self.A
        self.w(r"\section{" + self.L("Resumen", "Summary") + r"}\label{sec:summary}" "\n")
        d = A["data"]
        self.p(r"Se ha extrapolado el campo magnético de <<region>> en una caja de <<nx>>$\times$<<ny>>$\times$<<nz>> puntos (<<fx>>$\times$<<fy>>$\times$<<h>>~Mm) con tres modelos de complejidad creciente: potencial (sin corrientes), lineal libre de fuerzas (una sola torsión $\alpha$ para toda la región) y no lineal libre de fuerzas ($\alpha$ variable).",
               r"The magnetic field of <<region>> was extrapolated in a box of <<nx>>$\times$<<ny>>$\times$<<nz>> grid points (<<fx>>$\times$<<fy>>$\times$<<h>>~Mm) with three models of increasing complexity: potential (current-free), linear force-free (a single twist $\alpha$ for the whole region) and nonlinear force-free (variable $\alpha$).",
               region=(self.L(r"la solución analítica de Low \& Lou (1990)", r"the Low \& Lou (1990) analytic solution")
                       if d["benchmark"] else self.region_label()), nx=d["nx"], ny=d["ny"], nz=d["nz"],
               fx=f"{d['fov_x']:.0f}", fy=f"{d['fov_y']:.0f}", h=f"{d['height']:.0f}")
        self.w(r"\subsection*{" + self.L("Resultados principales", "Main results") + "}\n\\begin{itemize}\n")
        for f in A["findings"]:
            self.w("\\item " + self.finding_text(f) + "\n")
        self.w("\\end{itemize}\n\n")
        self.w(r"\subsection*{" + self.L("Semáforo de calidad", "Quality traffic light") + "}\n")
        self.assessment_table(A["assessments"])

    def finding_text(self, f):
        k = f["key"]
        nm = self.mname
        if k == "free_energy":
            return fill(self.L(
                r"\textbf{Energía libre:} el NLFFF almacena $E_{\rm free}=$ <<free>>~erg, un <<pct>> más que el campo potencial ($E/E_{\rm pot}=$ <<ratio>>). En regiones activas suele estar entre el 10 y el 40\%; es la energía disponible para fulguraciones y eyecciones de masa.",
                r"\textbf{Free energy:} the NLFFF stores $E_{\rm free}=$ <<free>>~erg, <<pct>> more than the potential field ($E/E_{\rm pot}=$ <<ratio>>). In active regions it is typically 10--40\%; it is the energy available for flares and coronal mass ejections."),
                free=num(f["free"]), pct=pct(f["ratio"] - 1), ratio=num(f["ratio"], 4))
        if k == "lfff_vs_nlfff":
            rel = self.L("más", "more") if f["r_l"] > f["r_n"] else self.L("menos", "less")
            return fill(self.L(
                r"\textbf{LFFF frente a NLFFF:} el LFFF da $E/E_{\rm pot}=$ <<rl>> frente a <<rn>> del NLFFF, es decir, <<rel>> energía. Con una sola $\alpha$ la torsión se reparte por toda la región en lugar de concentrarse donde están las corrientes, así que la energía del LFFF no es una estimación fiable.",
                r"\textbf{LFFF versus NLFFF:} the LFFF gives $E/E_{\rm pot}=$ <<rl>> versus <<rn>> for the NLFFF, i.e.\ <<rel>> energy. With a single $\alpha$ the twist is spread over the whole region instead of being concentrated where the currents are, so the LFFF energy is not a reliable estimate."),
                rl=num(f["r_l"], 4), rn=num(f["r_n"], 4), rel=rel)
        if k == "helicity":
            a = f["alpha"]
            sign = self.L("positiva (torsión dextrógira)", "positive (right-handed twist)") if a > 0 else \
                self.L("negativa (torsión levógira)", "negative (left-handed twist)")
            txt = fill(self.L(r"\textbf{Helicidad:} el $\alpha$ global es <<a>>~Mm$^{-1}$, helicidad <<sign>>.",
                              r"\textbf{Helicity:} the global $\alpha$ is <<a>>~Mm$^{-1}$, <<sign>> helicity."),
                       a=num(a), sign=sign)
            lat = f.get("lat")
            if lat is not None and np.isfinite(lat) and abs(lat) > 2:
                expected = -1 if lat > 0 else 1
                ok = np.sign(a) == expected
                txt += " " + (self.L(
                    "Cumple la regla hemisférica (negativa en el norte, positiva en el sur; Pevtsov et al. 1995).",
                    "It follows the hemispheric rule (negative in the north, positive in the south; Pevtsov et al. 1995).")
                    if ok else self.L(
                    "No sigue la regla hemisférica (negativa en el norte, positiva en el sur), lo que ocurre en un 20--40\\,\\% de las regiones.",
                    "It does not follow the hemispheric rule (negative in the north, positive in the south), as happens in 20--40\\,\\% of regions."))
            return txt
        if k == "closest":
            ref = nm(f["ref"])
            items = ", ".join(f"{nm(m)} {num(v, 3)}" for m, v in f["cvec"].items())
            return fill(self.L(
                r"\textbf{Parecido con «<<ref>>»:} el modelo más próximo es el <<best>> (correlación vectorial $C_{\rm vec}$: <<items>>; 1 = idéntico).",
                r"\textbf{Similarity to ``<<ref>>'':} the closest model is <<best>> (vector correlation $C_{\rm vec}$: <<items>>; 1 = identical)."),
                ref=ref, best=nm(f["best"]), items=items)
        if k == "height_trend":
            h = f["heights"]
            a = f["angles"]["nlfff"]
            return fill(self.L(
                r"\textbf{Fiabilidad con la altura:} el error angular medio del NLFFF pasa de <<a0>>$^\circ$ a <<h0>>~Mm a <<a2>>$^\circ$ a <<h2>>~Mm. Cuanto más alto, más pesan los contornos laterales y superior no observados.",
                r"\textbf{Reliability with height:} the mean NLFFF angular error goes from <<a0>>$^\circ$ at <<h0>>~Mm to <<a2>>$^\circ$ at <<h2>>~Mm. The higher up, the more the unobserved side and top boundaries matter."),
                a0=f"{a[0]:.1f}", a2=f"{a[-1]:.1f}", h0=f"{h[0]:.0f}", h2=f"{h[-1]:.0f}")
        if k == "twist":
            if f["max_abs"] is None:
                return ""
            return fill(self.L(
                r"\textbf{Torsión:} en el <<m>>, el 99\,\% de las líneas cerradas tiene $|T_w|<$ <<p99>> (máximo <<mx>>, sensible a píxeles ruidosos); un <<fr>> tiene $|T_w|\geq1$, el umbral habitual de cuerda de flujo. Lo relevante es si esas líneas forman una zona compacta (ver sección~\ref{sec:physics}).",
                r"\textbf{Twist:} in the <<m>>, 99\,\% of closed lines have $|T_w|<$ <<p99>> (maximum <<mx>>, sensitive to noisy pixels); <<fr>> have $|T_w|\geq1$, the usual flux-rope threshold. What matters is whether those lines form a compact patch (see Section~\ref{sec:physics})."),
                m=nm(f["model"]), mx=f"{f['max_abs']:.2f}", p99=f"{f['p99_abs']:.2f}", fr=pct(f["frac_gt1"]))
        if k == "decay":
            hc = f["h_crit"]
            if hc is None or not np.isfinite(hc):
                return self.L(r"\textbf{Confinamiento:} el índice de decaimiento no alcanza 1.5 en la caja: el campo externo confina con fuerza.",
                              r"\textbf{Confinement:} the decay index does not reach 1.5 in the box: the external field confines strongly.")
            return fill(self.L(
                r"\textbf{Confinamiento:} la altura crítica de la inestabilidad de toro es $h_{\rm crit}=$ <<hc>>~Mm; por encima, el campo externo ya no puede sujetar una cuerda de flujo.",
                r"\textbf{Confinement:} the critical height for the torus instability is $h_{\rm crit}=$ <<hc>>~Mm; above it the external field can no longer hold a flux rope down."),
                hc=f"{hc:.1f}")
        if k == "qsl":
            if f.get("frac_logq_gt2") is None:
                return ""
            return fill(self.L(
                r"\textbf{QSL:} en el <<m>>, un <<fr>> del área de campo fuerte tiene $Q>100$ (máx.\ $\log_{10}Q=$ <<mx>>): ahí es más probable la reconexión.",
                r"\textbf{QSLs:} in the <<m>>, <<fr>> of the strong-field area has $Q>100$ (max.\ $\log_{10}Q=$ <<mx>>): reconnection is most likely there."),
                m=nm(f["model"]), fr=pct(f["frac_logq_gt2"]), mx=f"{f['max_logq']:.1f}")
        if k == "methods":
            S = f["spread"]
            if "E_free" not in S:
                return self.L(
                    r"\textbf{Comparación de métodos:} las soluciones Grad--Rubin no convergieron con estos datos, así que no se ha podido estimar la incertidumbre de método (sección~\ref{sec:methods_cmp}).",
                    r"\textbf{Method comparison:} the Grad--Rubin solutions did not converge with these data, so the method uncertainty could not be estimated (Section~\ref{sec:methods_cmp}).")
            return fill(self.L(
                r"\textbf{Incertidumbre de método (optimización frente a Grad--Rubin P y N):} la energía libre varía entre <<a>> y <<b>>~erg y $H_V$ entre <<c>> y <<d>>~Mx$^2$ según el método (sección~\ref{sec:methods_cmp}).",
                r"\textbf{Method uncertainty (optimisation versus Grad--Rubin P and N):} the free energy ranges from <<a>> to <<b>>~erg and $H_V$ from <<c>> to <<d>>~Mx$^2$ depending on the method (Section~\ref{sec:methods_cmp})."),
                a=num(S["E_free"]["min"]), b=num(S["E_free"]["max"]), c=num(S["H_V"]["min"]), d=num(S["H_V"]["max"]))
        if k == "mc":
            s = f["summary"]
            def iv(key):
                d = s.get(key)
                return f"{num(d['p50'])} [{num(d['p16'])}, {num(d['p84'])}]" if d else "--"
            return fill(self.L(
                r"\textbf{Incertidumbre por ruido de medida (Monte Carlo, <<n>> realizaciones; mediana [16--84\%]):} energía libre <<ef>>~erg (pot. FFT) y <<ev>>~erg (pot. de volumen); $H_V=$ <<hv>>~Mx$^2$; $h_{\rm crit}=$ <<hc>>~Mm. Son cotas inferiores: no incluyen errores sistemáticos ni de método.",
                r"\textbf{Uncertainty from measurement noise (Monte Carlo, <<n>> realisations; median [16--84\%]):} free energy <<ef>>~erg (FFT pot.) and <<ev>>~erg (volume pot.); $H_V=$ <<hv>>~Mx$^2$; $h_{\rm crit}=$ <<hc>>~Mm. These are lower bounds: they exclude systematic and method errors."),
                n=f["n"], ef=iv("E_free"), ev=iv("E_free_vol"), hv=iv("H_V"), hc=iv("h_crit"))
        if k == "helicity_rel":
            sgn = self.L("positiva", "positive") if f["H_V"] > 0 else self.L("negativa", "negative")
            return fill(self.L(
                r"\textbf{Helicidad y energía de volumen:} el <<m>> tiene helicidad relativa <<s>> ($H_V=$ <<hv>>~Mx$^2$) e índice $|H_J|/|H_V|=$ <<r>>. Respecto al potencial con el mismo flujo en las seis caras, su energía libre es <<fv>>~erg, con un <<ed>> de energía debida a la divergencia.",
                r"\textbf{Helicity and volume energy:} the <<m>> has <<s>> relative helicity ($H_V=$ <<hv>>~Mx$^2$) and index $|H_J|/|H_V|=$ <<r>>. Relative to the potential field with the same flux on all six faces, its free energy is <<fv>>~erg, with <<ed>> of the energy due to divergence."),
                m=nm(f["model"]), s=sgn, hv=num(f["H_V"]), r=num(f["ratio"], 2), fv=num(f["free_vol"]),
                ed=pct(f["e_div"]))
        if k == "reliability":
            return fill(self.L(
                r"\textbf{Fiabilidad:} el <<fr>> de las líneas del <<m>> tiene confianza suficiente (sección~\ref{sec:reliability}); el resto depende de bordes no observados o no está bien resuelto.",
                r"\textbf{Reliability:} <<fr>> of the <<m>> field lines have sufficient confidence (Section~\ref{sec:reliability}); the rest depend on unobserved boundaries or are not well resolved."),
                fr=pct(f["frac_reliable"], 0), m=nm(f["model"]))
        if k == "combined_reliable":
            return r"\textbf{" + self.L("Lectura conjunta (sólo líneas fiables):", "Combined reading (reliable lines only):") + "} " + self.combined_text(f)
        if k == "combined":
            return r"\textbf{" + self.L("Lectura conjunta:", "Combined reading:") + "} " + self.combined_text(f)
        if k == "connectivity":
            items = ", ".join(f"{nm(m)} {pct(v, 0)}" for m, v in f["topo"].items())
            return fill(self.L(
                r"\textbf{Conectividad:} fracción de líneas, trazadas desde las mismas semillas, con la misma topología (cerrada/abierta) que en «<<ref>>»: <<items>>.",
                r"\textbf{Connectivity:} fraction of field lines, traced from the same seeds, with the same topology (closed/open) as in ``<<ref>>'': <<items>>."),
                ref=nm(f["ref"]), items=items)
        return ""

    def assessment_table(self, items, with_advice=False):
        self.w(r"\begin{longtable}{>{\raggedright}p{6.2cm} r p{2.2cm} >{\raggedright\arraybackslash}p{4.6cm}}" "\n\\toprule\n")
        self.w(self.L("Indicador & Valor & Estado & Criterio", "Indicator & Value & Status & Criterion")
               + r"\\ \midrule\endhead" "\n")
        for a in items:
            lab = ASSESS[a["key"]]
            v = a["value"]
            if v is None:
                vs = "--"
            elif a["fmt"] == "{:.1%}":
                vs = pct(v)
            elif a["fmt"] == "{:.0%}":
                vs = pct(v, 0)
            elif a["fmt"] == "{:.0f}°":
                vs = f"${v:.0f}^\\circ$"
            elif a["fmt"] == "{:d}":
                vs = str(int(v))
            else:
                vs = num(v)
            crit = lab[2] if self.lang == "es" else lab[3]
            self.w(f"{lab[0] if self.lang == 'es' else lab[1]} & {vs} & {self.status(a['status'])} & {{\\small {crit}}}\\\\\n")
        self.w("\\bottomrule\n\\end{longtable}\n\n")

    def background(self):
        self.w(r"\section{" + self.L("Fundamentos físicos", "Physical background") + r"}\label{sec:background}" "\n")
        self.w(r"\subsection{" + self.L("¿Por qué extrapolar?", "Why extrapolate?") + "}\n")
        self.p(
            r"""El campo magnético controla la estructura y la dinámica de la corona solar: los bucles brillantes que vemos en ultravioleta extremo trazan líneas de campo, y las fulguraciones y eyecciones de masa coronal (CMEs) liberan energía magnética almacenada. Sin embargo, el campo coronal \textbf{no se puede medir de forma rutinaria}: el plasma es tenue y caliente, y el desdoblamiento Zeeman de sus líneas es demasiado débil.

Lo que sí se mide con precisión es el campo en la \textbf{fotosfera}, la superficie visible. El instrumento HMI a bordo de SDO observa la línea de Fe\,\textsc{i} a 6173\,\AA{} en luz polarizada; la polarización (parámetros de Stokes) revela, mediante el efecto Zeeman, la intensidad y la dirección del campo. La \textbf{extrapolación} consiste en calcular el campo tridimensional por encima de la fotosfera usando este mapa como condición de contorno y una hipótesis física sobre la corona.""",
            r"""The magnetic field controls the structure and dynamics of the solar corona: the bright loops seen in extreme ultraviolet trace field lines, and flares and coronal mass ejections (CMEs) release stored magnetic energy. However, the coronal field \textbf{cannot be measured routinely}: the plasma is tenuous and hot, and the Zeeman splitting of its lines is too weak.

What can be measured accurately is the field in the \textbf{photosphere}, the visible surface. The HMI instrument on board SDO observes the Fe\,\textsc{i} 6173\,\AA{} line in polarised light; the polarisation (Stokes parameters) reveals, through the Zeeman effect, the strength and direction of the field. \textbf{Extrapolation} means computing the three-dimensional field above the photosphere using this map as a boundary condition plus a physical assumption about the corona.""")
        self.w(r"\subsection{" + self.L("La hipótesis libre de fuerzas", "The force-free assumption") + "}\n")
        self.p(
            r"""En la corona baja la presión magnética domina sobre la presión del gas: el parámetro $\beta = p_{\rm gas}/(B^2/2\mu_0)$ es mucho menor que 1 (Gary 2001). Si además el campo evoluciona lentamente, la única fuerza importante, la de Lorentz, debe anularse:""",
            r"""In the low corona magnetic pressure dominates over gas pressure: the parameter $\beta = p_{\rm gas}/(B^2/2\mu_0)$ is much smaller than 1 (Gary 2001). If the field also evolves slowly, the only important force, the Lorentz force, must vanish:""")
        self.w(r"""\begin{equation}
\mathbf{J}\times\mathbf{B} = 0,\qquad \nabla\cdot\mathbf{B}=0,\qquad \mathbf{J}=\frac{\nabla\times\mathbf{B}}{\mu_0}.
\end{equation}
""")
        self.p(
            r"""Que $\mathbf{J}\times\mathbf{B}=0$ significa que las corrientes fluyen \emph{paralelas} al campo, así que $\nabla\times\mathbf{B} = \alpha\,\mathbf{B}$. Tomando la divergencia se obtiene $\mathbf{B}\cdot\nabla\alpha = 0$: \textbf{$\alpha$ es constante a lo largo de cada línea de campo}, aunque puede cambiar de una línea a otra. Según cómo se trate $\alpha$ hay tres modelos:""",
            r"""$\mathbf{J}\times\mathbf{B}=0$ means currents flow \emph{parallel} to the field, so $\nabla\times\mathbf{B} = \alpha\,\mathbf{B}$. Taking the divergence gives $\mathbf{B}\cdot\nabla\alpha = 0$: \textbf{$\alpha$ is constant along each field line}, although it may change from line to line. Depending on how $\alpha$ is treated there are three models:""")
        self.w(r"\begin{center}\small\begin{tabular}{>{\raggedright}p{2.3cm}>{\raggedright}p{2.6cm}>{\raggedright}p{2.6cm}>{\raggedright\arraybackslash}p{6.4cm}}\toprule" "\n")
        self.w(self.L(r"Modelo & Ecuación & Datos necesarios & Ventajas / limitaciones",
                      r"Model & Equation & Required data & Advantages / limitations") + r"\\\midrule" "\n")
        self.w(self.L(
            r"Potencial & $\nabla\times\mathbf{B}=0$ & $B_z$ & Único y rápido; mínima energía. Sin corrientes: no almacena energía libre ni reproduce estructuras torsionadas.\\" "\n"
            r"LFFF & $\nabla\times\mathbf{B}=\alpha\mathbf{B}$, $\alpha$ cte. & $B_z$ (+ $B_x,B_y$ para elegir $\alpha$) & Rápido (FFT). Una misma torsión en toda la región; $|\alpha|$ limitado por el tamaño del dominio.\\" "\n"
            r"NLFFF & $\nabla\times\mathbf{B}=\alpha(\mathbf{r})\mathbf{B}$ & $B_x, B_y, B_z$ & El más realista: corrientes localizadas. Problema no lineal, costoso y sensible al ruido y a los contornos.\\" "\n",
            r"Potential & $\nabla\times\mathbf{B}=0$ & $B_z$ & Unique and fast; minimum energy. No currents: stores no free energy, cannot reproduce twisted structures.\\" "\n"
            r"LFFF & $\nabla\times\mathbf{B}=\alpha\mathbf{B}$, constant $\alpha$ & $B_z$ (+ $B_x,B_y$ to choose $\alpha$) & Fast (FFT). Same twist everywhere; $|\alpha|$ limited by the domain size.\\" "\n"
            r"NLFFF & $\nabla\times\mathbf{B}=\alpha(\mathbf{r})\mathbf{B}$ & $B_x, B_y, B_z$ & Most realistic: localised currents. Nonlinear, expensive and sensitive to noise and boundaries.\\" "\n"))
        self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
        self.box("key",
                 r"""\textbf{Energía libre.} Para un $B_z$ fotosférico dado, el campo potencial es el de \emph{mínima energía} (teorema de Thomson). Cualquier corriente añade energía. La diferencia $E_{\rm free}=E_{\rm NLFFF}-E_{\rm pot}$ es la energía que el campo podría liberar sin cambiar la fotosfera: la reserva de las fulguraciones y CMEs. Por eso el NLFFF, que localiza las corrientes, es el modelo de referencia para estudiar la actividad eruptiva.""",
                 r"""\textbf{Free energy.} For a given photospheric $B_z$, the potential field has the \emph{minimum energy} (Thomson's theorem). Any current adds energy. The difference $E_{\rm free}=E_{\rm NLFFF}-E_{\rm pot}$ is the energy the field could release without changing the photosphere: the reservoir for flares and CMEs. That is why the NLFFF, which localises the currents, is the reference model for studying eruptive activity.""")
        self.box("warn",
                 r"""La fotosfera \emph{no} está libre de fuerzas ($\beta\sim1$ allí), y las componentes transversales del campo son mucho más ruidosas que la vertical. Por eso los datos fotosféricos se \emph{preprocesan} antes del NLFFF y los resultados de distintos métodos NLFFF pueden diferir apreciablemente con datos reales (DeRosa et al. 2009).""",
                 r"""The photosphere is \emph{not} force-free ($\beta\sim1$ there), and the transverse field components are much noisier than the vertical one. That is why photospheric data are \emph{preprocessed} before the NLFFF, and why different NLFFF methods can disagree noticeably on real data (DeRosa et al. 2009).""")

    def data(self):
        A, d = self.A, self.A["data"]
        self.w(r"\section{" + self.L("Datos de entrada", "Input data") + r"}\label{sec:data}" "\n")
        if d["benchmark"]:
            self.p(
                r"""En este caso los datos no son observaciones sino el plano $z=0$ de la solución analítica de Low \& Lou (1990), un campo libre de fuerzas no lineal conocido exactamente. Se usa como \textbf{prueba de validación}: como la solución en todo el volumen es conocida, se puede medir el error de cada modelo. Es el procedimiento estándar para comprobar códigos NLFFF (Schrijver et al. 2006).""",
                r"""Here the data are not observations but the $z=0$ plane of the Low \& Lou (1990) analytic solution, a nonlinear force-free field known exactly. It is used as a \textbf{validation test}: since the solution is known throughout the volume, the error of each model can be measured. It is the standard procedure for testing NLFFF codes (Schrijver et al. 2006).""")
        else:
            self.p(
                r"""Los datos proceden de la serie \texttt{hmi.sharp\_cea\_720s} de SDO/HMI (Schou et al. 2012; Hoeksema et al. 2014; Bobra et al. 2014). Cada SHARP es un recorte de una región activa, cada 12 minutos, con el campo vectorial ya invertido a partir de los parámetros de Stokes, con la ambigüedad de $180^\circ$ resuelta y reproyectado a una malla cilíndrica de igual área (CEA) centrada en la región. Los segmentos se convierten así a la caja cartesiana local: $B_z = B_r$, $B_x = B_\phi$, $B_y = -B_\theta$.""",
                r"""The data come from the SDO/HMI \texttt{hmi.sharp\_cea\_720s} series (Schou et al. 2012; Hoeksema et al. 2014; Bobra et al. 2014). Each SHARP is a cut-out of an active region, every 12 minutes, with the vector field already inverted from the Stokes parameters, the $180^\circ$ ambiguity resolved, and remapped onto a cylindrical equal-area (CEA) grid centred on the region. The segments are converted to the local Cartesian box as $B_z = B_r$, $B_x = B_\phi$, $B_y = -B_\theta$.""")
            self.box("explain",
                     r"""El ruido típico de HMI es de unos 10~G en $B_z$ pero de unos 100~G en el campo transversal, porque la señal de polarización lineal es más débil. Por eso conviene trabajar con regiones cerca del centro del disco y con campos fuertes, y por eso el preprocesado suaviza sobre todo $B_x$ y $B_y$.""",
                     r"""Typical HMI noise is about 10~G in $B_z$ but about 100~G in the transverse field, because the linear polarisation signal is weaker. That is why it is best to work with regions near disk centre and with strong fields, and why preprocessing mostly smooths $B_x$ and $B_y$.""")
        self.w(r"\begin{center}\begin{tabular}{lr}\toprule" "\n")
        rows = [
            (self.L("Malla (nx × ny × nz)", "Grid (nx × ny × nz)").replace("×", r"$\times$"),
             f"{d['nx']}$\\times${d['ny']}$\\times${d['nz']}"),
            (self.L("Tamaño de píxel", "Pixel size"), f"{d['dx']:.3f}~Mm"),
            (self.L("Campo de visión", "Field of view"), f"{d['fov_x']:.0f}$\\times${d['fov_y']:.0f}~Mm"),
            (self.L("Altura de la caja", "Box height"), f"{d['height']:.0f}~Mm"),
            (r"$B_z$ " + self.L("mín / máx", "min / max"), f"{d['bz_min']:.0f} / {d['bz_max']:.0f}~G"),
            (self.L("Flujo positivo / negativo", "Positive / negative flux"),
             f"{num(d['flux_pos'])} / {num(d['flux_neg'])}~Mx"),
            (self.L("Desequilibrio de flujo", "Flux imbalance"), pct(d["imbalance"])),
        ]
        if d["bt_max"] is not None:
            rows.append((r"$|B_t|$ " + self.L("máx", "max"), f"{d['bt_max']:.0f}~G"))
        if d.get("lon") is not None:
            rows.append((self.L("Posición (lon, lat)", "Position (lon, lat)"),
                         f"${d['lon']:.1f}^\\circ$, ${d['lat']:.1f}^\\circ$"))
        for a, b in rows:
            self.w(f"{a} & {b}\\\\\n")
        self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
        self.figure("magnetogram",
                    r"Magnetograma de entrada. Fondo: $B_z$ en grises (claro = positivo, oscuro = negativo); flechas: campo transversal $(B_x, B_y)$.",
                    r"Input magnetogram. Background: $B_z$ in grey (light = positive, dark = negative); arrows: transverse field $(B_x, B_y)$.",
                    width=r"0.75\textwidth")
        self.box("read",
                 r"""Las zonas claras son polaridad positiva (el campo sale de la superficie) y las oscuras, negativa (entra). Los contornos continuos y discontinuos marcan el 20 y el 60\% del campo máximo de cada polaridad. La línea negra es la \textbf{línea de inversión de polaridad} (PIL), donde $B_z=0$. Las flechas muestran el campo horizontal: si cruzan la PIL perpendicularmente, el campo es casi potencial; si corren \emph{paralelas} a ella, hay \textbf{cizalla}, señal de corrientes y de energía libre almacenada.""",
                 r"""Light areas are positive polarity (field leaving the surface) and dark areas negative (entering). Solid and dashed contours mark 20 and 60\% of the peak field of each polarity. The black line is the \textbf{polarity inversion line} (PIL), where $B_z=0$. Arrows show the horizontal field: if they cross the PIL perpendicularly the field is nearly potential; if they run \emph{parallel} to it there is \textbf{shear}, a sign of currents and stored free energy.""")
        if A["aly"]:
            self.w(r"\subsection{" + self.L("¿Son los datos compatibles con un campo libre de fuerzas?",
                                            "Are the data compatible with a force-free field?") + "}\n")
            self.p(r"""Aly (1989) demostró que un campo libre de fuerzas debe tener fuerza y torque netos nulos sobre la frontera. Con los datos fotosféricos se miden dos números adimensionales:""",
                   r"""Aly (1989) showed that a force-free field must exert zero net force and torque on the boundary. Two dimensionless numbers are measured on the photospheric data:""")
            self.w(r"""\begin{gather}
\epsilon_F=\frac{|\sum B_xB_z|+|\sum B_yB_z|+|\sum D|}{\sum B^2},\qquad D=B_z^2-B_x^2-B_y^2,\\
\epsilon_T=\frac{|\sum x\,D|+|\sum y\,D|+|\sum(yB_xB_z-xB_yB_z)|}{\sum \sqrt{x^2+y^2}\,B^2}.
\end{gather}
""")
            a0 = A["aly"]
            self.p(r"Para estos datos, $\epsilon_F=$ <<f>> y $\epsilon_T=$ <<t>>. Valores $\ll 1$ (en la práctica $<0.1$) indican datos consistentes con un campo libre de fuerzas.",
                   r"For these data, $\epsilon_F=$ <<f>> and $\epsilon_T=$ <<t>>. Values $\ll 1$ (in practice $<0.1$) indicate data consistent with a force-free field.",
                   f=num(a0["eps_force"]), t=num(a0["eps_torque"]))
        if self.s.mag_pp is not None:
            self.w(r"\subsection{" + self.L("Preprocesado", "Preprocessing") + "}\n")
            cfg = self.s.config.get("preprocess_kw", {})
            self.p(r"""El preprocesado (Wiegelmann et al. 2006) modifica el magnetograma minimizando $L = \mu_1 L_1 + \mu_2 L_2 + \mu_3 L_3 + \mu_4 L_4$, donde $L_1$ y $L_2$ son la fuerza y el torque netos (los criterios de Aly), $L_3$ penaliza alejarse de las observaciones y $L_4$ penaliza las variaciones a pequeña escala (suaviza). Se ha usado $\mu_3=$ <<m3>> y $\mu_4=$ <<m4>>. El resultado: $\epsilon_F$ pasa de <<f0>> a <<f1>> y $\epsilon_T$ de <<t0>> a <<t1>>, con un cambio rms del <<ch>> respecto al campo original (<<chz>> en $B_z$).""",
                   r"""Preprocessing (Wiegelmann et al. 2006) modifies the magnetogram by minimising $L = \mu_1 L_1 + \mu_2 L_2 + \mu_3 L_3 + \mu_4 L_4$, where $L_1$ and $L_2$ are the net force and torque (Aly's criteria), $L_3$ penalises departures from the observations and $L_4$ penalises small-scale variations (smoothing). We used $\mu_3=$ <<m3>> and $\mu_4=$ <<m4>>. Result: $\epsilon_F$ goes from <<f0>> to <<f1>> and $\epsilon_T$ from <<t0>> to <<t1>>, with an rms change of <<ch>> relative to the original field (<<chz>> in $B_z$).""",
                   m3=num(cfg.get("mu3", 1e-2)), m4=num(cfg.get("mu4", 1e-2)),
                   f0=num(A["aly"]["eps_force"]), f1=num(A["aly_pp"]["eps_force"]),
                   t0=num(A["aly"]["eps_torque"]), t1=num(A["aly_pp"]["eps_torque"]),
                   ch=pct(A["pp_change"]), chz=pct(A["pp_change_bz"]))
            self.figure("preprocess",
                        r"Efecto del preprocesado: $B_z$ antes y después, y módulo del cambio introducido en el vector $\mathbf{B}$.",
                        r"Effect of preprocessing: $B_z$ before and after, and magnitude of the change introduced in the vector $\mathbf{B}$.")
            self.box("read",
                     r"""Los dos primeros paneles deberían parecer casi iguales: el preprocesado no debe cambiar la estructura de la región. El tercero muestra dónde se ha modificado más el campo. Lo esperable es que los cambios sean pequeños y estén repartidos, con algo más de cambio en zonas de campo débil y ruidoso. Si los cambios grandes se concentran en los núcleos de las manchas, el preprocesado es demasiado agresivo (aumenta $\mu_3$).""",
                     r"""The first two panels should look almost identical: preprocessing must not change the structure of the region. The third shows where the field was modified most. Changes are expected to be small and spread out, somewhat larger in weak, noisy areas. If large changes concentrate in sunspot cores, preprocessing is too aggressive (increase $\mu_3$).""")

    def methods(self):
        A, s = self.A, self.s
        self.w(r"\section{" + self.L("Métodos aplicados", "Methods applied") + r"}\label{sec:methods}" "\n")
        self.w(r"\subsection{" + self.L("Campo potencial y LFFF por transformada de Fourier",
                                        "Potential field and LFFF by Fourier transform") + "}\n")
        self.p(r"""Si $\alpha$ es constante, la ecuación $\nabla\times\mathbf{B}=\alpha\mathbf{B}$ es lineal y se resuelve exactamente en el espacio de Fourier (Nakagawa \& Raadu 1972; Alissandrakis 1981). Cada modo horizontal de número de onda $\mathbf{k}=(k_x,k_y)$ decae con la altura como $e^{-\ell z}$, con""",
               r"""If $\alpha$ is constant, the equation $\nabla\times\mathbf{B}=\alpha\mathbf{B}$ is linear and can be solved exactly in Fourier space (Nakagawa \& Raadu 1972; Alissandrakis 1981). Each horizontal mode with wavenumber $\mathbf{k}=(k_x,k_y)$ decays with height as $e^{-\ell z}$, with""")
        self.w(r"""\begin{equation}
\ell=\sqrt{k^2-\alpha^2},\quad
\hat B_z=\hat B_{z0}\,e^{-\ell z},\quad
\hat B_x=-i\,\frac{k_x\ell-\alpha k_y}{k^2}\,\hat B_z,\quad
\hat B_y=-i\,\frac{k_y\ell+\alpha k_x}{k^2}\,\hat B_z .
\end{equation}
""")
        self.p(r"""Sólo hace falta $B_z$ en la fotosfera. Con $\alpha=0$ se obtiene el campo potencial. Para que todos los modos decaigan con la altura hace falta $|\alpha|<k_{\min}=2\pi/L$, siendo $L$ el lado mayor del dominio: aquí $\alpha_{\max}=$ <<amax>>~Mm$^{-1}$. Las condiciones laterales son periódicas, como si la región se repitiera indefinidamente.""",
               r"""Only $B_z$ at the photosphere is needed. Setting $\alpha=0$ gives the potential field. For every mode to decay with height, $|\alpha|<k_{\min}=2\pi/L$ is required, $L$ being the longest side of the domain: here $\alpha_{\max}=$ <<amax>>~Mm$^{-1}$. The lateral boundaries are periodic, as if the region repeated indefinitely.""",
               amax=num(A["alpha"]["max"]))
        al = A["alpha"]
        if al.get("used") is not None:
            self.w(r"\subsection{" + self.L("Elección de $\\alpha$ para el LFFF", "Choosing $\\alpha$ for the LFFF") + "}\n")
            self.p(r"""El LFFF necesita un único valor de $\alpha$. Se han calculado dos estimaciones a partir del magnetograma vectorial:
\begin{itemize}[nosep]
\item \textbf{Ajuste del campo transversal:} el $\alpha$ cuyo LFFF reproduce mejor el $(B_x,B_y)$ observado en $z=0$, $\alpha_{\rm fit}=$ <<fit>>~Mm$^{-1}$.
\item \textbf{Promedio de corrientes} (Hagino \& Sakurai 2004): $\alpha_{J_z}=\sum J_zB_z/\sum B_z^2$ $=$ <<jz>>~Mm$^{-1}$, usando sólo píxeles de campo fuerte.
\end{itemize}
Se ha usado <<used>>~Mm$^{-1}$ (método: <<method>>).""",
                   r"""The LFFF needs a single value of $\alpha$. Two estimates were computed from the vector magnetogram:
\begin{itemize}[nosep]
\item \textbf{Transverse-field fit:} the $\alpha$ whose LFFF best reproduces the observed $(B_x,B_y)$ at $z=0$, $\alpha_{\rm fit}=$ <<fit>>~Mm$^{-1}$.
\item \textbf{Current average} (Hagino \& Sakurai 2004): $\alpha_{J_z}=\sum J_zB_z/\sum B_z^2$ $=$ <<jz>>~Mm$^{-1}$, using only strong-field pixels.
\end{itemize}
The value used is <<used>>~Mm$^{-1}$ (method: <<method>>).""",
                   fit=num(al.get("fit")), jz=num(al.get("jz")), used=num(al["used"]),
                   method={"fit": self.L("ajuste de $B_t$", "$B_t$ fit"),
                           "jz": self.L("promedio de corrientes", "current average"),
                           "user": self.L("fijado por el usuario", "user-defined")}.get(al.get("method"), esc(al.get("method", ""))))
            if al.get("clipped"):
                self.box("warn",
                         r"""El $\alpha$ estimado (<<raw>>) supera el máximo admisible y se ha recortado a $0.98\,\alpha_{\max}$. Señal de que la región necesita más torsión de la que un LFFF puede dar en este dominio, o de que la torsión no es uniforme. El LFFF resultante es una aproximación pobre.""",
                         r"""The estimated $\alpha$ (<<raw>>) exceeds the admissible maximum and was clipped to $0.98\,\alpha_{\max}$. This indicates that the region needs more twist than an LFFF can provide in this domain, or that the twist is not uniform. The resulting LFFF is a poor approximation.""",
                         raw=num(al.get("raw")))
            self.figure("alpha_fit",
                        r"Error relativo entre el campo transversal observado y el del LFFF en función de $\alpha$.",
                        r"Relative misfit between the observed and the LFFF transverse field as a function of $\alpha$.",
                        width=r"0.75\textwidth")
            self.box("read",
                     r"""La curva muestra cuánto se parece el campo horizontal del LFFF al observado para cada $\alpha$; el mínimo es el mejor ajuste. Un mínimo profundo y bien definido indica que un único $\alpha$ describe razonablemente la región. Una curva plana, o con el mínimo pegado a $\pm\alpha_{\max}$ (líneas de puntos), indica que \emph{ningún} $\alpha$ constante la describe bien. La línea discontinua es la estimación por corrientes $\alpha_{J_z}$; si ambas difieren mucho, la torsión no es uniforme.""",
                     r"""The curve shows how closely the LFFF horizontal field matches the observed one for each $\alpha$; the minimum is the best fit. A deep, well-defined minimum means a single $\alpha$ describes the region reasonably. A flat curve, or a minimum stuck at $\pm\alpha_{\max}$ (dotted lines), means that \emph{no} constant $\alpha$ describes it well. The dashed line is the current-based estimate $\alpha_{J_z}$; if the two differ a lot, the twist is not uniform.""")
        if s.nlfff is not None and s.nlfff.meta.get("backend") == "lbfgs":
            self.methods_lbfgs(A["convergence"])
            self.methods_divergence_weight()
        elif s.nlfff is not None:
            c = A["convergence"]
            self.w(r"\subsection{" + self.L("NLFFF por optimización", "NLFFF by optimisation") + "}\n")
            self.p(r"""El NLFFF se calcula con el método de optimización (Wheatland et al. 2000) con función de peso (Wiegelmann 2004). Se busca el campo que minimiza""",
                   r"""The NLFFF is computed with the optimisation method (Wheatland et al. 2000) with a weighting function (Wiegelmann 2004). It seeks the field that minimises""")
            self.w(r"""\begin{equation}
L=\int_V w(\mathbf{r})\left[\frac{|(\nabla\times\mathbf{B})\times\mathbf{B}|^2}{B^2}+|\nabla\cdot\mathbf{B}|^2\right]dV ,
\end{equation}
""")
            self.p(r"""que es cero sólo para un campo libre de fuerzas y sin divergencia. Para evitar que los puntos casi nulos ($|\mathbf{B}|\approx0$, frecuentes en datos reales) hagan explotar el integrando, se usa $B^2+\varepsilon^2$ en el denominador, con $\varepsilon=$ <<eps>> veces el campo máximo; la fuerza de iteración sigue siendo el gradiente exacto del funcional así regularizado. Partiendo del campo potencial, se itera $\mathbf{B}\leftarrow\mathbf{B}+\mu\mathbf{F}$, donde $\mathbf{F}$ es la dirección en la que $L$ decrece más rápido y el paso $\mu$ se adapta automáticamente (se reduce a la mitad si $L$ aumenta). La base $z=0$ se fija al magnetograma vectorial<<pp>>; las caras laterales y superior, al campo potencial. La función $w$ vale 1 en el interior y cae a 0 en una capa de <<nd>> puntos junto a esas caras, para que su influencia (no son datos reales) sea pequeña.

Resultado: <<it>> iteraciones; $L$ se redujo en un factor <<red>> (de <<l0>> a <<lf>>). Estado: \emph{<<msg>>}.""",
                   r"""which vanishes only for a force-free, divergence-free field. To prevent near-null points ($|\mathbf{B}|\approx0$, common in real data) from blowing up the integrand, $B^2+\varepsilon^2$ is used in the denominator, with $\varepsilon=$ <<eps>> times the peak field; the iteration force remains the exact gradient of the regularised functional. Starting from the potential field, the iteration $\mathbf{B}\leftarrow\mathbf{B}+\mu\mathbf{F}$ is applied, where $\mathbf{F}$ is the direction in which $L$ decreases fastest and the step $\mu$ adapts automatically (halved whenever $L$ increases). The base $z=0$ is fixed to the vector magnetogram<<pp>>; the side and top faces, to the potential field. The function $w$ equals 1 in the interior and drops to 0 in a layer of <<nd>> points next to those faces, so that their influence (they are not real data) is small.

Result: <<it>> iterations; $L$ was reduced by a factor <<red>> (from <<l0>> to <<lf>>). Status: \emph{<<msg>>}.""",
                   pp=self.L(" (preprocesado)", " (preprocessed)") if s.mag_pp is not None else "",
                   eps=num(s.nlfff.meta.get("eps", 0.0)),
                   nd=A["nd"], it=c["iterations"], red=num(c["reduction"]), l0=num(c["L0"]),
                   lf=num(c["Lf"]), msg=self.conv_status(c))
            self.figure("convergence",
                        r"Convergencia del NLFFF: funcional $L$ y paso $\mu$ frente a la iteración.",
                        r"NLFFF convergence: functional $L$ and step $\mu$ versus iteration.")
            self.box("read",
                     r"""A la izquierda, $L$ debe bajar varios órdenes de magnitud y acabar \emph{aplanándose}: entonces el método ha llegado a un mínimo. Si aún baja con pendiente clara al final, conviene más iteraciones. A la derecha, el paso $\mu$: crece mientras todo va bien y se reduce a la mitad cada vez que un paso empeora $L$. Un $\mu$ que cae continuamente indica que el problema se vuelve rígido (normal cerca del mínimo, sobre todo con datos ruidosos).""",
                     r"""On the left, $L$ should drop by several orders of magnitude and finally \emph{flatten}: the method has then reached a minimum. If it is still clearly decreasing at the end, more iterations are advisable. On the right, the step $\mu$: it grows while things go well and is halved whenever a step makes $L$ worse. A steadily falling $\mu$ means the problem is becoming stiff (normal near the minimum, especially with noisy data).""")

    def methods_lbfgs(self, c):
        A, s = self.A, self.s
        m = s.nlfff.meta
        relax = m.get("relax_bottom")
        self.w(r"\subsection{" + self.L("NLFFF por optimización con gradiente exacto (L-BFGS)",
                                        "NLFFF by optimisation with exact gradient (L-BFGS)") + "}\n")
        self.p(r"""El NLFFF se obtiene minimizando un funcional que es cero sólo para un campo libre de fuerzas y sin divergencia (Wheatland et al. 2000; Wiegelmann 2004), al que se añaden un término de datos y uno de suavidad:""",
               r"""The NLFFF is obtained by minimising a functional that vanishes only for a force-free, divergence-free field (Wheatland et al. 2000; Wiegelmann 2004), with an added data term and a smoothness term:""")
        self.w(r"""\begin{equation}
L=\Big\langle w\Big[\frac{|(\nabla\times\mathbf{B})\times\mathbf{B}|^2}{B^2+\varepsilon^2}+w_d|\nabla\cdot\mathbf{B}|^2\Big]\Big\rangle_V
+\nu\,\big\langle W\,|\mathbf{B}_t-\mathbf{B}_{t,\rm obs}|^2\big\rangle_{z=0}
+\lambda\,\big\langle|\nabla^2\mathbf{B}|^2\big\rangle_V .
\end{equation}
""")
        self.box("explain",
                 r"""\textbf{Cada término tiene una función.} El primero mide cuánto se aparta el campo de estar libre de fuerzas y sin divergencia; $\varepsilon=$ <<eps>>$\,B_{\max}$ evita que los puntos casi nulos lo hagan explotar, y $w_d=$ <<wd>> es el peso de la divergencia (sección~\ref{sec:divweight}). El segundo es la \textbf{base relajada} (Wiegelmann \& Inhester 2010): el campo transversal de la fotosfera puede ajustarse, con un coste $\nu=$ <<nu>> que pesa más donde el campo transversal observado es fuerte y fiable ($W=|\mathbf{B}_{t,\rm obs}|/\max|\mathbf{B}_{t,\rm obs}|$). Así el modelo no tiene que reproducir el ruido ni las fuerzas de la fotosfera; $B_z$ se mantiene fijo. El tercero ($\lambda=$ <<lam>>) suprime oscilaciones de escala de malla que las diferencias finitas centradas no detectan y que, sin él, el optimizador aprovecharía para bajar $L$ sin mejorar la física.""",
                 r"""\textbf{Each term has a purpose.} The first measures how far the field is from being force-free and divergence-free; $\varepsilon=$ <<eps>>$\,B_{\max}$ prevents near-null points from blowing it up, and $w_d=$ <<wd>> is the weight of the divergence (Section~\ref{sec:divweight}). The second is the \textbf{relaxed base} (Wiegelmann \& Inhester 2010): the photospheric transverse field may adjust, at a cost $\nu=$ <<nu>> weighted more where the observed transverse field is strong and reliable ($W=|\mathbf{B}_{t,\rm obs}|/\max|\mathbf{B}_{t,\rm obs}|$). The model thus need not reproduce the noise and forces of the photosphere; $B_z$ stays fixed. The third ($\lambda=$ <<lam>>) suppresses grid-scale oscillations that centred finite differences cannot detect and which, without it, the optimiser would exploit to lower $L$ without improving the physics.""",
                 eps=num(m.get("eps", 0.0)), nu=num(relax) if relax else "--", lam=num(m.get("smooth", 0.0)),
                 wd=fx(m.get("div_weight", 1.0), 0))
        self.p(r"""El gradiente de $L$ respecto a todas las incógnitas (el campo en el interior y, si la base está relajada, $\mathbf{B}_t$ en $z=0$) se calcula de forma \textbf{exacta} por diferenciación automática (PyTorch; Paszke et al. 2019), y $L$ se minimiza con el algoritmo cuasi-Newton L-BFGS (Liu \& Nocedal 1989), que aprovecha la curvatura del funcional y converge en muchas menos iteraciones que el descenso de gradiente clásico. Las caras laterales y superior se fijan al campo potencial, con una capa de amortiguamiento de <<nd>> puntos ($w\to0$).

Resultado: <<it>> iteraciones (<<ev>> evaluaciones del funcional); $L$ se redujo en un factor <<red>> (de <<l0>> a <<lf>>). Estado: \emph{<<msg>>}.""",
               r"""The gradient of $L$ with respect to all unknowns (the interior field and, if the base is relaxed, $\mathbf{B}_t$ at $z=0$) is computed \textbf{exactly} by automatic differentiation (PyTorch; Paszke et al. 2019), and $L$ is minimised with the quasi-Newton L-BFGS algorithm (Liu \& Nocedal 1989), which exploits the curvature of the functional and converges in far fewer iterations than classical gradient descent. The side and top faces are fixed to the potential field, with a buffer layer of <<nd>> points ($w\to0$).

Result: <<it>> iterations (<<ev>> functional evaluations); $L$ was reduced by a factor <<red>> (from <<l0>> to <<lf>>). Status: \emph{<<msg>>}.""",
               nd=A["nd"], it=c["iterations"], ev=m.get("evaluations", "--"), red=num(c["reduction"]),
               l0=num(c["L0"]), lf=num(c["Lf"]), msg=self.conv_status(c))
        self.figure("convergence",
                    r"Convergencia: funcional total y sus términos (fuerza + divergencia, datos) frente a la iteración.",
                    r"Convergence: total functional and its terms (force + divergence, data) versus iteration.",
                    width=r"0.8\textwidth")
        self.box("read",
                 r"""La curva total debe bajar y acabar aplanándose. El término de fuerza y divergencia es el que indica cuán libre de fuerzas es la solución. El término de datos \emph{sube} al principio: es el precio de dejar que la base se aparte del ruido de las observaciones. Si acabara dominando sobre el de fuerza, $\nu$ sería demasiado grande; si se mantiene en cero, la base apenas se ha movido.""",
                 r"""The total curve should drop and finally flatten. The force-and-divergence term tells how force-free the solution is. The data term \emph{rises} at first: it is the price of letting the base depart from the noise in the observations. If it ended up dominating the force term, $\nu$ would be too large; if it stays at zero, the base has barely moved.""")
        rb = A.get("relax")
        if rb:
            self.p(r"""El campo transversal de la base ha cambiado un <<ch>> (rms, relativo al observado); $B_z$ no cambia. Donde el campo transversal es fuerte el cambio es del <<chs>>, y en el campo débil (el más ruidoso) del <<chw>>.""",
                   r"""The transverse base field changed by <<ch>> (rms, relative to the observed one); $B_z$ is unchanged. Where the transverse field is strong the change is <<chs>>, and in the weak (noisiest) field <<chw>>.""",
                   ch=pct(rb["change"]), chs=pct(rb["change_strong"]), chw=pct(rb["change_weak"]))
            self.figure("bottom",
                        r"Base relajada: campo transversal observado, campo transversal del NLFFF en $z=0$ y diferencia.",
                        r"Relaxed base: observed transverse field, NLFFF transverse field at $z=0$, and difference.")
            self.box("read",
                     r"""Los dos primeros paneles deberían parecerse mucho en las zonas de campo fuerte (manchas y PIL), que son las fiables. El tercero muestra dónde se ha corregido la base: lo esperable son cambios sobre todo en el campo débil y ruidoso. Si hay cambios grandes en los núcleos de campo fuerte, $\nu$ es demasiado pequeño y el modelo está ignorando datos buenos.""",
                     r"""The first two panels should look very similar in the strong-field areas (sunspots and PIL), which are the reliable ones. The third shows where the base was corrected: changes are expected mainly in the weak, noisy field. Large changes in strong-field cores mean $\nu$ is too small and the model is ignoring good data.""")

    # ------------------------------------------- divergence weight and tracing
    def methods_divergence_weight(self):
        import json
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            D = json.load(open(os.path.join(base, "divergence_weight.json"), encoding="utf-8"))
        except (OSError, ValueError):
            D = None
        try:
            T = json.load(open(os.path.join(base, "tracing_step.json"), encoding="utf-8"))
        except (OSError, ValueError):
            T = None
        if D:
            th = D["thresholds"]
            scan = D["scan"]
            w1 = next(r for r in scan if r["div_weight"] == 1.0)
            wt = D.get("meets_thresholds", 4.0)
            w4 = next(r for r in scan if r["div_weight"] == wt)
            st = D.get("stricter_tol") or {}
            C, X = D.get("calibration") or {}, D.get("exact") or {}
            self.w(r"\subsection{" + self.L("Peso de la divergencia: prueba de sensibilidad",
                                            "Divergence weight: sensitivity test") + r"}\label{sec:divweight}" "\n")
            self.p(r"""El funcional suma con el mismo peso la fuerza de Lorentz y la divergencia ($w_d=1$, la forma estándar). Con él, la solución de HARP 377 queda algo por encima de los umbrales que \citet{Thalmann2019} recomiendan para que la energía y la helicidad sean fiables:
(1) $E_{\rm div}/E=$ <<e1>> (umbral $\lesssim$ <<te>>) y $\langle|f_i|\rangle=$ <<f1>> (umbral $\lesssim$ <<tf>>).
(2) La helicidad relativa, que sólo es independiente del gauge si $\nabla\cdot\mathbf{B}=0$, cambia un <<g1>> entre dos gauges.
(3) Dejar converger más no lo cambia: con un criterio de parada 100 veces más estricto, $E_{\rm div}/E=$ <<est>> y el error de gauge es del <<gs>>.""",
                   r"""The functional weights the Lorentz force and the divergence equally ($w_d=1$, the standard form). With it, the HARP 377 solution lies somewhat above the thresholds that \citet{Thalmann2019} recommend for reliable energy and helicity:
(1) $E_{\rm div}/E=$ <<e1>> (threshold $\lesssim$ <<te>>) and $\langle|f_i|\rangle=$ <<f1>> (threshold $\lesssim$ <<tf>>).
(2) The relative helicity, which is gauge-independent only if $\nabla\cdot\mathbf{B}=0$, changes by <<g1>> between two gauges.
(3) Letting it converge further does not change this: with a stopping criterion 100 times stricter, $E_{\rm div}/E=$ <<est>> and the gauge error is <<gs>>.""",
                   e1=fx(w1["E_div_frac"], 3), f1=num(w1["fi"], 2), te=fx(th["E_div_frac"], 2), tf=num(th["fi"], 1),
                   g1=pct(w1["gauge"], 0), est=fx(st.get("E_div_frac", float("nan")), 3), gs=pct(st.get("gauge", float("nan")), 0))
            self.p(r"""Por eso se probó a dar más peso a la divergencia, $w_d|\nabla\cdot\mathbf{B}|^2$ con la fuerza a peso 1, como en \citet{Wiegelmann2012} y \citet{Thalmann2019}. Primero en HARP 377, con todo lo demás igual:""",
                   r"""We therefore tested giving more weight to the divergence, $w_d|\nabla\cdot\mathbf{B}|^2$ with the force at weight 1, as in \citet{Wiegelmann2012} and \citet{Thalmann2019}. First on HARP 377, with everything else unchanged:""")
            self.w(r"\begin{center}\resizebox{\textwidth}{!}{\begin{tabular}{rcccccc}\toprule" "\n")
            self.w(r"$w_d$ & $E_{\rm div}/E$ & $\langle|f_i|\rangle\times10^4$ & $\mathrm{CW}_{\sin}$ & "
                   + self.L("Error de gauge de $H_V$", "Gauge error of $H_V$")
                   + r" & $E_{\rm free}$ " + self.L("(pot. de volumen)", "(volume pot.)")
                   + r" [$10^{32}$ erg] & $H_V$ [$10^{42}$ Mx$^2$]\\\midrule" "\n")
            for r in scan:
                mark = lambda ok: r"\,\checkmark" if ok else r"\,$\times$"
                row = (f"{fx(r['div_weight'], 0)} & {fx(r['E_div_frac'], 3)}{mark(r['E_div_frac'] <= th['E_div_frac'])} & "
                       f"{fx(r['fi'] * 1e4, 1)}{mark(r['fi'] <= th['fi'])} & {fx(r['CWsin'], 3)} & {pct(r['gauge'], 1)} & "
                       f"{fx(r['E_free'] / 1e32, 2)} & {fx(abs(r['H_V']) / 1e42, 2)}")
                if r["div_weight"] == 1.0:
                    row = r"\textbf{" + row.replace(" & ", r"} & \textbf{") + "}"
                self.w(row + r"\\" "\n")
            self.w(r"\bottomrule\end{tabular}}\end{center}" "\n\n")
            self.p(r"""(1) Con $w_d=$ <<w4>> se cumplen los dos umbrales y el error de gauge baja del <<g1>> al <<g4>>.
(2) El \textbf{valor} de la helicidad apenas cambia con $w_d$ (de <<h1>> a <<h4>>$\times10^{42}$~Mx$^2$): lo que cambia es su incertidumbre.
(3) La energía libre sí cambia: un <<dE>> más con $w_d=$ <<w4>>. En una región real no se puede saber cuál está más cerca de la verdad, así que se repitió la comparación con campos exactos.""",
                   r"""(1) With $w_d=$ <<w4>> both thresholds are met and the gauge error drops from <<g1>> to <<g4>>.
(2) The \textbf{value} of the helicity barely changes with $w_d$ (from <<h1>> to <<h4>>$\times10^{42}$~Mx$^2$): what changes is its uncertainty.
(3) The free energy does change: <<dE>> more with $w_d=$ <<w4>>. In a real region one cannot tell which is closer to the truth, so the comparison was repeated with exact fields.""",
                   w4=fx(wt, 0), g1=pct(w1["gauge"], 0), g4=pct(w4["gauge"], 0), h1=fx(abs(w1["H_V"]) / 1e42, 1),
                   h4=fx(abs(w4["H_V"]) / 1e42, 1), dE=pct(w4["E_free"] / w1["E_free"] - 1, 0))
            if C and X:
                c1, c4 = C["1.0"], C[str(float(wt))]
                x1, x4 = X["1.0"], X[str(float(wt))]
                self.p(r"""Con los campos exactos se conoce la verdad. La tabla compara $w_d=1$ y $w_d=$ <<w4>> en los <<nc>> casos de la calibración (líneas) y en <<nx>> casos de las cinco familias de campos con dos ruidos (energía y helicidad frente a sus valores exactos):""",
                       r"""With exact fields the truth is known. The table compares $w_d=1$ and $w_d=$ <<w4>> on the <<nc>> calibration cases (lines) and on <<nx>> cases of the five field families with two noise models (energy and helicity against their exact values):""",
                       w4=fx(wt, 0), nc=c1["n_cases"], nx=x1["n"])
                self.w(r"\begin{center}\begin{tabular}{lcc}\toprule" "\n")
                self.w(" & $w_d=1$ & $w_d=$ " + fx(wt, 0) + r"\\\midrule" "\n")
                rows = [
                    (self.L("Líneas malas del NLFFF", "Wrong NLFFF lines"), pct(c1["bad_all"]), pct(c4["bad_all"])),
                    (self.L("Error medio de $E_{\\rm free}$ (sin la cuerda)", "Mean $E_{\\rm free}$ error (without the rope)"),
                     pct(x1["err_E_free"]), pct(x4["err_E_free"])),
                    (self.L("Error medio de $H_V$ frente a la verdad", "Mean $H_V$ error against the truth"),
                     pct(x1["err_H_V"]), pct(x4["err_H_V"])),
                    (self.L("Error de gauge medio de $H_V$", "Mean gauge error of $H_V$"), pct(x1["gauge"]), pct(x4["gauge"])),
                ]
                for lab, a, b in rows:
                    self.w(f"{lab} & {a} & {b}" + r"\\" "\n")
                self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
                self.p(r"""(1) \textbf{Las líneas empeoran con $w_d=$ <<w4>>}: más líneas malas (la confianza por línea no se compara aquí: su calibración de esta batería es anterior al modelo publicado).
(2) \textbf{La energía no es más precisa} con $w_d=$ <<w4>> (<<x1>> frente a <<x4>> de error medio).
(3) \textbf{Que los dos gauges coincidan más no hace la helicidad más correcta}: su error frente a la verdad es <<v1>> con $w_d=1$ y <<v4>> con $w_d=$ <<w4>>.""",
                       r"""(1) \textbf{The lines get worse with $w_d=$ <<w4>>}: more wrong lines (the per-line confidence is not compared here: its calibration on this suite predates the published model).
(2) \textbf{The energy is not more accurate} with $w_d=$ <<w4>> (<<x1>> versus <<x4>> mean error).
(3) \textbf{Better agreement between the two gauges does not make the helicity more correct}: its error against the truth is <<v1>> with $w_d=1$ and <<v4>> with $w_d=$ <<w4>>.""",
                       w4=fx(wt, 0), x1=pct(x1["err_E_free"]), x4=pct(x4["err_E_free"]), v1=pct(x1["err_H_V"]), v4=pct(x4["err_H_V"]))
            self.box("key",
                     r"""Se mantiene $w_d=1$, la forma estándar: da mejores líneas y la energía y la helicidad no mejoran con más peso en la divergencia. Cumplir los umbrales de \citet{Thalmann2019} no es aquí un indicio de un modelo más correcto. La helicidad se da con su error de gauge como incertidumbre, y su valor es robusto: apenas cambia aunque la divergencia se reduzca a la mitad.""",
                     r"""We keep $w_d=1$, the standard form: it gives better lines, and neither the energy nor the helicity improves with more weight on the divergence. Meeting the thresholds of \citet{Thalmann2019} is not, here, a sign of a more correct model. The helicity is quoted with its gauge error as the uncertainty, and its value is robust: it barely changes even when the divergence is halved.""")
        if T:
            r = T["steps"].get("0.5") or next(iter(T["steps"].values()))
            self.w(r"\subsection{" + self.L("Trazado de las líneas de campo", "Field-line tracing") + r"}\label{sec:tracing}" "\n")
            self.p(r"""Las líneas se integran con Runge--Kutta de cuarto orden de paso fijo (medio píxel), con el campo interpolado de forma trilineal entre los nodos de la malla, y se detienen al salir de la caja (ajustando el extremo exactamente a la cara) o donde el campo es casi nulo. Un integrador adaptativo (como RK45) no ganaría precisión: con interpolación trilineal la derivada del campo salta entre celdas, y es la interpolación, no el integrador, la que limita la precisión. Para comprobar que el paso es suficiente se trazaron <<n>> líneas del NLFFF de HARP 377 con pasos de 0.5, 0.25 y 0.125 píxeles: con el paso usado, los extremos de las líneas se desplazan una mediana de <<de>> píxeles (percentil 95: <<d95>>) respecto al paso más fino, la longitud cambia un <<dl>> y \textbf{ninguna línea cambia de conectividad}. El error del trazado es despreciable frente a cualquier otra fuente de incertidumbre. Además, el campo exacto y el NLFFF se trazan con el mismo integrador en la calibración, así que lo que quede ya está incluido en la confianza.""",
                   r"""Lines are integrated with a fixed-step fourth-order Runge--Kutta scheme (half a pixel), with the field interpolated trilinearly between grid nodes, and stop when they leave the box (with the end point placed exactly on the face) or where the field is nearly null. An adaptive integrator (such as RK45) would not gain accuracy: with trilinear interpolation the field derivative jumps between cells, and it is the interpolation, not the integrator, that limits the accuracy. To check that the step is small enough, <<n>> lines of the HARP 377 NLFFF were traced with steps of 0.5, 0.25 and 0.125 pixels: with the step used, line end points move by a median of <<de>> pixels (95th percentile: <<d95>>) relative to the finest step, the length changes by <<dl>> and \textbf{no line changes its connectivity}. The tracing error is negligible compared with any other source of uncertainty. Moreover, the exact field and the NLFFF are traced with the same integrator in the calibration, so whatever remains is already included in the confidence.""",
                   n=T["n_lines"], de=fx(r["end_shift_px_median"], 3), d95=fx(r["end_shift_px_p95"], 3),
                   dl=pct(r["length_rel_median"], 3))

    def quality(self):
        A = self.A
        self.w(r"\section{" + self.L("Calidad de las soluciones", "Quality of the solutions") + r"}\label{sec:quality}" "\n")
        self.p(r"""Estas métricas miden cuánto se acerca cada campo a lo que debería ser (libre de fuerzas y sin divergencia), sin compararlo con ningún otro. Se calculan excluyendo la capa de amortiguamiento (<<nd>> puntos).""",
               r"""These metrics measure how close each field is to what it should be (force-free and divergence-free), without comparing it with anything else. They exclude the buffer layer (<<nd>> points).""", nd=A["nd"])
        self.w(r"""\begin{align}
\mathrm{CW}_{\sin}&=\frac{\sum_i |\mathbf{J}_i|\,\sigma_i}{\sum_i|\mathbf{J}_i|},\quad \sigma_i=\frac{|\mathbf{J}_i\times\mathbf{B}_i|}{|\mathbf{J}_i||\mathbf{B}_i|}=\sin\theta_i,\\
\langle|f_i|\rangle&=\left\langle\frac{|\nabla\cdot\mathbf{B}|_i\,\Delta x}{6|\mathbf{B}_i|}\right\rangle .
\end{align}
""")
        self.box("explain",
                 r"""\textbf{$\mathrm{CW}_{\sin}$} es el seno medio del ángulo $\theta$ entre la corriente y el campo, ponderado por la corriente: 0 si son paralelos en todas partes (libre de fuerzas). $\theta_J=\arcsin(\mathrm{CW}_{\sin})$ es ese ángulo en grados. \textbf{$\langle|f_i|\rangle$} es la fracción de flujo que ``se crea o se destruye'' en cada celda por la divergencia numérica: debe ser $\ll 1$. \textbf{Ojo con el campo potencial}: no tiene corrientes, así que su $\mathrm{CW}_{\sin}$ mide sólo ruido numérico y suele salir alto; no indica un problema.""",
                 r"""\textbf{$\mathrm{CW}_{\sin}$} is the mean sine of the angle $\theta$ between current and field, weighted by the current: 0 if they are parallel everywhere (force-free). $\theta_J=\arcsin(\mathrm{CW}_{\sin})$ is that angle in degrees. \textbf{$\langle|f_i|\rangle$} is the fraction of flux ``created or destroyed'' in each cell by numerical divergence: it should be $\ll 1$. \textbf{Beware of the potential field}: it has no currents, so its $\mathrm{CW}_{\sin}$ only measures numerical noise and is usually high; this does not indicate a problem.""")
        keys = list(A["quality"])
        cols = "l" + "r" * len(keys)
        self.w(f"\\begin{{center}}\\begin{{tabular}}{{{cols}}}\\toprule\n & "
               + " & ".join(self.mname(k) for k in keys) + r"\\\midrule" "\n")
        rows = [(r"$\mathrm{CW}_{\sin}$", "CWsin", 3), (r"$\theta_J$ [$^\circ$]", "theta_J_deg", 3),
                (r"$\langle|f_i|\rangle$", "<|f_i|>", 2)]
        for lab, key, sg in rows:
            self.w(lab + " & " + " & ".join(num(A["quality"][k][key], sg) for k in keys) + r"\\" "\n")
        self.w(r"$E$ [erg] & " + " & ".join(num(A["energy"][k]["E"]) for k in keys) + r"\\" "\n")
        self.w(r"$E/E_{\rm pot}$ & " + " & ".join(num(A["energy"][k]["ratio"], 4) for k in keys) + r"\\" "\n")
        self.w(r"$E_{\rm free}$ [erg] & " + " & ".join(num(A["energy"][k]["free"]) for k in keys) + r"\\" "\n")
        self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
        self._quality_energy_box()
        self._helicity_quality()
        return
        self.box("explain",
                 r"""\textbf{Energía.} $E=\int B^2/8\pi\,dV$ (en erg, unidades CGS). $E/E_{\rm pot}$ indica cuánta energía hay por encima del mínimo; un modelo libre de fuerzas correcto debe dar $E/E_{\rm pot}\ge1$. La energía libre de un LFFF depende mucho del tamaño del dominio (en un dominio infinito sería infinita), así que sólo es comparable dentro de este mismo dominio.""",
                 r"""\textbf{Energy.} $E=\int B^2/8\pi\,dV$ (in erg, CGS units). $E/E_{\rm pot}$ tells how much energy lies above the minimum; a correct force-free model must give $E/E_{\rm pot}\ge1$. The free energy of an LFFF depends strongly on the domain size (it would be infinite in an infinite domain), so it is only comparable within this same domain.""")

    def _quality_energy_box(self):
        self.box("explain",
                 r"""\textbf{Energía.} $E=\int B^2/8\pi\,dV$ (en erg, unidades CGS). $E/E_{\rm pot}$ indica cuánta energía hay por encima del mínimo; un modelo libre de fuerzas correcto debe dar $E/E_{\rm pot}\ge1$. La energía libre de un LFFF depende mucho del tamaño del dominio (en un dominio infinito sería infinita), así que sólo es comparable dentro de este mismo dominio.""",
                 r"""\textbf{Energy.} $E=\int B^2/8\pi\,dV$ (in erg, CGS units). $E/E_{\rm pot}$ tells how much energy lies above the minimum; a correct force-free model must give $E/E_{\rm pot}\ge1$. The free energy of an LFFF depends strongly on the domain size (it would be infinite in an infinite domain), so it is only comparable within this same domain.""")

    def _helicity_quality(self):
        hel = self.A.get("helicity")
        if hel:
            self.w(r"\subsection{" + self.L("¿Es fiable la energía? Descomposición de Valori",
                                            "Is the energy reliable? Valori decomposition") + "}\n")
            self.p(r"""El potencial de referencia correcto para medir la energía libre es el que tiene la misma componente normal que el modelo en las \emph{seis} caras del dominio (no sólo en la base): sólo para él se cumple el teorema de Thomson, $E\geq E_p$, si el campo no tiene divergencia. Se calcula resolviendo $\nabla^2\phi=0$ con $\partial\phi/\partial n=B_n$ en todas las caras. Después, siguiendo a Valori et al. (2013), cada parte del campo se separa en solenoidal y no solenoidal:""",
                   r"""The correct reference potential for measuring the free energy is the one with the same normal component as the model on all \emph{six} faces of the domain (not just the base): only for it does Thomson's theorem, $E\geq E_p$, hold if the field is divergence-free. It is computed by solving $\nabla^2\phi=0$ with $\partial\phi/\partial n=B_n$ on every face. Then, following Valori et al. (2013), each part of the field is split into solenoidal and non-solenoidal components:""")
            self.w(r"""\begin{equation}
    E=E_{p,s}+E_{J,s}+\underbrace{E_{p,ns}+E_{J,ns}+E_{\rm mix}}_{\to\ E_{\rm div}} .
    \end{equation}
    """)
            keys = list(hel)
            self.w(f"\\begin{{center}}\\begin{{tabular}}{{l{'r' * len(keys)}}}\\toprule\n & "
                   + " & ".join(self.mname(k) for k in keys) + r"\\\midrule" "\n")
            for lab, f in ((r"$E$ [erg]", lambda h: num(h["energy"]["E"])),
                           (self.L(r"$E_p$ (volumen) [erg]", r"$E_p$ (volume) [erg]"), lambda h: num(h["energy"]["E_p"])),
                           (self.L(r"$E-E_p$ = energía libre [erg]", r"$E-E_p$ = free energy [erg]"), lambda h: num(h["energy"]["E_free"])),
                           (r"$E_{J,s}$ [erg]", lambda h: num(h["energy"]["E_J_s"])),
                           (r"$E_{\rm div}/E$", lambda h: pct(h["energy"]["E_div_frac"]))):
                self.w(lab + " & " + " & ".join(f(hel[k]) for k in keys) + r"\\" "\n")
            self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
            self.figure("energy_helicity",
                        r"Izquierda: descomposición de la energía. Derecha: helicidad relativa y su reparto entre la parte con corrientes ($H_J$) y la mutua ($H_{PJ}$).",
                        r"Left: energy decomposition. Right: relative helicity and its split into the current-carrying ($H_J$) and mutual ($H_{PJ}$) parts.")
            self.box("read",
                     r"""\textbf{Izquierda:} gris = energía del campo potencial; azul = energía de las corrientes (la energía libre ``limpia''); naranja = energía espuria debida a la divergencia numérica. La cifra sobre cada barra es $E_{\rm div}/E$: si supera unos pocos por ciento, la energía libre debe tomarse con cautela, porque el naranja puede ser comparable al azul. \textbf{Derecha:} $H_J$ es la helicidad de la parte con corrientes y $H_{PJ}$ la que comparten el campo potencial y el de corrientes. Su suma es la helicidad relativa total $H_V$.""",
                     r"""\textbf{Left:} grey = potential-field energy; blue = current energy (the ``clean'' free energy); orange = spurious energy due to numerical divergence. The figure above each bar is $E_{\rm div}/E$: if it exceeds a few per cent, the free energy should be treated with caution, because the orange part may be comparable to the blue one. \textbf{Right:} $H_J$ is the helicity of the current-carrying part and $H_{PJ}$ the one shared by the potential and current-carrying fields. Their sum is the total relative helicity $H_V$.""")

    def comparison(self):
        A = self.A
        ref = A["reference"]
        self.w(r"\section{" + self.L("Comparación entre modelos", "Comparison of models") + r"}\label{sec:comparison}" "\n")
        if ref == "exact":
            self.p(r"""La referencia es la solución exacta: todas las diferencias son errores de los modelos.""",
                   r"""The reference is the exact solution: all differences are model errors.""")
        elif ref == "nlfff":
            self.p(r"""Con datos reales no hay solución exacta. Se toma el NLFFF como referencia, por ser el modelo físicamente más completo, y se mide cuánto se alejan de él el potencial y el LFFF. Esto mide \emph{diferencias}, no errores: el NLFFF también tiene incertidumbres.""",
                   r"""With real data there is no exact solution. The NLFFF is taken as reference, being the most complete physical model, and we measure how far the potential and LFFF fields depart from it. This measures \emph{differences}, not errors: the NLFFF also has uncertainties.""")
        self.w(r"\subsection{" + self.L("Líneas de campo", "Field lines") + "}\n")
        self.figure("lines",
                    r"Líneas de campo de cada modelo (columnas) en perspectiva, planta y perfil. Todas parten de las mismas semillas y el color indica su altura máxima, con la misma escala en todos los paneles.",
                    r"Field lines of each model (columns) in perspective, top and side views. All start from the same seeds and colour gives their apex height, with the same scale in all panels.")
        self.box("read",
                 r"""Cada columna es un modelo; las filas son tres vistas del mismo conjunto de líneas: en perspectiva, desde arriba (planta, sobre $B_z$) y de lado (perfil, proyección $x$--$z$). Las semillas están en la polaridad positiva, repartidas en rejilla, y son las mismas para todos los modelos. Qué mirar:
\begin{itemize}[nosep]
\item \textbf{En planta}, el ángulo con que los bucles cruzan la PIL (línea negra). Perpendicular = sin cizalla (potencial); oblicuo o casi paralelo = cizalla. Una forma en S o S invertida (\emph{sigmoide}) es señal de torsión.
\item \textbf{En perfil}, la altura de los bucles. Las líneas que llegan al borde lateral o al techo salen del dominio: su conectividad depende de contornos no observados.
\item \textbf{Comparando columnas}, cuánto cambia la geometría al añadir corrientes. El LFFF aplica la misma torsión en todas partes, por eso a menudo retuerce de más los bucles grandes y de menos los del núcleo.
\end{itemize}""",
                 r"""Each column is a model; the rows are three views of the same set of lines: perspective, from above (top view, over $B_z$) and from the side (side view, $x$--$z$ projection). The seeds are in the positive polarity, on a regular grid, and are the same for every model. What to look at:
\begin{itemize}[nosep]
\item \textbf{In the top view}, the angle at which loops cross the PIL (black line). Perpendicular = no shear (potential); oblique or nearly parallel = shear. An S or inverse-S shape (\emph{sigmoid}) is a sign of twist.
\item \textbf{In the side view}, the height of the loops. Lines reaching the side or top boundary leave the domain: their connectivity depends on unobserved boundaries.
\item \textbf{Comparing columns}, how much the geometry changes when currents are added. The LFFF applies the same twist everywhere, so it often over-twists large loops and under-twists those in the core.
\end{itemize}""")
        self.figure("connectivity",
                    r"Conectividad: seis semillas numeradas (mismo número = misma semilla en todos los modelos); el resto de líneas en gris.",
                    r"Connectivity: six numbered seeds (same number = same seed in every model); other lines in grey.")
        self.box("read",
                 r"""Sigue cada número de una columna a otra. Si la línea acaba en el mismo sitio en dos modelos, ambos predicen la misma \emph{conectividad} para ese punto; si acaba en otra zona (o sale del dominio), los modelos discrepan sobre qué regiones están conectadas magnéticamente. La conectividad importa porque determina, por ejemplo, dónde aparecen las cintas de una fulguración o qué regiones pueden interactuar.""",
                 r"""Follow each number from one column to the next. If the line ends at the same place in two models, both predict the same \emph{connectivity} for that point; if it ends elsewhere (or leaves the domain), the models disagree on which regions are magnetically connected. Connectivity matters because it determines, for example, where flare ribbons appear or which regions can interact.""")
        if A["lines"]:
            keys = list(A["lines"])
            self.w(f"\\begin{{center}}\\begin{{tabular}}{{l{'r' * len(keys)}}}\\toprule\n & "
                   + " & ".join(self.mname(k) for k in keys) + r"\\\midrule" "\n")
            self.w(self.L("Líneas cerradas", "Closed lines") + " & "
                   + " & ".join(pct(A["lines"][k]["closed"], 0) for k in keys) + r"\\" "\n")
            self.w(self.L("Altura media del ápice [Mm]", "Mean apex height [Mm]") + " & "
                   + " & ".join(f"{A['lines'][k]['apex_mean']:.1f}" for k in keys) + r"\\" "\n")
            if ref:
                self.w(self.L("Misma topología que la referencia", "Same topology as reference") + " & "
                       + " & ".join(pct(A["lines"][k].get("same_topology", 1.0), 0) for k in keys) + r"\\" "\n")
                self.w(self.L("Distancia media entre extremos [Mm]", "Mean endpoint distance [Mm]") + " & "
                       + " & ".join(f"{A['lines'][k].get('mean_endpoint_dist_Mm', 0.0):.1f}" for k in keys) + r"\\" "\n")
            self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")

        self.w(r"\subsection{" + self.L("El parámetro $\\alpha$: dónde están las corrientes",
                                        "The parameter $\\alpha$: where the currents are") + "}\n")
        self.figure("alpha_maps",
                    r"$\alpha=(\nabla\times\mathbf{B})\cdot\mathbf{B}/B^2$ en un corte horizontal a baja altura para cada modelo.",
                    r"$\alpha=(\nabla\times\mathbf{B})\cdot\mathbf{B}/B^2$ in a low horizontal slice for each model.")
        self.box("read",
                 r"""El color indica la torsión local: rojo $\alpha>0$ (dextrógira), azul $\alpha<0$ (levógira), gris claro $\alpha\approx0$ (sin corrientes). El gris oscuro marca campo demasiado débil para definir $\alpha$. Por construcción, el potencial es todo gris claro y el LFFF un único color uniforme. El NLFFF muestra \emph{dónde} se concentran las corrientes, normalmente a lo largo de la PIL y en los núcleos de las manchas. Esta diferencia es la razón de ser del NLFFF. Cerca de los bordes, el $\alpha$ del NLFFF puede ser ruidoso por la capa de amortiguamiento.""",
                 r"""Colour shows the local twist: red $\alpha>0$ (right-handed), blue $\alpha<0$ (left-handed), light grey $\alpha\approx0$ (no currents). Dark grey marks field too weak to define $\alpha$. By construction, the potential field is all light grey and the LFFF a single uniform colour. The NLFFF shows \emph{where} the currents concentrate, usually along the PIL and in sunspot cores. That difference is the whole point of the NLFFF. Near the edges the NLFFF $\alpha$ can be noisy because of the buffer layer.""")
        self.figure("alpha_hist",
                    r"Izquierda: $\alpha$ medio de cada línea de campo. Derecha: desviación típica de $\alpha$ a lo largo de cada línea.",
                    r"Left: mean $\alpha$ of each field line. Right: standard deviation of $\alpha$ along each line.")
        self.box("read",
                 r"""\textbf{Izquierda:} en el potencial todas las líneas tienen $\alpha\approx0$ y en el LFFF todas el mismo valor (un pico estrecho). En el NLFFF las líneas tienen torsiones distintas: un histograma ancho es justo lo que un LFFF no puede reproducir. \textbf{Derecha:} como $\alpha$ debe ser constante en cada línea de un campo libre de fuerzas, esta desviación es una prueba de calidad: cuanto más concentrada cerca de 0, mejor. Incluso una solución exacta da valores algo mayores que 0 por la discretización.""",
                 r"""\textbf{Left:} in the potential field every line has $\alpha\approx0$ and in the LFFF they all share one value (a narrow spike). In the NLFFF lines carry different twists: a broad histogram is precisely what an LFFF cannot reproduce. \textbf{Right:} since $\alpha$ must be constant along each line of a force-free field, this spread is a quality test: the closer to 0, the better. Even an exact solution gives values slightly above 0 because of discretisation.""")
        if A["vector"]:
            self.w(r"\subsection{" + self.L("Comparación vectorial", "Vector comparison") + "}\n")
            self.p(r"""Siguiendo a Schrijver et al. (2006), cada campo $\mathbf{b}$ se compara punto a punto con la referencia $\mathbf{B}$:""",
                   r"""Following Schrijver et al. (2006), each field $\mathbf{b}$ is compared point by point with the reference $\mathbf{B}$:""")
            self.w(r"""\begin{gather}
C_{\rm vec}=\frac{\sum\mathbf{B}\cdot\mathbf{b}}{\sqrt{\sum B^2\sum b^2}},\quad
C_{\rm CS}=\left\langle\frac{\mathbf{B}\cdot\mathbf{b}}{|\mathbf{B}||\mathbf{b}|}\right\rangle,\quad
E'_n=1-\frac{\sum|\mathbf{b}-\mathbf{B}|}{\sum|\mathbf{B}|},\quad
E'_m=1-\left\langle\frac{|\mathbf{b}-\mathbf{B}|}{|\mathbf{B}|}\right\rangle,\quad
\epsilon=\frac{\sum b^2}{\sum B^2}.
\end{gather}
""")
            keys = list(A["vector"])
            self.w(f"\\begin{{center}}\\begin{{tabular}}{{l{'r' * len(keys)}}}\\toprule\n & "
                   + " & ".join(self.mname(k) for k in keys) + r"\\\midrule" "\n")
            for lab, key in ((r"$C_{\rm vec}$", "C_vec"), (r"$C_{\rm CS}$", "C_CS"), (r"$E'_n$", "E'_n"),
                             (r"$E'_m$", "E'_m"), (r"$\epsilon$", "eps")):
                self.w(lab + " & " + " & ".join(num(A["vector"][k][key], 3) for k in keys) + r"\\" "\n")
            self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
            self.box("explain",
                     r"""Todas valen 1 para campos idénticos. $C_{\rm vec}$ y $E'_n$ están dominadas por el campo fuerte (donde está casi toda la energía); $C_{\rm CS}$ y $E'_m$ dan el mismo peso a cada punto, así que son sensibles a las zonas débiles y de los bordes. Por eso $C_{\rm vec}$ suele ser cercano a 1 aunque $E'_m$ sea bajo. $\epsilon$ es el cociente de energías: $>1$ si el modelo tiene más energía que la referencia. Como orientación, un buen NLFFF frente a una solución exacta da $C_{\rm vec}>0.99$ y $E'_n>0.8$.""",
                     r"""All equal 1 for identical fields. $C_{\rm vec}$ and $E'_n$ are dominated by the strong field (where nearly all the energy is); $C_{\rm CS}$ and $E'_m$ give equal weight to every point, so they are sensitive to weak and boundary regions. That is why $C_{\rm vec}$ is often close to 1 even when $E'_m$ is low. $\epsilon$ is the energy ratio: $>1$ if the model has more energy than the reference. As a guide, a good NLFFF against an exact solution gives $C_{\rm vec}>0.99$ and $E'_n>0.8$.""")
        if A.get("angle_by_height"):
            self.w(r"\subsection{" + self.L("¿Dónde difieren los modelos?", "Where do the models differ?") + "}\n")
            self.figure("errors",
                        r"Ángulo entre el campo de cada modelo (filas) y el de la referencia en cortes a varias alturas (columnas).",
                        r"Angle between each model's field (rows) and the reference field in slices at several heights (columns).")
            self.box("read",
                     r"""Azul claro = el modelo apunta casi en la misma dirección que la referencia; azul oscuro = difieren mucho (hasta $90^\circ$). $\langle\theta\rangle_B$, en la esquina, es el ángulo medio ponderado por la intensidad del campo. Lo típico es que el NLFFF acierte cerca de la base (donde se le imponen los datos) y que su ventaja se diluya con la altura y hacia los bordes, donde dominan los contornos tomados del campo potencial.""",
                     r"""Light blue = the model points almost in the same direction as the reference; dark blue = they differ strongly (up to $90^\circ$). $\langle\theta\rangle_B$, in the corner, is the mean angle weighted by field strength. Typically the NLFFF is accurate near the base (where the data are imposed) and its advantage fades with height and towards the edges, where the boundaries taken from the potential field dominate.""")
            keys = list(A["angle_by_height"])
            self.w(f"\\begin{{center}}\\begin{{tabular}}{{l{'r' * len(A['heights'])}}}\\toprule\n"
                   + self.L("Modelo", "Model") + " & "
                   + " & ".join(f"$z={h:.0f}$~Mm" for h in A["heights"]) + r"\\\midrule" "\n")
            for k in keys:
                self.w(self.mname(k) + " & " + " & ".join(f"${v:.1f}^\\circ$" for v in A["angle_by_height"][k]) + r"\\" "\n")
            self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")

    # ------------------------------------------------- is the region eruptive?
    def physics(self):
        A = self.A
        ph = A.get("physics") or {}
        if not ph.get("twist") and not ph.get("decay") and not ph.get("q"):
            return
        self.w(r"\section{" + self.L("¿Es la región eruptiva?", "Is the region eruptive?") + r"}\label{sec:physics}" "\n")
        self.p(r"""Las fulguraciones y eyecciones de masa coronal más energéticas suelen implicar una \textbf{cuerda de flujo} (\emph{flux rope}): un haz de líneas de campo enrolladas unas alrededor de otras. Tres diagnósticos complementarios, muy usados en la literatura, ayudan a valorar si una región puede entrar en erupción:
\begin{enumerate}[nosep]
\item \textbf{Twist $T_w$}: ¿hay líneas suficientemente enrolladas para formar una cuerda de flujo?
\item \textbf{Índice de decaimiento $n$}: si la cuerda sube, ¿la deja escapar el campo que tiene encima?
\item \textbf{Factor de squashing $Q$}: ¿dónde cambia bruscamente la conectividad, es decir, dónde es más probable la reconexión?
\end{enumerate}
Ninguno de ellos \emph{predice} una erupción por sí solo: son indicadores de la configuración magnética.""",
               r"""The most energetic flares and coronal mass ejections usually involve a \textbf{flux rope}: a bundle of field lines wound around each other. Three complementary diagnostics, widely used in the literature, help assess whether a region may erupt:
\begin{enumerate}[nosep]
\item \textbf{Twist $T_w$}: are there field lines wound enough to form a flux rope?
\item \textbf{Decay index $n$}: if the rope rises, does the overlying field let it escape?
\item \textbf{Squashing factor $Q$}: where does the connectivity change abruptly, i.e.\ where is reconnection most likely?
\end{enumerate}
None of them \emph{predicts} an eruption on its own: they are indicators of the magnetic configuration.""")

        # --- Twist
        if ph.get("twist"):
            self.w(r"\subsection{" + self.L("Número de torsión $T_w$", "Twist number $T_w$") + "}\n")
            self.w(r"""\begin{equation}
T_w=\frac{1}{4\pi}\int_{\rm línea}\alpha\,dl,\qquad \alpha=\frac{(\nabla\times\mathbf{B})\cdot\mathbf{B}}{B^2}.
\end{equation}
""".replace("línea", self.L("línea", "line")))
            self.box("explain",
                     r"""$T_w$ cuenta cuántas vueltas dan las líneas vecinas alrededor de una línea dada (Berger \& Prior 2006). Se calcula para la línea que sale de cada punto de la base. Una cuerda de flujo se identifica habitualmente por $|T_w|\gtrsim1$ (Liu et al. 2016); valores de $\sim$1.25--1.75 vueltas se asocian a la inestabilidad \emph{kink}, en la que la cuerda se retuerce sobre sí misma (Török et al. 2004). El signo indica el sentido del enrollamiento (el mismo que la helicidad). \textbf{Ojo con el LFFF}: como $\alpha$ es la misma en todas partes, $T_w=\alpha L/4\pi$ crece con la longitud $L$ de la línea, y los bucles largos tienen un $T_w$ grande que es un artefacto del modelo, no una cuerda real.""",
                     r"""$T_w$ counts how many turns neighbouring field lines make around a given line (Berger \& Prior 2006). It is computed for the line rooted at each point of the base. A flux rope is usually identified by $|T_w|\gtrsim1$ (Liu et al. 2016); values of $\sim$1.25--1.75 turns are associated with the \emph{kink} instability, in which the rope twists upon itself (Török et al. 2004). The sign gives the sense of winding (the same as the helicity). \textbf{Beware of the LFFF}: since $\alpha$ is the same everywhere, $T_w=\alpha L/4\pi$ grows with the line length $L$, so long loops have a large $T_w$ that is a model artefact, not a real rope.""")
            keys = list(ph["twist"])
            self.w(f"\\begin{{center}}\\begin{{tabular}}{{l{'r' * len(keys)}}}\\toprule\n"
                   + self.L(r"(líneas cerradas)", r"(closed lines)") + " & "
                   + " & ".join(self.mname(k) for k in keys) + r"\\\midrule" "\n")
            T = ph["twist"]
            self.w(r"$\max|T_w|$ & " + " & ".join(num(T[k]["max_abs"]) for k in keys) + r"\\" "\n")
            self.w(self.L(r"Percentil 99 de $|T_w|$", r"99th percentile of $|T_w|$") + " & "
                   + " & ".join(num(T[k]["p99_abs"]) for k in keys) + r"\\" "\n")
            self.w(self.L(r"Fracción con $|T_w|\geq1$", r"Fraction with $|T_w|\geq1$") + " & "
                   + " & ".join(pct(T[k]["frac_gt1"]) for k in keys) + r"\\" "\n")
            self.w(self.L(r"Mediana de $T_w$", r"Median $T_w$") + " & "
                   + " & ".join(num(T[k]["median"]) for k in keys) + r"\\" "\n")
            self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
            self.figure("twist",
                        r"Mapa de $T_w$: cada punto de la base tiene el color del $T_w$ de la línea que sale de él.",
                        r"$T_w$ map: each base point is coloured by the $T_w$ of the field line rooted there.")
            self.box("read",
                     r"""Rojo = enrollamiento dextrógiro ($T_w>0$), azul = levógiro ($T_w<0$), gris claro = sin torsión. El contorno punteado encierra los puntos con $|T_w|\geq1$: si forma una zona compacta y alargada a ambos lados de la PIL, es la huella de una cuerda de flujo (sus dos pies están en polaridades opuestas). Manchas pequeñas y dispersas con $|T_w|\geq1$ suelen ser ruido. Compara el NLFFF con el LFFF: en el LFFF el color crece hacia los bordes porque las líneas son más largas, no porque haya más torsión.""",
                     r"""Red = right-handed winding ($T_w>0$), blue = left-handed ($T_w<0$), light grey = no twist. The dotted contour encloses points with $|T_w|\geq1$: if it forms a compact, elongated patch on both sides of the PIL, it is the footprint of a flux rope (its two legs sit in opposite polarities). Small scattered patches with $|T_w|\geq1$ are usually noise. Compare the NLFFF with the LFFF: in the LFFF the colour grows towards the edges because lines are longer, not because there is more twist.""")

        # --- Decay index
        if ph.get("decay"):
            d = ph["decay"]
            self.w(r"\subsection{" + self.L("Índice de decaimiento e inestabilidad de toro",
                                            "Decay index and torus instability") + "}\n")
            self.w(r"""\begin{equation}
n(z)=-\frac{d\ln B_h}{d\ln z},
\end{equation}
""")
            self.p(r"""donde $B_h$ es el campo horizontal del campo \emph{potencial}, que representa el campo externo que sujeta la cuerda desde arriba, promediado sobre las columnas situadas encima de la PIL. Una cuerda de flujo que asciende sufre una fuerza hacia fuera (su propia corriente) y otra hacia abajo (el campo externo). Si el campo externo decae con la altura más deprisa que $n_{\rm crit}\approx1.5$, la fuerza de sujeción disminuye más rápido que la que la empuja y la cuerda se escapa: es la \textbf{inestabilidad de toro} (Kliem \& Török 2006). La altura a la que $n$ alcanza 1.5 se llama \textbf{altura crítica} $h_{\rm crit}$.""",
                   r"""where $B_h$ is the horizontal component of the \emph{potential} field, which represents the external field holding the rope down, averaged over the columns above the PIL. A rising flux rope feels an outward force (its own current) and a downward one (the external field). If the external field decays with height faster than $n_{\rm crit}\approx1.5$, the restraining force drops faster than the driving one and the rope escapes: this is the \textbf{torus instability} (Kliem \& Török 2006). The height at which $n$ reaches 1.5 is the \textbf{critical height} $h_{\rm crit}$.""")
            hc = d["h_crit"]
            if hc is not None and np.isfinite(hc):
                self.p(r"Para esta región, $h_{\rm crit}=$ <<hc>>~Mm (calculado sobre <<npx>> columnas de la PIL).",
                       r"For this region, $h_{\rm crit}=$ <<hc>>~Mm (computed over <<npx>> PIL columns).",
                       hc=f"{hc:.1f}", npx=d["n_pil_pixels"])
            else:
                self.p(r"En esta caja $n$ no llega a 1.5: el campo externo confina con fuerza en toda la altura calculada.",
                       r"In this box $n$ never reaches 1.5: the external field confines strongly over the whole computed height.")
            self.figure("decay",
                        r"Izquierda: índice de decaimiento $n(z)$ del campo potencial sobre la PIL. Derecha: columnas de la PIL usadas.",
                        r"Left: decay index $n(z)$ of the potential field above the PIL. Right: PIL columns used.")
            self.box("read",
                     r"""La curva azul es $n(z)$. La banda gris marca el rango de valores críticos citados en la literatura (1.1--1.75) y la línea discontinua el valor teórico 1.5. Donde la curva está por debajo de la banda, el campo externo confina bien. La línea vertical marca $h_{\rm crit}$: una cuerda de flujo cuyo eje esté por encima de esa altura sería inestable al toro. Cuanto más baja sea $h_{\rm crit}$, más fácil le resulta a una cuerda escapar. El tramo punteado está demasiado cerca del techo de la caja y es menos fiable. Para este cálculo el campo potencial se obtiene en un dominio ampliado al doble, porque las copias periódicas de la FFT falsean el decaimiento en altura (ver Apéndice~\ref{sec:validation}).""",
                     r"""The blue curve is $n(z)$. The grey band marks the range of critical values quoted in the literature (1.1--1.75) and the dashed line the theoretical value 1.5. Where the curve is below the band, the external field confines well. The vertical line marks $h_{\rm crit}$: a flux rope whose axis lies above that height would be torus-unstable. The lower $h_{\rm crit}$, the easier it is for a rope to escape. The dotted section is too close to the top of the box and less reliable. For this calculation the potential field is computed in a domain enlarged by a factor of two, because the periodic copies of the FFT distort the decay with height (see Appendix~\ref{sec:validation}).""")

        # --- Q
        if ph.get("q"):
            self.w(r"\subsection{" + self.L("Factor de squashing $Q$ y QSL", "Squashing factor $Q$ and QSLs") + "}\n")
            self.w(r"""\begin{equation}
Q=\frac{a^2+b^2+c^2+d^2}{|B_{n,0}/B_{n,1}|},\qquad
\begin{pmatrix}a&b\\c&d\end{pmatrix}=\frac{\partial(X,Y)}{\partial(x,y)} .
\end{equation}
""")
            self.box("explain",
                     r"""Imagina un tubo de flujo muy fino que sale de un círculo pequeño en la base. En su otro extremo, el círculo se ha deformado en una elipse. $Q$ mide cuánto se ha estirado esa elipse: $Q=2$ si no se deforma, $Q\gg2$ si se aplana mucho (Titov et al. 2002). Las zonas con $Q$ muy alto son las \textbf{capas cuasi-separatrices} (QSL, Démoulin et al. 1996): allí, puntos vecinos de la base están conectados con puntos muy alejados, las corrientes se concentran y la reconexión es más fácil. Las \textbf{cintas de las fulguraciones} suelen aparecer a lo largo de las huellas de las QSL en la base. Aquí se calcula con el método de Pariat \& Démoulin (2012), válido también para líneas que salen por los lados o el techo de la caja.""",
                     r"""Picture a very thin flux tube rooted in a small circle at the base. At its other end, the circle has been deformed into an ellipse. $Q$ measures how stretched that ellipse is: $Q=2$ if it is not deformed, $Q\gg2$ if it is strongly flattened (Titov et al. 2002). Regions of very high $Q$ are \textbf{quasi-separatrix layers} (QSLs, Démoulin et al. 1996): there, neighbouring base points connect to widely separated points, currents concentrate and reconnection is easier. \textbf{Flare ribbons} usually appear along the footprints of QSLs at the base. Here it is computed with the method of Pariat \& Démoulin (2012), which also works for lines leaving through the sides or the top of the box.""")
            keys = list(ph["q"])
            self.w(f"\\begin{{center}}\\begin{{tabular}}{{l{'r' * len(keys)}}}\\toprule\n & "
                   + " & ".join(self.mname(k) for k in keys) + r"\\\midrule" "\n")
            Qd = ph["q"]
            self.w(self.L(r"Área con $Q>100$", r"Area with $Q>100$") + " & "
                   + " & ".join(pct(Qd[k]["frac_logq_gt2"]) for k in keys) + r"\\" "\n")
            self.w(r"$\max\log_{10}Q$ & " + " & ".join(num(Qd[k]["max_logq"]) for k in keys) + r"\\" "\n")
            self.w(self.L(r"Área con $Q$ definido", r"Area with $Q$ defined") + " & "
                   + " & ".join(pct(Qd[k]["frac_defined"]) for k in keys) + r"\\" "\n")
            self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
            self.figure("qmap",
                        r"$\log_{10}Q$ en la base para cada modelo. Las bandas oscuras y estrechas son las huellas de las QSL.",
                        r"$\log_{10}Q$ at the base for each model. Narrow dark bands are the footprints of QSLs.")
            self.box("read",
                     r"""Fíjate en las \emph{bandas} oscuras finas: son las QSL. Compara con la PIL (línea negra) y entre modelos: añadir corrientes (NLFFF) suele desplazar y alargar las QSL respecto al potencial, y a menudo aparecen en forma de J o de S alrededor de una cuerda de flujo. Los puntos grises son píxeles de campo débil o líneas cuyos vecinos acaban en caras distintas de la caja (allí $Q$ no está definido). Si tienes imágenes de la fulguración (AIA 1600\,Å o H$\alpha$), las cintas deberían coincidir con estas bandas.""",
                     r"""Look at the thin dark \emph{bands}: those are the QSLs. Compare them with the PIL (black line) and across models: adding currents (NLFFF) usually shifts and elongates the QSLs relative to the potential field, often into J or S shapes around a flux rope. Grey points are weak-field pixels or lines whose neighbours end on different faces of the box (where $Q$ is undefined). If you have flare images (AIA 1600\,Å or H$\alpha$), the ribbons should match these bands.""")

        hel = (A.get("helicity") or {})
        if hel:
            self.w(r"\subsection{" + self.L("Helicidad magnética relativa", "Relative magnetic helicity") + "}\n")
            self.w(r"""\begin{equation}
H_V=\int_V(\mathbf{A}+\mathbf{A}_p)\cdot(\mathbf{B}-\mathbf{B}_p)\,dV=H_J+H_{PJ},\qquad
H_J=\int_V(\mathbf{A}-\mathbf{A}_p)\cdot(\mathbf{B}-\mathbf{B}_p)\,dV .
\end{equation}
""")
            self.box("explain",
                     r"""La helicidad magnética mide cuánto están retorcidas y entrelazadas las líneas de campo en conjunto. Se conserva muy bien incluso durante la reconexión, por lo que una región que acumula helicidad tiene que deshacerse de ella de algún modo, y las eyecciones de masa coronal son una vía. En un volumen finito se usa la helicidad \emph{relativa} al campo potencial (Berger \& Field 1984; Finn \& Antonsen 1985), que no depende del gauge del potencial vector. Aquí se calcula en el gauge de DeVore ($A_z=0$; DeVore 2000; Valori et al. 2012), y se repite tomando la referencia en el techo y en la base: la diferencia mide el error numérico. \textbf{El índice $|H_J|/|H_V|$} (Pariat et al. 2017) indica qué parte de la helicidad está en la componente con corrientes; en simulaciones, las erupciones se producen cuando este índice es alto. No hay un umbral universal, así que es más útil para comparar regiones o seguir su evolución en el tiempo.""",
                     r"""Magnetic helicity measures how twisted and interlinked field lines are as a whole. It is very well conserved even during reconnection, so a region accumulating helicity must shed it somehow, and coronal mass ejections are one way. In a finite volume the helicity \emph{relative} to the potential field is used (Berger \& Field 1984; Finn \& Antonsen 1985), which is independent of the vector-potential gauge. It is computed here in the DeVore gauge ($A_z=0$; DeVore 2000; Valori et al. 2012) and repeated with the reference at the top and at the base: the difference measures the numerical error. \textbf{The index $|H_J|/|H_V|$} (Pariat et al. 2017) gives the fraction of helicity in the current-carrying component; in simulations, eruptions occur when this index is high. There is no universal threshold, so it is most useful for comparing regions or following their time evolution.""")
            keys = list(hel)
            self.w(f"\\begin{{center}}\\begin{{tabular}}{{l{'r' * len(keys)}}}\\toprule\n & "
                   + " & ".join(self.mname(k) for k in keys) + r"\\\midrule" "\n")
            for lab, f in ((r"$H_V$ [Mx$^2$]", lambda h: num(h["helicity"]["H_V"])),
                           (r"$H_J$ [Mx$^2$]", lambda h: num(h["helicity"]["H_J"])),
                           (r"$H_V/\Phi^2$", lambda h: num(h["H_norm"])),
                           (r"$|H_J|/|H_V|$", lambda h: num(h["helicity"]["ratio_HJ_HV"])),
                           (self.L("Diferencia entre gauges", "Gauge difference"), lambda h: pct(h["gauge_diff"]))):
                self.w(lab + " & " + " & ".join(f(hel[k]) for k in keys) + r"\\" "\n")
            self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
            self.p(r"""$\Phi$ es el flujo no signado de la base dividido entre dos; $H_V/\Phi^2$ es una helicidad adimensional, comparable entre regiones de distinto tamaño. El signo de $H_V$ debe coincidir con el de $\alpha$ y el de $T_w$.""",
                   r"""$\Phi$ is half the unsigned flux of the base; $H_V/\Phi^2$ is a dimensionless helicity, comparable between regions of different size. The sign of $H_V$ should match those of $\alpha$ and $T_w$.""")

        # --- Joint reading
        c = ph.get("combined")
        if c:
            self.w(r"\subsection{" + self.L("Lectura conjunta", "Combined reading") + "}\n")
            self.w(r"\textbf{" + self.L("Con todas las líneas.", "With all lines.") + "} " + self.combined_text(c) + "\n\n")
            cr = (A.get("reliability") or {}).get("combined")
            if cr:
                self.w(r"\textbf{" + self.L("Sólo con líneas fiables", "Only with reliable lines")
                       + f" ({cr['n_used']}).}} " + self.combined_text(cr) + "\n\n")
                if cr["is_rope"] != c["is_rope"] or (
                        c.get("h_crit") is not None and np.isfinite(c["h_crit"])
                        and (cr["apex_median"] >= c["h_crit"]) != (c["apex_median"] >= c["h_crit"])):
                    self.p(r"\emph{Las dos lecturas difieren: la conclusión depende de líneas poco fiables, así que la versión restringida a líneas fiables es la más defendible.}",
                           r"\emph{The two readings differ: the conclusion depends on unreliable lines, so the version restricted to reliable lines is the more defensible one.}")
                else:
                    self.p(r"\emph{Ambas lecturas coinciden: la conclusión no depende de líneas poco fiables.}",
                           r"\emph{Both readings agree: the conclusion does not depend on unreliable lines.}")
            self.box("warn",
                     r"""Esta lectura combina umbrales teóricos simplificados (toro delgado, $n_{\rm crit}=1.5$; cuerda si $|T_w|\geq1$). Las erupciones reales dependen también de la evolución de la región (emergencia, cancelación de flujo) y de la reconexión. Úsala como orientación para tu análisis, no como predicción.""",
                     r"""This reading combines simplified theoretical thresholds (thin torus, $n_{\rm crit}=1.5$; rope if $|T_w|\geq1$). Real eruptions also depend on the evolution of the region (emergence, flux cancellation) and on reconnection. Use it as guidance for your analysis, not as a prediction.""")

    def combined_text(self, c):
        hc = c.get("h_crit")
        has_hc = hc is not None and np.isfinite(hc)
        base = fill(self.L(
            r"Las 30 líneas más torsionadas del <<m>> tienen una mediana de $|T_w|=$ <<tw>> y alcanzan una altura mediana de <<ap>>~Mm",
            r"The 30 most twisted field lines of the <<m>> have a median $|T_w|=$ <<tw>> and reach a median height of <<ap>>~Mm"),
            m=self.mname(c["model"]), tw=f"{c['tw_top_median']:.2f}", ap=f"{c['apex_median']:.1f}")
        if has_hc:
            base += fill(self.L(r", frente a una altura crítica $h_{\rm crit}=$ <<hc>>~Mm. ",
                                r", compared with a critical height $h_{\rm crit}=$ <<hc>>~Mm. "), hc=f"{hc:.1f}")
        else:
            base += self.L(r"; $n$ no alcanza el umbral en la caja. ", r"; $n$ does not reach the threshold in the box. ")
        if not c["is_rope"]:
            return base + self.L(
                r"\textbf{No se identifica una cuerda de flujo coherente} ($|T_w|<1$): no hay una estructura que pueda escapar por inestabilidad de toro. La energía libre, si se libera, lo haría más probablemente en fulguraciones confinadas.",
                r"\textbf{No coherent flux rope is identified} ($|T_w|<1$): there is no structure that could escape through the torus instability. Free energy, if released, would more likely go into confined flares.")
        if has_hc and c["apex_median"] >= hc:
            return base + self.L(
                r"\textbf{Hay una cuerda de flujo que ya está a la altura crítica o por encima}: configuración potencialmente inestable al toro, compatible con una erupción.",
                r"\textbf{There is a flux rope already at or above the critical height}: a potentially torus-unstable configuration, compatible with an eruption.")
        return base + self.L(
            r"\textbf{Hay una cuerda de flujo, pero por debajo de la altura crítica}: el campo externo la confina. Tendría que ascender (p.\,ej., por emergencia o cancelación de flujo) para volverse inestable.",
            r"\textbf{There is a flux rope, but below the critical height}: the external field confines it. It would have to rise (e.g.\ through flux emergence or cancellation) to become unstable.")

    # ---------------------------------------------------------- validation
    def validation_section(self):
        V = self.validation
        if not V:
            return
        self.w(r"\appendix" "\n" r"\section{" + self.L("Validación analítica de las herramientas",
                                                       "Analytic validation of the tools") + r"}\label{sec:validation}" "\n")
        self.p(r"""Cada herramienta usada en este informe se ha comprobado, en el momento de generarlo, frente a una solución analítica cuya respuesta se conoce exactamente. Si todos los tests pasan, los números de las secciones anteriores no contienen errores de implementación, aunque sí las incertidumbres propias de los datos y de las hipótesis físicas.""",
               r"""Every tool used in this report was checked, at the time of generating it, against an analytic solution whose answer is known exactly. If all tests pass, the numbers in the previous sections contain no implementation errors, although they still carry the uncertainties of the data and of the physical assumptions.""")
        self.w(r"\begin{center}\small\begin{tabular}{>{\raggedright}p{5.6cm}rrrr l}\toprule" "\n")
        self.w(self.L(r"Test & Esperado & Obtenido & Error & Tolerancia & Estado",
                      r"Test & Expected & Obtained & Error & Tolerance & Status") + r"\\\midrule" "\n")
        for t in V:
            name = t["name_es"] if self.lang == "es" else t["name_en"]
            st = "good" if t["passed"] else "poor"
            lab = self.L("Pasa", "Pass") if t["passed"] else self.L("Falla", "Fail")
            self.w(f"{name} & {num(t['expected'], 4)} & {num(t['obtained'], 4)} & "
                   f"{num(t['error'], 2)} & {num(t['tol'], 2)} & \\st{st}{{{lab}}}\\\\\n")
        self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
        self.p(r"""\begin{itemize}[nosep]
\item \textbf{Potencial}: el campo de un dipolo enterrado es potencial; el método FFT (dominio ampliado $\times2$) debe reproducirlo. Error: $1-C_{\rm vec}$.
\item \textbf{LFFF}: se comprueba que el campo calculado cumple $\nabla\times\mathbf{B}=\alpha\mathbf{B}$ (residuo relativo máximo).
\item \textbf{NLFFF}: reconstrucción de la solución de Low \& Lou (1990) a partir de su plano $z=0$. Error: $1-C_{\rm vec}$.
\item \textbf{NLFFF L-BFGS}: lo mismo con el método de gradiente exacto y base relajada, partiendo de un magnetograma con 30~G de ruido y preprocesado, como con datos reales.
\item \textbf{Potencial de volumen}: Laplace con Neumann en las 6 caras; con una solución cuadrática la discretización es exacta.
\item \textbf{Helicidad}: debe ser 0 para un campo potencial y no depender del gauge (se compara el gauge de DeVore con referencia arriba y abajo); la energía no solenoidal de una solución exacta debe ser pequeña.
\item \textbf{Grad--Rubin P y N}: reconstrucción de Low \& Lou con el método de Grad--Rubin a partir de $\alpha$ en cada polaridad. Error: $1-C_{\rm vec}$.
\item \textbf{Confianza por línea}: se aplica la calibración a una realización de ruido que no se usó al calibrar y se mide su AUC frente al error real de cada línea. Error: $1-\mathrm{AUC}$.
\item \textbf{$T_w$}: en el tubo de Gold \& Hoyle (1960) cada línea es una hélice y $T_w(r)=bH/(2\pi\sqrt{1+b^2r^2})$ exactamente.
\item \textbf{Índice de decaimiento}: sobre un dipolo horizontal enterrado a profundidad $d$, $n(z)=3z/(z+d)$, así que $h_{\rm crit}=d$.
\item \textbf{$Q$}: en el campo $\mathbf{B}=(\lambda x,-\lambda y,B_0)$, $Q=2\cosh(2\lambda H/B_0)$ en todo el dominio.
\end{itemize}""",
               r"""\begin{itemize}[nosep]
\item \textbf{Potential}: the field of a buried dipole is potential; the FFT method (domain enlarged $\times2$) must reproduce it. Error: $1-C_{\rm vec}$.
\item \textbf{LFFF}: checks that the computed field satisfies $\nabla\times\mathbf{B}=\alpha\mathbf{B}$ (maximum relative residual).
\item \textbf{NLFFF}: reconstruction of the Low \& Lou (1990) solution from its $z=0$ plane. Error: $1-C_{\rm vec}$.
\item \textbf{L-BFGS NLFFF}: the same with the exact-gradient method and relaxed base, starting from a magnetogram with 30~G noise, preprocessed, as with real data.
\item \textbf{Volume potential}: Laplace with Neumann on all 6 faces; for a quadratic solution the discretisation is exact.
\item \textbf{Helicity}: must vanish for a potential field and be gauge-independent (DeVore gauge with reference at the top and at the bottom); the non-solenoidal energy of an exact solution must be small.
\item \textbf{Grad--Rubin P and N}: reconstruction of Low \& Lou with the Grad--Rubin method from $\alpha$ in each polarity. Error: $1-C_{\rm vec}$.
\item \textbf{Line confidence}: the calibration is applied to a noise realisation not used for calibrating, and its AUC against the true error of each line is measured. Error: $1-\mathrm{AUC}$.
\item \textbf{$T_w$}: in the Gold \& Hoyle (1960) tube each line is a helix and $T_w(r)=bH/(2\pi\sqrt{1+b^2r^2})$ exactly.
\item \textbf{Decay index}: above a horizontal dipole buried at depth $d$, $n(z)=3z/(z+d)$, so $h_{\rm crit}=d$.
\item \textbf{$Q$}: in the field $\mathbf{B}=(\lambda x,-\lambda y,B_0)$, $Q=2\cosh(2\lambda H/B_0)$ throughout the domain.
\end{itemize}""")
        self.figure("calibration",
                    r"Calibración de la confianza: poder predictivo de cada indicador y error real frente a la confianza en una realización de ruido nueva.",
                    r"Confidence calibration: predictive power of each indicator, and true error versus confidence on a new noise realisation.")
        self.figure("validation",
                    r"Resultados de \texttt{solarff} (azul) frente a las soluciones exactas (negro).",
                    r"\texttt{solarff} results (blue) versus the exact solutions (black).")
        self.box("read",
                 r"""Los puntos y curvas azules deben caer sobre las líneas negras. En el panel de $T_w$, la pequeña dispersión se debe a la discretización y disminuye al aumentar la resolución. En el índice de decaimiento, la curva FFT se separa de la exacta cerca del techo de la caja, por la influencia de las copias periódicas: por eso el informe marca como menos fiable el tramo superior de $n(z)$. En $Q$, el histograma debe ser un pico estrecho sobre el valor exacto.""",
                 r"""Blue points and curves should fall on the black lines. In the $T_w$ panel, the small scatter comes from discretisation and decreases with resolution. In the decay index, the FFT curve departs from the exact one near the top of the box because of the periodic copies: that is why the report flags the upper part of $n(z)$ as less reliable. In $Q$, the histogram should be a narrow spike on the exact value.""")

    # ------------------------------------------------- per-line reliability
    def reliability(self):
        R = self.A.get("reliability")
        if not R:
            return
        self.w(r"\section{" + self.L("¿Qué partes del modelo son fiables?", "Which parts of the model are reliable?")
               + r"}\label{sec:reliability}" "\n")
        self.p(r"""Las métricas globales (CW$_{\sin}$, $E/E_{\rm pot}$...) resumen todo el volumen en un número, pero un modelo puede ser bueno en unas zonas y malo en otras. Esta sección asigna a \textbf{cada línea de campo} una \textbf{confianza} entre 0 y 1, para analizar sólo las partes significativas del modelo.""",
               r"""Global metrics (CW$_{\sin}$, $E/E_{\rm pot}$...) summarise the whole volume in one number, but a model can be good in some regions and poor in others. This section assigns \textbf{each field line} a \textbf{confidence} between 0 and 1, so that only the significant parts of the model are analysed.""")
        self.w(r"\subsection{" + self.L("Indicadores y calibración", "Indicators and calibration") + "}\n")
        self.w(r"\begin{center}\small\begin{tabular}{>{\raggedright}p{3.6cm}>{\raggedright}p{7.6cm}rc}\toprule" "\n")
        self.w(self.L(r"Indicador & Qué detecta & AUC & Usado", r"Indicator & What it detects & AUC & Used") + r"\\\midrule" "\n")
        rows = [
            ("sin_jb", self.L("Ángulo J--B a lo largo de la línea", "J--B angle along the line"),
             self.L("si la línea no está libre de fuerzas", "whether the line is not force-free")),
            ("alpha_spread", self.L(r"Variación de $\alpha$ en la línea", r"Variation of $\alpha$ along the line"),
             self.L(r"$\alpha$ debe ser constante en cada línea", r"$\alpha$ must be constant along each line")),
            ("buffer", self.L("Paso por la capa de amortiguamiento", "Time in the buffer layer"),
             self.L("dependencia de contornos no observados", "dependence on unobserved boundaries")),
            ("apex", self.L("Altura máxima", "Apex height"),
             self.L("cercanía al techo (no observado)", "proximity to the (unobserved) top")),
            ("length", self.L("Longitud", "Length"), self.L("líneas largas acumulan error", "long lines accumulate error")),
            ("length_h", self.L("Longitud en alturas de caja", "Length in box heights"),
             self.L("una línea larga frente a la altura de la caja depende del techo (no observado)",
                    "a line that is long compared with the height of the box depends on the (unobserved) top")),
            ("length_sq", self.L("Longitud (término cuadrático)", "Length (quadratic term)"),
             self.L("las líneas muy cortas también son sospechosas (bajo una cuerda de flujo)",
                    "very short lines are also suspicious (under a flux rope)")),
            ("div", self.L("Divergencia", "Divergence"), self.L("errores numéricos", "numerical errors")),
            ("open", self.L("Sale del dominio", "Leaves the domain"), self.L("conectividad incompleta", "incomplete connectivity")),
            ("foot_snr", self.L(r"$|B_t|$ en el pie / ruido", r"$|B_t|$ at footpoint / noise"),
             self.L(r"si el $\alpha$ del pie es medible", r"whether the footpoint $\alpha$ is measurable")),
            ("base_change", self.L("Corrección de la base en el pie", "Base correction at footpoint"),
             self.L("cuánto ignoró el modelo los datos", "how much the model ignored the data")),
        ]
        from ..confidence import calibration_metrics, select_calibration
        extras = tuple(x for x in ("mc_instability", "method_disagreement") if x in (R.get("calibration") or ""))
        cal_used = select_calibration(extras)
        CM = calibration_metrics(cal_used)
        if "mc_instability" in cal_used["features"]:
            rows.append(("mc_instability", self.L("Inestabilidad en el Monte Carlo", "Monte Carlo instability"),
                         self.L("si la línea cambia con el ruido de medida", "whether the line changes with measurement noise")))
        if "method_disagreement" in cal_used["features"] or "method_disagreement" in (R.get("extras") or ()):
            rows.append(("method_disagreement", self.L("Desacuerdo con Grad--Rubin", "Disagreement with Grad--Rubin"),
                         self.L("si otro método traza la línea distinta", "whether another method traces the line differently")))
        used = set(cal_used["features"])
        from ..calibration import load_results
        _S = load_results()
        from ..confidence import LEGACY_LOWLOU_METRICS
        legacy_used = False
        for k, name, what in rows:
            v = CM["auc_features"].get(k)
            mark = ""
            if k == "length_sq":  # not a predictor on its own: it modifies the effect of the length
                self.w(f"{name} & {what} & -- & {'$\\checkmark$' if k in used else '--'}\\\\\n")
                continue
            if v is None:  # indicator only evaluated in the earlier calibration (Low & Lou)
                v, mark, legacy_used = LEGACY_LOWLOU_METRICS["auc_features"].get(k), r"$^\dagger$", True
            self.w(f"{name} & {what} & {num(v, 2)}{mark} & {'$\\checkmark$' if k in used else '--'}\\\\\n")
        self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
        if legacy_used:
            self.p(r"""$^\dagger$ Indicador evaluado sólo en la calibración anterior con Low \& Lou (requiere un Monte Carlo o Grad--Rubin por caso, aún no incluidos en la batería diversa).""",
                   r"""$^\dagger$ Indicator evaluated only in the earlier Low \& Lou calibration (it requires a Monte Carlo or Grad--Rubin per case, not yet included in the diverse suite).""")
        self.p(r"""\textbf{Cómo se ha calibrado.} En campos libres de fuerzas exactos se conoce el \emph{error real} de cada línea del modelo: se traza desde la misma semilla en el modelo y en la solución exacta, y se mide cuánto se separan. Una línea es \emph{mala} si su desviación media supera el 10\% de su longitud o si cambia de topología. Una regresión logística \citep{Cox1958} combina los indicadores marcados en la tabla, con un término cuadrático en el logaritmo de la longitud para que tanto las líneas muy largas como las muy cortas puedan ser sospechosas. Se ha ajustado con una batería de <<nc>> casos (<<nf>> familias de campos, incluida una cuerda de flujo, tres modelos de ruido y tres resoluciones; ver más abajo) y todas las cifras que siguen son \emph{fuera de muestra}: cada familia de campos se evalúa con un modelo entrenado sin ella. La columna AUC mide cuánto predice cada indicador por sí solo (0.5 = nada, 1 = perfecto). Sólo se usan los indicadores que aportan algo en la validación cruzada: quitar el ángulo J--B o la altura máxima no la empeora (ablación), porque repiten información de la variación de $\alpha$ y de la longitud, así que se omiten; los demás tampoco la mejoran. Combinados dan AUC $=$ <<auc>>, y la probabilidad está bien calibrada (ECE $=$ <<ece>>), así que la confianza se puede leer literalmente: una línea con confianza 0.9 está mal con un 10\% de probabilidad. Se consideran \emph{fiables} las líneas con confianza $\geq$ <<thr>>: entre ellas la fracción de líneas malas baja del <<b0>> al <<b1>> (<<bw>> en el peor campo nunca visto), y el error mediano pasa de <<er>> (rechazadas) a <<ea>> (aceptadas).""",
               r"""\textbf{How it was calibrated.} In exact force-free fields the \emph{true error} of each model field line is known: it is traced from the same seed in the model and in the exact solution, and their separation is measured. A line is \emph{bad} if its mean deviation exceeds 10\% of its length or if it changes topology. A logistic regression \citep{Cox1958} combines the indicators ticked in the table, with a quadratic term in the logarithm of the length so that both very long and very short lines can be suspicious. It was fitted on a suite of <<nc>> cases (<<nf>> field families, including a flux rope, three noise models and three resolutions; see below) and all the figures below are \emph{out of sample}: each field family is evaluated with a model trained without it. The AUC column measures how much each indicator predicts on its own (0.5 = nothing, 1 = perfect). Only the indicators that add something in the cross-validation are used: removing the J--B angle or the apex height does not worsen it (ablation), because they repeat information carried by the variation of $\alpha$ and by the length, so they are omitted; the rest do not improve it either. Combined they give AUC $=$ <<auc>>, and the probability is well calibrated (ECE $=$ <<ece>>), so the confidence can be read literally: a line with confidence 0.9 is wrong with 10\% probability. Lines with confidence $\geq$ <<thr>> are considered \emph{reliable}: among them the fraction of bad lines drops from <<b0>> to <<b1>> (<<bw>> in the worst unseen field), and the median error goes from <<er>> (rejected) to <<ea>> (accepted).""",
               auc=num(CM["auc_combined"], 2), thr=num(R["threshold"], 2), b0=pct(CM["bad_fraction_all"], 0),
               b1=pct(CM["bad_fraction_accepted"], 0), er=num(CM["error_median_rejected"], 2),
               ea=num(CM["error_median_accepted"], 2), ece=num(CM.get("ece", float("nan")), 2),
               bw=pct(CM.get("bad_accepted_worst_field", float("nan")), 0),
               nc=(_S or {}).get("n_cases", "--"), nf=len((_S or {}).get("fields", [])) or "--")
        if "mc_instability" in cal_used["features"] and CM.get("auc_gain_mc") is not None:
            g0, g1 = CM["auc_gain_mc_ci"]
            self.p(r"""\textbf{Con Monte Carlo.} Como esta región tiene un Monte Carlo, la confianza incluye además la \emph{estabilidad} de cada línea frente al ruido. Esta versión se calibró con <<mcn>> casos de la misma batería (32$^2$ y 48$^2$, <<nm>> realizaciones de ruido por caso) y en esos mismos casos mejora el AUC en <<g>> (intervalo al 95\% por bootstrap pareado: <<g0>>--<<g1>>): una mejora pequeña pero real. El ángulo J--B se retira de esta versión porque, junto a los demás indicadores, su peso es nulo. Las cifras de arriba corresponden a esta versión.""",
                   r"""\textbf{With Monte Carlo.} Since this region has a Monte Carlo, the confidence also includes the \emph{stability} of each line against noise. This version was calibrated with <<mcn>> cases of the same suite (32$^2$ and 48$^2$, <<nm>> noise realisations per case) and on those same cases it improves the AUC by <<g>> (95\% interval from a paired bootstrap: <<g0>>--<<g1>>): a small but real improvement. The J--B angle is dropped in this version because, together with the other indicators, its weight is zero. The figures above refer to this version.""",
                   nm=CM.get("mc_realizations", "--"), g=fx(CM["auc_gain_mc"], 3), g0=fx(g0, 3), g1=fx(g1, 3),
                   mcn=CM.get("n_cases", "--"))
        elif "mc_instability" in (R.get("extras") or ()):
            from ..confidence import CALIBRATION_METRICS_MC as CMM
            if CMM.get("auc_gain_mc_ci"):
                self.p(r"""\textbf{¿Y el Monte Carlo?} Esta región tiene un Monte Carlo, pero su indicador (la \emph{estabilidad} de cada línea frente al ruido) no se usa. Con las cinco primeras familias mejoraba el AUC en <<g5>>; al añadir la cuerda de flujo la mejora desaparece (<<g>>, intervalo al 95\%: <<g0>> a <<g1>>, en <<n>> casos con 4 realizaciones de ruido cada uno). En la cuerda el NLFFF se equivoca de forma \emph{estable}: las líneas apenas cambian con el ruido y, sin embargo, son incorrectas, porque el error es sistemático. La estabilidad frente al ruido no implica que la línea sea correcta. La calibración con Monte Carlo sólo se usa si su ganancia es significativa.""",
                       r"""\textbf{What about the Monte Carlo?} This region has a Monte Carlo, but its indicator (the \emph{stability} of each line against noise) is not used. With the first five families it improved the AUC by <<g5>>; when the flux rope is added the improvement vanishes (<<g>>, 95\% interval: <<g0>> to <<g1>>, on <<n>> cases with 4 noise realisations each). In the rope the NLFFF is wrong in a \emph{stable} way: the lines barely change with noise and yet they are incorrect, because the error is systematic. Stability against noise does not imply that the line is correct. The Monte Carlo calibration is only used if its gain is significant.""",
                       g5=("$+$" + fx(CMM["auc_gain_mc_5fam"], 3)) if CMM.get("auc_gain_mc_5fam") is not None else "--",
                       g=("$+$" if CMM["auc_gain_mc"] >= 0 else "$-$") + fx(abs(CMM["auc_gain_mc"]), 3),
                       g0=fx(CMM["auc_gain_mc_ci"][0], 3), g1=fx(CMM["auc_gain_mc_ci"][1], 3), n=CMM.get("n_cases", "--"))
        if "method_disagreement" in (R.get("extras") or ()) and "method_disagreement" not in cal_used["features"]:
            self.p(r"""\textbf{Desacuerdo entre métodos.} Se ha medido también si cada línea coincide en el NLFFF por optimización y en las soluciones Grad--Rubin. En la calibración anterior con Low \& Lou este indicador predice el error por sí solo (AUC $=0.68$), pero \emph{no mejora} la combinación de los demás (AUC $0.919\to0.915$; con Monte Carlo $0.942\to0.939$): los dos métodos fallan en las mismas zonas mal determinadas, que los otros indicadores ya detectan. Por eso no se incluye en la confianza. Con datos reales, en los que la fotosfera no es libre de fuerzas, podría ser más informativo.""",
                   r"""\textbf{Method disagreement.} Whether each line agrees between the optimisation NLFFF and the Grad--Rubin solutions was also measured. In the earlier Low \& Lou calibration this indicator predicts the error on its own (AUC $=0.68$), but it does \emph{not improve} the combination of the others (AUC $0.919\to0.915$; with Monte Carlo $0.942\to0.939$): both methods fail in the same poorly constrained regions, which the other indicators already detect. It is therefore not included in the confidence. With real data, where the photosphere is not force-free, it might be more informative.""")
        self.calibration_cv()
        self.box("warn",
                 r"""Estos indicadores son condiciones \emph{necesarias}, no suficientes: una línea puede estar libre de fuerzas y lejos de los bordes y aun así ser incorrecta si los datos de sus pies son erróneos de forma consistente. Además, la calibración se ha hecho con campos analíticos libres de fuerzas; en datos reales la fotosfera no es libre de fuerzas y el ruido depende de la intensidad del campo, por lo que indicadores como el campo en el pie o el desacuerdo entre métodos podrían ganar importancia.<<aia>>""",
                 r"""These indicators are \emph{necessary}, not sufficient conditions: a line can be force-free and far from the boundaries and still be wrong if the data at its footpoints are consistently wrong. Moreover, the calibration used analytic force-free fields; in real data the photosphere is not force-free and the noise depends on field strength, so indicators such as the footpoint field or the disagreement between methods could gain importance.<<aia>>""",
                 aia=self.L(r" La comparación con los bucles observados por AIA (sección~\ref{sec:aia}) es una primera prueba con datos reales.",
                            r" The comparison with the loops observed by AIA (Section~\ref{sec:aia}) is a first test with real data.")
                 if self.A.get("aia") else "")
        self.w(r"\subsection{" + self.L("Resultado para esta región", "Result for this region") + "}\n")
        self.p(r"""De las <<n>> líneas que salen de la base (una por cada <<st>> píxel/es con campo apreciable), el <<fr>> tiene confianza suficiente (mediana de la confianza: <<cm>>).""",
               r"""Of the <<n>> field lines rooted at the base (one every <<st>> pixel(s) with appreciable field), <<fr>> have sufficient confidence (median confidence: <<cm>>).""",
               n=R["n_lines"], st=self.A["physics"]["step"], fr=pct(R["frac_reliable"], 0), cm=num(R["conf_median"], 2))
        self.figure("confidence",
                    r"Izquierda: confianza de la línea que sale de cada punto de la base. Derecha: distribución de la confianza.",
                    r"Left: confidence of the field line rooted at each base point. Right: distribution of the confidence.")
        self.box("read",
                 r"""Azul oscuro = líneas fiables; azul claro = poco fiables. El contorno naranja separa ambas zonas según el umbral calibrado. Lo habitual es que los núcleos de las manchas y los bucles bajos sobre la PIL sean fiables, y que las líneas que nacen cerca de los bordes o suben mucho (y por tanto dependen del techo) no lo sean. Si la cuerda de flujo o la zona que te interesa cae en una región poco fiable, sus resultados deben tomarse con cautela.""",
                 r"""Dark blue = reliable lines; light blue = unreliable. The orange contour separates both regions at the calibrated threshold. Typically sunspot cores and low loops over the PIL are reliable, while lines rooted near the edges or rising high (and thus depending on the top boundary) are not. If the flux rope or your region of interest falls in an unreliable area, its results should be treated with caution.""")
        self.figure("reliable_lines",
                    r"Líneas del modelo: fiables en color (según su altura) y no fiables en gris.",
                    r"Model field lines: reliable ones in colour (by height) and unreliable ones in grey.")
        E = R.get("energy")
        if E:
            self.p(r"""\textbf{Energía en el volumen bien determinado.} Las celdas recorridas por líneas fiables ocupan el <<vf>> del volumen. En ellas el modelo tiene $E/E_{\rm pot}=$ <<r>> y una energía de las corrientes $|\mathbf{B}-\mathbf{B}_p|^2/8\pi$ (potencial de volumen) de <<fr>>~erg, el <<pc>> del total. Se usa esta densidad porque es la única bien definida localmente: $B^2-B_p^2$ sólo da la energía libre integrada en todo el volumen. El reparto depende de cuántas líneas se trazan, así que es orientativo.""",
                   r"""\textbf{Energy in the well-determined volume.} The cells crossed by reliable lines fill <<vf>> of the volume. There the model has $E/E_{\rm pot}=$ <<r>> and a current-carrying energy $|\mathbf{B}-\mathbf{B}_p|^2/8\pi$ (volume potential) of <<fr>>~erg, <<pc>> of the total. This density is used because it is the only one defined locally: $B^2-B_p^2$ gives the free energy only when integrated over the whole volume. The split depends on how many lines are traced, so it is indicative.""",
                   vf=pct(R["volume_frac"], 0), r=num(E["ratio"], 4), fr=num(E["free"]),
                   pc=pct(E["free_frac_of_total"], 0) if E.get("free_frac_of_total") is not None else "--")
        ta, tr = R["twist_all"], R["twist_reliable"]
        self.w(r"\begin{center}\begin{tabular}{lrr}\toprule" "\n" + self.L(r"(líneas cerradas) & Todas & Fiables", r"(closed lines) & All & Reliable")
               + r"\\\midrule" "\n")
        self.w(self.L("Número de líneas", "Number of lines") + f" & {ta['n']} & {tr['n']}\\\\\n")
        self.w(self.L(r"Percentil 99 de $|T_w|$", r"99th percentile of $|T_w|$") + f" & {num(ta['p99_abs'])} & {num(tr['p99_abs'])}\\\\\n")
        self.w(self.L(r"Fracción con $|T_w|\geq1$", r"Fraction with $|T_w|\geq1$") + f" & {pct(ta['frac_gt1'])} & {pct(tr['frac_gt1'])}\\\\\n")
        self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")

    def calibration_cv(self):
        """Subsection: calibration with diverse data and cross-validation (if there are results)."""
        from ..calibration import CLASSIFIER_NAMES, load_results
        S = load_results()
        if not S:
            return
        cvf = S["cv"]["field"]["models"]
        chosen = S.get("chosen", "logistic")
        nm = lambda m: CLASSIFIER_NAMES[m][0 if self.lang == "es" else 1]
        self.w(r"\subsection{" + self.L("Validación con datos diversos", "Validation with diverse data") + "}\n")
        self.p(r"""Para comprobar que la confianza no está ajustada a un único caso, se ha generado una batería de <<nc>> casos con verdad conocida: <<nf>> familias de campos libres de fuerzas exactos (tres variantes de Low \& Lou \citep{LowLou1990}, dos campos lineales exactos sobre regiones bipolar y cuadrupolar \citep{Alissandrakis1981} y, como sexta familia, la cuerda de flujo de Titov--Démoulin relajada a un equilibrio; ver la prueba externa más abajo), <<nn>> modelos de ruido (uniforme de 10 y 30~G, y uno tipo HMI con el campo transversal seis veces más ruidoso que el longitudinal) y <<ns>> resoluciones (<<sz>> puntos por lado), con <<nr>> realizaciones de ruido cada uno: <<nl>> líneas en total. Los clasificadores se evalúan con \textbf{validación cruzada dejando fuera un tipo de campo entero}: se entrenan con todas las familias menos una y se evalúan en la que falta, que nunca han visto. Así se mide lo que importa con datos reales, donde la geometría del campo es siempre nueva.""",
               r"""To check that the confidence is not tuned to a single case, a suite of <<nc>> cases with known truth was generated: <<nf>> families of exact force-free fields (three Low \& Lou variants \citep{LowLou1990}, two exact linear fields over bipolar and quadrupolar regions \citep{Alissandrakis1981} and, as a sixth family, the Titov--Démoulin flux rope relaxed to an equilibrium; see the external test below), <<nn>> noise models (uniform 10 and 30~G, and an HMI-like one with the transverse field six times noisier than the longitudinal one) and <<ns>> resolutions (<<sz>> points per side), with <<nr>> noise realisations each: <<nl>> lines in total. Classifiers are evaluated with \textbf{cross-validation leaving out a whole field type}: they are trained on all families but one and evaluated on the missing one, which they have never seen. This measures what matters with real data, where the field geometry is always new.""",
               nc=S["n_cases"], nf=len(S["fields"]), nn=len(S["noise_models"]), ns=len(S["sizes"]),
               sz=", ".join(map(str, S["sizes"])), nr=S["realizations"], nl=S["n_lines"])
        self.p(r"""Se comparan una regresión logística \citep{Cox1958}, un modelo aditivo generalizado con splines \citep[GAM;][]{HastieTibshirani1986}, gradient boosting con restricciones de monotonía \citep{Friedman2001}, un random forest \citep{Breiman2001} y versiones recalibradas con regresión isotónica \citep{ZadroznyElkan2002,NiculescuMizil2005}, implementados con scikit-learn \citep{Pedregosa2011}. Se miden tres cosas: el \textbf{AUC} \citep{Hanley1982}, que dice si el clasificador \emph{ordena} bien las líneas (las malas, con menos confianza); el \textbf{Brier} \citep{Brier1950}, error cuadrático de la probabilidad; y el \textbf{ECE} \citep{Guo2017}, que dice si la probabilidad es \emph{literal} (de las líneas con confianza 0.8, ¿son buenas el 80\%?). Los intervalos al 95\% se obtienen por bootstrap \citep{Efron1979} remuestreando casos completos, porque las líneas de un mismo caso no son independientes.""",
               r"""A logistic regression \citep{Cox1958}, a generalised additive model with splines \citep[GAM;][]{HastieTibshirani1986}, gradient boosting with monotonicity constraints \citep{Friedman2001}, a random forest \citep{Breiman2001} and versions recalibrated with isotonic regression \citep{ZadroznyElkan2002,NiculescuMizil2005} are compared, implemented with scikit-learn \citep{Pedregosa2011}. Three things are measured: the \textbf{AUC} \citep{Hanley1982}, i.e.\ whether the classifier \emph{ranks} lines well (bad ones with lower confidence); the \textbf{Brier} score \citep{Brier1950}, the squared error of the probability; and the \textbf{ECE} \citep{Guo2017}, i.e.\ whether the probability is \emph{literal} (of the lines with confidence 0.8, are 80\% good?). The 95\% intervals are obtained by bootstrap \citep{Efron1979} resampling whole cases, because lines from the same case are not independent.""")
        self.w(r"\begin{center}\small\begin{tabular}{lccccc}\toprule" "\n")
        self.w(self.L(r"Clasificador & AUC global & AUC por campo & Brier & ECE \\",
                      r"Classifier & Pooled AUC & AUC per field & Brier & ECE \\") + r" \midrule" "\n")
        ci = lambda r, k: f"{fx(r[k], 3)} [{fx(r[k + '_ci'][0], 3)}, {fx(r[k + '_ci'][1], 3)}]"
        for m, r in cvf.items():
            name = nm(m) + (r" $\star$" if m == chosen else "")
            self.w(f"{name} & {ci(r, 'auc')} & {fx(r['auc_groups_mean'], 2)} $\\pm$ {fx(r['auc_groups_std'], 2)}"
                   f" & {fx(r['brier'], 3)} & {fx(r['ece'], 3)}\\\\\n")
        base = S["cv"]["field"]["base_rate"]
        self.w(r"\midrule " + self.L("Sin información (tasa base)", "No information (base rate)")
               + f" & 0.5 & 0.5 & {fx(base * (1 - base), 3)} & --\\\\\n")
        self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
        self.p(r"""$\star$: clasificador elegido. \emph{AUC global}: todas las líneas juntas, con intervalo al 95\%. \emph{AUC por campo}: media y dispersión del AUC dentro de cada familia excluida; mide si el orden de las líneas se transfiere a una geometría nueva aunque cambie la proporción de líneas malas.""",
               r"""$\star$: chosen classifier. \emph{Pooled AUC}: all lines together, with 95\% interval. \emph{AUC per field}: mean and spread of the AUC within each left-out family; it measures whether the ranking of lines transfers to a new geometry even if the proportion of bad lines changes.""")
        self.figure("calibration_cv",
                    r"Validación cruzada dejando fuera cada familia de campos. (a) Diagrama de fiabilidad: probabilidad predicha frente a frecuencia observada (la diagonal es la calibración perfecta). (b) AUC en cada familia excluida. (c) Fracción de líneas malas frente a la resolución de la malla.",
                    r"Cross-validation leaving out each field family. (a) Reliability diagram: predicted probability versus observed frequency (the diagonal is perfect calibration). (b) AUC on each left-out family. (c) Fraction of bad lines versus grid resolution.")
        self.cv_findings(S)
        self.robustness()
        self.flux_rope_test()

    def flux_rope_test(self):
        """External test with the Titov–Démoulin flux rope (calibrations that have not seen it)."""
        from ..calibration import load_flux_rope_test
        T, Tr = load_flux_rope_test(), load_flux_rope_test(relaxed=True)
        if not T:
            return
        a, ref, sl = T["all"], T["reference"], T.get("short_lines") or {}
        cw = list(T["exact_cwsin"].values())
        cwr = list(Tr["exact_cwsin"].values()) if Tr else None
        self.w(r"\subsection{" + self.L("Prueba externa: una cuerda de flujo",
                                        "External test: a flux rope") + "}\n")
        self.p(r"""Las cinco primeras familias de la batería no contienen ninguna \emph{cuerda de flujo}, que es justamente la estructura que más interesa en regiones eruptivas. Por eso se aplicaron calibraciones ajustadas \textbf{sin ella} al modelo de Titov \& Démoulin \citep{TitovDemoulin1999}: un toro de corriente torsionado y sujeto por el campo de dos polaridades, con los parámetros de \citet{WiegelmannTD2006}, con los mismos tres modelos de ruido, mallas de <<sz>> puntos por lado y dos realizaciones (<<nc>> casos, <<nl>> líneas). El modelo analítico sólo es libre de fuerzas en primer orden ($\mathrm{CW}_{\sin}=$ <<cw0>>--<<cw1>>, frente a 0.017 de Low \& Lou), así que también se usó una versión \emph{relajada} a un equilibrio con las seis caras fijas, como hacen \citet{Valori2010} (<<cwr>>): la cuerda sobrevive (torsión y altura casi iguales) y el campo cambia muy poco.""",
               r"""The first five families of the suite contain no \emph{flux rope}, which is precisely the structure of most interest in eruptive regions. Calibrations fitted \textbf{without it} were therefore applied to the Titov \& Démoulin model \citep{TitovDemoulin1999}: a twisted current torus held down by the field of two polarities, with the parameters of \citet{WiegelmannTD2006}, the same three noise models, grids of <<sz>> points per side and two realisations (<<nc>> cases, <<nl>> lines). The analytic model is force-free only to first order ($\mathrm{CW}_{\sin}=$ <<cw0>>--<<cw1>>, against 0.017 for Low \& Lou), so a version \emph{relaxed} to an equilibrium with all six faces fixed, as in \citet{Valori2010}, was also used (<<cwr>>): the rope survives (almost the same twist and height) and the field changes very little.""",
               sz=", ".join(map(str, T["sizes"])), nc=len(T["sizes"]) * len(T["noise_models"]) * T["realizations"],
               nl=a["n_lines"], cw0=fx(min(cw), 3), cw1=fx(max(cw), 3),
               cwr=(r"$\mathrm{CW}_{\sin}=$ " + fx(min(cwr), 3) + "--" + fx(max(cwr), 3)) if cwr else "--")
        self.w(r"\begin{center}\resizebox{\textwidth}{!}{\begin{tabular}{llccccc}\toprule" "\n")
        self.w(self.L(r"Campo & Calibración & Líneas malas & AUC & ECE & Fiables & Malas entre fiables\\",
                      r"Field & Calibration & Bad lines & AUC & ECE & Reliable & Bad among reliable\\") + r"\midrule" "\n")
        lin, quad = self.L("lineal", "linear"), self.L(r"con $\log^2 L$", r"with $\log^2 L$")

        def row(field, cal, v, ci=True):
            auc_ = fx(v["auc"], 3) + (f" [{fx(v['auc_ci'][0], 2)}, {fx(v['auc_ci'][1], 2)}]" if ci and v.get("auc_ci") else "")
            self.w(f"{field} & {cal} & {pct(v['bad_rate'], 0)} & {auc_} & {fx(v['ece'], 3)} & "
                   f"{pct(v['accepted'], 0)} & {pct(v['bad_accepted'], 1)}\\\\\n")
        row(self.L("Batería (familia no vista)", "Suite (unseen family)"), lin, ref, ci=False)
        row(self.L("Cuerda TD analítica", "Analytic TD rope"), lin, a)
        if Tr:
            row(self.L("Cuerda TD relajada", "Relaxed TD rope"), lin, Tr["all"])
            if Tr.get("quadratic"):
                row(self.L("Batería (familia no vista)", "Suite (unseen family)"), quad, Tr["quadratic"]["reference"], ci=False)
                row(self.L("Cuerda TD relajada", "Relaxed TD rope"), quad, Tr["quadratic"]["all"])
        self.w(r"\bottomrule\end{tabular}}\end{center}" "\n\n")
        self.p(r"""Ninguna de las calibraciones de la tabla ha visto la cuerda (todas se ajustaron con las cinco primeras familias); \emph{lineal}: la calibración anterior; \emph{con $\log^2 L$}: la misma regresión con el término cuadrático en el logaritmo de la longitud.""",
               r"""None of the calibrations in the table has seen the rope (all were fitted on the first five families); \emph{linear}: the earlier calibration; \emph{with $\log^2 L$}: the same regression with the quadratic term in the logarithm of the length.""")
        bn = [v["bad_rate"] for v in T["by_noise"].values()] + [v["bad_rate"] for v in T["by_size"].values()]
        self.p(r"""(1) \textbf{El propio NLFFF reproduce peor la cuerda}: el <<br>> de las líneas son malas, frente al <<r0>>--<<r1>> de las otras familias, con cualquier ruido o resolución (<<bn>>). La cifra es la misma con la cuerda relajada, así que el fallo no se debe a que el modelo analítico sea sólo aproximado: es una limitación del método al reconstruir una cuerda a partir de la base. (2) \textbf{La calibración lineal ordena las líneas} (AUC <<auc>>) \textbf{pero su probabilidad deja de ser literal} (ECE <<ece>>): el <<ba>> de las líneas que declara fiables son malas.""",
               r"""(1) \textbf{The NLFFF itself reproduces the rope less well}: <<br>> of the lines are bad, against <<r0>>--<<r1>> for the other families, with any noise or resolution (<<bn>>). The figure is the same for the relaxed rope, so the failure is not due to the analytic model being approximate: it is a limitation of the method when reconstructing a rope from the base. (2) \textbf{The linear calibration ranks the lines} (AUC <<auc>>) \textbf{but its probability is no longer literal} (ECE <<ece>>): <<ba>> of the lines it declares reliable are bad.""",
               br=pct(a["bad_rate"], 0), r0=pct(min(ref.get("bad_rate_by_field", {"x": ref["bad_rate"]}).values()), 0),
               r1=pct(max(ref.get("bad_rate_by_field", {"x": ref["bad_rate"]}).values()), 0),
               bn=(pct(min(bn), 0) + "--" + pct(max(bn), 0) if pct(max(bn), 0) != pct(min(bn), 0)
                   else pct(min(bn), 0) + self.L(" en todos los casos", " in every case")),
               auc=fx(a["auc"], 2), ece=fx(a["ece"], 2), ba=pct(a["bad_accepted"], 0))
        if sl:
            self.p(r"""(3) \textbf{Diagnóstico.} El <<fr>> de esas líneas fiables pero malas pertenecen al cuartil de líneas \emph{más cortas}. En las otras familias las líneas cortas eran las más seguras (sólo el <<tr>> son malas), y una regresión \emph{lineal} sólo puede aprender que cuanto más corta, más fiable; en la cuerda, en cambio, el <<sr>> de las líneas cortas son malas. No es un artefacto del umbral: el <<tp>> de ellas cambia de conectividad y su desviación mediana es el <<sh>> de su longitud.""",
                   r"""(3) \textbf{Diagnosis.} <<fr>> of those reliable-but-bad lines belong to the quartile of \emph{shortest} lines. In the other families short lines were the safest (only <<tr>> are bad), and a \emph{linear} regression can only learn that the shorter, the more reliable; in the rope, instead, <<sr>> of the short lines are bad. This is not a threshold artefact: <<tp>> of them change connectivity and their median deviation is <<sh>> of their length.""",
                   fr=pct(sl["frac_of_reliable_bad"], 0), tr=pct(sl["bad_rate_short_train"], 0) if sl.get("bad_rate_short_train") is not None else "--",
                   sr=pct(sl["bad_rate_short"], 0), tp=pct(sl["topology_frac"], 0), sh=pct(sl["shape_median"], 0))
        if Tr and Tr.get("quadratic"):
            q = Tr["quadratic"]["all"]
            self.p(r"""(4) \textbf{Solución.} Basta con añadir un término cuadrático en el logaritmo de la longitud, que permite que tanto las líneas muy largas como las muy cortas sean sospechosas. Ajustada \emph{sólo con las cinco primeras familias}, esta versión no empeora en ellas y en la cuerda, que nunca ha visto, ordena mejor (AUC <<qa>>) y sus líneas fiables vuelven a ser correctas: sólo el <<qb>> son malas, frente al <<ba>> de la lineal. Su probabilidad media sigue siendo algo optimista dentro de la cuerda (predice un <<qp>> de líneas malas cuando hay un <<br>>; ECE <<qe>>), pero el conjunto de líneas fiables está limpio. La calibración por defecto usa este término y, además, incluye la cuerda relajada como sexta familia de la batería.""",
                   r"""(4) \textbf{Solution.} It is enough to add a quadratic term in the logarithm of the length, which allows both very long and very short lines to be suspicious. Fitted \emph{only on the first five families}, this version does not get worse on them, and on the rope, which it has never seen, it ranks better (AUC <<qa>>) and its reliable lines are correct again: only <<qb>> are bad, against <<ba>> for the linear one. Its mean probability is still somewhat optimistic inside the rope (it predicts <<qp>> bad lines when there are <<br>>; ECE <<qe>>), but the set of reliable lines is clean. The default calibration uses this term and also includes the relaxed rope as the sixth family of the suite.""",
                   qa=fx(q["auc"], 2), qb=pct(q["bad_accepted"], 1), ba=pct(a["bad_accepted"], 0),
                   qp=pct(q["mean_predicted_bad"], 0), br=pct(Tr["all"]["bad_rate"], 0), qe=fx(q["ece"], 2))
            self.box("key",
                     r"""Una calibración ajustada sin cuerdas de flujo falló en una cuerda, y el fallo tenía una causa concreta: no podía representar que las líneas muy cortas también sean sospechosas. Con ese término la confianza se transfiere a la cuerda nunca vista (sus líneas fiables son correctas), aunque el NLFFF reproduzca mal buena parte de la cuerda. La lección general: hay que probar la confianza en geometrías realmente distintas de las de entrenamiento.""",
                     r"""A calibration fitted without flux ropes failed on a rope, and the failure had a specific cause: it could not represent that very short lines are also suspicious. With that term the confidence transfers to the unseen rope (its reliable lines are correct), even though the NLFFF reproduces much of the rope poorly. The general lesson: the confidence must be tested on geometries truly different from the training ones.""")
        else:
            self.box("warn",
                     r"""La confianza por línea \textbf{no es universal}: en una cuerda de flujo subestima los errores de las líneas cortas y bajas.""",
                     r"""The per-line confidence is \textbf{not universal}: in a flux rope it underestimates the errors of short, low-lying lines.""")

    def robustness(self):
        """Robustness against the "wrong line" threshold and the NLFFF/preprocessing parameters."""
        from ..calibration import load_robustness
        R = load_robustness()
        if not R:
            return
        self.w(r"\subsection{" + self.L("Robustez frente a las decisiones arbitrarias",
                                        "Robustness against arbitrary choices") + "}\n")
        T = R.get("bad_threshold")
        if T:
            self.p(r"""\textbf{Definición de «línea mala».} El umbral del 10\% de la longitud es una convención. La tabla repite la evaluación con el 5\% y el 20\%: con la calibración por defecto (ajustada con el 10\%) y reajustando la regresión con cada definición (validación cruzada dejando fuera cada familia).""",
                   r"""\textbf{Definition of a ``bad line''.} The 10\% length threshold is a convention. The table repeats the evaluation with 5\% and 20\%: with the default calibration (fitted with 10\%) and refitting the regression for each definition (cross-validation leaving out each family).""")
            self.w(r"\begin{center}\small\begin{tabular}{lccccc}\toprule" "\n")
            self.w(self.L(r"Umbral & Líneas malas & AUC (por defecto) & ECE (por defecto) & AUC (reajustada) & ECE (reajustada)\\",
                          r"Threshold & Bad lines & AUC (default) & ECE (default) & AUC (refitted) & ECE (refitted)\\")
                   + r"\midrule" "\n")
            for t, v in T.items():
                self.w(f"{pct(float(t), 0)} & {pct(v['bad_rate'], 0)} & {fx(v['default_auc'], 3)} & {fx(v['default_ece'], 3)}"
                       f" & {fx(v['refit_auc'], 3)} & {fx(v['refit_ece'], 3)}\\\\\n")
            self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
            aucs = [v["default_auc"] for v in T.values()]
            self.p(r"""El \emph{orden} de las líneas no depende de la definición (AUC <<a0>>--<<a1>> con la calibración por defecto). Lo que cambia es el significado literal de la probabilidad: está calibrada para el 10\%, y con otra definición hay que reajustarla (con ello el ECE vuelve a ser pequeño).""",
                   r"""The \emph{ranking} of the lines does not depend on the definition (AUC <<a0>>--<<a1>> with the default calibration). What changes is the literal meaning of the probability: it is calibrated for 10\%, and with another definition it must be refitted (which brings the ECE back down).""",
                   a0=fx(min(aucs), 2), a1=fx(max(aucs), 2))
        P = R.get("parameters")
        if P:
            names = {"relax_bottom": r"$\nu$", "smooth": r"$\lambda$", "eps": r"$\varepsilon$",
                     "nd": self.L("capa amort.", "buffer"), "mu3": r"$\mu_3$", "mu4": r"$\mu_4$",
                     "size": self.L("malla", "grid")}

            def lab(v):
                if v == "default":
                    return self.L("por defecto", "default")
                k, x = v.split("=")
                if k == "size":
                    return f"{names[k]} {x}$^2$"
                return f"{names.get(k, k)} $=$ {x}"
            self.p(r"""\textbf{Parámetros del método.} Se ha repetido el modelo cambiando, de uno en uno, los parámetros del NLFFF (relajación de la base $\nu$, suavizado $\lambda$, regularización $\varepsilon$, anchura de la capa de amortiguamiento) y del preprocesado ($\mu_3$, $\mu_4$) a valores la mitad o el doble de grandes (o más), en <<nc>> casos con verdad conocida (<<cs>>; <<sz>>$^2$ puntos). Se usa siempre la calibración por defecto.""",
                   r"""\textbf{Method parameters.} The model was recomputed changing, one at a time, the NLFFF parameters (base relaxation $\nu$, smoothing $\lambda$, regularisation $\varepsilon$, buffer-layer width) and the preprocessing ones ($\mu_3$, $\mu_4$) to values half or twice as large (or more), on <<nc>> cases with known truth (<<cs>>; <<sz>>$^2$ points). The default calibration is always used.""",
                   nc=len(R["cases"]), cs=", ".join(R["cases"]), sz=R["size"])
            ext = "efree_vol_abs" in P["default"]
            self.w(r"\begin{center}\resizebox{\textwidth}{!}{\begin{tabular}{lcccc" + ("ccc" if ext else "c") + r"}\toprule" "\n")
            hdr_e = (r" & $|\Delta E_{\rm free}|$ FFT & $|\Delta E_{\rm free}|$ vol. & $|\Delta H_V|$"
                     if ext else r" & $|\Delta E_{\rm free}|$")
            self.w(self.L(r"Variante & AUC & ECE & Fiables & Malas entre fiables",
                          r"Variant & AUC & ECE & Reliable & Bad among reliable") + hdr_e + r"\\\midrule" "\n")
            for v, d in P.items():
                tail = f" & {pct(d['efree_abs'], 0)}"
                if ext:
                    tail += f" & {pct(d['efree_vol_abs'], 0)} & {pct(d['hv_abs'], 0)}"
                self.w(f"{lab(v)} & {fx(d['auc'], 3)} & {fx(d['ece'], 3)} & {pct(d['reliable'], 0)} & "
                       f"{pct(d['bad_reliable'], 0)}" + tail + "\\\\\n")
            self.w(r"\bottomrule\end{tabular}}\end{center}" "\n\n")
            d0 = P["default"]

            def grp(prefixes):
                return {k: v for k, v in P.items() if k.split("=")[0] in prefixes}

            def rng(g, key, f):
                vals = [v[key] for v in g.values()] + [d0[key]]
                return f(min(vals)), f(max(vals))
            nl, bu, pp = grp(("relax_bottom", "smooth", "eps")), grp(("nd",)), grp(("mu3", "mu4"))
            worst_pp = max(pp, key=lambda k: pp[k]["bad_reliable"]) if pp else None
            p2 = lambda v: fx(v, 2)
            p0 = lambda v: pct(v, 0)
            self.p(r"""$|\Delta E_{\rm free}|$: error relativo medio de la energía libre (potencial FFT) frente a la del campo exacto; ECE calculado caso a caso (dentro de un único campo, ver la sección anterior). Los parámetros se comportan de forma muy distinta. (1) \textbf{Parámetros del NLFFF} ($\nu$, $\lambda$, $\varepsilon$): la confianza apenas cambia (AUC <<na0>>--<<na1>>; líneas malas entre las fiables <<nb0>>--<<nb1>>), y la energía libre varía entre <<ne0>> y <<ne1>> de error. (2) \textbf{Capa de amortiguamiento}: cambia cuántas líneas se declaran fiables (<<br0>>--<<br1>>) porque el indicador de paso por la capa depende de su anchura; con una capa estrecha se cuelan más líneas malas (hasta <<bb1>>). (3) \textbf{Preprocesado} ($\mu_3$, $\mu_4$): es la decisión \textbf{más crítica}. Lejos de los valores por defecto el propio modelo empeora (más líneas malas y energía libre con errores de hasta <<pe1>>), y la confianza pierde capacidad (AUC hasta <<pa0>>; con <<wp>> el <<wpb>> de las líneas «fiables» son malas). La confianza mide la calidad del modelo \emph{dado} el preprocesado: no detecta un preprocesado inadecuado, que debe comprobarse aparte (sección de preprocesado y criterios de Aly).""",
                   r"""$|\Delta E_{\rm free}|$: mean relative error of the free energy (FFT potential) against that of the exact field; ECE computed case by case (within a single field, see the previous section). The parameters behave very differently. (1) \textbf{NLFFF parameters} ($\nu$, $\lambda$, $\varepsilon$): the confidence barely changes (AUC <<na0>>--<<na1>>; bad lines among the reliable ones <<nb0>>--<<nb1>>), and the free-energy error ranges between <<ne0>> and <<ne1>>. (2) \textbf{Buffer layer}: it changes how many lines are declared reliable (<<br0>>--<<br1>>) because the buffer indicator depends on its width; with a narrow layer more bad lines get through (up to <<bb1>>). (3) \textbf{Preprocessing} ($\mu_3$, $\mu_4$): it is the \textbf{most critical} choice. Far from the default values the model itself gets worse (more bad lines and free-energy errors of up to <<pe1>>), and the confidence loses power (AUC down to <<pa0>>; with <<wp>>, <<wpb>> of the ``reliable'' lines are bad). The confidence measures the quality of the model \emph{given} the preprocessing: it does not detect an inadequate preprocessing, which must be checked separately (preprocessing section and Aly criteria).""",
                   na0=rng(nl, "auc", p2)[0], na1=rng(nl, "auc", p2)[1],
                   nb0=rng(nl, "bad_reliable", p0)[0], nb1=rng(nl, "bad_reliable", p0)[1],
                   ne0=rng(nl, "efree_abs", p0)[0], ne1=rng(nl, "efree_abs", p0)[1],
                   br0=rng(bu, "reliable", p0)[0], br1=rng(bu, "reliable", p0)[1],
                   bb1=rng(bu, "bad_reliable", p0)[1],
                   pe1=rng(pp, "efree_abs", p0)[1], pa0=rng(pp, "auc", p2)[0],
                   wp=lab(worst_pp) if worst_pp else "--",
                   wpb=p0(pp[worst_pp]["bad_reliable"]) if worst_pp else "--")
            rs = grp(("size",))
            if rs and "efree_vol_abs" in d0:
                self.p(r"""(4) \textbf{Resolución} (el mismo campo con <<sz>> puntos por lado): la confianza conserva su capacidad (AUC <<a0>>--<<a1>>; malas entre las fiables <<b0>>--<<b1>>), y el error de la energía libre respecto al potencial de volumen queda entre <<e0>> y <<e1>> (<<f0>>--<<f1>> con el potencial FFT). La resolución, por tanto, influye sobre todo en la energía calculada con el potencial de la base, en línea con lo encontrado con datos reales \citep{DeRosa2015}.""",
                       r"""(4) \textbf{Resolution} (the same field with <<sz>> points per side): the confidence keeps its power (AUC <<a0>>--<<a1>>; bad among reliable <<b0>>--<<b1>>), and the error of the free energy relative to the volume potential stays between <<e0>> and <<e1>> (<<f0>>--<<f1>> with the FFT potential). Resolution therefore mainly affects the energy computed with the base potential, in line with what has been found with real data \citep{DeRosa2015}.""",
                       sz=", ".join(sorted([k.split("=")[1] for k in rs] + [str(R["size"])], key=int)),
                       a0=rng(rs, "auc", p2)[0], a1=rng(rs, "auc", p2)[1],
                       b0=rng(rs, "bad_reliable", p0)[0], b1=rng(rs, "bad_reliable", p0)[1],
                       e0=rng(rs, "efree_vol_abs", p0)[0], e1=rng(rs, "efree_vol_abs", p0)[1],
                       f0=rng(rs, "efree_abs", p0)[0], f1=rng(rs, "efree_abs", p0)[1])
            if "efree_vol_abs" in d0:
                self.p(r"""\textbf{Energía libre de volumen y helicidad.} En \emph{todas} las variantes, incluidas las de preprocesado extremo, el error medio de la energía libre respecto al potencial de volumen queda entre <<v0>> y <<v1>>, y el de $H_V$ entre <<h0>> y <<h1>>, mientras que el de la energía libre con el potencial de la base va de <<f0>> a <<f1>>. La razón es que el potencial de la base se calcula con el magnetograma observado, de modo que cualquier cambio del flujo en la base del modelo (por el preprocesado, la relajación de la base o la discretización) aparece como energía libre ficticia; el potencial de volumen, en cambio, tiene las mismas componentes normales que el propio modelo en todas las caras \citep{Valori2013}, y su error es sobre todo el del método.""",
                       r"""\textbf{Volume free energy and helicity.} In \emph{all} variants, including those with extreme preprocessing, the mean error of the free energy relative to the volume potential stays between <<v0>> and <<v1>>, and that of $H_V$ between <<h0>> and <<h1>>, whereas that of the free energy with the base potential ranges from <<f0>> to <<f1>>. The reason is that the base potential is computed from the observed magnetogram, so any change of the flux at the base of the model (by the preprocessing, the base relaxation or the discretisation) appears as spurious free energy; the volume potential, instead, has the same normal components as the model itself on all faces \citep{Valori2013}, and its error is mainly that of the method.""",
                       v0=rng(P, "efree_vol_abs", p0)[0], v1=rng(P, "efree_vol_abs", p0)[1],
                       h0=rng(P, "hv_abs", p0)[0], h1=rng(P, "hv_abs", p0)[1],
                       f0=rng(P, "efree_abs", p0)[0], f1=rng(P, "efree_abs", p0)[1])
            self.box("key",
                     r"""Los resultados no dependen de forma crítica de los parámetros del NLFFF, pero sí del preprocesado: en un artículo conviene justificar $\mu_3$ y $\mu_4$ (aquí $10^{-2}$, como en \citealt{Wiegelmann2006}; sus valores óptimos para HMI dependen de los datos, \citealt{Wiegelmann2012}) y mostrar que la base preprocesada sigue siendo compatible con los datos.""",
                     r"""The results do not depend critically on the NLFFF parameters, but they do depend on the preprocessing: in a paper $\mu_3$ and $\mu_4$ should be justified (here $10^{-2}$, as in \citealt{Wiegelmann2006}; their optimal values for HMI depend on the data, \citealt{Wiegelmann2012}) and the preprocessed base should be shown to remain compatible with the data.""")

    def cv_findings(self, S):
        """Conclusions of the cross-validation (with the numbers of the summary)."""
        cv = S["cv"]
        lf = cv["field"]["models"]["logistic"]
        per = lf["per_group"]
        aucs = [v["auc"] for v in per.values()]
        eces = [v["ece"] for v in per.values()]
        allm = cv["field"]["models"]
        lo = min(m["auc_ci"][0] for m in allm.values())
        hi = max(m["auc_ci"][1] for m in allm.values())
        br = S["bad_rate"]["field"]
        leg = S.get("legacy_lowlou") or {}
        bfs = S["bad_rate_field_size"]
        sizes = [str(n) for n in S["sizes"]]
        trend = {f: v[sizes[-1]] - v[sizes[0]] for f, v in bfs.items()}
        down = [f for f, d in trend.items() if d < -0.02]
        up = [f for f, d in trend.items() if d > 0.02]
        self.p(r"""\textbf{Resultados.} (1) \emph{Ningún clasificador es mejor que la regresión logística} en campos nunca vistos: todos los AUC caen en el intervalo <<lo>>--<<hi>> y sus intervalos al 95\% se solapan. Los modelos flexibles sólo ganan algo cuando se deja fuera un nivel de ruido o una resolución (es decir, cuando la geometría del campo ya se ha visto), lo que indica que aprenden detalles de cada campo que no se transfieren. Se elige la regresión logística: es la más simple, es interpretable (un coeficiente por indicador, todos con el signo físico esperado) y no necesita dependencias. (2) \emph{La confianza ordena bien las líneas en cualquier geometría}: el AUC dentro de cada familia excluida va de <<amin>> a <<amax>>, y apenas cambia al dejar fuera un modelo de ruido (<<an>>) o una resolución (<<as>>). (3) \emph{La probabilidad absoluta es menos transferible que el orden}: la fracción de líneas malas varía mucho entre familias (del <<bmin>> al <<bmax>>), y dentro de una sola familia nunca vista el ECE es <<emin>>--<<emax>>, frente a <<eall>> en conjunto. Por eso la confianza de una región concreta debe leerse como una probabilidad aproximada (con un margen de unos 0.1) y, sobre todo, como un \emph{orden} de qué líneas son más fiables. (4) \emph{Entrenar con datos diversos es necesario}: la calibración anterior, ajustada sólo con Low \& Lou, ordenaba casi igual de bien (AUC <<lauc>> sobre esta batería) pero su probabilidad estaba muy mal calibrada (ECE <<lece>>).""",
               r"""\textbf{Results.} (1) \emph{No classifier beats logistic regression} on unseen fields: all AUCs fall within <<lo>>--<<hi>> and their 95\% intervals overlap. Flexible models only gain somewhat when a noise level or a resolution is left out (i.e.\ when the field geometry has already been seen), which indicates that they learn details of each field that do not transfer. Logistic regression is chosen: it is the simplest, it is interpretable (one coefficient per indicator, all with the physically expected sign) and it needs no dependencies. (2) \emph{The confidence ranks lines well in any geometry}: the AUC within each left-out family ranges from <<amin>> to <<amax>>, and it barely changes when a noise model (<<an>>) or a resolution (<<as>>) is left out. (3) \emph{The absolute probability transfers less well than the ranking}: the fraction of bad lines varies widely between families (from <<bmin>> to <<bmax>>), and within a single unseen family the ECE is <<emin>>--<<emax>>, against <<eall>> overall. Hence the confidence of a specific region should be read as an approximate probability (within about 0.1) and, above all, as a \emph{ranking} of which lines are more reliable. (4) \emph{Training on diverse data is necessary}: the earlier calibration, fitted on Low \& Lou only, ranked almost as well (AUC <<lauc>> on this suite) but its probability was badly calibrated (ECE <<lece>>).""",
               lo=fx(lo, 2), hi=fx(hi, 2), amin=fx(min(aucs), 2), amax=fx(max(aucs), 2),
               an=fx(cv["noise_model"]["models"]["logistic"]["auc"], 2) if "noise_model" in cv else "--",
               **{"as": fx(cv["size"]["models"]["logistic"]["auc"], 2) if "size" in cv else "--"},
               bmin=pct(min(br.values()), 0), bmax=pct(max(br.values()), 0),
               emin=fx(min(eces), 2), emax=fx(max(eces), 2), eall=fx(lf["ece"], 2),
               lauc=fx(leg.get("auc"), 2), lece=fx(leg.get("ece"), 2))
        lin = S.get("linear_cv")
        if lin:
            self.p(r"""(5) \emph{El término cuadrático de la longitud es necesario}: la misma regresión sin él, en los mismos casos, da AUC <<l>> [<<l0>>, <<l1>>], frente a <<q>> [<<q0>>, <<q1>>]. La diferencia viene sobre todo de la cuerda de flujo, donde las líneas muy cortas fallan a menudo (prueba externa, más abajo).""",
                   r"""(5) \emph{The quadratic length term is necessary}: the same regression without it, on the same cases, gives AUC <<l>> [<<l0>>, <<l1>>], against <<q>> [<<q0>>, <<q1>>]. The difference comes mainly from the flux rope, where very short lines often fail (external test, below).""",
                   l=fx(lin["auc"], 3), l0=fx(lin["auc_ci"][0], 3), l1=fx(lin["auc_ci"][1], 3),
                   q=fx(lf["auc"], 3), q0=fx(lf["auc_ci"][0], 3), q1=fx(lf["auc_ci"][1], 3))
        if down and up:
            self.p(r"""\textbf{Resolución.} Al pasar de <<n0>> a <<n1>> puntos por lado, la fracción de líneas malas baja en <<down>>, pero \emph{sube} en <<up>> (panel c). El error de una línea no es sólo de discretización: con más resolución el modelo recibe más detalle de la base, pero las líneas largas y altas siguen dependiendo de los contornos no observados. Es una razón más para calibrar con varias resoluciones y para comprobar la convergencia en resolución con datos reales, donde también se ha visto que los resultados NLFFF cambian con la resolución \citep{DeRosa2015}.""",
                   r"""\textbf{Resolution.} Going from <<n0>> to <<n1>> points per side, the fraction of bad lines decreases for <<down>>, but \emph{increases} for <<up>> (panel c). The error of a line is not only discretisation error: with more resolution the model receives more detail from the base, but long and high lines still depend on the unobserved boundaries. This is one more reason to calibrate with several resolutions and to check resolution convergence with real data, where NLFFF results have also been found to change with resolution \citep{DeRosa2015}.""",
                   n0=sizes[0], n1=sizes[-1], down=", ".join(down), up=", ".join(up))
        self.box("key",
                 r"""La confianza por línea se ha validado en <<nf>> geometrías distintas (incluida una cuerda de flujo), tres modelos de ruido y tres resoluciones, siempre evaluando en casos que el modelo no había visto. Ordena las líneas con AUC $\approx$ <<a>> y su probabilidad es fiable en promedio (ECE <<e>>); en una región concreta, úsala sobre todo para separar las partes fiables de las que no lo son.""",
                 r"""The per-line confidence has been validated on <<nf>> different geometries (including a flux rope), three noise models and three resolutions, always evaluating on cases the model had not seen. It ranks lines with AUC $\approx$ <<a>> and its probability is reliable on average (ECE <<e>>); in a specific region, use it mainly to separate the reliable parts from the unreliable ones.""",
                 a=fx(lf["auc"], 2), e=fx(lf["ece"], 2), nf=len(S["fields"]))

    # ------------------------------------------------- Monte Carlo
    def montecarlo(self):
        M = self.A.get("mc")
        if not M:
            return
        self.w(r"\section{" + self.L("Incertidumbres: Monte Carlo con los errores de medida",
                                     "Uncertainties: Monte Carlo with measurement errors") + r"}\label{sec:mc}" "\n")
        src = self.L("los mapas de error de HMI (segmentos \\texttt{Br\\_err}, \\texttt{Bp\\_err}, \\texttt{Bt\\_err})",
                     "the HMI error maps (\\texttt{Br\\_err}, \\texttt{Bp\\_err}, \\texttt{Bt\\_err} segments)") \
            if M["source"].startswith("HMI") else self.L("un ruido gaussiano uniforme", "a uniform Gaussian noise")
        self.p(r"""Todo lo anterior se ha calculado con \emph{un} magnetograma, pero ese magnetograma tiene errores de medida. Para saber cuánto cambian los resultados por ellos, se ha repetido la extrapolación completa (preprocesado y NLFFF) <<n>> veces, añadiendo cada vez al magnetograma un ruido aleatorio gaussiano con la amplitud indicada por <<src>><<scale>>. La dispersión de los resultados es su incertidumbre debida al ruido de los datos.""",
               r"""Everything above was computed from \emph{one} magnetogram, but that magnetogram has measurement errors. To find out how much the results change because of them, the full extrapolation (preprocessing and NLFFF) was repeated <<n>> times, each time adding to the magnetogram random Gaussian noise with the amplitude given by <<src>><<scale>>. The spread of the results is their uncertainty due to data noise.""",
               n=M["n"], src=src, scale=(self.L(f", multiplicada por {M['error_scale']:g}", f", multiplied by {M['error_scale']:g}")
                                         if M["error_scale"] != 1 else ""))
        s = M["summary"]
        rows = [("E_free", self.L("Energía libre (pot. FFT) [erg]", "Free energy (FFT pot.) [erg]")),
                ("E_free_vol", self.L("Energía libre (pot. de volumen) [erg]", "Free energy (volume pot.) [erg]")),
                ("E_ratio", r"$E/E_{\rm pot}$"),
                ("E_div_frac", r"$E_{\rm div}/E$"),
                ("H_V", r"$H_V$ [Mx$^2$]"), ("HJ_HV", r"$|H_J|/|H_V|$"),
                ("h_crit", r"$h_{\rm crit}$ [Mm]"), ("tw_p99", self.L(r"Percentil 99 de $|T_w|$", r"99th percentile of $|T_w|$")),
                ("CWsin", r"$\mathrm{CW}_{\sin}$")]
        self.w(r"\begin{center}\small\begin{tabular}{lrrrrr}\toprule" "\n")
        self.w(self.L(r"Magnitud & Nominal & Mediana MC & Intervalo 16--84\% & Incert. & Sesgo MC",
                      r"Quantity & Nominal & MC median & 16--84\% interval & Unc. & MC bias") + r"\\\midrule" "\n")
        for k, lab in rows:
            if k not in s:
                continue
            d = s[k]
            rel = (d["p84"] - d["p16"]) / 2 / abs(d["p50"]) if d["p50"] else float("nan")
            mark = r"$^\dagger$" if d.get("outside") else ""
            bias = (d["p50"] / d["nominal"] - 1) if d.get("nominal") else float("nan")
            bias_txt = (("$+$" if bias > 0 else "$-$" if bias < 0 else "") + pct(abs(bias), 0)
                        if np.isfinite(bias) else "--")
            self.w(f"{lab} & {num(d['nominal'])}{mark} & {num(d['p50'])} & [{num(d['p16'])}, {num(d['p84'])}] & {pct(rel, 0)} & {bias_txt}\\\\\n")
        self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
        outs = [k for k, _ in rows if k in s and s[k].get("outside")]
        if outs:
            self.box("warn",
                     r"""\textbf{Sensibilidad al ruido} (marcado con $^\dagger$). En estas magnitudes el valor nominal cae \emph{fuera} del intervalo del Monte Carlo: añadir ruido las desplaza sistemáticamente (columna \emph{Sesgo MC}: mediana del Monte Carlo respecto al nominal). Es una señal de que el ruido que ya contienen los datos también puede haber desplazado el resultado nominal. La corrección \emph{bootstrap} habitual ($2\times\text{nominal}-\text{mediana MC}$; \citealt{Efron1979}) \textbf{no se aplica}: con campos exactos funciona con ruido isótropo, pero empeora el resultado con ruido de tipo HMI (ver la validación más abajo). Trata el sesgo como una incertidumbre sistemática adicional.""",
                     r"""\textbf{Noise sensitivity} (marked with $^\dagger$). For these quantities the nominal value falls \emph{outside} the Monte Carlo interval: adding noise shifts them systematically (\emph{MC bias} column: Monte Carlo median relative to the nominal value). This indicates that the noise already present in the data may also have shifted the nominal result. The usual \emph{bootstrap} correction ($2\times\text{nominal}-\text{MC median}$; \citealt{Efron1979}) is \textbf{not applied}: with exact fields it works for isotropic noise, but it worsens the result for HMI-like noise (see the validation below). Treat the bias as an additional systematic uncertainty.""")
        self.bias_validation()
        self.figure("montecarlo",
                    r"Distribución de las magnitudes principales en las realizaciones del Monte Carlo; línea negra: valor nominal.",
                    r"Distribution of the main quantities over the Monte Carlo realisations; black line: nominal value.")
        self.box("read",
                 r"""Cada histograma muestra cómo varía una magnitud al cambiar el ruido de los datos. La banda gris contiene el 68\% central de las realizaciones (equivalente a $\pm1\sigma$). Si la línea negra (resultado nominal) cae dentro de la banda, el resultado nominal es representativo. Una distribución ancha indica que esa magnitud está mal determinada por los datos. Las magnitudes que dependen del campo transversal débil (helicidad, torsión) suelen ser más sensibles que la energía.""",
                 r"""Each histogram shows how a quantity varies when the data noise changes. The grey band contains the central 68\% of the realisations (equivalent to $\pm1\sigma$). If the black line (nominal result) falls inside the band, the nominal result is representative. A broad distribution means the quantity is poorly constrained by the data. Quantities depending on the weak transverse field (helicity, twist) are usually more sensitive than the energy.""")
        self.p(r"""\textbf{Estabilidad de cada línea.} En cada realización se trazan las líneas desde las mismas semillas y se comprueba si coinciden con las del modelo nominal (desviación $<10\%$ de su longitud y misma topología). La mediana de la estabilidad es <<sm>>, y el <<sf>> de las líneas coincide en al menos el 80\% de las realizaciones. Esta estabilidad se ha añadido a la confianza de cada línea (sección~\ref{sec:reliability}).""",
               r"""\textbf{Stability of each line.} In each realisation the lines are traced from the same seeds and compared with those of the nominal model (deviation $<10\%$ of their length and same topology). The median stability is <<sm>>, and <<sf>> of the lines match in at least 80\% of the realisations. This stability has been added to each line's confidence (Section~\ref{sec:reliability}).""",
               sm=num(M["stability_median"], 2), sf=pct(M["stable_frac"], 0))
        self.figure("stability",
                    r"Estabilidad frente al ruido de la línea que sale de cada punto de la base.",
                    r"Stability against noise of the field line rooted at each base point.",
                    width=r"0.8\textwidth")
        self.box("warn",
                 r"""Este Monte Carlo sólo recoge el ruido \emph{aleatorio} de la medida tal como lo estima la inversión de HMI. No incluye errores sistemáticos, errores en la resolución de la ambigüedad de $180^\circ$, el efecto de la resolución ni la dependencia del método NLFFF, que suelen ser mayores. Por eso las incertidumbres de esta tabla son \emph{cotas inferiores} (véase \citealt{Gilchrist2012} para un análisis más amplio de la incertidumbre de la energía libre). La fiabilidad de la energía y de la helicidad depende además de cuánto se aparte el modelo de un campo solenoidal \citep{Thalmann2019, Thalmann2020}. Además, los errores formales de HMI subestiman la incertidumbre real; el parámetro \texttt{error\_scale} permite repetir el análisis con errores mayores.""",
                 r"""This Monte Carlo only captures the \emph{random} measurement noise as estimated by the HMI inversion. It does not include systematic errors, errors in the $180^\circ$ ambiguity resolution, resolution effects or the dependence on the NLFFF method, which are usually larger. The uncertainties in this table are therefore \emph{lower bounds} (see \citealt{Gilchrist2012} for a broader analysis of free-energy uncertainty). The reliability of the energy and helicity also depends on how far the model departs from a solenoidal field \citep{Thalmann2019, Thalmann2020}. Moreover, the formal HMI errors underestimate the true uncertainty; the \texttt{error\_scale} parameter allows repeating the analysis with larger errors.""")

    def bias_validation(self):
        """Validation with exact fields of the bootstrap correction of the noise bias."""
        from ..montecarlo import load_bias_validation
        S = load_bias_validation()
        if not S:
            return
        noise_lab = {"10": self.L("uniforme 10~G", "uniform 10~G"), "30": self.L("uniforme 30~G", "uniform 30~G"),
                     "60": self.L("uniforme 60~G", "uniform 60~G"),
                     "10/60": self.L("tipo HMI (10/60~G)", "HMI-like (10/60~G)")}
        bc = S["by_case"]

        def mean_over_fields(q, nm, key):
            v = [bc[q][f"{f}|{nm}"][key] for f in S["fields"] if f"{f}|{nm}" in bc[q]]
            return float(np.mean(v)) if v else float("nan")
        self.w(r"\subsection{" + self.L("¿Se puede corregir el sesgo? Validación con campos exactos",
                                        "Can the bias be corrected? Validation with exact fields") + "}\n")
        self.p(r"""Para saber si la corrección \emph{bootstrap} acerca el resultado al valor sin ruido, se repitió todo el procedimiento (estudio + Monte Carlo de <<mc>> realizaciones) sobre <<no>> observaciones ruidosas independientes de <<nf>> campos exactos (<<fl>>) en mallas de <<sz>>$^2$ puntos, con cuatro modelos de ruido. La referencia es el mismo método aplicado a la base \emph{sin ruido}: así se aísla el efecto del ruido del error propio del método. La tabla da el error relativo medio (en valor absoluto) del valor nominal y del corregido.""",
               r"""To find out whether the \emph{bootstrap} correction brings the result closer to the noise-free value, the whole procedure (study + Monte Carlo with <<mc>> realisations) was repeated on <<no>> independent noisy observations of <<nf>> exact fields (<<fl>>) on <<sz>>$^2$ grids, with four noise models. The reference is the same method applied to the \emph{noise-free} base: this isolates the effect of noise from the method's own error. The table gives the mean relative error (in absolute value) of the nominal and corrected values.""",
               mc=S["mc"], no=S["obs"], nf=len(S["fields"]), fl=", ".join(S["fields"]), sz=S["size"])
        self.w(r"\begin{center}\small\begin{tabular}{lcccccc}\toprule" "\n")
        self.w(self.L(r" & \multicolumn{2}{c}{$E_{\rm free}$ (pot. FFT)} & \multicolumn{2}{c}{$E_{\rm free}$ (pot. volumen)} & \multicolumn{2}{c}{$H_V$}\\",
                      r" & \multicolumn{2}{c}{$E_{\rm free}$ (FFT pot.)} & \multicolumn{2}{c}{$E_{\rm free}$ (volume pot.)} & \multicolumn{2}{c}{$H_V$}\\")
               + r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n")
        self.w(self.L(r"Ruido & nominal & corregido & nominal & corregido & nominal & corregido\\",
                      r"Noise & nominal & corrected & nominal & corrected & nominal & corrected\\") + r"\midrule" "\n")
        for nm in S["noises"]:
            cells = []
            for q in ("E_free", "E_free_vol", "H_V"):
                a_, b_ = mean_over_fields(q, nm, "nominal_abs"), mean_over_fields(q, nm, "corrected_abs")
                cells += [pct(a_, 0), (r"\textbf{" + pct(b_, 0) + "}") if b_ < a_ else pct(b_, 0)]
            self.w(noise_lab.get(nm, nm) + " & " + " & ".join(cells) + r"\\" "\n")
        self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
        e30 = [mean_over_fields("E_free", nm, k) for nm in ("30", "60") for k in ("nominal_abs", "corrected_abs")]
        hmi = [mean_over_fields("E_free", "10/60", k) for k in ("nominal_abs", "corrected_abs")]
        vol = max(mean_over_fields("E_free_vol", nm, "nominal_abs") for nm in S["noises"])
        hv = max(mean_over_fields("H_V", nm, "nominal_abs") for nm in S["noises"])
        self.p(r"""En negrita, los casos en que la corrección mejora. (1) Con ruido \emph{isótropo} de 30--60~G el ruido \emph{reduce} la energía libre calculada con el potencial FFT (en torno al <<a30>> y <<a60>>), y la corrección recupera parte del sesgo (<<c30>> y <<c60>>). (2) Con ruido \emph{tipo HMI} (transversal mucho más ruidoso que el longitudinal), que es el caso real, la corrección \textbf{empeora} el resultado (del <<h0>> al <<h1>>): añadir ruido a datos que ya son ruidosos no desplaza el resultado igual que el ruido original desplazó el valor limpio, que es la hipótesis de la corrección. (3) La energía libre con el potencial de volumen y la helicidad son mucho \textbf{menos sensibles al ruido} (error nominal $\leq$ <<vol>> y $\leq$ <<hv>>), y la corrección tampoco las mejora. \textbf{Recomendación}: citar el valor nominal, usar como cifra principal de energía libre la del potencial de volumen, y dar el sesgo del Monte Carlo como incertidumbre sistemática, sin corregirlo.""",
               r"""Bold marks the cases where the correction helps. (1) With \emph{isotropic} noise of 30--60~G the noise \emph{reduces} the free energy computed with the FFT potential (by about <<a30>> and <<a60>>), and the correction recovers part of the bias (<<c30>> and <<c60>>). (2) With \emph{HMI-like} noise (transverse much noisier than longitudinal), which is the real case, the correction \textbf{worsens} the result (from <<h0>> to <<h1>>): adding noise to already noisy data does not shift the result in the same way as the original noise shifted the clean value, which is the assumption behind the correction. (3) The free energy with the volume potential and the helicity are much \textbf{less sensitive to noise} (nominal error $\leq$ <<vol>> and $\leq$ <<hv>>), and the correction does not improve them either. \textbf{Recommendation}: quote the nominal value, use the volume-potential free energy as the main free-energy figure, and give the Monte Carlo bias as a systematic uncertainty, without correcting it.""",
               a30=pct(e30[0], 0), c30=pct(e30[1], 0), a60=pct(e30[2], 0), c60=pct(e30[3], 0),
               h0=pct(hmi[0], 0), h1=pct(hmi[1], 0), vol=pct(vol, 0), hv=pct(hv, 0))
        self.box("key",
                 r"""El ruido de medida sesga sobre todo la energía libre calculada con el potencial FFT. La corrección bootstrap no es fiable con el ruido de HMI, así que no se aplica: la energía libre con el potencial de volumen es la estimación más robusta frente al ruido.""",
                 r"""Measurement noise mainly biases the free energy computed with the FFT potential. The bootstrap correction is not reliable for HMI noise, so it is not applied: the free energy with the volume potential is the most noise-robust estimate.""")

    # ------------------------------------------------- comparison of methods
    def methods_comparison(self):
        M = self.A.get("methods")
        if not M:
            return
        GN = {"es": {"nlfff": "NLFFF (optim.)", "gr_p": "Grad--Rubin P", "gr_n": "Grad--Rubin N"},
              "en": {"nlfff": "NLFFF (optim.)", "gr_p": "Grad--Rubin P", "gr_n": "Grad--Rubin N"}}[self.lang]
        self.w(r"\section{" + self.L("¿Depende el resultado del método? Comparación con Grad--Rubin",
                                     "Does the result depend on the method? Comparison with Grad--Rubin") + r"}\label{sec:methods_cmp}" "\n")
        self.p(r"""Con los mismos datos, distintos códigos NLFFF pueden dar resultados apreciablemente distintos (DeRosa et al. 2009). Para medir esta \textbf{incertidumbre de método} se ha calculado también el NLFFF con un método de otra familia: \textbf{Grad--Rubin} (Grad \& Rubin 1958; Amari et al. 2006; Wheatland 2007). En lugar de minimizar un funcional, toma $\alpha=J_z/B_z$ de la fotosfera en \emph{una} polaridad, lo propaga a lo largo de las líneas de campo, calcula el campo que generan esas corrientes y repite hasta converger. Usa el mismo magnetograma, el mismo dominio y el mismo campo potencial que el NLFFF por optimización.""",
               r"""With the same data, different NLFFF codes can give appreciably different results (DeRosa et al. 2009). To measure this \textbf{method uncertainty}, the NLFFF was also computed with a method of a different family: \textbf{Grad--Rubin} (Grad \& Rubin 1958; Amari et al. 2006; Wheatland 2007). Instead of minimising a functional, it takes $\alpha=J_z/B_z$ from the photosphere in \emph{one} polarity, propagates it along field lines, computes the field generated by those currents and repeats until convergence. It uses the same magnetogram, domain and potential field as the optimisation NLFFF.""")
        self.box("explain",
                 r"""\textbf{Soluciones P y N.} La solución P usa sólo el $\alpha$ medido en la polaridad positiva, y la N sólo el de la negativa. En un campo libre de fuerzas, los dos pies de cada línea deben tener el mismo $\alpha$, así que ambas soluciones deberían coincidir. Con datos reales no coinciden, porque la fotosfera no está libre de fuerzas y los datos tienen ruido: \textbf{la diferencia entre P y N mide cuán inconsistentes son los datos} con un campo libre de fuerzas (Wheatland \& Régnier 2009).""",
                 r"""\textbf{P and N solutions.} The P solution uses only the $\alpha$ measured in the positive polarity, and N only that of the negative one. In a force-free field both footpoints of each line must have the same $\alpha$, so both solutions should agree. With real data they do not, because the photosphere is not force-free and the data are noisy: \textbf{the difference between P and N measures how inconsistent the data are} with a force-free field (Wheatland \& Régnier 2009).""")
        keys = list(M["models"])
        R = M["models"]
        self.w(f"\\begin{{center}}\\small\\begin{{tabular}}{{l{'r' * len(keys)}}}\\toprule\n & "
               + " & ".join(GN[k] for k in keys) + r"\\\midrule" "\n")
        rows = [(r"$\mathrm{CW}_{\sin}$", lambda r: num(r["CWsin"])),
                (r"$\langle|f_i|\rangle$", lambda r: num(r["fi"], 2)),
                (r"$E/E_{\rm pot}$", lambda r: num(r["E_ratio"], 4)),
                (self.L("E libre (pot. FFT) [erg]", "Free E (FFT pot.) [erg]"), lambda r: num(r["E_free"])),
                (self.L("E libre (pot. volumen) [erg]", "Free E (volume pot.) [erg]"), lambda r: num(r["E_free_vol"])),
                (r"$E_{\rm div}/E$", lambda r: pct(r["E_div_frac"])),
                (r"$H_V$ [Mx$^2$]", lambda r: num(r["H_V"])),
                (r"$|H_J|/|H_V|$", lambda r: num(r["HJ_HV"])),
                (self.L(r"Percentil 99 de $|T_w|$", r"99th percentile of $|T_w|$"), lambda r: num(r["tw_p99"])),
                (self.L("Iteraciones", "Iterations"), lambda r: str(r["iterations"]))]
        if any("vs_exact" in R[k] for k in keys):
            rows.append((r"$C_{\rm vec}$ " + self.L("frente al exacto", "vs. exact"), lambda r: num(r["vs_exact"]["C_vec"], 4) if "vs_exact" in r else "--"))
            rows.append((r"$E'_n$ " + self.L("frente al exacto", "vs. exact"), lambda r: num(r["vs_exact"]["E'_n"], 3) if "vs_exact" in r else "--"))
        rows.append((r"$C_{\rm vec}$ " + self.L("frente al NLFFF", "vs. NLFFF"), lambda r: num(r["vs_nlfff"]["C_vec"], 4) if "vs_nlfff" in r else "--"))
        rows.append((self.L("Misma topología que el NLFFF", "Same topology as NLFFF"), lambda r: pct(r["same_topology_nlfff"], 0) if "same_topology_nlfff" in r else "--"))
        for lab, f in rows:
            self.w(lab + " & " + " & ".join(f(R[k]) for k in keys) + r"\\" "\n")
        self.w(r"\bottomrule\end{tabular}\end{center}" "\n\n")
        exc = M.get("excluded") or []
        if exc:
            self.box("warn",
                     r"""\textbf{Soluciones no convergidas: <<ex>>.} No alcanzaron la convergencia en el número máximo de iteraciones, así que sus valores (tabla) \emph{no son soluciones libres de fuerzas válidas} y se excluyen de la incertidumbre de método. Es un problema conocido de Grad--Rubin con datos reales: el $\alpha$ medido en la fotosfera es ruidoso y las dos polaridades dan valores inconsistentes, porque la fotosfera no está libre de fuerzas; con corrientes intensas la iteración no converge (Wheatland \& Régnier 2009). Una gran diferencia entre las soluciones P y N indica, en sí misma, que los datos de esta región son poco consistentes con un campo libre de fuerzas.""",
                     r"""\textbf{Non-converged solutions: <<ex>>.} They did not converge within the maximum number of iterations, so their values (table) are \emph{not valid force-free solutions} and are excluded from the method uncertainty. This is a known problem of Grad--Rubin with real data: the photospheric $\alpha$ is noisy and the two polarities give inconsistent values, because the photosphere is not force-free; with strong currents the iteration does not converge (Wheatland \& Régnier 2009). A large difference between the P and N solutions is itself an indication that the data of this region are poorly consistent with a force-free field.""",
                     ex=", ".join(GN[k] for k in exc))
        S = M["spread"]
        if "E_free" in S:
            self.p(r"""\textbf{Incertidumbre de método.} Entre los tres modelos, la energía libre (potencial FFT) varía entre <<a>> y <<b>>~erg (semiamplitud del <<ra>>), la energía libre respecto al potencial de volumen entre <<c>> y <<d>>~erg (<<rc>>) y $H_V$ entre <<e>> y <<f>>~Mx$^2$ (<<re>>). Compárese con la incertidumbre por ruido de medida de la sección~\ref{sec:mc}, si se calculó: normalmente la de método es mucho mayor.""",
                   r"""\textbf{Method uncertainty.} Among the three models, the free energy (FFT potential) ranges from <<a>> to <<b>>~erg (half-range <<ra>>), the free energy relative to the volume potential from <<c>> to <<d>>~erg (<<rc>>) and $H_V$ from <<e>> to <<f>>~Mx$^2$ (<<re>>). Compare with the measurement-noise uncertainty of Section~\ref{sec:mc}, if computed: the method uncertainty is usually much larger.""",
                   a=num(S["E_free"]["min"]), b=num(S["E_free"]["max"]), ra=pct(S["E_free"]["rel_halfrange"], 0),
                   c=num(S["E_free_vol"]["min"]), d=num(S["E_free_vol"]["max"]), rc=pct(S["E_free_vol"]["rel_halfrange"], 0),
                   e=num(S["H_V"]["min"]), f=num(S["H_V"]["max"]), re=pct(S["H_V"]["rel_halfrange"], 0))
        if M.get("p_vs_n"):
            pn = M["p_vs_n"]
            self.p(r"""\textbf{Consistencia P--N.} Las soluciones P y N tienen $C_{\rm vec}=$ <<c>> entre sí y el <<t>> de sus líneas tienen la misma topología.""",
                   r"""\textbf{P--N consistency.} The P and N solutions have $C_{\rm vec}=$ <<c>> between them and <<t>> of their lines share the same topology.""",
                   c=num(pn["C_vec"], 4), t=pct(pn["same_topology"], 0))
        self.figure("methods_bars", r"Energía libre y helicidad según el método.", r"Free energy and helicity by method.")
        self.figure("methods_lines",
                    r"Mismas semillas numeradas en los tres métodos: conectividad según el método.",
                    r"Same numbered seeds in the three methods: connectivity by method.")
        self.box("read",
                 r"""Sigue cada número de un panel a otro: si la línea conecta los mismos puntos en los tres métodos, esa conectividad es robusta frente al método. Las líneas que cambian de destino dependen de cómo se resuelve el problema, no sólo de los datos. En la figura de barras, la dispersión entre métodos es la incertidumbre de método de la energía libre y de la helicidad.""",
                 r"""Follow each number from one panel to the next: if the line connects the same points in all three methods, that connectivity is robust against the method. Lines that change destination depend on how the problem is solved, not just on the data. In the bar chart, the spread among methods is the method uncertainty of the free energy and helicity.""")

    # ------------------------------------------------- uncertainty budget
    BUDGET_LABELS = {
        "noise_random": ("Ruido de medida (Monte Carlo)", "Measurement noise (Monte Carlo)"),
        "noise_bias": ("Sesgo por ruido (Monte Carlo)", "Noise bias (Monte Carlo)"),
        "method_exact": ("Método frente al campo exacto", "Method vs.\\ exact field"),
        "preprocessing": (r"Preprocesado ($\mu_3$, $\mu_4$ $\times10^{\pm1}$)", r"Preprocessing ($\mu_3$, $\mu_4$ $\times10^{\pm1}$)"),
        "nlfff_params": (r"Parámetros NLFFF ($\nu$, $\lambda$, $\varepsilon$)", r"NLFFF parameters ($\nu$, $\lambda$, $\varepsilon$)"),
        "buffer": ("Capa de amortiguamiento (2--8 puntos)", "Buffer layer (2--8 points)"),
        "resolution": ("Resolución (28$^2$--56$^2$)", "Resolution (28$^2$--56$^2$)"),
        "method_gr": ("Método (optimización frente a Grad--Rubin)", "Method (optimisation vs.\\ Grad--Rubin)"),
    }

    @staticmethod
    def budget_cell(v):
        if v is None:
            return "--"
        lo, hi = (0.0 if abs(x) < 0.005 else x for x in v)  # < 0.5 % is shown as 0
        if abs(lo + hi) < 1e-9 and hi > 0:
            return r"$\pm$" + pct(hi, 0 if hi >= 0.095 else 1)
        f = lambda x: ("$+$" if x > 0 else "$-$" if x < 0 else "") + pct(abs(x), 0)
        if lo == 0:
            return f(hi)
        if hi == 0:
            return f(lo)
        return f(lo) + " / " + f(hi)

    def uncertainty_budget(self):
        B = self.A.get("budget")
        if not B or not B["rows"]:
            return
        self.w(r"\section{" + self.L("Presupuesto de incertidumbres", "Uncertainty budget") + r"}\label{sec:budget}" "\n")
        self.p(r"""Esta tabla reúne todas las fuentes de incertidumbre de la energía libre y de la helicidad que se han podido medir. Las filas marcadas \emph{este caso} se han calculado con los datos de esta región; las marcadas \emph{sintético}, con campos libres de fuerzas exactos, variando una sola elección alrededor de su valor por defecto y midiendo cuánto cambia el error respecto al campo exacto. Cada celda es el desplazamiento relativo: $\pm$ para una dispersión aleatoria ($1\sigma$) y un rango con signo para una elección o un sesgo.""",
               r"""This table gathers all the sources of uncertainty of the free energy and helicity that could be measured. Rows marked \emph{this case} were computed with the data of this region; rows marked \emph{synthetic}, with exact force-free fields, varying a single choice around its default value and measuring how much the error relative to the exact field changes. Each cell is the relative shift: $\pm$ for a random spread ($1\sigma$) and a signed range for a choice or a bias.""")
        kind = {"random": self.L("aleatoria", "random"), "systematic": self.L("sistemática", "systematic")}
        scope = {"case": self.L("este caso", "this case"), "synthetic": self.L("sintético", "synthetic")}
        self.w(r"\begin{table}[htbp]\centering\small\caption{"
               + self.L("Presupuesto de incertidumbres (desplazamientos relativos).",
                        "Uncertainty budget (relative shifts).")
               + r"}\label{tab:budget}" "\n" r"\resizebox{\textwidth}{!}{\begin{tabular}{lllccc}\toprule" "\n")
        self.w(self.L(r"Fuente & Tipo & Medida en & $E_{\rm free}$ (pot. FFT) & $E_{\rm free}$ (pot. volumen) & $H_V$",
                      r"Source & Type & Measured on & $E_{\rm free}$ (FFT pot.) & $E_{\rm free}$ (volume pot.) & $H_V$")
               + r"\\\midrule" "\n")
        dom = set(B["dominant"].values())
        for r in B["rows"]:
            lab = self.L(*self.BUDGET_LABELS.get(r["key"], (r["key"], r["key"])))
            cells = []
            for q in ("E_free", "E_free_vol", "H_V"):
                c = self.budget_cell(r["values"].get(q))
                if r["key"] == "method_gr" and not r["detail"].get("estimated"):
                    c = self.L("no conv.", "not conv.")
                cells.append(r"\textbf{" + c + "}" if B["dominant"].get(q) == r["key"] else c)
            self.w(f"{lab} & {kind[r['kind']]} & {scope[r['scope']]} & " + " & ".join(cells) + r"\\" "\n")
        self.w(r"\bottomrule\end{tabular}}\end{table}" "\n\n")
        names = {k: self.L(*self.BUDGET_LABELS[k]).split(" (")[0].lower() if k in self.BUDGET_LABELS else k
                 for k in dom if k}
        self.p(r"""En negrita, la fuente dominante de cada magnitud: <<d1>> para la energía libre con el potencial FFT, <<d2>> para la del potencial de volumen y <<d3>> para la helicidad. Las fuentes \textbf{no se suman}: no son independientes y no tienen el mismo significado estadístico (el Monte Carlo da una desviación típica; las demás filas, el efecto de cambiar una elección en un orden de magnitud o de duplicar la resolución), así que un total sería engañoso. Las filas sintéticas proceden de unos pocos campos y de mallas pequeñas: indican el orden de magnitud, no el valor exacto en esta región.""",
               r"""In bold, the dominant source of each quantity: <<d1>> for the free energy with the FFT potential, <<d2>> for that with the volume potential and <<d3>> for the helicity. The sources are \textbf{not added}: they are not independent and do not have the same statistical meaning (the Monte Carlo gives a standard deviation; the other rows, the effect of changing a choice by an order of magnitude or of doubling the resolution), so a total would be misleading. The synthetic rows come from a few fields and small grids: they give the order of magnitude, not the exact value for this region.""",
               d1=names.get(B["dominant"]["E_free"], "--"), d2=names.get(B["dominant"]["E_free_vol"], "--"),
               d3=names.get(B["dominant"]["H_V"], "--"))
        gr = next((r for r in B["rows"] if r["key"] == "method_gr"), None)
        if gr and not gr["detail"].get("estimated"):
            self.p(r"""La incertidumbre de método con datos reales no se ha podido estimar porque las soluciones Grad--Rubin no convergieron (sección~\ref{sec:methods_cmp}). Con datos reales, distintos códigos NLFFF difieren en decenas de por ciento \citep{DeRosa2009}, así que es probable que esta fuente sea comparable a la del preprocesado.""",
                   r"""The method uncertainty with real data could not be estimated because the Grad--Rubin solutions did not converge (Section~\ref{sec:methods_cmp}). With real data, different NLFFF codes differ by tens of per cent \citep{DeRosa2009}, so this source is probably comparable to the preprocessing one.""")
        if B.get("definition"):
            d = B["definition"]
            self.p(r"""\textbf{Definición de la energía libre.} En esta región, la energía libre respecto al potencial de volumen (<<v>>~erg) es <<r>> veces la calculada con el potencial de la base (<<f>>~erg). No es una incertidumbre sino una elección: los dos potenciales tienen condiciones de contorno distintas en las caras laterales y superior \citep{Valori2013}. Indica siempre cuál se usa; aquí la cifra principal es la de volumen, que es la más robusta frente al ruido.""",
                   r"""\textbf{Definition of the free energy.} In this region, the free energy relative to the volume potential (<<v>>~erg) is <<r>> times that computed with the base potential (<<f>>~erg). This is not an uncertainty but a choice: the two potentials have different boundary conditions on the side and top faces \citep{Valori2013}. Always state which one is used; here the main figure is the volume one, which is the most robust against noise.""",
                   v=num(d["E_free_vol"]), f=num(d["E_free"]), r=num(d["ratio"], 2))
        if dom & {"noise_random", "noise_bias"}:
            return
        self.box("key",
                 r"""El ruido de medida es una fuente \emph{menor}. La energía libre calculada con el potencial de la base depende sobre todo del preprocesado \citep{Wiegelmann2012} y de la resolución; la calculada con el potencial de volumen y la helicidad apenas cambian con esas elecciones, y su incertidumbre principal es el error propio del método. Por eso en un artículo no basta con la barra de error del Monte Carlo: hay que justificar las elecciones, dar su efecto (por ejemplo, con esta tabla) y preferir la energía libre de volumen como cifra principal.""",
                 r"""Measurement noise is a \emph{minor} source. The free energy computed with the base potential depends mainly on the preprocessing \citep{Wiegelmann2012} and on the resolution; the one computed with the volume potential and the helicity barely change with those choices, and their main uncertainty is the method's own error. In a paper the Monte Carlo error bar is therefore not enough: the choices must be justified, their effect given (for instance, with this table), and the volume free energy preferred as the main figure.""")

    # ------------------------------------------------- comparison with AIA
    def aia_section(self):
        S = self.A.get("aia")
        if not S:
            return
        W = S["by_wavelength"]
        ws = sorted(W, key=int)
        self.w(r"\section{" + self.L("Comparación con los bucles observados (AIA)",
                                     "Comparison with the observed loops (AIA)") + r"}\label{sec:aia}" "\n")
        self.p(r"""Todo lo anterior se ha validado con campos sintéticos. Aquí se comprueba con el Sol real: las líneas del modelo se proyectan sobre las imágenes de SDO/AIA en <<wl>>~Å más cercanas a la observación de HMI, que muestran los bucles coronales, y se mide el \textbf{ángulo de desalineamiento} entre cada línea proyectada y los bucles \citep[como][]{Warren2018}. La orientación local de los bucles se obtiene de la matriz hessiana de la imagen (las crestas brillantes), y para cada línea se toma la mediana del ángulo en los puntos donde hay una cresta clara (el 10\% más intenso). En una imagen sintética con bucles conocidos el método recupera la orientación con 0.4° de mediana, y una línea sin relación con la imagen da 45°. Se comparan las líneas fiables con las no fiables, y el NLFFF con el campo potencial y el LFFF trazados desde las mismas semillas.""",
               r"""Everything above was validated with synthetic fields. Here it is checked against the real Sun: the model lines are projected onto the SDO/AIA <<wl>>~Å images closest in time to the HMI observation, which show the coronal loops, and the \textbf{misalignment angle} between each projected line and the loops is measured \citep[as in][]{Warren2018}. The local orientation of the loops is obtained from the Hessian matrix of the image (the bright ridges), and for each line the median angle is taken over the points where there is a clear ridge (the brightest 10\%). On a synthetic image with known loops the method recovers the orientation with a 0.4° median error, and a line unrelated to the image gives 45°. Reliable lines are compared with unreliable ones, and the NLFFF with the potential and LFFF fields traced from the same seeds.""",
               wl=self.L(" y ", " and ").join(ws))
        self.figure("aia_steps",
                    r"El procesado paso a paso sobre la imagen de 171~Å: (1) imagen AIA; (2) líneas del NLFFF proyectadas; (3) fuerza de las crestas brillantes según la hessiana; (4) crestas claras (recuadro: zona ampliada en 5); (5) dirección local de los bucles (trazos amarillos) y ángulo de una línea de ejemplo en cada punto sobre una cresta clara; (6) distribución de esos ángulos: su mediana es el ángulo de desalineamiento de la línea.",
                    r"The processing step by step on the 171~Å image: (1) AIA image; (2) projected NLFFF lines; (3) bright-ridge strength from the Hessian; (4) clear ridges (box: area enlarged in 5); (5) local loop direction (yellow sticks) and the angle of an example line at each point on a clear ridge; (6) distribution of those angles: their median is the line's misalignment angle.")
        self.box("read",
                 r"""(1) La imagen muestra los bucles coronales como trazos brillantes. (2) Cada línea del modelo se proyecta como la vería AIA: sus puntos 3D pasan a coordenadas solares y de ahí a píxeles de la imagen. (3) La hessiana resalta las crestas, es decir, las estructuras finas más brillantes que su entorno. (4) Sólo se usan las crestas claras, para no medir ruido. (5) En cada punto de la línea que cae sobre una cresta clara se mide el ángulo entre la línea y el bucle: 0° si van paralelos, 90° si se cruzan. (6) La mediana de esos ángulos es el ángulo de la línea; una línea sin relación con la imagen da unos 45° (línea de puntos).""",
                 r"""(1) The image shows the coronal loops as bright strands. (2) Each model line is projected as AIA would see it: its 3D points go to solar coordinates and then to image pixels. (3) The Hessian highlights ridges, i.e. thin structures brighter than their surroundings. (4) Only clear ridges are used, to avoid measuring noise. (5) At each point of the line lying on a clear ridge, the angle between line and loop is measured: 0° if parallel, 90° if crossing. (6) The median of those angles is the line's angle; a line unrelated to the image gives about 45° (dotted line).""")
        self.figure("aia",
                    r"Validación con AIA. (a) Líneas del NLFFF proyectadas sobre AIA. (b) Distribución del ángulo de desalineamiento de cada línea con los bucles. (c) Diferencia de ángulo entre el NLFFF y el campo potencial con las mismas semillas, para líneas fiables y no fiables, con todas las alturas y sólo en el tercio intermedio de alturas.",
                    r"Validation with AIA. (a) NLFFF lines projected onto AIA. (b) Distribution of each line's misalignment angle with the loops. (c) Angle difference between the NLFFF and the potential field from the same seeds, for reliable and unreliable lines, for all heights and for the middle third of heights only.")
        self.w(r"\begin{center}\resizebox{\textwidth}{!}{\begin{tabular}{l" + "c" * len(ws) + r"}\toprule" "\n")
        self.w(" & " + " & ".join(f"AIA {w}~Å" for w in ws) + r"\\\midrule" "\n")

        def ci(v, c):
            return f"{'$+$' if v >= 0 else '$-$'}{fx(abs(v), 1)}° [{fx(c[0], 1)}, {fx(c[1], 1)}]"

        def pv(p):
            return ("$p<0.001$" if p < 1e-3 else f"$p={fx(p, 3)}$") if p is not None else "--"
        rows = [
            (self.L("Líneas medidas (fiables / no fiables)", "Lines measured (reliable / unreliable)"),
             lambda r: f"{r['n_measured']} ({r['n_reliable']} / {r['n_unreliable']})"),
            (self.L("Ángulo mediano: fiables / no fiables", "Median angle: reliable / unreliable"),
             lambda r: f"{fx(r['median_reliable'], 1)}° / {fx(r['median_unreliable'], 1)}°"),
            (self.L("Diferencia (IC 95\\%), Mann--Whitney", "Difference (95\\% CI), Mann--Whitney"),
             lambda r: ci(r["diff"], r["diff_ci"]) + ", " + pv(r["p_mannwhitney"])),
            (self.L("Diferencia a igual altura (regresión)", "Difference at equal height (regression)"),
             lambda r: ci(r["ols_reliable"], r["ols_reliable_ci"])),
            (self.L("NLFFF $-$ potencial, líneas fiables", "NLFFF $-$ potential, reliable lines"),
             lambda r: f"{fx(r['models']['potential']['reliable']['median'], 1)}°, {pv(r['models']['potential']['reliable']['p'])}"),
            (self.L("NLFFF $-$ potencial, líneas no fiables", "NLFFF $-$ potential, unreliable lines"),
             lambda r: f"{fx(r['models']['potential']['unreliable']['median'], 1)}°, {pv(r['models']['potential']['unreliable']['p'])}"),
            (self.L("Ídem a altura intermedia: fiables / no fiables", "Same at intermediate height: reliable / unreliable"),
             lambda r: f"{fx(r['terciles'][1]['nlfff_minus_potential_reliable']['median'], 1)}° ({pv(r['terciles'][1]['nlfff_minus_potential_reliable']['p'])}) / "
                       f"{fx(r['terciles'][1]['nlfff_minus_potential_unreliable']['median'], 1)}°"),
        ]
        for lab, f in rows:
            self.w(lab + " & " + " & ".join(f(W[w]) for w in ws) + r"\\" "\n")
        NB = (S.get("null") or {}).get("by_wavelength") or {}

        def nul(w):
            return NB.get(w) or NB.get(str(w)) or NB.get(int(w))
        has_null = all(nul(w) for w in ws)
        if has_null:
            cells = [f"{fx(nul(w)['null']['shift']['all']['median'], 1)}° [{fx(nul(w)['null']['shift']['all']['p2.5'], 1)}, "
                     f"{fx(nul(w)['null']['shift']['all']['p97.5'], 1)}]" for w in ws]
            self.w(self.L(r"Azar: líneas desplazadas (IC 95\%)", r"Chance: shifted lines (95\% CI)")
                   + " & " + " & ".join(cells) + r"\\" "\n")
            gain = {g: [nul(w)["null"]["shift"][g]["median"] - nul(w)["observed"][g] for w in ws]
                    for g in ("reliable", "unreliable")}
            cells = [f"{fx(a, 1)}° / {fx(b, 1)}°" for a, b in zip(gain["reliable"], gain["unreliable"])]
            self.w(self.L("Mejora sobre el azar: fiables / no fiables", "Gain over chance: reliable / unreliable")
                   + " & " + " & ".join(cells) + r"\\" "\n")
        self.w(r"\bottomrule\end{tabular}}\end{center}" "\n\n")
        t = [W[w]["terciles"] for w in ws]
        low_rel = sum(tt[0]["n_reliable"] for tt in t) / max(1, sum(tt[0]["n_reliable"] + tt[0]["n_unreliable"] for tt in t))
        high_rel = sum(tt[2]["n_reliable"] for tt in t) / max(1, sum(tt[2]["n_reliable"] + tt[2]["n_unreliable"] for tt in t))
        ols_sig = [W[w]["ols_reliable_ci"][1] < 0 for w in ws]
        sens = [s for w in ws for s in W[w]["sensitivity"]]
        mids = [W[w]["terciles"][1] for w in ws]
        nlfff_helps = all((m["nlfff_minus_potential_reliable"]["p"] or 1) < 0.05 for m in mids) and \
            all((W[w]["models"]["potential"]["unreliable"]["p"] or 0) > 0.05 for w in ws)
        n_better = sum(s["median_reliable"] < s["median_unreliable"] for s in sens)
        n_better_mid = sum(s["mid_reliable"] is not None and s["mid_unreliable"] is not None
                           and s["mid_reliable"] < s["mid_unreliable"] for s in sens)
        self.p(r"""(1) \textbf{Las líneas fiables se alinean mejor con los bucles} que las no fiables (tabla), y lo hacen con cualquier escala y umbral de detección de los bucles (<<nb>> de <<ns>> combinaciones). (2) \textbf{Pero en esta región la confianza y la altura van casi juntas}: en el tercio de líneas más bajas, el <<lr>> son fiables; en el más alto, el <<hr>>. Las líneas bajas siguen mejor los bucles por sí mismas, así que hay que separar ambos efectos. A igual altura la diferencia es pequeña y <<sig>> (regresión con la altura en la tabla), aunque tiene el mismo signo en <<nm>> de <<ns>> combinaciones. <<p3>> <<p4>>""",
               r"""(1) \textbf{Reliable lines align better with the loops} than unreliable ones (table), with any scale and threshold of the loop detection (<<nb>> of <<ns>> combinations). (2) \textbf{But in this region confidence and height go almost together}: in the lowest third of lines, <<lr>> are reliable; in the highest, <<hr>>. Low lines follow the loops better by themselves, so both effects must be separated. At equal height the difference is small and <<sig>> (regression on height in the table), although it has the same sign in <<nm>> of <<ns>> combinations. <<p3>> <<p4>>""",
               nb=n_better, ns=len(sens), nm=n_better_mid, lr=pct(low_rel, 0), hr=pct(high_rel, 0),
               sig=self.L("significativa", "significant") if all(ols_sig) else self.L("no significativa", "not significant"),
               p3=self.L(r"(3) \textbf{La confianza sí señala dónde el NLFFF aporta información}: en las líneas fiables el NLFFF se alinea mejor que el campo potencial con las mismas semillas, también a igual altura, mientras que en las no fiables no lo mejora.",
                         r"(3) \textbf{The confidence does flag where the NLFFF adds information}: on reliable lines the NLFFF aligns better than the potential field from the same seeds, also at equal height, whereas on unreliable lines it does not improve on it.")
               if nlfff_helps else self.L(r"(3) La comparación entre el NLFFF y el campo potencial no separa con claridad las líneas fiables de las no fiables (tabla).",
                                          r"(3) The comparison between the NLFFF and the potential field does not clearly separate reliable from unreliable lines (table)."),
               p4=self.L(r"(4) \textbf{No es casualidad}: las mismas líneas desplazadas al azar sobre la imagen dan unos <<nz>>°, así que el abanico de la región apenas alinea por sí solo. Las líneas fiables mejoran al azar en <<gr>>° y las no fiables en <<gu>>° (media de los filtros).",
                         r"(4) \textbf{It is not chance}: the same lines shifted at random over the image give about <<nz>>°, so the region's fan pattern barely aligns by itself. Reliable lines beat chance by <<gr>>° and unreliable ones by <<gu>>° (mean over filters).")
               .replace("<<nz>>", fx(np.mean([nul(w)["null"]["shift"]["all"]["median"] for w in ws]), 0))
               .replace("<<gr>>", fx(np.mean(gain["reliable"]), 0))
               .replace("<<gu>>", fx(np.mean(gain["unreliable"]), 0)) if has_null else "")
        if nlfff_helps:
            self.box("key",
                 r"""Con datos reales, la confianza identifica las líneas en las que el NLFFF mejora al campo potencial frente a los bucles observados. Que las líneas fiables se alineen mejor que las no fiables a igual altura no se puede afirmar con una sola región, porque aquí confianza y altura están muy ligadas; más regiones lo aclararán.""",
                 r"""With real data, the confidence identifies the lines on which the NLFFF beats the potential field against the observed loops. That reliable lines align better than unreliable ones at equal height cannot be claimed from a single region, because here confidence and height are tightly linked; more regions will clarify it.""")
        self.box("warn",
                 r"""Limitaciones: la imagen EUV integra a lo largo de la línea de visión, así que cada cresta puede mezclar varios bucles; los bucles visibles en 171 y 193~Å son sólo los de cierto rango de temperaturas; la proyección supone la caja cartesiana local del SHARP CEA; y la orientación de una cresta no dice a qué altura está el bucle.""",
                 r"""Limitations: the EUV image integrates along the line of sight, so each ridge may mix several loops; the loops visible at 171 and 193~Å are only those within a certain temperature range; the projection assumes the local Cartesian box of the SHARP CEA; and the orientation of a ridge does not tell the height of the loop.""")

    # ------------------------------------------------- three regions and flares
    def regions_section(self):
        R = self.A.get("regions") or {}
        if not R.get("summary") or not R.get("flares"):
            return
        G, F = R["summary"]["regions"], R["flares"]
        self.w(r"\section{" + self.L("Tres regiones y tres fulguraciones", "Three regions and three flares")
               + r"}\label{sec:regions}" "\n")
        self.p(r"""Para comprobar que los resultados no dependen de una sola región, el mismo estudio se repitió en tres regiones con fulguraciones X: NOAA 11158 (X2.2), NOAA 11429 (X5.4) y NOAA 12673 (X9.3). En cada una se extrapoló el campo poco antes de la fulguración, una o dos horas después y en un instante de referencia previo (sin fulguraciones grandes entre medias), con el mismo método y la misma malla. Las hipótesis se escribieron \textbf{antes} de ver los resultados, para que ningún hallazgo sea fruto de buscar hasta encontrar.""",
               r"""To check that the results do not depend on a single region, the same study was repeated in three regions with X-class flares: NOAA 11158 (X2.2), NOAA 11429 (X5.4) and NOAA 12673 (X9.3). In each one the field was extrapolated shortly before the flare, one or two hours after it and at an earlier reference time (with no large flares in between), with the same method and grid. The hypotheses were written \textbf{before} seeing the results, so that no finding comes from searching until something turns up.""")

        def e32(v):
            return fx(v / 1e32, 2) if v is not None else "--"
        cols = "l" + "c" * len(G)
        self.w(r"\begin{center}\resizebox{\textwidth}{!}{\begin{tabular}{" + cols + r"}\toprule" "\n")
        self.w(" & " + " & ".join(f"NOAA {r['noaa']}" for r in G) + r"\\\midrule" "\n")
        rows = [
            (self.L("Píxel; malla", "Pixel; grid"), lambda r: f"{fx(r['dx_Mm'], 2)}~Mm; {r['grid'][0]}$\\times${r['grid'][1]}$\\times${r['grid'][2]}"),
            (r"$\mathrm{CW}_{\sin}$; $E_{\rm div}/E$", lambda r: f"{fx(r['CWsin'], 3)}; {fx(r['E_div_frac'], 3)}"),
            (self.L(r"$E_{\rm free}$ (pot. de volumen) [$10^{32}$ erg]", r"$E_{\rm free}$ (volume pot.) [$10^{32}$ erg]"), lambda r: e32(r["E_free_vol"])),
            (self.L("Ruido en $E_{\\rm free}$ (vol.)", "Noise in $E_{\\rm free}$ (vol.)"), lambda r: pct(r["noise_E_free_vol"]) if r["noise_E_free_vol"] else "--"),
            (self.L("Error de gauge de $H_V$", "Gauge error of $H_V$"), lambda r: pct(r["gauge"], 0) if r["gauge"] is not None else "--"),
            (self.L("Líneas fiables", "Reliable lines"), lambda r: pct(r["frac_reliable"], 0)),
            (self.L("AIA 171~Å: fiables / no fiables / azar", "AIA 171~Å: reliable / unreliable / chance"),
             lambda r: f"{fx(r['aia']['171']['reliable'], 1)}° / {fx(r['aia']['171']['unreliable'], 1)}° / {fx(r['aia']['171']['chance'], 1)}°"),
            (self.L("AIA 171~Å: fiables $-$ no fiables a igual altura", "AIA 171~Å: reliable $-$ unreliable at equal height"),
             lambda r: f"{fx(r['aia']['171']['equal_height'], 1)}° [{fx(r['aia']['171']['equal_height_ci'][0], 1)}, {fx(r['aia']['171']['equal_height_ci'][1], 1)}]"),
            (self.L("AIA 171~Å: NLFFF $-$ potencial, fiables / no fiables", "AIA 171~Å: NLFFF $-$ potential, reliable / unreliable"),
             lambda r: f"{fx(r['aia']['171']['nlfff_minus_pot_reliable'], 1)}° / {fx(r['aia']['171']['nlfff_minus_pot_unreliable'], 1)}°"),
            (self.L("Fulguración", "Flare"), lambda r: r["flare"]["name"].split(" ")[0]),
            (self.L(r"$\Delta E_{\rm free}$ (vol.) [$10^{32}$ erg]", r"$\Delta E_{\rm free}$ (vol.) [$10^{32}$ erg]"),
             lambda r: f"{e32(r['flare']['dE_vol'])} ({pct(r['flare']['dE_vol_rel'], 0)})"),
            (self.L("Variación natural previa; ruido $\\sigma$", "Natural variation before; noise $\\sigma$"),
             lambda r: f"{e32(r['flare']['dE_natural'])}; {e32(r['flare']['sigma_noise'])}"),
            (self.L("Fracción de la caída en la zona fiable", "Fraction of the drop in the reliable zone"),
             lambda r: pct(r["flare"]["frac_from_reliable"], 0)),
            (self.L("No fiables cerca / lejos de la PIL (razón)", "Unreliable near / far from the PIL (ratio)"),
             lambda r: f"{fx(r['flare']['pil_ratio'], 2)} [{fx(r['flare']['pil_ci'][0], 2)}, {fx(r['flare']['pil_ci'][1], 2)}]"),
            (self.L("Líneas fiables antes $\\to$ después", "Reliable lines before $\\to$ after"),
             lambda r: f"{pct(r['flare']['frac_rel_pre_post'][0], 0)} $\\to$ {pct(r['flare']['frac_rel_pre_post'][1], 0)}"),
        ]
        for lab, f in rows:
            self.w(lab + " & " + " & ".join(f(r) for r in G) + r"\\" "\n")
        self.w(r"\bottomrule\end{tabular}}\end{center}" "\n\n")
        sm = F.get("summary", {})
        self.p(r"""Resultado de cada hipótesis:
(1) \textbf{H1, la energía liberada sale de las líneas fiables: se cumple en <<h1>> de 3 fulguraciones, pero no es robusto}. Con la densidad $|\mathbf{B}-\mathbf{B}_p|^2/8\pi$, la fracción de la caída en la zona fiable cambia mucho según cuántas líneas se tracen, cuánto se dilaten y el umbral de confianza (de 17 a 107\% según el caso), así que no se puede afirmar dónde se libera la energía.
(2) \textbf{H2, las líneas no fiables se concentran en la línea de inversión (PIL): sale al revés en las tres}. Cerca de la PIL las líneas son \emph{más} fiables: son cortas, bajas y de campo fuerte.
(3) \textbf{H3, después de la fulguración hay más líneas fiables: no se cumple} (<<h3>> de 3; los cambios son de 1--2 puntos).
(4) \textbf{H4}: la energía respecto al potencial de la base y la de volumen bajan en las tres; coinciden en el signo.
(5) \textbf{H5, las fiables se alinean mejor con AIA a igual altura: no se cumple} al combinar las tres regiones (tabla de arriba).""",
               r"""Outcome of each hypothesis:
(1) \textbf{H1, the released energy comes from the reliable lines: met in <<h1>> of 3 flares, but not robust}. With the density $|\mathbf{B}-\mathbf{B}_p|^2/8\pi$, the fraction of the drop in the reliable zone changes strongly with how many lines are traced, how much they are dilated and the confidence threshold (from 17 to 107\% depending on the case), so where the energy is released cannot be stated.
(2) \textbf{H2, unreliable lines concentrate at the polarity inversion line (PIL): the opposite holds in all three}. Near the PIL lines are \emph{more} reliable: they are short, low and in strong field.
(3) \textbf{H3, more reliable lines after the flare: not met} (<<h3>> of 3; the changes are 1--2 points).
(4) \textbf{H4}: the free energy relative to the base potential and the volume one both drop in all three; their signs agree.
(5) \textbf{H5, reliable lines align better with AIA at equal height: not met} when the three regions are combined (table above).""",
               h1=(sm.get("H1") or {}).get("n_pass", "--"), h3=(sm.get("H3") or {}).get("n_pass", "--"))
        self.box("key",
                 r"""La línea de inversión, donde se dispara la fulguración, es la zona que el NLFFF reproduce mejor. La caída total de energía es clara en dos de las tres fulguraciones, pero su localización depende de cómo se construyan las zonas: el método no permite decir de qué parte del volumen sale la energía.""",
                 r"""The polarity inversion line, where the flare is triggered, is the region the NLFFF reproduces best. The total energy drop is clear in two of the three flares, but its location depends on how the zones are built: the method cannot tell which part of the volume the energy comes from.""")
        self.box("warn",
                 r"""Precauciones: sólo tres fulguraciones; las zonas se definen con las líneas de antes de la fulguración; NOAA 11429 tiene peor resolución (2.55~Mm por píxel) y entre sus dos instantes hay además una X1.3 de NOAA 11430; y en NOAA 12673 la energía varía antes de la fulguración casi tanto como cae en ella.""",
                 r"""Caveats: only three flares; the zones are defined by the pre-flare lines; NOAA 11429 has a coarser resolution (2.55~Mm per pixel) and an X1.3 flare from NOAA 11430 also occurs between its two times; and in NOAA 12673 the energy varies before the flare almost as much as it drops during it.""")

    def warnings_section(self):
        A = self.A
        self.w(r"\section{" + self.L("Avisos y limitaciones", "Warnings and limitations") + r"}\label{sec:warnings}" "\n")
        if A["warnings"]:
            self.p(r"Problemas detectados automáticamente en este caso, con una recomendación para cada uno:",
                   r"Problems automatically detected in this case, each with a recommendation:")
            self.w("\\begin{itemize}\n")
            for wv in A["warnings"]:
                lab = ASSESS[wv["key"]]
                name = lab[0] if self.lang == "es" else lab[1]
                advice = lab[4] if self.lang == "es" else lab[5]
                self.w(f"\\item {self.status(wv['status'])} \\textbf{{{name}.}} {advice}\n")
            self.w("\\end{itemize}\n\n")
        else:
            self.p(r"No se ha detectado ningún problema con los umbrales usados.",
                   r"No problems were detected with the thresholds used.")
        self.w(r"\subsection*{" + self.L("Limitaciones generales del método", "General limitations of the method") + "}\n")
        self.p(r"""\begin{itemize}[nosep]
\item \textbf{Geometría cartesiana:} se ignora la curvatura solar; es válido para regiones de hasta $\sim$200--300~Mm.
\item \textbf{Contornos no observados:} las caras laterales y superior proceden del campo potencial. La solución es más fiable cerca de la base y en el centro de la caja.
\item \textbf{Fotosfera no libre de fuerzas:} el preprocesado lo mitiga, pero introduce cambios en los datos. Lo ideal serían medidas cromosféricas.
\item \textbf{Dependencia del método:} con datos reales, distintos códigos NLFFF pueden dar energías libres que difieren en decenas de por ciento \citep{DeRosa2009}, y las soluciones de cada polaridad pueden no coincidir \citep{WheatlandLeka2011}. Conviene acompañar los resultados de barras de error (p.\,ej., variando el preprocesado o la resolución) y, si es posible, compararlos con métodos de otra familia, como las redes neuronales con física \citep{Jarolim2023}.
\item \textbf{Validación externa:} la confianza por línea se ha calibrado con campos libres de fuerzas exactos. En una cuerda de flujo (Titov--Démoulin) ordena bien las líneas pero subestima sus errores (sección~\ref{sec:reliability}). Falta comprobarla frente a bucles coronales observados \citep{Warren2018} y frente a simulaciones MHD, que no son libres de fuerzas en la base \citep{Fleishman2017}.
\item \textbf{LFFF:} el dominio es periódico y $|\alpha|<\alpha_{\max}$; su energía depende del tamaño de la caja.
\end{itemize}""",
               r"""\begin{itemize}[nosep]
\item \textbf{Cartesian geometry:} solar curvature is ignored; valid for regions up to $\sim$200--300~Mm.
\item \textbf{Unobserved boundaries:} the side and top faces come from the potential field. The solution is most reliable near the base and in the centre of the box.
\item \textbf{Non-force-free photosphere:} preprocessing mitigates this but changes the data. Chromospheric measurements would be ideal.
\item \textbf{Method dependence:} with real data, different NLFFF codes can give free energies differing by tens of per cent \citep{DeRosa2009}, and the solutions from each polarity may disagree \citep{WheatlandLeka2011}. Results should come with error bars (e.g.\ by varying the preprocessing or the resolution) and, where possible, be compared with methods of another family, such as physics-informed neural networks \citep{Jarolim2023}.
\item \textbf{External validation:} the per-line confidence has been calibrated with exact force-free fields. In a flux rope (Titov--Démoulin) it ranks lines well but underestimates their errors (Section~\ref{sec:reliability}). It still has to be tested against observed coronal loops \citep{Warren2018} and against MHD simulations, which are not force-free at the base \citep{Fleishman2017}.
\item \textbf{LFFF:} the domain is periodic and $|\alpha|<\alpha_{\max}$; its energy depends on the box size.
\end{itemize}""")

    def paper(self):
        self.w(r"\section{" + self.L("Material para el artículo", "Material for the paper") + r"}\label{sec:paper}" "\n")
        self.p(r"""Párrafo de métodos generado con los valores de este caso. Revísalo y adáptalo a tu artículo. Está también en \texttt{paper/methods\_es.tex} y \texttt{paper/methods\_en.tex}; las tablas, en \texttt{paper/table\_*.tex}; las referencias, en \texttt{paper/references.bib}; todos los números, en \texttt{paper/results.json}; y las figuras a 200~dpi, en \texttt{figures/}.""",
               r"""Methods paragraph generated with the values of this case. Review it and adapt it to your paper. It is also in \texttt{paper/methods\_en.tex} and \texttt{paper/methods\_es.tex}; tables in \texttt{paper/table\_*.tex}; references in \texttt{paper/references.bib}; all numbers in \texttt{paper/results.json}; and figures at 200~dpi in \texttt{figures/}.""")
        self.w(r"\begin{readbox}{" + self.L("Métodos (borrador)", "Methods (draft)") + "}\n"
               + methods_paragraph(self.s, self.A, self.lang) + "\n\\end{readbox}\n\n")

    def glossary(self):
        self.w(r"\section{" + self.L("Glosario", "Glossary") + r"}\label{sec:glossary}" "\n\\begin{description}[style=nextline]\n")
        for es_t, en_t, es_d, en_d in GLOSSARY:
            self.w(f"\\item[{self.L(es_t, en_t)}] {self.L(es_d, en_d)}\n")
        self.w("\\end{description}\n\n")

    def references(self):
        self.w(r"\section*{" + self.L("Referencias", "References") + "}\n"
               r"\addcontentsline{toc}{section}{" + self.L("Referencias", "References") + "}\n"
               "\\begin{itemize}[leftmargin=*,label={},nosep]\n")
        for _, ref in REFERENCES:
            self.w(f"\\item {ref}\n")
        self.w("\\end{itemize}\n")


def methods_paragraph(study, A, lang) -> str:
    """Methods paragraph in LaTeX with the actual values (es/en)."""
    d = A["data"]
    L = (lambda es, en: es) if lang == "es" else (lambda es, en: en)
    parts = []
    if d["benchmark"]:
        parts.append(L(
            r"Como prueba de validación, se tomó como condición de contorno el plano $z=0$ de la solución analítica de Low \& Lou (1990)",
            r"As a validation test, the $z=0$ plane of the Low \& Lou (1990) analytic solution was used as the boundary condition"))
    else:
        reg = []
        if d.get("noaa"):
            reg.append(f"NOAA {esc(d['noaa'])}")
        if d.get("harpnum"):
            reg.append(f"HARP {esc(d['harpnum'])}")
        parts.append(fill(L(
            r"El campo magnético coronal de <<reg>> se reconstruyó a partir del magnetograma vectorial SHARP CEA de SDO/HMI (\texttt{hmi.sharp\_cea\_720s}; Bobra et al. 2014) de <<t>>",
            r"The coronal magnetic field of <<reg>> was reconstructed from the SDO/HMI SHARP CEA vector magnetogram (\texttt{hmi.sharp\_cea\_720s}; Bobra et al. 2014) at <<t>>"),
            reg=" / ".join(reg) or L("la región", "the region"), t=esc(d.get("t_rec") or "--")))
    parts.append(fill(L(
        r", en una caja cartesiana de <<nx>>$\times$<<ny>>$\times$<<nz>> puntos con un paso de <<dx>>~Mm (<<fx>>$\times$<<fy>>$\times$<<h>>~Mm$^3$). ",
        r", in a Cartesian box of <<nx>>$\times$<<ny>>$\times$<<nz>> grid points with a spacing of <<dx>>~Mm (<<fx>>$\times$<<fy>>$\times$<<h>>~Mm$^3$). "),
        nx=d["nx"], ny=d["ny"], nz=d["nz"], dx=f"{d['dx']:.2f}", fx=f"{d['fov_x']:.0f}",
        fy=f"{d['fov_y']:.0f}", h=f"{d['height']:.0f}"))
    al = A["alpha"]
    if al.get("used") is not None:
        how = L("ajustando el campo transversal observado", "by best-fitting the observed transverse field") \
            if al.get("method") == "fit" else (L("como $\\sum J_zB_z/\\sum B_z^2$ (Hagino \\& Sakurai 2004)",
                                                   "as $\\sum J_zB_z/\\sum B_z^2$ (Hagino \\& Sakurai 2004)")
                                                 if al.get("method") == "jz" else L("fijado por el usuario", "set by the user"))
        parts.append(fill(L(
            r"Se calcularon un campo potencial y un campo libre de fuerzas lineal (LFFF) mediante el método de Fourier de Alissandrakis (1981) con contornos laterales periódicos; el parámetro libre de fuerzas, $\alpha=$ <<a>>~Mm$^{-1}$, se obtuvo <<how>>. ",
            r"A potential field and a linear force-free field (LFFF) were computed with the Fourier method of Alissandrakis (1981) using periodic lateral boundaries; the force-free parameter, $\alpha=$ <<a>>~Mm$^{-1}$, was obtained <<how>>. "),
            a=num(al["used"]), how=how))
    if study.nlfff is not None:
        c = A["convergence"]
        q = A["quality"]["nlfff"]
        e = A["energy"]["nlfff"]
        pp = ""
        if study.mag_pp is not None:
            cfg = study.config.get("preprocess_kw", {})
            pp = fill(L(
                r", tras preprocesar los datos fotosféricos según Wiegelmann et al. (2006) ($\mu_3=$ <<m3>>, $\mu_4=$ <<m4>>), lo que redujo la fuerza neta de $\epsilon_F=$ <<f0>> a <<f1>>",
                r", after preprocessing the photospheric data following Wiegelmann et al. (2006) ($\mu_3=$ <<m3>>, $\mu_4=$ <<m4>>), which reduced the net force from $\epsilon_F=$ <<f0>> to <<f1>>"),
                m3=num(cfg.get("mu3", 1e-2)), m4=num(cfg.get("mu4", 1e-2)),
                f0=num(A["aly"]["eps_force"]), f1=num(A["aly_pp"]["eps_force"]))
        meta = study.nlfff.meta
        if meta.get("backend") == "lbfgs":
            rb = meta.get("relax_bottom")
            base_es = (fill(r"con la base relajada (Wiegelmann \& Inhester 2010): el campo transversal en $z=0$ puede ajustarse con un término de datos de peso $\nu=$ <<nu>>, ponderado por la intensidad del campo transversal observado, mientras $B_z$ se mantiene fijo", nu=num(rb))
                       if rb else "con la base fija al magnetograma")
            base_en = (fill(r"with a relaxed base (Wiegelmann \& Inhester 2010): the transverse field at $z=0$ may adjust through a data term of weight $\nu=$ <<nu>>, weighted by the observed transverse field strength, while $B_z$ is kept fixed", nu=num(rb))
                       if rb else "with the base fixed to the magnetogram")
            parts.append(fill(L(
                r"El campo libre de fuerzas no lineal (NLFFF) se calculó minimizando el funcional de Wheatland et al. (2000) con la función de peso de Wiegelmann (2004) (capa de amortiguamiento de <<nd>> puntos), una regularización $B^2+\varepsilon^2$ ($\varepsilon=$ <<eps>>\,$B_{\max}$) cerca de los nulos, un peso $w_d=$ <<wd>> del término de divergencia (la forma estándar; con un peso mayor se cumplen los umbrales de solenoidalidad de Thalmann et al. 2019, pero las líneas reproducen peor los campos exactos y la energía no mejora), y un término de suavidad $\lambda\langle|\nabla^2\mathbf{B}|^2\rangle$ ($\lambda=$ <<lam>>), <<base>><<pp>>. El gradiente exacto del funcional discreto se obtuvo por diferenciación automática (PyTorch; Paszke et al. 2019) y la minimización se realizó con L-BFGS (Liu \& Nocedal 1989), iniciada desde el campo potencial, durante <<it>> iteraciones, con una reducción del funcional en un factor <<red>>. La solución tiene $\mathrm{CW}_{\sin}=$ <<cw>> ($\theta_J=$ <<th>>$^\circ$) y $\langle|f_i|\rangle=$ <<fi>> (Wheatland et al. 2000; Schrijver et al. 2006). Su energía magnética es $E=$ <<E>>~erg, con $E/E_{\rm pot}=$ <<r>>, lo que implica una energía libre de <<free>>~erg. ",
                r"The nonlinear force-free field (NLFFF) was computed by minimising the functional of Wheatland et al. (2000) with the weighting function of Wiegelmann (2004) (buffer layer of <<nd>> grid points), a regularisation $B^2+\varepsilon^2$ ($\varepsilon=$ <<eps>>\,$B_{\max}$) near nulls, a weight $w_d=$ <<wd>> on the divergence term (the standard form; a larger weight meets the solenoidality thresholds of Thalmann et al. 2019, but the lines reproduce exact fields less well and the energy does not improve), and a smoothness term $\lambda\langle|\nabla^2\mathbf{B}|^2\rangle$ ($\lambda=$ <<lam>>), <<base>><<pp>>. The exact gradient of the discrete functional was obtained by automatic differentiation (PyTorch; Paszke et al. 2019) and the minimisation was performed with L-BFGS (Liu \& Nocedal 1989), started from the potential field, for <<it>> iterations, the functional decreasing by a factor <<red>>. The solution has $\mathrm{CW}_{\sin}=$ <<cw>> ($\theta_J=$ <<th>>$^\circ$) and $\langle|f_i|\rangle=$ <<fi>> (Wheatland et al. 2000; Schrijver et al. 2006). Its magnetic energy is $E=$ <<E>>~erg, with $E/E_{\rm pot}=$ <<r>>, implying a free energy of <<free>>~erg. "),
                eps=num(meta.get("eps", 0.0)), lam=num(meta.get("smooth", 0.0)), base=L(base_es, base_en),
                wd=fx(meta.get("div_weight", 1.0), 0),
                nd=A["nd"], pp=pp, it=c["iterations"], red=num(c["reduction"]), cw=num(q["CWsin"]),
                th=f"{q['theta_J_deg']:.1f}", fi=num(q["<|f_i|>"], 2), E=num(e["E"]), r=num(e["ratio"], 4),
                free=num(e["free"])))
        else:
            parts.append(fill(L(
                r"El campo libre de fuerzas no lineal (NLFFF) se calculó con el método de optimización (Wheatland et al. 2000) con la función de peso de Wiegelmann (2004), una capa de amortiguamiento de <<nd>> puntos y una regularización $B^2+\varepsilon^2$ ($\varepsilon=$ <<eps>>\,$B_{\max}$) cerca de los nulos magnéticos<<pp>>. La optimización, iniciada desde el campo potencial, se detuvo tras <<it>> iteraciones, con una reducción del funcional en un factor <<red>>. La solución tiene $\mathrm{CW}_{\sin}=$ <<cw>> ($\theta_J=$ <<th>>$^\circ$) y $\langle|f_i|\rangle=$ <<fi>> (Wheatland et al. 2000; Schrijver et al. 2006). Su energía magnética es $E=$ <<E>>~erg, con $E/E_{\rm pot}=$ <<r>>, lo que implica una energía libre de <<free>>~erg. ",
                r"The nonlinear force-free field (NLFFF) was computed with the optimisation method (Wheatland et al. 2000) with the weighting function of Wiegelmann (2004), a buffer layer of <<nd>> grid points and a regularisation $B^2+\varepsilon^2$ ($\varepsilon=$ <<eps>>\,$B_{\max}$) near magnetic nulls<<pp>>. The optimisation, started from the potential field, was stopped after <<it>> iterations, the functional having decreased by a factor <<red>>. The solution has $\mathrm{CW}_{\sin}=$ <<cw>> ($\theta_J=$ <<th>>$^\circ$) and $\langle|f_i|\rangle=$ <<fi>> (Wheatland et al. 2000; Schrijver et al. 2006). Its magnetic energy is $E=$ <<E>>~erg, with $E/E_{\rm pot}=$ <<r>>, implying a free energy of <<free>>~erg. "),
                eps=num(study.nlfff.meta.get("eps", 0.0)), nd=A["nd"], pp=pp, it=c["iterations"], red=num(c["reduction"]), cw=num(q["CWsin"]),
                th=f"{q['theta_J_deg']:.1f}", fi=num(q["<|f_i|>"], 2), E=num(e["E"]), r=num(e["ratio"], 4),
                free=num(e["free"])))
    if A["reference"] == "exact" and A["vector"].get("nlfff"):
        v = A["vector"]["nlfff"]
        parts.append(fill(L(
            r"Frente a la solución exacta, el NLFFF alcanza $C_{\rm vec}=$ <<cv>>, $C_{\rm CS}=$ <<cs>>, $E'_n=$ <<en>> y $E'_m=$ <<em>> (Schrijver et al. 2006). ",
            r"Compared with the exact solution, the NLFFF reaches $C_{\rm vec}=$ <<cv>>, $C_{\rm CS}=$ <<cs>>, $E'_n=$ <<en>> and $E'_m=$ <<em>> (Schrijver et al. 2006). "),
            cv=num(v["C_vec"]), cs=num(v["C_CS"]), en=num(v["E'_n"]), em=num(v["E'_m"])))
    ph = A.get("physics") or {}
    if ph.get("twist") or ph.get("decay") or ph.get("q"):
        s_ = []
        if ph.get("twist"):
            s_.append(L(r"el número de torsión $T_w=(4\pi)^{-1}\int\alpha\,dl$ de las líneas ancladas en la base (Berger \& Prior 2006; Liu et al. 2016)",
                        r"the twist number $T_w=(4\pi)^{-1}\int\alpha\,dl$ of field lines rooted at the base (Berger \& Prior 2006; Liu et al. 2016)"))
        if ph.get("decay"):
            hc = ph["decay"]["h_crit"]
            s_.append(fill(L(r"el índice de decaimiento $n=-d\ln B_h/d\ln z$ del campo potencial (calculado en un dominio ampliado al doble) promediado sobre la PIL, que da una altura crítica $h_{\rm crit}=$ <<hc>>~Mm para $n=1.5$ (Kliem \& Török 2006)",
                             r"the decay index $n=-d\ln B_h/d\ln z$ of the potential field (computed in a domain enlarged by a factor of two) averaged over the PIL, giving a critical height $h_{\rm crit}=$ <<hc>>~Mm for $n=1.5$ (Kliem \& Török 2006)"),
                           hc=f"{hc:.1f}" if hc is not None and np.isfinite(hc) else "--"))
        if ph.get("q"):
            s_.append(L(r"el factor de squashing $Q$ en la base (Titov et al. 2002; Pariat \& Démoulin 2012)",
                        r"the squashing factor $Q$ at the base (Titov et al. 2002; Pariat \& Démoulin 2012)"))
        parts.append(L("Se calcularon además ", "We also computed ") + "; ".join(s_) + ". "
                     + L(r"Todas las herramientas se validaron frente a soluciones analíticas (Gold \& Hoyle 1960; Low \& Lou 1990). ",
                         r"All tools were validated against analytic solutions (Gold \& Hoyle 1960; Low \& Lou 1990). "))
    hel = (A.get("helicity") or {}).get("nlfff")
    if hel:
        parts.append(fill(L(
            r"La energía se descompuso según Valori et al. (2013) respecto al campo potencial con la misma componente normal en las seis caras, obteniendo $E_{\rm div}/E=$ <<ed>> y una energía libre de <<fv>>~erg. La helicidad relativa (Berger \& Field 1984; Finn \& Antonsen 1985) se calculó en el gauge de DeVore (DeVore 2000; Valori et al. 2012): $H_V=$ <<hv>>~Mx$^2$, con $|H_J|/|H_V|=$ <<r>> (Pariat et al. 2017) y una diferencia del <<gd>> entre gauges. ",
            r"The energy was decomposed following Valori et al. (2013) with respect to the potential field having the same normal component on all six faces, giving $E_{\rm div}/E=$ <<ed>> and a free energy of <<fv>>~erg. The relative helicity (Berger \& Field 1984; Finn \& Antonsen 1985) was computed in the DeVore gauge (DeVore 2000; Valori et al. 2012): $H_V=$ <<hv>>~Mx$^2$, with $|H_J|/|H_V|=$ <<r>> (Pariat et al. 2017) and a <<gd>> difference between gauges. "),
            ed=pct(hel["energy"]["E_div_frac"]), fv=num(hel["energy"]["E_free"]), hv=num(hel["helicity"]["H_V"]),
            r=num(hel["helicity"]["ratio_HJ_HV"], 2), gd=pct(hel["gauge_diff"])))
    Mt = A.get("methods")
    if Mt and "E_free" in Mt.get("spread", {}):
        S = Mt["spread"]
        parts.append(fill(L(
            r"Para estimar la incertidumbre de método se calcularon también las soluciones Grad--Rubin (Grad \& Rubin 1958; Amari et al. 2006; Wheatland 2007) a partir de $\alpha$ en la polaridad positiva (P) y negativa (N), con los mismos datos y dominio; la energía libre varía entre <<a>> y <<b>>~erg y $H_V$ entre <<c>> y <<d>>~Mx$^2$ según el método. ",
            r"To estimate the method uncertainty, Grad--Rubin solutions (Grad \& Rubin 1958; Amari et al. 2006; Wheatland 2007) were also computed from $\alpha$ in the positive (P) and negative (N) polarities with the same data and domain; the free energy ranges from <<a>> to <<b>>~erg and $H_V$ from <<c>> to <<d>>~Mx$^2$ depending on the method. "),
            a=num(S["E_free"]["min"]), b=num(S["E_free"]["max"]), c=num(S["H_V"]["min"]), d=num(S["H_V"]["max"])))

    def _pm(d):
        """'$a \\pm b$' from the median and the half-width of the 16-84 % interval."""
        if not d:
            return "--"
        return ("$" + num(d["p50"]).strip("$") + r"\pm" + num((d["p84"] - d["p16"]) / 2).strip("$") + "$")

    M = A.get("mc")
    if M:
        sm = M["summary"]
        parts.append(fill(L(
            r"Las incertidumbres debidas al ruido de medida se estimaron con un Monte Carlo de <<n>> realizaciones, añadiendo al magnetograma ruido gaussiano según los mapas de error de HMI<<sc>> y repitiendo el preprocesado y la extrapolación: la energía libre respecto al potencial de volumen resulta <<efv>>~erg y $H_V=$ <<hv>>~Mx$^2$ (mediana y semiancho del intervalo 16--84\%). La energía libre respecto al potencial FFT (<<ef>>~erg) es más sensible al ruido: la mediana del Monte Carlo difiere del valor nominal en un <<bias>>, que se trata como incertidumbre sistemática; la corrección \emph{bootstrap} de ese sesgo no se aplica porque, en pruebas con campos exactos, empeora el resultado con ruido de tipo HMI. ",
            r"Uncertainties due to measurement noise were estimated with a Monte Carlo of <<n>> realisations, adding Gaussian noise to the magnetogram according to the HMI error maps<<sc>> and repeating the preprocessing and extrapolation: the free energy relative to the volume potential is <<efv>>~erg and $H_V=$ <<hv>>~Mx$^2$ (median and half-width of the 16--84\% interval). The free energy relative to the FFT potential (<<ef>>~erg) is more sensitive to noise: the Monte Carlo median differs from the nominal value by <<bias>>, which is treated as a systematic uncertainty; the \emph{bootstrap} correction of this bias is not applied because, in tests with exact fields, it worsens the result for HMI-like noise. "),
            n=M["n"], sc=(f" ($\\times${M['error_scale']:g})" if M["error_scale"] != 1 else ""),
            ef=_pm(sm.get("E_free")), efv=_pm(sm.get("E_free_vol")), hv=_pm(sm.get("H_V")),
            bias=(pct(sm["E_free"]["p50"] / sm["E_free"]["nominal"] - 1, 0)
                  if sm.get("E_free", {}).get("nominal") else "--")))
    B = A.get("budget")
    if B and B["rows"]:
        from ..budget import magnitude
        rw = {r["key"]: r for r in B["rows"]}
        pre = magnitude((rw.get("preprocessing") or {}).get("values", {}).get("E_free"))
        dom = B["dominant"].get("E_free_vol")
        lab = {"preprocessing": ("del preprocesado", "the preprocessing"),
               "method_exact": ("del propio método", "the method itself"),
               "buffer": ("de la capa de amortiguamiento", "the buffer layer"),
               "resolution": ("de la resolución", "the resolution"),
               "nlfff_params": ("de los parámetros del NLFFF", "the NLFFF parameters"),
               "noise_random": ("del ruido de medida", "measurement noise"),
               "noise_bias": ("del sesgo por ruido", "the noise bias"),
               "method_gr": ("del método", "the method")}
        if dom and dom not in ("noise_random", "noise_bias"):
            parts.append(fill(L(
                r"El ruido de medida no es la fuente de incertidumbre dominante. En pruebas con campos exactos, variando una a una las elecciones del modelo, la mayor contribución a la energía libre respecto al potencial de volumen procede <<d>>, y cambiar los pesos del preprocesado un orden de magnitud altera la energía libre respecto al potencial FFT hasta en un <<p>> (presupuesto completo en la Tabla~\ref{tab:budget}). ",
                r"Measurement noise is not the dominant source of uncertainty. In tests with exact fields, varying the modelling choices one at a time, the largest contribution to the free energy relative to the volume potential comes from <<d>>, and changing the preprocessing weights by an order of magnitude alters the free energy relative to the FFT potential by up to <<p>> (full budget in Table~\ref{tab:budget}). "),
                d=L(*lab.get(dom, (dom, dom))), p=pct(pre, 0) if pre is not None else "--"))
    R = A.get("reliability")
    if R:
        from ..confidence import calibration_metrics, select_calibration
        use_mc = "mc_instability" in (R.get("calibration") or "")
        CMp = calibration_metrics(select_calibration(("mc_instability",) if use_mc else ()))
        mc_es = (" y su estabilidad en el Monte Carlo" if use_mc else "")
        mc_en = (" and its stability in the Monte Carlo" if use_mc else "")
        cases = CMp.get("n_cases", "--")
        parts.append(fill(L(
            r"La fiabilidad de cada línea de campo se estimó combinando, mediante una regresión logística, el ángulo J--B, la variación de $\alpha$ a lo largo de la línea, su paso por la capa de amortiguamiento, su altura máxima y su longitud (con un término cuadrático en su logaritmo)<<mces>>. La regresión se calibró con <<nc>> casos de campos libres de fuerzas exactos (tres variantes de Low \& Lou \citep{LowLou1990}, dos campos lineales \citep{Alissandrakis1981} y la cuerda de flujo de \citet{TitovDemoulin1999} relajada a un equilibrio, con tres modelos de ruido y varias resoluciones); en validación cruzada dejando fuera cada familia de campos alcanza AUC $=$ <<auc>> y un error de calibración ECE $=$ <<ece>>. Con el umbral de confianza <<thr>>, el <<fr>> de las líneas resultó fiable. ",
            r"The reliability of each field line was estimated by combining, through a logistic regression, the J--B angle, the variation of $\alpha$ along the line, its passage through the buffer layer, its apex height and its length (with a quadratic term in its logarithm)<<mcen>>. The regression was calibrated on <<nc>> cases of exact force-free fields (three Low \& Lou variants \citep{LowLou1990}, two linear fields \citep{Alissandrakis1981} and the \citet{TitovDemoulin1999} flux rope relaxed to an equilibrium, with three noise models and several resolutions); in cross-validation leaving out each field family it reaches AUC $=$ <<auc>> and a calibration error ECE $=$ <<ece>>. With a confidence threshold of <<thr>>, <<fr>> of the lines were found reliable. "),
            auc=num(CMp["auc_combined"], 2), ece=num(CMp.get("ece"), 2), nc=cases,
            mces=mc_es, mcen=mc_en, thr=num(R["threshold"], 2),
            fr=pct(R["frac_reliable"], 0)))
    G = A.get("aia")
    if G:
        ws = sorted(G["by_wavelength"], key=int)
        parts.append(fill(L(
            r"Las líneas se compararon con los bucles de las imágenes de SDO/AIA en <<wl>>~Å más cercanas a la observación, midiendo el ángulo entre cada línea proyectada y la orientación local de las crestas brillantes de la imagen \citep[como][]{Warren2018}. ",
            r"The lines were compared with the loops in the SDO/AIA <<wl>>~Å images closest to the observation, by measuring the angle between each projected line and the local orientation of the bright ridges in the image \citep[as in][]{Warren2018}. "),
            wl=L(" y ", " and ").join(ws)))
    from ..utils import code_version
    cv = (study.config.get("code_version") or code_version())["label"]
    parts.append(fill(L(r"Los cálculos se realizaron con \texttt{solarff} <<v>>.",
                        r"Computations were performed with \texttt{solarff} <<v>>."), v=esc(cv)))
    return "".join(parts)


def build_tex(study, A, figs, lang, fig_rel, title=None, validation=None) -> str:
    return ReportBuilder(study, A, figs, lang, fig_rel, title, validation).build()
