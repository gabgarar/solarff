"""Full extrapolation workflow with everything the report needs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .alpha import alpha_best_fit, alpha_from_jz
from .field import VectorField3D
from .lfff import alpha_max, lfff
from .magnetogram import Magnetogram
from .nlfff import nlfff
from .potential import potential_field
from .preprocess import preprocess as _preprocess
from .tracing import seeds_from_magnetogram
from .utils import code_version, keeps_awake


@dataclass
class Study:
    """Result of ``run_study``: data, models, parameters and diagnostics.

    Passed as is to ``solarff.generate_report``.
    """

    mag: Magnetogram
    nz: int
    potential: VectorField3D
    lfff: VectorField3D | None = None
    nlfff: VectorField3D | None = None
    exact: VectorField3D | None = None
    mag_pp: Magnetogram | None = None
    alpha: dict = field(default_factory=dict)
    nlfff_history: list = field(default_factory=list)
    nlfff_converged: bool = False
    nlfff_terms: list = field(default_factory=list)  # L-BFGS: (it, L, force+div, data)
    nlfff_message: str = ""
    seeds: np.ndarray | None = None
    timings: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    name: str = ""
    grad_rubin: dict = field(default_factory=dict)  # {"P": VectorField3D, "N": VectorField3D}

    def models(self, names: dict | None = None) -> dict:
        """Available models in order (exact, potential, LFFF, NLFFF).

        ``names`` allows renaming, e.g. ``{"exact": "Exact", "potential": "Potential"}``.
        """
        names = {"exact": "Exacto", "potential": "Potencial", "lfff": "LFFF",
                 "nlfff": "NLFFF", **(names or {})}
        out = {}
        for key in ("exact", "potential", "lfff", "nlfff"):
            f = getattr(self, key)
            if f is not None:
                out[names[key]] = f
        return out


@keeps_awake
def run_study(mag: Magnetogram, nz: int, *, exact: VectorField3D | None = None,
              preprocess: bool = True, preprocess_kw: dict | None = None,
              alpha: float | None = None, alpha_method: str = "fit",
              nlfff_kw: dict | None = None, nlfff_method: str = "auto", n_seeds: int = 80,
              name: str = "", verbose: bool = True) -> Study:
    """Run potential, LFFF and NLFFF on ``mag`` and store the diagnostics.

    **Parameters**

    - ``mag``: magnetogram (already cropped / rebinned). If it only has Bz, the
        potential is computed and, if ``alpha`` is given, the LFFF; the NLFFF requires a vector.
    - ``nz``: levels in height.
    - ``exact``: exact solution (benchmarks) to compare against.
    - ``preprocess``: apply Wiegelmann's preprocessing before the NLFFF.
    - ``alpha``: α for the LFFF; if None it is estimated with ``alpha_method``.
    - ``nlfff_method``: ``"lbfgs"`` (exact gradient with PyTorch, relaxed base;
        recommended), ``"classic"`` (Wheatland/Wiegelmann with fixed base, no
        dependencies) or ``"auto"`` (lbfgs if PyTorch is installed).
    - ``nlfff_kw``: extra arguments for the chosen NLFFF method.
    """
    t = {}
    say = print if verbose else (lambda *a, **k: None)
    cfg = {"nz": nz, "preprocess": preprocess, "preprocess_kw": preprocess_kw or {},
           "alpha_method": alpha_method, "nlfff_kw": nlfff_kw or {}, "n_seeds": n_seeds,
           "alpha_user": alpha, "nlfff_method": nlfff_method, "code_version": code_version()}

    t0 = time.time()
    say("[1/4] Campo potencial")
    pot = potential_field(mag, nz)
    t["potential"] = time.time() - t0

    # --- α ---
    amax = alpha_max(mag)
    ainfo = {"max": amax, "fit": None, "jz": None, "curve": None, "method": alpha_method}
    if mag.has_vector:
        best, grid, costs = alpha_best_fit(mag, return_curve=True)
        ainfo.update(fit=best, curve=(grid, costs))
        try:
            ainfo["jz"] = alpha_from_jz(mag)
        except ZeroDivisionError:
            ainfo["jz"] = None
    if alpha is not None:
        used, ainfo["method"] = float(alpha), "user"
    elif alpha_method == "jz" and ainfo["jz"] is not None:
        used = ainfo["jz"]
    else:
        used = ainfo["fit"]
    lin = None
    if used is not None:
        clipped = abs(used) > 0.98 * amax
        used_c = float(np.clip(used, -0.98 * amax, 0.98 * amax))
        ainfo.update(raw=used, used=used_c, clipped=clipped)
        say(f"[2/4] LFFF con alpha = {used_c:.4g} Mm^-1 (alpha_max = {amax:.4g})")
        t0 = time.time()
        lin = lfff(mag, used_c, nz)
        t["lfff"] = time.time() - t0
    else:
        say("[2/4] LFFF omitido: no hay campo transversal para estimar alpha")

    # --- NLFFF ---
    nl, mag_pp, hist, conv, msg = None, None, [], False, ""
    if mag.has_vector:
        if preprocess:
            say("[3/4] Preprocesado")
            t0 = time.time()
            mag_pp = _preprocess(mag, verbose=verbose, **(preprocess_kw or {}))
            t["preprocess"] = time.time() - t0
        method = nlfff_method
        if method == "auto":
            try:
                import torch  # noqa: F401
                method = "lbfgs"
            except ImportError:
                method = "classic"
        cfg["nlfff_method"] = method
        say(f"[4/4] NLFFF ({method})")
        t0 = time.time()
        src = mag_pp if mag_pp is not None else mag
        if method == "lbfgs":
            from .nlfff_torch import nlfff_lbfgs
            res = nlfff_lbfgs(src, nz, return_result=True, verbose=verbose, **(nlfff_kw or {}))
            terms = res.terms
        else:
            res = nlfff(src, nz, return_result=True, verbose=verbose, **(nlfff_kw or {}))
            terms = []
        t["nlfff"] = time.time() - t0
        nl, hist, conv, msg = res.field, res.history, res.converged, res.message
    else:
        say("[3-4/4] NLFFF omitido: requiere magnetograma vectorial")

    seeds = seeds_from_magnetogram(mag, n=n_seeds)
    return Study(mag=mag, nz=nz, potential=pot, lfff=lin, nlfff=nl, exact=exact,
                 mag_pp=mag_pp, alpha=ainfo, nlfff_history=hist, nlfff_converged=conv,
                 nlfff_terms=terms if mag.has_vector else [],
                 nlfff_message=msg, seeds=seeds, timings=t, config=cfg,
                 name=name or str(mag.meta.get("source", "")))


def reference_potential(study: Study) -> VectorField3D:
    """Reference potential field for the free energy: that of the SAME magnetogram used by the
    NLFFF (the preprocessed one, if there was one). E − E_pot is a free energy only if both fields
    share B_z at the base (Thomson's theorem); with the potential of the unpreprocessed
    observation the effect of the preprocessing would be mixed in (revision R13). ``study.potential`` (of the
    observation) is still used for the decay index and the comparisons with AIA."""
    pot = getattr(study, "_reference_potential", None)
    if pot is None:
        src = study.mag_pp if study.mag_pp is not None else study.mag
        pot = potential_field(src, study.nz) if src is not study.mag else study.potential
        try:
            study._reference_potential = pot
        except AttributeError:
            pass
    return pot


@keeps_awake
def add_grad_rubin(study: Study, polarities=("P", "N"), verbose: bool = True, **kw) -> Study:
    """Compute the Grad–Rubin solutions of ``study`` and store them in ``study.grad_rubin``.

    Uses the same magnetogram (preprocessed if there was one), the same height and the
    same potential field as the NLFFF, so that the comparison only reflects
    the difference of method.
    """
    from .grad_rubin import nlfff_grad_rubin
    src = study.mag_pp if study.mag_pp is not None else study.mag
    if not src.has_vector:
        raise ValueError("Grad–Rubin necesita un magnetograma vectorial")
    if not hasattr(study, "grad_rubin") or study.grad_rubin is None:
        study.grad_rubin = {}
    for pol in polarities:
        t0 = time.time()
        if verbose:
            print(f"[Grad-Rubin {pol}]")
        study.grad_rubin[pol] = nlfff_grad_rubin(src, study.nz, polarity=pol, initial=study.potential,
                                                 verbose=verbose, **kw)
        study.timings[f"grad_rubin_{pol}"] = time.time() - t0
    return study
