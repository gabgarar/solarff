"""Monte Carlo uncertainties with the HMI measurement errors.

The full extrapolation (preprocessing + NLFFF) is repeated on ``n``
realizations of the magnetogram, each with added Gaussian noise of
standard deviation equal to the measurement error of each pixel (``*_err`` segments of
SHARP), multiplied by ``error_scale``. The spread of the results gives
the uncertainty due to the noise of the data (not that of the method).

The **stability of each field line** is measured too: the fraction of
realizations in which the line traced from the same seed still
matches that of the nominal model (deviation < 10 % of its length and
same topology). An unstable line depends on the noise and is not reliable.

The summary includes ``bias`` (Monte Carlo median − nominal) and
``corrected`` (2·nominal − median, the bootstrap correction). The validation
with exact fields (``examples/05_bias_validation.py``) shows that this
correction helps with isotropic noise but **worsens** the free energy with
HMI-like noise (transverse ≫ longitudinal), so the report does not
apply it: the bias is given as a systematic uncertainty.

Limitations: the formal errors of the HMI inversion underestimate the real
uncertainty (they include neither systematics nor errors of the 180°
ambiguity) and are treated as independent between pixels and components.
"""

from __future__ import annotations

import time

import numpy as np

from .confidence import line_errors
from .field import VectorField3D
from .magnetogram import Magnetogram
from .metrics import energy, force_free_metrics
from .potential import potential_field
from .utils import code_version, keeps_awake, load_checkpoint, save_checkpoint

QUANTITIES = ("E", "E_pot", "E_free", "E_ratio", "E_free_vol", "E_div_frac", "H_V", "H_J",
              "HJ_HV", "gauge_diff", "CWsin", "fi", "h_crit", "tw_p99", "tw_frac_gt1")


def perturb(mag: Magnetogram, rng: np.random.Generator, error_scale: float = 1.0,
            noise: float | None = None) -> Magnetogram:
    """One realization: mag + N(0, σ) in each component (σ = pixel error or ``noise``)."""
    def sig(e, b):
        if noise is not None:
            return np.full_like(b, noise)
        if e is None:
            raise ValueError("El magnetograma no tiene mapas de error: pasa noise=<G> o "
                             "descarga los segmentos *_err (download_sharp los incluye por defecto)")
        return e
    bz = mag.bz + rng.normal(0, 1, mag.shape) * sig(mag.bz_err, mag.bz) * error_scale
    bx = mag.bx + rng.normal(0, 1, mag.shape) * sig(mag.bx_err, mag.bx) * error_scale
    by = mag.by + rng.normal(0, 1, mag.shape) * sig(mag.by_err, mag.by) * error_scale
    out = Magnetogram(bz=bz - bz.mean() + mag.bz.mean(), bx=bx, by=by, dx=mag.dx,
                      origin=mag.origin, meta=dict(mag.meta),
                      bx_err=mag.bx_err, by_err=mag.by_err, bz_err=mag.bz_err)
    return out


def base_seeds(field: VectorField3D, step: int, threshold: float = 0.05) -> np.ndarray:
    """Seeds at the base pixels with |Bz| > ``threshold``·max (as in the confidence map)."""
    bz = field.bz[:, :, 0]
    ix = np.arange(0, field.shape[0], step)
    iy = np.arange(0, field.shape[1], step)
    IX, IY = np.meshgrid(ix, iy, indexing="ij")
    strong = np.abs(bz[IX, IY]) > threshold * np.abs(bz).max()
    x0, y0, z0 = field.origin
    return np.column_stack([x0 + IX[strong] * field.dx, y0 + IY[strong] * field.dx,
                            np.full(strong.sum(), z0 + 0.5 * field.dx)])


def _quantities(model, pot, mag_k, nz, nd, seeds):
    from .helicity import helicity_report
    from .stability import decay_index_profile, pil_mask
    from .topology import twist_number
    from .tracing import trace_field_lines
    q = {}
    e, ep = energy(model, nd), energy(pot, nd)
    q.update(E=e, E_pot=ep, E_free=e - ep, E_ratio=e / ep)
    ff = force_free_metrics(model, nd)
    q.update(CWsin=ff["CWsin"], fi=ff["<|f_i|>"])
    hr = helicity_report(model)
    q.update(E_free_vol=hr["energy"]["E_free"], E_div_frac=hr["energy"]["E_div_frac"],
             H_V=hr["helicity"]["H_V"], H_J=hr["helicity"]["H_J"],
             HJ_HV=hr["helicity"]["ratio_HJ_HV"], gauge_diff=hr["gauge_diff"])
    q["h_crit"] = np.nan
    if pil_mask(mag_k.bz).any():
        q["h_crit"] = decay_index_profile(potential_field(mag_k, nz, pad=2))["h_crit"]
    lines = trace_field_lines(model, seeds)
    tw = np.abs(twist_number(model, lines))
    closed = np.array([l.closed for l in lines])
    v = tw[closed & np.isfinite(tw)]
    q["tw_p99"] = float(np.percentile(v, 99)) if v.size else np.nan
    q["tw_frac_gt1"] = float((v >= 1).mean()) if v.size else np.nan
    return q, lines


@keeps_awake
def monte_carlo(study, n: int = 10, error_scale: float = 1.0, noise: float | None = None,
                seed: int = 0, step: int | None = None, checkpoint: str | None = None,
                verbose: bool = True) -> dict:
    """Monte Carlo of the NLFFF of ``study`` (see the module docstring).

    **Parameters**

    - ``study``: result of ``solarff.run_study`` (its configuration is used).
    - ``n``: number of realizations (each costs the same as the nominal NLFFF).
    - ``error_scale``: factor on the HMI errors (>1 to account for the fact that
        the formal errors underestimate the real uncertainty).
    - ``noise``: alternative to the error maps: uniform σ in G.
    - ``step``: step of the base seeds (by default ~80 along the longer side).
    - ``checkpoint``: file where the progress is saved after each realization. If
        it exists (and corresponds to the same configuration and the same study), it
        resumes from there. Each realization uses its own seed (seed, k), so
        the result is identical with or without interruptions.

    Returns a dict with: ``values`` (realization × quantity), ``nominal``,
    ``summary`` (mean, standard deviation, 16/50/84 percentiles of each quantity),
    ``stability`` (per line), ``seeds`` and the configuration.
    """
    from .nlfff import nlfff
    from .preprocess import preprocess as _pp
    say = print if verbose else (lambda *a, **k: None)
    if study.nlfff is None:
        raise ValueError("El estudio no tiene NLFFF")
    cfg = study.config
    nz = study.nz
    mag = study.mag
    nd = int(study.nlfff.meta.get("nd", 4))
    step = step or max(1, int(round(max(mag.shape) / 80)))
    seeds = base_seeds(study.nlfff, step)
    method = cfg.get("nlfff_method", "lbfgs")
    nkw = dict(cfg.get("nlfff_kw") or {})
    nkw.pop("verbose", None)

    def boundary(m):
        return _pp(m, verbose=False, **(cfg.get("preprocess_kw") or {})) if cfg.get("preprocess", True) else m

    def run_nlfff(src):
        if method == "lbfgs":
            from .nlfff_torch import nlfff_lbfgs
            return nlfff_lbfgs(src, nz, verbose=False, **nkw)
        return nlfff(src, nz, verbose=False, **nkw)

    t0 = time.time()
    from .study import reference_potential
    nominal, lines_nom = _quantities(study.nlfff, reference_potential(study), mag, nz, nd, seeds)
    signature = {"n": n, "error_scale": error_scale, "noise": noise, "seed": seed, "step": step,
                 "method": method, "nominal_E": round(float(nominal["E"]), -25), "shape": mag.shape, "nz": nz}
    values = {k: [] for k in QUANTITIES}
    good = np.zeros((n, len(seeds)), dtype=bool)
    start = 0
    state = load_checkpoint(checkpoint, signature)
    if state is not None:
        values, good, start = state["values"], state["good"], state["done"]
        say(f"  Monte Carlo: reanudando desde {checkpoint} ({start}/{n} realizaciones hechas)")
    for k in range(start, n):
        tk = time.time()
        rng = np.random.default_rng([seed, k])  # own seed: resuming does not change the result
        mk = perturb(mag, rng, error_scale, noise)
        src = boundary(mk)  # the same magnetogram for the NLFFF and for its reference potential
        model = run_nlfff(src)
        pot = potential_field(src, nz)
        q, lines_k = _quantities(model, pot, mk, nz, nd, seeds)
        for key in QUANTITIES:
            values[key].append(q[key])
        err = line_errors(lines_k, lines_nom)
        good[k] = (err["shape"] <= 0.1) & (err["topology"] == 0)
        say(f"  Monte Carlo {k + 1}/{n}: E_free = {q['E_free']:.3e} erg, H_V = {q['H_V']:.3e} Mx², "
            f"líneas estables {good[k].mean():.0%} ({time.time() - tk:.0f}s)", flush=True)
        if checkpoint:
            save_checkpoint(checkpoint, {"signature": signature, "values": values, "good": good, "done": k + 1})
    values = {k: np.array(v, dtype=float) for k, v in values.items()}
    summary = {}
    for k, v in values.items():
        f = v[np.isfinite(v)]
        if f.size:
            p16, p50, p84 = np.percentile(f, [16, 50, 84])
            nom = float(nominal[k]) if np.isfinite(nominal[k]) else None
            d = {"mean": float(f.mean()), "std": float(f.std(ddof=1)) if f.size > 1 else 0.0,
                 "p16": float(p16), "p50": float(p50), "p84": float(p84), "nominal": nom}
            if nom is not None:
                # Bootstrap bias: the added noise shifts the result the way the real
                # noise shifted the nominal value with respect to the noise-free one.
                d["bias"] = float(p50 - nom)
                d["corrected"] = float(2 * nom - p50)
                d["outside"] = bool(nom < p16 or nom > p84)
            summary[k] = d
    stability = good.mean(axis=0)
    say(f"  Monte Carlo terminado: {n} realizaciones en {time.time() - t0:.0f}s")
    return {"values": values, "nominal": nominal, "summary": summary, "stability": stability,
            "seeds": seeds, "lines_nominal": lines_nom, "step": step,
            "config": {"n": n, "error_scale": error_scale, "noise": noise, "seed": seed,
                       "source": "noise" if noise is not None else "HMI *_err",
                       "code_version": code_version()}}


def load_bias_validation(path: str | None = None) -> dict | None:
    """Summary of the validation of the bias correction with exact fields
    (``examples/05_bias_validation.py``), or None if it does not exist."""
    import json
    import os
    path = path or os.path.join(os.path.dirname(__file__), "bias_validation.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
