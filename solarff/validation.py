"""Validation suite (the test names are in LaTeX notation) against exact analytical solutions.

Each test reproduces a field whose answer is known exactly and measures the
error of the corresponding tool of the library. The report includes
these results in an appendix, so that every analysis documents that the
tools used reproduce the exact solutions.
"""

from __future__ import annotations

import time

import numpy as np

from . import analytic as an
from .lfff import lfff, alpha_max
from .magnetogram import Magnetogram
from .metrics import compare_vector_fields
from .nlfff import nlfff
from .potential import potential_field
from .stability import decay_index_profile
from .topology import squashing_q, twist_map


def _test(key, name_es, name_en, expected, obtained, error, tol, unit_es, unit_en, **extra):
    return {"key": key, "name_es": name_es, "name_en": name_en, "expected": expected,
            "obtained": obtained, "error": float(error), "tol": tol, "passed": bool(error <= tol),
            "error_es": unit_es, "error_en": unit_en, **extra}


def run_validation(quick: bool = False, verbose: bool = False) -> list[dict]:
    """Run all the analytical tests and return a list of results.

    ``quick=True`` uses smaller grids (for unit tests).
    """
    say = print if verbose else (lambda *a, **k: None)
    out = []

    # 1. Potential by FFT against the exact dipole
    t0 = time.time()
    n = 40 if quick else 64
    dip = an.dipole_field(nx=n, ny=n, nz=n // 2, half_width=50, depth=10)
    mag = Magnetogram(bz=dip.bz[:, :, 0], dx=dip.dx, origin=dip.origin[:2])
    pot = potential_field(mag, nz=n // 2, pad=2)
    m = n // 8
    cv = compare_vector_fields(dip, pot, margin=m)["C_vec"]
    out.append(_test("potential", "Campo potencial (FFT) frente a dipolo exacto",
                     "Potential field (FFT) versus exact dipole", 1.0, cv, 1 - cv, 0.02,
                     "1 − C_vec", "1 − C_vec", seconds=time.time() - t0))
    say(f"potential C_vec={cv:.4f}")

    # 2. LFFF: ∇×B = αB
    t0 = time.time()
    n = 64 if quick else 96
    x = np.arange(n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    c, w, d = n // 2, (n / 6.4) ** 2, n // 7
    bz = 1000 * (np.exp(-((X - c + d) ** 2 + (Y - c) ** 2) / w) - np.exp(-((X - c - d) ** 2 + (Y - c) ** 2) / w))
    bip = Magnetogram(bz=bz, dx=1.0)
    a = 0.7 * alpha_max(bip)
    f = lfff(bip, a, nz=24)
    j = f.current()
    res = max(np.abs(j[i] - a * cc)[3:-3, 3:-3, 1:-2].max() for i, cc in enumerate((f.bx, f.by, f.bz)))
    rel = res / (abs(a) * f.babs.max())
    out.append(_test("lfff", r"LFFF: residuo de $\nabla\times\mathbf{B}=\alpha\mathbf{B}$", r"LFFF: residual of $\nabla\times\mathbf{B}=\alpha\mathbf{B}$",
                     0.0, rel, rel, 0.05, "residuo relativo máx.", "max. relative residual",
                     seconds=time.time() - t0))
    say(f"lfff residual={rel:.4f}")

    # 3. NLFFF against Low & Lou
    t0 = time.time()
    n, nz = (24, 16) if quick else (32, 24)
    mag, exact = an.low_lou_magnetogram(nx=n, ny=n, nz=nz)
    nl = nlfff(mag, nz=nz, max_iter=300 if quick else 1500, verbose=False)
    cvn = compare_vector_fields(exact, nl, margin=nl.meta["nd"])["C_vec"]
    out.append(_test("nlfff", r"NLFFF frente a Low \& Lou (1990)", r"NLFFF versus Low \& Lou (1990)",
                     1.0, cvn, 1 - cvn, 0.03, "1 − C_vec", "1 − C_vec",
                     grid=f"{n}x{n}x{nz}", seconds=time.time() - t0))
    say(f"nlfff C_vec={cvn:.4f}")

    # 3b. NLFFF with exact gradient (L-BFGS, relaxed base) on Low & Lou with noise
    try:
        import torch  # noqa: F401
        from .nlfff_torch import nlfff_lbfgs
        from .preprocess import preprocess
        t0 = time.time()
        n, nz = (24, 16) if quick else (32, 24)
        mag, exact = an.low_lou_magnetogram(noise=30, nx=n, ny=n, nz=nz)
        nl = nlfff_lbfgs(preprocess(mag, verbose=False), nz=nz, max_iter=300 if quick else 1500, verbose=False)
        cvl = compare_vector_fields(exact, nl, margin=nl.meta["nd"])["C_vec"]
        out.append(_test("nlfff_lbfgs", r"NLFFF L-BFGS (base relajada) frente a Low \& Lou con 30~G de ruido",
                         r"L-BFGS NLFFF (relaxed base) versus Low \& Lou with 30~G noise",
                         1.0, cvl, 1 - cvl, 0.07 if quick else 0.04, "1 − C_vec", "1 − C_vec",
                         grid=f"{n}x{n}x{nz}", seconds=time.time() - t0))
        say(f"nlfff_lbfgs C_vec={cvl:.4f}")
    except ImportError:
        pass

    # 3b'. Grad–Rubin against Low & Lou (P and N solutions)
    from .grad_rubin import nlfff_grad_rubin
    t0 = time.time()
    n, nz = (32, 22) if quick else (40, 28)
    mag, exact = an.low_lou_magnetogram(nx=n, ny=n, nz=nz)
    for pol in ("P", "N"):
        g = nlfff_grad_rubin(mag, nz, polarity=pol, verbose=False)
        cg = compare_vector_fields(exact, g, margin=g.meta["nd"])["C_vec"]
        out.append(_test(f"grad_rubin_{pol.lower()}",
                         rf"Grad--Rubin ({pol}) frente a Low \& Lou (1990)", rf"Grad--Rubin ({pol}) versus Low \& Lou (1990)",
                         1.0, cg, 1 - cg, 0.04, "1 − C_vec", "1 − C_vec", grid=f"{n}x{n}x{nz}",
                         seconds=time.time() - t0))
        say(f"grad-rubin {pol} C_vec={cg:.4f}")

    # 3c. Per-line confidence: generalisation to a new noise realization
    try:
        import torch  # noqa: F401
        from .confidence import assess_lines, line_errors, auc
        from .nlfff_torch import nlfff_lbfgs
        from .preprocess import preprocess
        from .tracing import seeds_from_magnetogram, trace_field_lines
        t0 = time.time()
        n, nz = (32, 22) if quick else (40, 28)
        mag, exact = an.low_lou_magnetogram(noise=30, rng=77, nx=n, ny=n, nz=nz)
        pp = preprocess(mag, verbose=False)
        nl = nlfff_lbfgs(pp, nz, max_iter=400 if quick else 2000, verbose=False)
        seeds = seeds_from_magnetogram(mag, n=150 if quick else 300, method="random",
                                       polarity="both", threshold=0.05, rng=5)
        lm, le = trace_field_lines(nl, seeds), trace_field_lines(exact, seeds)
        a = assess_lines(nl, lm, mag=pp, noise_bt=30)
        e = line_errors(lm, le)
        bad = (e["shape"] > 0.1) | (e["topology"] > 0)
        A_ = auc(-a["confidence"], bad)
        acc = a["reliable"]
        out.append(_test("confidence", r"Confianza por línea: AUC en una realización de ruido nueva",
                         r"Line confidence: AUC on a new noise realisation",
                         1.0, A_, 1 - A_, 0.25 if quick else 0.2, "1 − AUC", "1 − AUC",
                         threshold=a["threshold"], bad_all=float(bad.mean()),
                         bad_accepted=float(bad[acc].mean()) if acc.any() else float("nan"),
                         accepted=float(acc.mean()), seconds=time.time() - t0,
                         curve={"conf": a["confidence"], "err": e["shape"], "bad": bad}))
        say(f"confidence AUC={A_:.3f}")
    except ImportError:
        pass

    # 3d. Energy and helicity
    from .helicity import poisson_neumann, helicity_report
    t0 = time.time()
    shape, hh = (17, 13, 11), 0.5
    xs = [hh * np.arange(k) for k in shape]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    phi = 0.5 * X ** 2 - 0.25 * Y ** 2 + 0.3 * Y * Z - 0.25 * Z ** 2 + 0.1 * X * Z
    gx, gy, gz = X + 0.1 * Z, -0.5 * Y + 0.3 * Z, 0.3 * Y - 0.5 * Z + 0.1 * X
    sol = poisson_neumann(np.zeros(shape), hh, {"x0": gx[0], "x1": gx[-1], "y0": gy[:, 0], "y1": gy[:, -1],
                                               "z0": gz[:, :, 0], "z1": gz[:, :, -1]})
    sol += (phi - sol).mean()
    err = float(np.abs(sol - phi).max() / np.abs(phi).max())
    out.append(_test("neumann", r"Potencial de volumen: Laplace con Neumann en las 6 caras (solución cuadrática)",
                     r"Volume potential: Laplace with Neumann on all 6 faces (quadratic solution)",
                     0.0, err, err, 1e-10, "error relativo máx.", "max. relative error", seconds=time.time() - t0))
    t0 = time.time()
    dip = an.dipole_field(nx=32 if quick else 40, ny=32 if quick else 40, nz=24, half_width=50, depth=10)
    r = helicity_report(dip)
    from .field import trapezoid_weights
    b0 = np.abs(dip.bz[:, :, 0])
    flux = float((trapezoid_weights(b0.shape) * b0).sum() * (dip.dx * 1e8) ** 2 / 2)
    hn = abs(r["helicity"]["H_V"]) / flux ** 2
    out.append(_test("helicity_potential", r"Helicidad de un campo potencial (dipolo): debe ser 0",
                     r"Helicity of a potential field (dipole): must be 0",
                     0.0, hn, hn, 1e-6, r"$|H_V|/\Phi^2$", r"$|H_V|/\Phi^2$", seconds=time.time() - t0))
    t0 = time.time()
    ll = an.low_lou_field(nx=40 if quick else 48, ny=40 if quick else 48, nz=28 if quick else 32)
    r = helicity_report(ll)
    out.append(_test("helicity_gauge", r"Invariancia de gauge de $H_V$ (Low \& Lou, gauges de DeVore desde arriba y desde abajo)",
                     r"Gauge invariance of $H_V$ (Low \& Lou, DeVore gauges from the top and the bottom)",
                     r["helicity"]["H_V"], r["helicity_bottom_gauge"]["H_V"], r["gauge_diff"], 0.05,
                     "diferencia relativa", "relative difference", seconds=time.time() - t0))
    out.append(_test("e_div", r"Energía no solenoidal de la solución exacta de Low \& Lou ($E_{\rm div}/E$)",
                     r"Non-solenoidal energy of the exact Low \& Lou solution ($E_{\rm div}/E$)",
                     0.0, r["energy"]["E_div_frac"], r["energy"]["E_div_frac"], 0.02,
                     "fracción", "fraction", seconds=0.0))
    say(f"helicity gauge diff={r['gauge_diff']:.3f}")

    # 4. Tw against the Gold-Hoyle tube
    t0 = time.time()
    n = 32 if quick else 48
    b = 0.3
    gh = an.gold_hoyle_field(nx=n, ny=n, nz=n, half_width=20, b=b)
    tm = twist_map(gh, step=1 if not quick else 2)
    XX, YY = np.meshgrid(tm["x"], tm["y"], indexing="ij")
    r = np.hypot(XX, YY)
    H = gh.z[-1]
    ex = an.gold_hoyle_twist(r, H, b)
    sel = (r < 12) & np.isfinite(tm["tw"])
    err = np.abs(tm["tw"][sel] - ex[sel]) / ex[sel]
    rs = np.linspace(0, 12, 60)
    out.append(_test("twist", r"Twist $T_w$ frente al tubo de Gold \& Hoyle (1960)",
                     r"Twist $T_w$ versus the Gold \& Hoyle (1960) flux tube",
                     # same point in both: the pixel closest to the axis of the tube
                     float(ex[sel][np.argmin(r[sel])]), float(tm["tw"][sel][np.argmin(r[sel])]),
                     float(np.median(err)), 0.03, "error relativo mediano", "median relative error",
                     p95=float(np.percentile(err, 95)), seconds=time.time() - t0,
                     curve={"r": r[sel], "tw": tm["tw"][sel], "r_exact": rs,
                            "tw_exact": an.gold_hoyle_twist(rs, H, b)}))
    say(f"twist median err={np.median(err):.4f}")

    # 5. Decay index against the dipole
    t0 = time.time()
    n = 48 if quick else 64
    depth = 10.0
    dip = an.dipole_field(nx=n, ny=n, nz=40, half_width=50, depth=depth)
    msk = np.zeros(dip.shape[:2], dtype=bool)
    msk[n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1] = True
    pa = decay_index_profile(dip, mask=msk)
    mag = Magnetogram(bz=dip.bz[:, :, 0], dx=dip.dx, origin=dip.origin[:2])
    pf = decay_index_profile(potential_field(mag, nz=40, pad=2), mask=msk)
    out.append(_test("decay", r"Altura crítica ($n=1.5$) sobre un dipolo enterrado",
                     r"Critical height ($n=1.5$) above a buried dipole",
                     depth, pf["h_crit"], abs(pf["h_crit"] - depth) / depth, 0.05,
                     "error relativo en h_crit", "relative error in h_crit",
                     h_crit_analytic_field=pa["h_crit"], seconds=time.time() - t0,
                     curve={"z": pf["z"], "n_fft": pf["n"], "n_field": pa["n"],
                            "n_exact": an.dipole_decay_index(pf["z"], depth)}))
    say(f"decay h_crit={pf['h_crit']:.3f}")

    # 6. Q against the hyperbolic field
    t0 = time.time()
    lam = 0.05
    hy = an.hyperbolic_field(nx=48, ny=48, nz=24, half_width=20, lam=lam)
    qm = squashing_q(hy, step=2)
    top = qm["face"] == "top"
    qex = an.hyperbolic_q(hy.z[-1], lam)
    qv = qm["q"][top]
    err = np.abs(qv - qex) / qex
    out.append(_test("q", r"Factor $Q$ frente al campo hiperbólico", r"$Q$ factor versus the hyperbolic field",
                     qex, float(np.median(qv)), float(np.median(err)), 0.03,
                     "error relativo mediano", "median relative error",
                     p95=float(np.percentile(err, 95)), seconds=time.time() - t0,
                     q_values=qv))
    say(f"q median={np.median(qv):.4f} exact={qex:.4f}")
    return out
