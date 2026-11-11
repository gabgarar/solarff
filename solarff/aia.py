"""Comparison of the field lines with the coronal loops observed by SDO/AIA.

External validation with real data (as in Warren et al. 2018): the model lines
are projected onto an AIA EUV image and the **misalignment angle** between the
tangent of the projected line and the local orientation of the bright structures
of the image (loops) is measured point by point.

Steps:

1. ``download_aia`` — level-1 image closest to the HMI observation
   (``aia.lev1_euv_12s``, requires ``drms`` and an email registered at JSOC, which
   is only passed as an argument and never stored).
2. ``project_points`` — from the local Cartesian coordinates of the model
   (x, y, z in Mm on the CEA projection of the SHARP) to AIA pixels: CEA →
   Carrington (longitude, latitude) with the WCS of the SHARP, radius R☉ + z, and
   transformation to the SDO viewpoint with ``sunpy.coordinates``. The Cartesian
   box is a local (flat) approximation of the sphere, the usual one in
   NLFFF with SHARP CEA.
3. ``ridge_orientation`` — local orientation of the bright ridges
   (loops) from the Hessian matrix of the smoothed image: the direction
   of the loop is the eigenvector of the eigenvalue of smallest modulus, and the
   ridge strength is −λ_min (negative curvature = bright ridge).
4. ``line_misalignment`` — for each line, median of the angle between its
   projected tangent and the ridge orientation at the points where there is
   a clear ridge (above a strength percentile).

The angle lies in [0°, 90°] (orientations have no sense); a line unrelated
to the image gives 45° on average.
"""
from __future__ import annotations

import glob
import os

import numpy as np

AIA_SERIES = "aia.lev1_euv_12s"


def download_aia(time_utc: str, wavelengths=(171, 193), email: str | None = None,
                 out_dir: str = "data/aia", window: str = "2m") -> dict:
    """Download the level-1 AIA image closest to ``time_utc`` for each wavelength.

    ``time_utc``: ISO time in UTC (the HMI T_REC are in TAI: in 2011
    TAI − UTC = 34 s). Records within ±``window`` with QUALITY = 0 are searched and
    the one with the closest T_OBS is chosen. Returns ``{wavelength: path}``. If the
    file already exists it is not downloaded again.
    """
    import drms
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    t = pd.Timestamp(time_utc.replace("Z", ""))
    half = pd.Timedelta(window) if window[-1] != "m" else pd.Timedelta(minutes=int(window[:-1]))
    q0 = (t - half).strftime("%Y-%m-%dT%H:%M:%SZ")
    span = f"{int(2 * half.total_seconds())}s"
    c = drms.Client(email=email) if email else drms.Client()
    out = {}
    for w in wavelengths:
        k = c.query(f"{AIA_SERIES}[{q0}/{span}][{w}]", key="T_REC,T_OBS,QUALITY")
        k = k[k.QUALITY == 0].copy()
        if k.empty:
            raise RuntimeError(f"No hay imágenes AIA {w} Å con QUALITY = 0 cerca de {time_utc}")
        k["dt"] = (pd.to_datetime(k.T_OBS.str.replace("Z", "")) - t).abs()
        rec = k.sort_values("dt").iloc[0]
        tag = rec.T_REC.replace(":", "").replace("-", "")
        existing = glob.glob(os.path.join(out_dir, f"*{w}*{tag[:15]}*.fits"))
        if existing:
            out[w] = existing[0]
            continue
        if not email:
            raise ValueError("Hace falta un email registrado en JSOC para descargar (argumento email)")
        r = c.export(f"{AIA_SERIES}[{rec.T_REC}][{w}]{{image}}", method="url", protocol="fits")
        r.wait()
        d = r.download(out_dir)
        path = d.download[0]
        new = os.path.join(out_dir, f"aia.lev1_euv_12s.{w}.{tag}.fits")
        os.replace(path, new)
        out[w] = new
    return out


def load_aia(path: str):
    """sunpy Map of a level-1 AIA image, normalised by the exposure time."""
    import sunpy.map
    m = sunpy.map.Map(path)
    exp = float(m.meta.get("exptime", 1.0)) or 1.0
    return sunpy.map.Map(m.data.astype(float) / exp, m.meta)


def sharp_wcs(sharp_fits: str):
    """WCS and header of a SHARP CEA segment (e.g. the Br one)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(sharp_fits) as hd:
        h = hd[1].header if len(hd) > 1 and hd[1].data is not None else hd[0].header
    return WCS(h), h


def project_points(points_mm: np.ndarray, sharp_fits: str, aia_map) -> np.ndarray:
    """AIA pixels (x, y) of points (N, 3) in Mm of the model (x, y on the CEA, z height).

    The native pixel of the SHARP is ``x / dx0`` with ``dx0 = CDELT1 · R☉`` (the
    convention of ``solarff.io.load_sharp_cea``, also after ``rebin``).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames
    w, h = sharp_wcs(sharp_fits)
    rsun_mm = float(h.get("RSUN_REF", 6.96e8)) / 1e6
    dx0 = np.deg2rad(float(h["CDELT1"])) * rsun_mm
    p = np.asarray(points_mm, float)
    lon, lat = w.wcs_pix2world(p[:, 0] / dx0, p[:, 1] / dx0, 0)
    obs = aia_map.observer_coordinate
    c = SkyCoord(lon * u.deg, lat * u.deg, (rsun_mm + p[:, 2]) * u.Mm,
                 frame=frames.HeliographicCarrington(observer=obs, obstime=aia_map.date))
    px, py = aia_map.wcs.world_to_pixel(c.transform_to(aia_map.coordinate_frame))
    return np.column_stack([px, py])


def ridge_orientation(img: np.ndarray, sigma: float = 2.0, log: bool = True):
    """Orientation (rad, in [0, π)) and strength of the bright ridges of an image.

    Hessian of the image (of its logarithm, by default) smoothed with a
    Gaussian of width ``sigma`` pixels. On a bright ridge the transverse
    curvature is strongly negative (λ_min ≪ 0) and the longitudinal one nearly
    zero: the ridge direction is the eigenvector of λ_max. Strength =
    max(−λ_min, 0). ``img[row=y, column=x]``; the angle is measured from +x.
    """
    from scipy.ndimage import gaussian_filter
    a = np.log(np.clip(img, np.nanpercentile(img, 1) + 1e-3, None)) if log else img.astype(float)
    a = np.nan_to_num(a)
    hxx = gaussian_filter(a, sigma, order=(0, 2))
    hyy = gaussian_filter(a, sigma, order=(2, 0))
    hxy = gaussian_filter(a, sigma, order=(1, 1))
    tr, det = hxx + hyy, hxx * hyy - hxy * hxy
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    lmin = tr / 2 - disc
    lmax = tr / 2 + disc
    # eigenvector of lmax: (hxy, lmax − hxx) or (lmax − hyy, hxy); the one with the larger norm is taken
    # (one of the two vanishes when the Hessian is diagonal)
    ax, ay = hxy, lmax - hxx
    bx, by = lmax - hyy, hxy
    use_a = ax * ax + ay * ay >= bx * bx + by * by
    theta = np.mod(np.arctan2(np.where(use_a, ay, by), np.where(use_a, ax, bx)), np.pi)
    strength = np.maximum(-lmin, 0.0) * sigma ** 2  # scale-normalised
    return theta, strength


def _sample(arr, px, py):
    ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < arr.shape[1]) & (iy < arr.shape[0])
    out = np.full(px.shape, np.nan)
    out[ok] = arr[iy[ok], ix[ok]]
    return out


def line_misalignment(pix_lines: list, theta: np.ndarray, strength: np.ndarray, threshold: float,
                      min_points: int = 5, step_px: float = 2.0) -> np.ndarray:
    """Median misalignment angle (degrees) of each projected line.

    ``pix_lines``: list of (N, 2) arrays of AIA pixels. Each line is
    resampled every ``step_px`` pixels along its projection; the points with
    ridge strength ≥ ``threshold`` are used. Lines with fewer than
    ``min_points`` valid points give NaN.
    """
    out = np.full(len(pix_lines), np.nan)
    for k, P in enumerate(pix_lines):
        if len(P) < 3:
            continue
        seg = np.hypot(*np.diff(P, axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        if s[-1] < 3 * step_px:
            continue
        si = np.arange(0, s[-1], step_px)
        x, y = np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])
        tx, ty = np.gradient(x), np.gradient(y)
        ang = np.mod(np.arctan2(ty, tx), np.pi)
        th, st = _sample(theta, x, y), _sample(strength, x, y)
        ok = np.isfinite(th) & (st >= threshold) & (np.hypot(tx, ty) > 1e-6)
        if ok.sum() < min_points:
            continue
        d = np.abs(ang[ok] - th[ok])
        d = np.minimum(d, np.pi - d)
        out[k] = np.degrees(np.median(d))
    return out


# ----------------------------------------------------------------- full analysis
def _boot_median_diff(x, y, n, rng):
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size < 3 or y.size < 3:
        return None, None
    d = [np.median(rng.choice(x, x.size)) - np.median(rng.choice(y, y.size)) for _ in range(n)]
    return float(np.median(x) - np.median(y)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def _paired(d):
    """Median of the per-line differences and Wilcoxon p (one-sided: NLFFF better)."""
    from scipy.stats import wilcoxon
    d = d[np.isfinite(d)]
    if d.size < 5:
        return {"n": int(d.size), "median": None, "p": None}
    return {"n": int(d.size), "median": float(np.median(d)), "p": float(wilcoxon(d, alternative="less")[1])}


def aia_validation(lines, conf, reliable, files: dict, sharp_fits: str, other_lines: dict | None = None,
                   sigma: float = 2.0, pct: float = 90, n_boot: int = 2000, seed: int = 0) -> dict:
    """Compare the lines of a model with the AIA loops (see the module docstring).

    ``lines``: model lines (objects with ``points`` and ``apex``); ``conf`` and
    ``reliable``: their confidence; ``files``: {wavelength: AIA FITS};
    ``other_lines``: {name: lines traced from the same seeds in other
    models (e.g. potential)}. Returns ``{"summary": ..., "plot": ...}``:
    the summary (JSON) and what the figure needs.

    Since in a region the confidence is usually strongly tied to the height of the
    line, the comparison is repeated at equal height (terciles of the maximum
    height) and with a regression angle ~ reliable + log(1 + height).
    """
    from scipy.stats import mannwhitneyu, spearmanr
    rng = np.random.default_rng(seed)
    other_lines = other_lines or {}
    conf = np.asarray(conf, float)
    rel = np.asarray(reliable, bool)
    apex = np.array([l.apex for l in lines])
    S = {"n_lines": len(lines), "reliable_frac": float(rel.mean()), "sigma_px": sigma, "ridge_pct": pct,
         "by_wavelength": {}}
    plot = {}
    for w, path in files.items():
        m = load_aia(path)
        pix = [project_points(np.asarray(l.points), sharp_fits, m) for l in lines]
        allp = np.vstack(pix)
        x0, y0 = np.maximum(np.floor(allp.min(0) - 30).astype(int), 0)
        x1, y1 = np.ceil(allp.max(0) + 30).astype(int)
        sub = m.data[y0:y1, x0:x1]
        shift = np.array([x0, y0])
        pl = [p - shift for p in pix]
        th, st = ridge_orientation(sub, sigma=sigma)
        thr = float(np.percentile(st, pct))
        ang = line_misalignment(pl, th, st, thr)
        ang_o = {k: line_misalignment([project_points(np.asarray(l.points), sharp_fits, m) - shift for l in ls],
                                      th, st, thr) for k, ls in other_lines.items()}
        ok = np.isfinite(ang)
        r = {"file": os.path.basename(path), "date": m.date.isot, "n_measured": int(ok.sum()),
             "median_reliable": float(np.nanmedian(ang[rel])), "median_unreliable": float(np.nanmedian(ang[~rel])),
             "n_reliable": int((ok & rel).sum()), "n_unreliable": int((ok & ~rel).sum())}
        r["diff"], r["diff_ci"] = _boot_median_diff(ang[rel], ang[~rel], n_boot, rng)
        r["p_mannwhitney"] = float(mannwhitneyu(ang[ok & rel], ang[ok & ~rel], alternative="less").pvalue)
        r["spearman_conf"] = float(spearmanr(conf[ok], ang[ok])[0])
        # Height control
        la = np.log1p(apex[ok])
        X = np.column_stack([np.ones(ok.sum()), rel[ok].astype(float), la])
        y = ang[ok]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        bs = [np.linalg.lstsq(X[i], y[i], rcond=None)[0][1]
              for i in (rng.integers(0, y.size, y.size) for _ in range(n_boot))]
        r["ols_reliable"] = float(beta[1])
        r["ols_reliable_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        Z = np.column_stack([np.ones_like(la), la])
        res_c = conf[ok] - Z @ np.linalg.lstsq(Z, conf[ok], rcond=None)[0]
        res_a = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        r["spearman_partial"] = float(spearmanr(res_c, res_a)[0])
        r["spearman_partial_p"] = float(spearmanr(res_c, res_a)[1])
        q = np.percentile(apex[ok], [100 / 3, 200 / 3])
        terc = np.digitize(apex, q)
        r["terciles"] = []
        for i in range(3):
            s = ok & (terc == i)
            row = {"apex_mm": [float(apex[s].min()), float(apex[s].max())],
                   "n_reliable": int((s & rel).sum()), "n_unreliable": int((s & ~rel).sum()),
                   "median_reliable": float(np.median(ang[s & rel])) if (s & rel).any() else None,
                   "median_unreliable": float(np.median(ang[s & ~rel])) if (s & ~rel).any() else None}
            if row["n_reliable"] >= 5 and row["n_unreliable"] >= 5:
                row["diff"], row["diff_ci"] = _boot_median_diff(ang[s & rel], ang[s & ~rel], n_boot, rng)
            for k, v in ang_o.items():
                row[f"nlfff_minus_{k}_reliable"] = _paired((ang - v)[s & rel])
                row[f"nlfff_minus_{k}_unreliable"] = _paired((ang - v)[s & ~rel])
            r["terciles"].append(row)
        r["models"] = {}
        for k, v in ang_o.items():
            b = ok & np.isfinite(v)
            r["models"][k] = {"all": _paired((ang - v)[b]), "reliable": _paired((ang - v)[b & rel]),
                              "unreliable": _paired((ang - v)[b & ~rel]),
                              "median_nlfff": float(np.median(ang[b])), "median_other": float(np.median(v[b]))}
        # Sensitivity to the loop detection
        sens = []
        for sg in (1.5, 2.0, 3.0):
            th2, st2 = (th, st) if sg == sigma else ridge_orientation(sub, sigma=sg)
            for pc in (85, 90, 95):
                a2 = line_misalignment(pl, th2, st2, float(np.percentile(st2, pc)))
                mid = np.isfinite(a2) & (terc == 1)
                sens.append({"sigma": sg, "pct": pc, "median_reliable": float(np.nanmedian(a2[rel])),
                             "median_unreliable": float(np.nanmedian(a2[~rel])),
                             "mid_reliable": float(np.nanmedian(a2[mid & rel])) if (mid & rel).any() else None,
                             "mid_unreliable": float(np.nanmedian(a2[mid & ~rel])) if (mid & ~rel).any() else None})
        r["sensitivity"] = sens
        S["by_wavelength"][str(w)] = r
        plot[w] = {"cutout": sub, "pix": pl, "angles": ang, "date": m.date.isot[:19]}
    return {"summary": S, "plot": {"wavelengths": plot, "reliable": rel}}
