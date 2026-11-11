"""Reading and downloading of SDO/HMI data.

Recommended product: **SHARP CEA** (``hmi.sharp_cea_720s``), segments

* ``Br`` : radial component            -> Bz
* ``Bp`` : φ component (westward)        -> Bx
* ``Bt`` : θ component (southward)       -> By = -Bt

They are already reprojected onto a cylindrical equal-area (CEA) grid
centred on the region, with the 180° ambiguity resolved and in heliographic
components: exactly the local Cartesian boundary condition that the
LFFF/NLFFF methods need.

For potential/LFFF ``Br`` is enough (or the ``magnetogram`` segment of the
``hmi.sharp_cea_720s`` series, LOS, if the region is near disc centre).
"""

from __future__ import annotations

import glob
import os
import re
import warnings

import numpy as np

from .magnetogram import Magnetogram

SHARP_CEA = "hmi.sharp_cea_720s"


def _read_fits(path):
    from astropy.io import fits

    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data, dtype=float), dict(hdu.header)
    raise ValueError(f"{path}: no contiene datos de imagen")


def load_sharp_cea(br: str, bp: str | None = None, bt: str | None = None,
                   rebin: int = 1) -> Magnetogram:
    """Load a SHARP CEA and convert it to the local Cartesian convention.

    **Parameters**

    - ``br, bp, bt``: paths to the FITS files of the Br, Bp, Bt segments. If only
        ``br`` is given, a Bz-only magnetogram is obtained (valid for potential and
        LFFF). A directory is also accepted in ``br``: the files
        ``*.Br.fits``, ``*.Bp.fits``, ``*.Bt.fits`` are looked for inside it.
    - ``rebin``: resolution reduction factor (see ``Magnetogram.rebin``).
    """
    if os.path.isdir(br):
        d = br
        found = {}
        for seg in ("Br", "Bp", "Bt", "Br_err", "Bp_err", "Bt_err"):
            m = sorted(glob.glob(os.path.join(d, f"*.{seg}.fits")))
            found[seg] = m[0] if m else None
        if found["Br"] is None:
            raise FileNotFoundError(f"No hay *.Br.fits en {d}")
        br, bp, bt = found["Br"], found["Bp"], found["Bt"]
        errs = (found["Br_err"], found["Bp_err"], found["Bt_err"])
    else:
        errs = (None, None, None)

    data_r, hdr = _read_fits(br)
    # FITS: data[row=y, column=x]  ->  our convention [ix, iy]
    bz = np.nan_to_num(data_r.T)
    bx = by = None
    if bp is not None and bt is not None:
        bx = np.nan_to_num(_read_fits(bp)[0].T)
        by = -np.nan_to_num(_read_fits(bt)[0].T)

    cdelt = float(hdr.get("CDELT1", 0.03))  # heliographic degrees per pixel
    rsun_m = float(hdr.get("RSUN_REF", 6.96e8))
    dx = np.deg2rad(cdelt) * rsun_m / 1e6  # Mm

    meta = {k: hdr.get(k) for k in ("T_REC", "HARPNUM", "NOAA_AR", "NOAA_ARS", "CRVAL1",
                                    "CRVAL2", "CRLN_OBS", "LON_FWT", "LAT_FWT", "USFLUX")
            if k in hdr}
    meta["source"] = SHARP_CEA
    lon = hdr.get("LON_FWT")
    if lon is not None and np.isfinite(lon) and abs(lon) > 45:
        warnings.warn(
            f"La región está a {lon:.0f}° del meridiano central: el ruido del campo "
            "transversal y los efectos de proyección son grandes. Preferible |lon| < 30-45°.")
    e = [None if f is None else np.nan_to_num(_read_fits(f)[0].T) for f in errs]
    mag = Magnetogram(bz=bz, bx=bx, by=by, dx=dx, meta=meta,
                      bz_err=e[0], bx_err=e[1], by_err=e[2])
    if mag.has_errors:
        meta["errors"] = "Br_err, Bp_err, Bt_err"
    return mag.rebin(rebin) if rebin > 1 else mag


def _header_get(hdr, key, default=None):
    """Case-insensitive lookup in a FITS header, a sunpy MetaDict or a plain dict."""
    for k in (key, key.lower(), key.upper()):
        try:
            if k in hdr and hdr[k] is not None:
                return hdr[k]
        except TypeError:
            pass
    return default


def from_sunpy(br, bp=None, bt=None, br_err=None, bp_err=None, bt_err=None,
               rebin: int = 1) -> Magnetogram:
    """Build a ``Magnetogram`` from sunpy Maps of SHARP CEA segments.

    Same conventions as ``load_sharp_cea``: ``Br -> Bz``, ``Bp -> Bx``,
    ``Bt -> -By``; ``Map.data`` is ``[y, x]`` and is transposed to ``[ix, iy]``;
    the pixel size in Mm comes from ``CDELT1`` (heliographic degrees) and
    ``RSUN_REF``. The full header of ``br`` is kept in ``meta["fits_header"]``
    so that ``to_sunpy`` can give it back.

    **Parameters**

    - ``br, bp, bt``: ``sunpy.map.Map`` of the ``Br``, ``Bp`` and ``Bt`` segments
        (``hmi.sharp_cea_720s``). With ``br`` alone the magnetogram has only ``Bz``.
    - ``br_err, bp_err, bt_err``: optional Maps of the ``*_err`` segments (1σ, G).
    - ``rebin``: resolution reduction factor (see ``Magnetogram.rebin``).
    """
    hdr = br.meta
    bz = np.nan_to_num(np.asarray(br.data, dtype=float).T)
    bx = by = None
    if bp is not None and bt is not None:
        bx = np.nan_to_num(np.asarray(bp.data, dtype=float).T)
        by = -np.nan_to_num(np.asarray(bt.data, dtype=float).T)

    cdelt = float(_header_get(hdr, "CDELT1", 0.03))
    rsun_m = float(_header_get(hdr, "RSUN_REF", 6.96e8))
    dx = np.deg2rad(cdelt) * rsun_m / 1e6  # Mm

    meta = {}
    for k in ("T_REC", "HARPNUM", "NOAA_AR", "NOAA_ARS", "CRVAL1", "CRVAL2", "CRLN_OBS",
              "LON_FWT", "LAT_FWT", "USFLUX"):
        v = _header_get(hdr, k)
        if v is not None:
            meta[k] = v
    meta["source"] = SHARP_CEA
    meta["fits_header"] = {str(k): v for k, v in dict(hdr).items()}

    def err(m):
        return None if m is None else np.nan_to_num(np.asarray(m.data, dtype=float).T)

    mag = Magnetogram(bz=bz, bx=bx, by=by, dx=dx, meta=meta,
                      bz_err=err(br_err), bx_err=err(bp_err), by_err=err(bt_err))
    if mag.has_errors:
        meta["errors"] = "Br_err, Bp_err, Bt_err"
    return mag.rebin(rebin) if rebin > 1 else mag


def to_sunpy(mag: Magnetogram) -> dict:
    """Return the magnetogram as sunpy Maps: ``{"Br": Map, "Bp": Map, "Bt": Map}``.

    ``Bp`` and ``Bt`` are present only for a vector magnetogram. The header is the
    one stored by ``from_sunpy`` (``meta["fits_header"]``) when available;
    otherwise a minimal CEA header is built from ``dx`` and the keys kept in
    ``meta``. In both cases ``CDELT``, ``CRPIX`` and ``NAXIS`` are recomputed
    from the current arrays, so a rebinned magnetogram gives a consistent Map
    (the reference pixel is the centre of the map, as in SHARP CEA).
    """
    import sunpy.map

    nx, ny = mag.bz.shape
    rsun_m = float(mag.meta.get("RSUN_REF", 6.96e8))
    base = mag.meta.get("fits_header")
    if base is not None:
        rsun_m = float(_header_get(base, "RSUN_REF", rsun_m))
    cdelt = float(np.rad2deg(mag.dx * 1e6 / rsun_m))
    hdr = {}
    if base is not None:
        hdr.update({k: v for k, v in base.items()
                    if not str(k).upper().startswith(("NAXIS", "BITPIX", "CRPIX", "CDELT"))})
    else:
        hdr.update({"CTYPE1": "CRLN-CEA", "CTYPE2": "CRLT-CEA", "CUNIT1": "deg", "CUNIT2": "deg",
                    "CRVAL1": float(mag.meta.get("CRVAL1", 0.0)), "CRVAL2": float(mag.meta.get("CRVAL2", 0.0)),
                    "RSUN_REF": rsun_m, "TELESCOP": "SDO/HMI", "INSTRUME": "HMI_SIDE1"})
        t_rec = mag.meta.get("T_REC")
        if t_rec:  # "2011.02.15_00:00:00_TAI" -> ISO (TAI differs from UTC by ~37 s)
            d, t = str(t_rec).replace("_TAI", "").split("_")[:2]
            hdr["DATE-OBS"] = d.replace(".", "-") + "T" + t
        for k in ("HARPNUM", "NOAA_AR", "NOAA_ARS", "LON_FWT", "LAT_FWT", "USFLUX"):
            if k in mag.meta:
                hdr[k] = mag.meta[k]
    hdr.update({"CDELT1": cdelt, "CDELT2": cdelt, "CRPIX1": (nx + 1) / 2, "CRPIX2": (ny + 1) / 2,
                "CUNIT1": "deg", "CUNIT2": "deg"})

    def mk(a):
        return sunpy.map.Map(np.asarray(a, dtype=float).T.copy(), dict(hdr))

    out = {"Br": mk(mag.bz)}
    if mag.bx is not None and mag.by is not None:
        out["Bp"] = mk(mag.bx)
        out["Bt"] = mk(-mag.by)
    return out


def _jsoc_time(t: str) -> str:
    """'2011-02-15T00:00:00' -> '2011.02.15_00:00:00_TAI' (leaves the JSOC format untouched)."""
    if "_" in t and "." in t.split("_")[0]:
        return t
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?", t)
    if not m:
        raise ValueError(f"Formato de fecha no reconocido: {t}")
    y, mo, d, hh, mm, ss = m.groups()
    return f"{y}.{mo}.{d}_{hh}:{mm}:{ss or '00'}_TAI"


def search_sharp(time: str, noaa: int | None = None, email: str | None = None):
    """List the SHARPs available at a given time (requires ``drms`` and ``pandas``).

    Returns a DataFrame with HARPNUM, NOAA_ARS, position and total flux.
    Filters by NOAA number if ``noaa`` is given.
    """
    import drms

    c = drms.Client(email=email) if email else drms.Client()
    keys = c.query(f"{SHARP_CEA}[][{_jsoc_time(time)}]",
                   key="HARPNUM,NOAA_ARS,T_REC,LON_FWT,LAT_FWT,USFLUX")
    if noaa is not None:
        keys = keys[keys["NOAA_ARS"].astype(str).str.split(",").apply(lambda l: str(noaa) in l)]
    return keys


def download_sharp(harpnum: int, time: str, email: str, out_dir: str = "data",
                   segments=("Br", "Bp", "Bt", "Br_err", "Bp_err", "Bt_err")) -> str:
    """Download the SHARP CEA segments from JSOC and return the directory.

    By default the error maps (``*_err``), needed for the Monte Carlo, are
    downloaded too. ``email`` must be registered at http://jsoc.stanford.edu/ajax/register_email.html.
    Then: ``load_sharp_cea(out_dir)``.
    """
    import drms

    os.makedirs(out_dir, exist_ok=True)
    c = drms.Client(email=email)
    ds = f"{SHARP_CEA}[{harpnum}][{_jsoc_time(time)}]{{{','.join(segments)}}}"
    req = c.export(ds, method="url", protocol="fits")
    req.wait()
    if req.status != 0:
        raise RuntimeError(f"Exportación JSOC fallida (status {req.status}) para {ds}")
    req.download(out_dir)
    return out_dir
