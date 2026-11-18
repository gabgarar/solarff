"""Reference fields from public MHD simulations.

The Hinode Science Data Centre Europe (https://sdc.uio.no/search/simulations) serves snapshots of radiative MHD
simulations as FITS cubes, one variable per file, with the height of each plane in an extension. ``reduced_field``
turns the magnetic field of a snapshot into a ``VectorField3D`` above the photosphere, on cubic cells, which can
be used as the truth of an extrapolation made from its own base (``examples/33_mhd_truth.py``).

A field with a photosphere out of force balance and a corona that is not in equilibrium is a harder reference
than the exact force-free solutions of ``solarff.analytic``, and a closer one to the Sun.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np

from .field import VectorField3D

URL = "https://sdc.uio.no/vol/simulations/{sim}/atmos/{code}_{sim}_{var}_{snap}.fits"

#: Runs that have been used with this module. Others of the catalogue work if their code name is given.
SIMULATIONS = {
    "ar098192": {"code": "MURaM", "what": "active region with a flare", "reference": "Cheung et al. (2019, Nature Astronomy 3, 160)"},
    "qs024048_by3363": {"code": "BIFROST", "what": "flux emergence", "reference": "Hansteen et al. (2017, ApJ 839, 22)"},
    "en024048_hion": {"code": "BIFROST", "what": "enhanced network", "reference": "Carlsson et al. (2016, A&A 585, A4)"},
}


def cube_path(sim: str, var: str, snap: int, raw: str, code: str | None = None, download: bool = True) -> str:
    """Path of the cube of ``var`` (``bx``, ``by``, ``bz``, ...) of a snapshot in the folder ``raw``.

    It is downloaded from the data centre if it is not there (about 0.4 to 0.5 GB per cube).
    """
    code = code or SIMULATIONS[sim]["code"]
    path = os.path.join(raw, f"{code}_{sim}_{var}_{snap}.fits")
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(path)
        os.makedirs(raw, exist_ok=True)
        print(f"  downloading {os.path.basename(path)}", flush=True)
        urllib.request.urlretrieve(URL.format(sim=sim, code=code, var=var, snap=snap), path + ".part")
        os.replace(path + ".part", path)
    return path


def _planes(z: np.ndarray, step: float):
    """Planes and weights that give the heights z0, z0 + step, ... above the plane closest to z = 0.

    Returns ``k0``, the heights and, for each one, ``(i, w)``: the field there is ``(1 - w) plane[i] + w plane[i + 1]``.
    """
    z = np.asarray(z, dtype=float)
    assert np.all(np.diff(z) > 0), "the heights of the planes must increase"
    k0 = int(np.argmin(np.abs(z)))
    heights = z[k0] + step * np.arange(int(np.floor((z[-1] - z[k0]) / step + 1e-6)) + 1)
    out = []
    for hgt in heights:
        i = min(int(np.searchsorted(z, hgt, side="right")) - 1, len(z) - 2)
        w = (hgt - z[i]) / (z[i + 1] - z[i])
        if abs(w) < 1e-4:
            w = 0.0
        elif abs(w - 1) < 1e-4:
            i, w = i + 1, 0.0
        out.append((i, float(w)))
    return k0, heights, out


def reduced_field(sim: str, snap: int, r: int, raw: str, code: str | None = None, download: bool = True) -> VectorField3D:
    """Magnetic field of a snapshot above the photosphere, on cubic cells of ``r`` pixels of the simulation (G, Mm).

    Horizontally the field is averaged in blocks of ``r`` x ``r`` pixels, as a magnetogram is rebinned. In height it
    is taken at the plane closest to z = 0, the photosphere of these runs, and at every cell size above it, with a
    linear interpolation between planes when the vertical grid of the run does not contain those heights. The base
    of the field is therefore the photospheric plane itself and not an average with higher layers.

    ``meta`` keeps the time of the snapshot (``elapsed_s``) and the plane of the photosphere (``k0``).
    """
    from astropy.io import fits

    comp, info = {}, {}
    for var in ("bx", "by", "bz"):
        with fits.open(cube_path(sim, var, snap, raw, code, download), memmap=True) as h:
            hd, z = h[0].header, np.asarray(h[1].data, dtype=float)
            assert hd["BUNIT"].strip() == "T", "the field is expected in tesla"
            assert abs(hd["CDELT1"] - hd["CDELT2"]) < 1e-6, "the pixels are expected square"
            step = float(hd["CDELT1"]) * r
            k0, heights, planes = _planes(z, step)
            data = h[0].data                                    # (z, y, x)
            ny, nx = data.shape[1] - data.shape[1] % r, data.shape[2] - data.shape[2] % r
            out = np.empty((len(planes), ny // r, nx // r), dtype=np.float32)
            for j, (i, w) in enumerate(planes):
                p = np.asarray(data[i], dtype=np.float32)
                if w:
                    p = (1 - w) * p + w * np.asarray(data[i + 1], dtype=np.float32)
                out[j] = p[:ny, :nx].reshape(ny // r, r, nx // r, r).mean(axis=(1, 3))
            comp[var] = 1e4 * out.transpose(2, 1, 0).astype(float)     # (x, y, z), gauss
            info = {"elapsed_s": float(hd.get("ELAPSED", np.nan)), "k0": k0, "pixel_Mm": float(hd["CDELT1"]),
                    "origin": (float(hd["CRVAL1"]) + 0.5 * (r - 1) * hd["CDELT1"], float(hd["CRVAL2"]) + 0.5 * (r - 1) * hd["CDELT2"], float(heights[0]))}
    code = code or SIMULATIONS[sim]["code"]
    return VectorField3D(comp["bx"], comp["by"], comp["bz"], step, info["origin"],
                         {"method": f"{code} {sim}, snapshot {snap}, r = {r}", "simulation": sim, "snapshot": int(snap), "r": int(r),
                          "elapsed_s": info["elapsed_s"], "k0": info["k0"], "pixel_Mm": info["pixel_Mm"]})
