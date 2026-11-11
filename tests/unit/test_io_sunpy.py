"""sunpy Map input/output (solarff.io.from_sunpy / to_sunpy).

The core loader (load_sharp_cea) is not changed; these tests check that the sunpy
path gives the same Magnetogram as the FITS path and that the round trip is exact.
"""
import numpy as np
import pytest

sunpy_map = pytest.importorskip("sunpy.map")
from astropy.io import fits  # noqa: E402

from solarff.io import from_sunpy, load_sharp_cea, to_sunpy  # noqa: E402

RSUN = 6.96e8
CDELT = 0.03


def _header():
    return {"CTYPE1": "CRLN-CEA", "CTYPE2": "CRLT-CEA", "CUNIT1": "deg", "CUNIT2": "deg",
            "CDELT1": CDELT, "CDELT2": CDELT, "CRPIX1": 4.5, "CRPIX2": 3.5,
            "CRVAL1": 100.0, "CRVAL2": -20.0, "RSUN_REF": RSUN, "HARPNUM": 377,
            "T_REC": "2011.02.15_00:00:00_TAI", "DATE-OBS": "2011-02-15T00:00:00",
            "TELESCOP": "SDO/HMI", "INSTRUME": "HMI_SIDE1"}


def _segments():
    rng = np.random.default_rng(0)
    ny, nx = 6, 8  # FITS / Map layout: [y, x]
    return {seg: rng.normal(size=(ny, nx)) * 100 for seg in ("Br", "Bp", "Bt")}


def test_from_sunpy_conventions():
    seg = _segments()
    maps = {k: sunpy_map.Map(v, _header()) for k, v in seg.items()}
    mag = from_sunpy(maps["Br"], maps["Bp"], maps["Bt"])
    assert mag.bz.shape == (8, 6)  # [ix, iy]
    np.testing.assert_array_equal(mag.bz, seg["Br"].T)
    np.testing.assert_array_equal(mag.bx, seg["Bp"].T)
    np.testing.assert_array_equal(mag.by, -seg["Bt"].T)
    assert mag.dx == pytest.approx(np.deg2rad(CDELT) * RSUN / 1e6)
    assert mag.meta["HARPNUM"] == 377 and mag.meta["CRVAL1"] == 100.0
    assert "fits_header" in mag.meta


def test_roundtrip_to_sunpy():
    seg = _segments()
    maps = {k: sunpy_map.Map(v, _header()) for k, v in seg.items()}
    mag = from_sunpy(maps["Br"], maps["Bp"], maps["Bt"])
    back = to_sunpy(mag)
    assert set(back) == {"Br", "Bp", "Bt"}
    for k in seg:
        np.testing.assert_array_equal(back[k].data, seg[k])
    assert back["Br"].meta["cdelt1"] == pytest.approx(CDELT)
    assert back["Br"].meta["crpix1"] == 4.5 and back["Br"].meta["crpix2"] == 3.5
    again = from_sunpy(back["Br"], back["Bp"], back["Bt"])
    np.testing.assert_array_equal(again.bz, mag.bz)
    np.testing.assert_array_equal(again.by, mag.by)
    assert again.dx == pytest.approx(mag.dx)


def test_to_sunpy_without_stored_header_and_after_rebin():
    seg = _segments()
    maps = {k: sunpy_map.Map(v, _header()) for k, v in seg.items()}
    mag = from_sunpy(maps["Br"], maps["Bp"], maps["Bt"], rebin=2)
    assert mag.bz.shape == (4, 3)
    m = to_sunpy(mag)["Br"]
    assert m.meta["cdelt1"] == pytest.approx(2 * CDELT)
    assert m.data.shape == (3, 4)
    # magnetogram built by hand (no fits_header): minimal CEA header
    mag.meta.pop("fits_header")
    m2 = to_sunpy(mag)["Br"]
    assert m2.meta["ctype1"] == "CRLN-CEA" and m2.meta["cdelt1"] == pytest.approx(2 * CDELT)
    assert m2.meta["crval1"] == 100.0
    assert str(m2.date).startswith("2011-02-15T00:00:00")


def test_from_sunpy_matches_load_sharp_cea(tmp_path):
    seg = _segments()
    for k, v in seg.items():
        fits.PrimaryHDU(v, header=fits.Header(_header())).writeto(tmp_path / f"hmi.sharp_cea_720s.377.{k}.fits")
    ref = load_sharp_cea(str(tmp_path))
    maps = {k: sunpy_map.Map(str(tmp_path / f"hmi.sharp_cea_720s.377.{k}.fits")) for k in seg}
    mag = from_sunpy(maps["Br"], maps["Bp"], maps["Bt"])
    for comp in ("bz", "bx", "by"):
        np.testing.assert_array_equal(getattr(mag, comp), getattr(ref, comp))
    assert mag.dx == pytest.approx(ref.dx)
    assert mag.meta["HARPNUM"] == ref.meta["HARPNUM"]
