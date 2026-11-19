"""Reduction of a simulation cube to a field on cubic cells (``solarff.simulations``), on small synthetic cubes."""
import numpy as np
import pytest

from solarff.simulations import _planes, reduced_field

fits = pytest.importorskip("astropy.io.fits")


def write_cubes(folder, z, nx=8, ny=4, dx=0.1, sim="test", code="CODE", snap=1, unit="T"):
    """Cubes with bz = 1e-4 T * (1 + z) at every pixel, bx = 1e-4 T * x index, by = 2e-4 T: easy to check after the reduction."""
    nz = len(z)
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    fields = {"bx": 1e-4 * i, "by": 2e-4 * np.ones_like(i, dtype=float), "bz": 1e-4 * (1 + np.asarray(z)[k])}
    for var, data in fields.items():
        hd = fits.Header()
        hd["BUNIT"], hd["CDELT1"], hd["CDELT2"], hd["CRVAL1"], hd["CRVAL2"], hd["ELAPSED"] = unit, dx, dx, -0.4, -0.2, 10.0
        fits.HDUList([fits.PrimaryHDU(data.astype(np.float32), hd), fits.ImageHDU(np.asarray(z, np.float32))]).writeto(
            folder / f"{code}_{sim}_{var}_{snap}.fits")


def test_planes_on_a_uniform_grid_are_the_planes_themselves():
    z = -0.2 + 0.1 * np.arange(12)                      # z = 0 at plane 2
    k0, heights, planes = _planes(z, 0.3)
    assert k0 == 2
    assert np.allclose(heights, [0.0, 0.3, 0.6, 0.9])
    assert planes == [(2, 0.0), (5, 0.0), (8, 0.0), (11, 0.0)]


def test_reduced_field_block_mean_and_heights(tmp_path):
    z = -0.2 + 0.1 * np.arange(12)
    write_cubes(tmp_path, z)
    f = reduced_field("test", 1, 2, str(tmp_path), code="CODE", download=False)
    assert f.shape == (4, 2, 5) and f.dx == pytest.approx(0.2)                  # heights 0, 0.2, ..., 0.8
    assert np.allclose(f.bz[:, :, 0], 1.0, atol=1e-4) and np.allclose(f.bz[:, :, 4], 1.8, atol=1e-4)      # gauss
    assert np.allclose(f.bx[:, 0, 0], [0.5, 2.5, 4.5, 6.5], atol=1e-4)          # block mean of the x index
    assert np.allclose(f.by, 2.0, atol=1e-4)
    assert f.origin == pytest.approx((-0.35, -0.15, 0.0), abs=1e-6)
    assert f.meta["k0"] == 2 and f.meta["elapsed_s"] == 10.0


def test_reduced_field_with_the_base_above_the_photosphere(tmp_path):
    z = -0.2 + 0.1 * np.arange(12)                      # z = 0.4 at plane 6
    write_cubes(tmp_path, z)
    f = reduced_field("test", 1, 2, str(tmp_path), code="CODE", download=False, z0=0.42)
    assert f.shape == (4, 2, 3) and f.meta["k0"] == 6 and f.origin[2] == pytest.approx(0.4, abs=1e-6)       # heights 0.4, 0.6, 0.8
    assert np.allclose(f.bz[:, :, 0], 1.4, atol=1e-4) and np.allclose(f.bz[:, :, 2], 1.8, atol=1e-4)


def test_reduced_field_accepts_the_unit_written_as_a_logical_value(tmp_path):
    """The Bifrost cubes write BUNIT = T without quotes, which is read as the logical value true."""
    write_cubes(tmp_path, -0.2 + 0.1 * np.arange(12), unit=True)
    f = reduced_field("test", 1, 2, str(tmp_path), code="CODE", download=False)
    assert np.allclose(f.by, 2.0, atol=1e-4)


def test_reduced_field_interpolates_a_non_uniform_height_grid(tmp_path):
    z = np.array([-0.3, -0.1, 0.0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.2])         # irregular: 0.2, 0.4, ... are between planes
    write_cubes(tmp_path, z)
    f = reduced_field("test", 1, 2, str(tmp_path), code="CODE", download=False)
    assert f.shape[2] == 7                                                    # 0, 0.2, ..., 1.2
    assert np.allclose(f.bz[0, 0, :], 1 + 0.2 * np.arange(7), atol=1e-4)      # bz is linear in z: the interpolation is exact
