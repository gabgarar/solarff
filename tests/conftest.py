"""Fixtures shared by the tests.

Structure:

* ``tests/unit``: each piece separately against an exact answer (quadratures, energy,
  helicity, analytical fields, tracing, metrics). Fast.
* ``tests/integration``: the full workflow (preprocessing → NLFFF → study → report, Monte Carlo)
  against exact fields, and the reproduction of the published figures.

Markers: ``slow`` (minutes; ``pytest -m slow``) and ``needs_data`` (use data outside git; skipped
if absent). By default everything but ``slow`` is run.
"""
import os
import pickle

import numpy as np
import pytest

import solarff as sff

ROOT = os.path.join(os.path.dirname(__file__), "..")
#: Suite with which the published calibration was fitted (examples/04_calibration_suite.py).
CALIBRATION_SUITE = os.path.join(ROOT, "data", "calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl")


def make_bipole(n=64, w=160.0, d=10):
    """Balanced Gaussian bipolar magnetogram (B_z only)."""
    x = np.arange(n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = n // 2
    bz = 1000 * (np.exp(-((X - c + d) ** 2 + (Y - c) ** 2) / w)
                 - np.exp(-((X - c - d) ** 2 + (Y - c) ** 2) / w))
    return sff.Magnetogram(bz=bz, dx=1.0)


@pytest.fixture
def bipole():
    return make_bipole


@pytest.fixture(scope="session")
def lowlou():
    """(magnetogram, exact field) of Low & Lou on 32×32×20."""
    return sff.low_lou_magnetogram(nx=32, ny=32, nz=20)


@pytest.fixture(scope="session")
def lowlou_study():
    """Full study (with preprocessing and L-BFGS NLFFF) of a small Low & Lou."""
    mag, _ = sff.low_lou_magnetogram(nx=24, ny=24, nz=16)
    return sff.run_study(mag, 16, nlfff_method="lbfgs", verbose=False)


@pytest.fixture(scope="session")
def calibration_suite():
    if not os.path.exists(CALIBRATION_SUITE):
        pytest.skip("falta la batería de calibración (examples/04_calibration_suite.py --extra-fields TD-relax)")
    data = pickle.load(open(CALIBRATION_SUITE, "rb"))
    return data["data"] if isinstance(data, dict) else data
