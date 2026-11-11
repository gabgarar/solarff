"""Line tracing and topology against known geometries."""
import numpy as np
import pytest

import solarff as sff
from solarff.analytic import dipole_field


def test_tracing_closed_loops_in_potential_bipole(bipole):
    f = sff.potential_field(bipole(), nz=30)
    line = sff.trace_field_lines(f, np.array([[22.0, 32.0, 1.0]]))[0]
    assert line.closed
    assert line.start[0] < 32 < line.end[0]  # leaves the positive polarity and enters the negative one


def test_tracing_stops_exactly_on_boundary():
    # Tilted uniform field: the line must end exactly at the top
    n = 20
    bx = np.full((n, n, n), 0.3)
    f = sff.VectorField3D(bx, np.zeros_like(bx), np.ones_like(bx), dx=1.0)
    from solarff.tracing import integrate_lines
    end, L, _ = integrate_lines(f, np.array([[5.0], [10.0], [0.0]]), +1.0, ds=0.7)
    assert end[2, 0] == pytest.approx(n - 1)
    assert end[0, 0] == pytest.approx(5 + 0.3 * (n - 1), abs=1e-6)


def test_dipole_field_lines_follow_r_sin2():
    # Dipole with moment along x buried at depth d: r = L sin²θ, θ from the x axis
    d = 10.0
    f = dipole_field(nx=64, ny=64, nz=40, half_width=50.0, depth=d)
    seeds = np.array([[-8.0, 3.0, 0.5], [-15.0, -6.0, 0.5], [-5.0, 0.0, 0.5]])
    for line in sff.trace_field_lines(f, seeds, ds=0.25):
        p = line.points.copy()
        p[:, 2] += d
        r = np.linalg.norm(p, axis=1)
        L = r / (1 - (p[:, 0] / r) ** 2)
        assert np.std(L) / np.mean(L) < 0.01, np.std(L) / np.mean(L)


def test_q_is_two_for_uniform_field():
    n = 16
    bx = np.full((n, n, n), 0.2)
    f = sff.VectorField3D(bx, np.zeros_like(bx), np.ones_like(bx), dx=1.0)
    q = sff.squashing_q(f, step=2)
    assert np.allclose(q["q"][q["face"] == "top"], 2.0, rtol=1e-3)


def test_analytic_validation_suite():
    """Tw, decay index, Q, potential, LFFF and NLFFF against exact solutions."""
    res = sff.run_validation(quick=True)
    failed = [(t["key"], t["error"], t["tol"]) for t in res if not t["passed"]]
    assert not failed, failed
