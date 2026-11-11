"""Quadrature, energy and helicity against exact answers (revisions R10, R12, R13)."""
import numpy as np

import solarff as sff
from solarff.analytic import dipole_field, low_lou_field
from solarff.field import VectorField3D, trapezoid_weights
from solarff.helicity import energy_decomposition, relative_helicity, volume_potential
from solarff.metrics import energy


def test_trapezoid_weights_integrate_exactly():
    # ∫ 1 dV over a box of nodes = (nx−1)(ny−1)(nz−1) h³; in 2D, (nx−1)(ny−1) h²
    assert np.isclose(trapezoid_weights((5, 7, 9)).sum(), 4 * 6 * 8)
    assert np.isclose(trapezoid_weights((5, 7)).sum(), 4 * 6)


def test_energy_of_uniform_field_is_exact():
    n, h, b = (10, 12, 8), 2.0, 100.0
    f = VectorField3D(np.zeros(n), np.zeros(n), np.full(n, b), dx=h)
    vol = (n[0] - 1) * (n[1] - 1) * (n[2] - 1) * (h * 1e8) ** 3
    assert np.isclose(energy(f), b * b / (8 * np.pi) * vol, rtol=1e-12)


def test_volume_potential_matches_normal_component_on_all_faces():
    f = low_lou_field(24, 24, 16)
    p = volume_potential(f)
    for a, b in ((f.bx[0], p.bx[0]), (f.bx[-1], p.bx[-1]), (f.by[:, 0], p.by[:, 0]), (f.by[:, -1], p.by[:, -1]),
                 (f.bz[:, :, 0], p.bz[:, :, 0]), (f.bz[:, :, -1], p.bz[:, :, -1])):
        assert np.allclose(a, b)


def test_potential_field_has_no_free_energy_or_helicity():
    # A potential field has no free energy. On the grid, E − E_p picks up the discrete divergence
    # of the sampled dipole and only tends to 0 when refining (2nd order: ÷4 when doubling the resolution);
    # E_J, which does not depend on solenoidality, is nearly zero already on the coarse grid.
    fr = []
    for n in (48, 96):
        f = dipole_field(nx=n, ny=n, nz=2 * n // 3)
        e = energy_decomposition(f)
        fr.append(e["E_free"] / e["E"])
        assert e["E_J"] / e["E"] < 1e-3, e["E_J"] / e["E"]
    assert fr[0] / fr[1] > 3.5 and abs(fr[1]) < 6e-3, fr
    f = dipole_field(nx=48, ny=48, nz=32)
    phi = np.abs(f.bz[:, :, 0]).sum() * (f.dx * 1e8) ** 2 / 2
    assert abs(relative_helicity(f)["H_V"]) / phi ** 2 < 1e-3


def test_energy_identity_and_helicity_gauge_converge_on_exact_field():
    # E − E_p = E_J and the gauge independence must improve when the grid is refined (R10, R12)
    out = []
    for n in (24, 40):
        f = low_lou_field(n, n, int(0.7 * n), n=1, m=1, l=0.3, phi=np.pi / 4)
        pv = volume_potential(f)
        e = energy_decomposition(f, pv)
        top, bot = relative_helicity(f, pv, "top")["H_V"], relative_helicity(f, pv, "bottom")["H_V"]
        out.append((abs(e["E_free"] - e["E_J"]) / e["E_J"], abs(top - bot) / abs(top)))
    assert out[1][0] < out[0][0] and out[1][1] < out[0][1], out
    assert out[1][0] < 0.03 and out[1][1] < 0.05, out


def test_base_potential_reproduces_base_bz(bipole):
    mag = bipole()
    p = sff.potential_field(mag, nz=30)
    assert np.abs(p.bz[:, :, 0] - mag.bz).sum() / np.abs(mag.bz).sum() < 1e-3


def test_energy_decomposition_sums_to_E_as_grid_is_refined():
    # E = E_p,s + E_J,s + E_p,ns + E_J,ns + E_mix (Valori et al. 2013) assumes ∫B_p,s·B_J,s and
    # ∫B_p,ns·B_J,ns vanish, which on the grid only happens when refining: 2.0 % (n = 24) → 0.4 % (n = 96)
    err = []
    for n in (32, 64):
        e = energy_decomposition(low_lou_field(n, n, int(0.7 * n)))
        parts = e["E_p_s"] + e["E_J_s"] + e["E_p_ns"] + e["E_J_ns"] + e["E_mix"]
        err.append(abs(parts / e["E"] - 1))
    assert err[1] < 0.6 * err[0] and err[1] < 0.01, err
