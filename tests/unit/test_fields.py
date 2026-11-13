"""Analytical fields, potential, LFFF and synthetic magnetograms."""
import numpy as np
import pytest

import solarff as sff


def test_lowlou_eigenvalue():
    from solarff.analytic import low_lou_eigenvalue
    assert low_lou_eigenvalue(1, 1) == pytest.approx(0.4274, abs=1e-3)


def test_lowlou_is_force_free(lowlou):
    _, exact = lowlou
    # Discretisation error at 32^3; a potential field with this Bz gives ~0.9
    assert sff.force_free_metrics(exact)["CWsin"] < 0.1


def test_potential_is_current_free(bipole):
    f = sff.potential_field(bipole(), nz=20)
    j = np.array(f.current())[:, 2:-2, 2:-2, 1:-2]
    assert np.abs(j).max() < 0.02 * np.abs(f.bz).max()
    assert np.allclose(f.bz[:, :, 0], bipole().bz, atol=1e-8)


def test_lfff_satisfies_curl_alpha_b(bipole):
    mag = bipole(n=96, w=300.0, d=14)
    a = 0.7 * sff.alpha_max(mag)
    f = sff.lfff(mag, a, nz=24)
    j = f.current()
    res = max(np.abs(j[i] - a * c)[3:-3, 3:-3, 1:-2].max() for i, c in enumerate((f.bx, f.by, f.bz)))
    assert res < 0.05 * abs(a) * f.babs.max()
    assert abs(f.divergence()[3:-3, 3:-3, 1:-2]).max() < 0.02 * f.babs.max()


def test_lfff_rejects_large_alpha(bipole):
    mag = bipole()
    with pytest.raises(ValueError):
        sff.lfff(mag, 1.1 * sff.alpha_max(mag), nz=4)


def test_alpha_fit_recovers_lfff_alpha(bipole):
    mag = bipole()
    a = 0.5 * sff.alpha_max(mag)
    f = sff.lfff(mag, a, nz=1)
    vmag = sff.Magnetogram(bz=f.bz[:, :, 0], bx=f.bx[:, :, 0], by=f.by[:, :, 0], dx=1.0)
    assert sff.alpha_best_fit(vmag, bt_min=10) == pytest.approx(a, rel=1e-3)


def test_origin_propagates(lowlou):
    mag, exact = lowlou
    pot = sff.potential_field(mag, nz=4)
    assert np.allclose(pot.origin, exact.origin)
    assert np.allclose(mag.rebin(2).x[0], mag.x[0] + 0.5 * mag.dx)


def test_lfff_multipole_is_exactly_linear_force_free():
    from solarff.analytic import lfff_multipole_field
    f = lfff_multipole_field(40, 40, 20, alpha_frac=0.7)
    jx, jy, jz = f.current()
    a = f.meta["alpha"]
    s = (slice(4, -4),) * 2 + (slice(1, -1),)
    res = np.sqrt((jx - a * f.bx) ** 2 + (jy - a * f.by) ** 2 + (jz - a * f.bz) ** 2)[s].sum()
    assert res / (abs(a) * f.babs[s].sum()) < 0.1  # discretisation error of the derivative
    assert abs(f.bz[:, :, 0].sum()) < 1e-6 * np.abs(f.bz[:, :, 0]).sum()


def test_field_magnetogram_anisotropic_noise():
    from solarff.analytic import field_magnetogram, lfff_multipole_field
    f = lfff_multipole_field(40, 40, 4)
    m = field_magnetogram(f, noise=10, noise_t=60, rng=1)
    assert abs(np.std(m.bz - f.bz[:, :, 0]) - 10) < 1.5
    assert abs(np.std(m.bx - f.bx[:, :, 0]) - 60) < 6
    assert m.has_errors and m.bx_err[0, 0] == 60


def test_titov_demoulin_equilibrium():
    from solarff.analytic import titov_demoulin_field
    from solarff.metrics import force_free_metrics
    cw = {c: force_free_metrics(titov_demoulin_field(40, 40, 28, current_factor=c))["CWsin"] for c in (0.5, 1.0, 2.0)}
    assert cw[1.0] < 0.12 and cw[1.0] < cw[0.5] and cw[1.0] < cw[2.0]  # equilibrium of Eq. (6)
    f = titov_demoulin_field(40, 40, 28)
    assert f.meta["I0"] < 0 < f.meta["I"]  # left-handed twist
    bz = f.bz[:, :, 0]
    assert abs(bz.sum()) < 0.05 * np.abs(bz).sum()  # nearly balanced flux at the base


def test_relaxed_titov_demoulin_is_more_force_free():
    from solarff.analytic import relaxed_titov_demoulin_field, titov_demoulin_field
    from solarff.metrics import force_free_metrics
    td, rl = titov_demoulin_field(32, 32, 22), relaxed_titov_demoulin_field(32, 32, 22)
    assert force_free_metrics(rl)["CWsin"] < 0.5 * force_free_metrics(td)["CWsin"]
    assert np.allclose(rl.bz[:, :, 0], td.bz[:, :, 0])  # fixed base


def test_linear_mhs_reduces_to_lfff(bipole):
    from solarff.lfff import linear_mhs
    mag = bipole(n=48, w=300.0, d=10)
    a = 0.5 * sff.alpha_max(mag)
    f, g = sff.lfff(mag, a, nz=12), linear_mhs(mag, a, 12, a=0.0, kappa=0.25)
    assert max(np.abs(getattr(f, c) - getattr(g, c)).max() for c in ("bx", "by", "bz")) < 1e-9 * f.babs.max()


def test_mhs_multipole_is_an_exact_equilibrium_with_a_forced_base():
    from solarff.analytic import mhs_multipole_field
    from solarff.field import curl
    from solarff.magnetogram import Magnetogram, aly_criteria
    f = mhs_multipole_field(64, 64, 44, alpha_frac=0.7, a=0.6, scale_mm=2.0)
    al, a, kappa = f.meta["alpha"], f.meta["a"], f.meta["kappa"]
    jx, jy, jz = curl(f.bx, f.by, f.bz, f.dx)
    fz = a * np.exp(-kappa * f.dx * np.arange(f.shape[2]))
    rx = jx - al * f.bx - fz * np.gradient(f.bz, f.dx, axis=1)
    ry = jy - al * f.by + fz * np.gradient(f.bz, f.dx, axis=0)
    rz = jz - al * f.bz
    s = (slice(4, -4), slice(4, -4), slice(1, -2))
    j = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2)
    assert np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)[s].sum() / j[s].sum() < 0.05      # satisfies its equation
    assert np.abs(f.divergence()[s]).max() * f.dx < 0.01 * f.babs.max()
    jxb = np.sqrt((jy * f.bz - jz * f.by) ** 2 + (jz * f.bx - jx * f.bz) ** 2 + (jx * f.by - jy * f.bx) ** 2)
    sin = lambda k: jxb[4:-4, 4:-4, k].sum() / (j * f.babs)[4:-4, 4:-4, k].sum()  # noqa: E731
    assert sin(1) > 0.3 and sin(24) < 0.02                                          # forced below, force-free above
    base = Magnetogram(bz=f.bz[:, :, 0], bx=f.bx[:, :, 0], by=f.by[:, :, 0], dx=f.dx)
    assert 0.05 < aly_criteria(base)["eps_force"] < 0.15                            # net force like that of HMI data
