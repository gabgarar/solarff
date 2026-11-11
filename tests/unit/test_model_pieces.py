"""Pieces of the confidence model against known answers (test revision, T1–T7).

Indicators, "wrong line" label, logistic fit, leak-free validation, H1 zones,
Monte Carlo perturbation and preprocessing on an already force-free base.
"""
import numpy as np
import pytest

import solarff as sff
from solarff.analytic import lfff_multipole_field, low_lou_field
from solarff.confidence import _fit_logistic, line_errors, line_indicators, reliable_volume
from solarff.field import VectorField3D
from solarff.tracing import FieldLine, seeds_from_magnetogram, trace_field_lines


# ------------------------------------------------------------------ T1: indicators
def test_alpha_spread_is_small_on_exact_force_free_fields():
    # In a force-free field α is constant along each line: alpha_spread ≈ 0
    # (measured: median 0.005 in the exact LFFF, 0.015 in Low & Lou; 0.07 in an NLFFF with 60 G of noise)
    for f in (lfff_multipole_field(40, 40, 28), low_lou_field(40, 40, 28)):
        s = seeds_from_magnetogram(f, n=120, method="random", polarity="both", threshold=0.05, rng=1)
        ind = line_indicators(f, trace_field_lines(f, s), nd=4)
        assert np.nanmedian(ind["alpha_spread"]) < 0.03


def test_buffer_length_apex_open_on_uniform_inclined_field():
    # Uniform B = (0.3, 0, 1): the line is a straight segment from the base to the top, of exact length
    n, h, nd = 24, 2.0, 4
    shape = (n, n, n)
    f = VectorField3D(np.full(shape, 0.3), np.zeros(shape), np.ones(shape), dx=h, meta={"nd": nd})
    # interior, and at y = 1 cell: B has no y component, so that line stays entirely in the side layer
    seeds = np.array([[8 * h, 12 * h, 0.0], [8 * h, 1 * h, 0.0]])
    lines = trace_field_lines(f, seeds, ds=0.25)
    ind = line_indicators(f, lines, nd=nd)
    L = (n - 1) * h * np.hypot(1.0, 0.3)
    diag = np.hypot((n - 1) * h, (n - 1) * h)
    assert ind["length"][0] == pytest.approx(L / diag, rel=1e-3)
    assert ind["apex"][0] == pytest.approx(1.0, abs=1e-6)  # reaches the top
    assert ind["open"][0] == 1.0                            # leaves through the top
    # The buffer layer occupies the top nd points: fraction ≈ nd/(n−1) of the length
    assert ind["buffer"][0] == pytest.approx(nd / (n - 1), abs=0.05)
    assert ind["buffer"][1] == pytest.approx(1.0, abs=0.02)  # the whole line inside the side layer


# ------------------------------------------------------------------ T2: label
def _line(points, closed):
    p = np.asarray(points, float)
    return FieldLine(points=p, seed=p[0], closed=closed)


def test_line_errors_on_shifted_lines():
    t = np.linspace(0, 1, 50)
    ref = np.column_stack([10 * t, np.zeros_like(t), 5 * np.sin(np.pi * t)])  # closed arc
    L = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
    d = 0.7
    moved = ref + [0.0, d, 0.0]
    e = line_errors([_line(moved, True)], [_line(ref, True)])
    assert e["shape_abs"][0] == pytest.approx(d, rel=1e-6)
    assert e["shape"][0] == pytest.approx(d / L, rel=1e-3)
    assert e["endpoint"][0] == pytest.approx(d, rel=1e-6)
    assert e["topology"][0] == 0
    e2 = line_errors([_line(moved, False)], [_line(ref, True)])
    assert e2["topology"][0] == 1


# ------------------------------------------------------------------ T3: logistic fit
def test_logistic_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 40000
    X = rng.normal(size=(n, 2)) * [2.0, 0.5] + [1.0, -3.0]
    Z = (X - X.mean(0)) / X.std(0)
    beta, b0 = np.array([1.5, -0.8]), -1.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(b0 + Z @ beta)))).astype(float)
    c = _fit_logistic(X, y, l2=0.0)
    assert c["intercept"] == pytest.approx(b0, abs=0.05)
    assert np.allclose(c["coef"], beta, atol=0.05)
    # The default penalty (L2 = 0.01 on the mean likelihood) shrinks the coefficients by
    # ~20 % regardless of n: it makes the model conservative (calibration slope > 1).
    cpen = _fit_logistic(X, y)
    shrink = np.array(cpen["coef"]) / np.array(c["coef"])
    assert np.all((shrink > 0.7) & (shrink < 0.9)), shrink


# ------------------------------------------------------------------ T4: leak-free validation
def test_leave_one_family_out_never_trains_on_the_left_out_family(monkeypatch):
    import solarff.calibration as cal
    rng = np.random.default_rng(1)
    fams = ["A", "B", "C", "D"]
    data = []
    for k, fam in enumerate(fams):
        m = 200
        data.append({"field": fam, "noise_model": "x", "size": 32,
                     "ind": {"buffer": np.full(m, float(k + 1))},  # the indicator encodes the family
                     "bad": rng.uniform(size=m) < 0.3})
    seen = []
    orig = cal._Logistic.fit

    def spy(self, X, y):
        seen.append(set(np.unique(X[:, 0]).tolist()))
        return orig(self, X, y)
    monkeypatch.setattr(cal._Logistic, "fit", spy)
    cal.cross_validate(data, features=["buffer"], models=["logistic"], group="field", n_boot=5, verbose=False)
    assert len(seen) == len(fams)
    for k, s in enumerate(seen):  # the families are visited in order: exactly the k-th one is missing
        assert s == {float(j + 1) for j in range(len(fams)) if j != k}


# ------------------------------------------------------------------ T5: H1 zones
def test_zones_are_disjoint_and_cover_the_box(bipole):
    f = sff.potential_field(bipole(n=40), nz=16)
    s = seeds_from_magnetogram(f, n=60, method="random", polarity="both", threshold=0.05, rng=2)
    lines = trace_field_lines(f, s)
    rel = np.arange(len(lines)) % 2 == 0
    v_rel = reliable_volume(f, lines, rel)
    v_unr = reliable_volume(f, lines, ~rel) & ~v_rel
    v_rest = ~(v_rel | v_unr)
    assert not (v_rel & v_unr).any() and not (v_rel & v_rest).any() and not (v_unr & v_rest).any()
    assert (v_rel | v_unr | v_rest).all()
    dens = np.random.default_rng(3).uniform(size=f.shape)
    assert dens[v_rel].sum() + dens[v_unr].sum() + dens[v_rest].sum() == pytest.approx(dens.sum(), rel=1e-12)


# ------------------------------------------------------------------ T6: Monte Carlo
def test_perturb_noise_level_and_flux():
    from solarff.montecarlo import perturb
    n = 200
    rng = np.random.default_rng(4)
    bz = rng.normal(0, 300, (n, n))
    mag = sff.Magnetogram(bz=bz, bx=np.zeros((n, n)), by=np.zeros((n, n)), dx=1.0,
                          bz_err=np.full((n, n), 10.0), bx_err=np.full((n, n), 60.0), by_err=np.full((n, n), 60.0))
    m = perturb(mag, np.random.default_rng(5), error_scale=2.0)
    assert np.std(m.bx - mag.bx) == pytest.approx(120.0, rel=0.02)
    assert np.std(m.bz - mag.bz) == pytest.approx(20.0, rel=0.02)
    assert m.bz.sum() == pytest.approx(mag.bz.sum(), abs=1e-6 * np.abs(mag.bz).sum())  # net flux preserved


# ------------------------------------------------------------------ T7: preprocessing
def test_preprocess_force_terms_do_not_distort_a_consistent_base():
    # On an already force-free base, the force and torque terms (without smoothing) hardly
    # change it (measured: 0.3 %). With the default values (μ3 = μ4 = 0.01) the smoothing does change it
    # by ~22 % in B_z and B_t: it is a choice of the method, not an error, and is discussed in the paper.
    mag, _ = sff.low_lou_magnetogram(nx=32, ny=32, nz=20)
    pp = sff.preprocess(mag, mu4=0.0, verbose=False)
    b0 = np.stack([mag.bx, mag.by, mag.bz])
    b1 = np.stack([pp.bx, pp.by, pp.bz])
    assert np.linalg.norm(b1 - b0) / np.linalg.norm(b0) < 0.01
    assert sff.aly_criteria(pp)["eps_force"] <= sff.aly_criteria(mag)["eps_force"] + 1e-3
