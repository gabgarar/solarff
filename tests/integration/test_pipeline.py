"""Full workflow against exact fields: preprocessing → NLFFF → study → report → Monte Carlo."""
import numpy as np
import pytest

import solarff as sff
from solarff.helicity import energy_decomposition
from solarff.metrics import compare_vector_fields, energy


def test_nlfff_improves_on_potential(lowlou):
    mag, exact = lowlou
    pot = sff.potential_field(mag, nz=20)
    nl = sff.nlfff(mag, nz=20, max_iter=600, verbose=False)
    m = nl.meta["nd"]
    c_pot = compare_vector_fields(exact, pot, margin=m)["C_vec"]
    c_nl = compare_vector_fields(exact, nl, margin=m)["C_vec"]
    assert c_nl > c_pot
    assert c_nl > 0.97
    assert np.allclose(nl.bx[:, :, 0], mag.bx)  # the base stays fixed


def test_nlfff_lbfgs_recovers_low_lou_32():
    """Low & Lou case (n = m = 1, l = 0.3, Φ = π/4; Schrijver et al. 2006) at low resolution."""
    mag, ex = sff.low_lou_magnetogram(nx=32, ny=32, nz=22)
    st = sff.run_study(mag, 22, nlfff_method="lbfgs", preprocess=False, verbose=False)
    nd = int(st.nlfff.meta.get("nd", 4))
    c = compare_vector_fields(ex, st.nlfff, margin=nd)
    assert c["C_vec"] > 0.96, c
    # 32 points resolve the field poorly: +7 % in the total energy (at 64³, 0.0 %; see the slow test)
    assert abs(energy(st.nlfff, nd) / energy(ex, nd) - 1) < 0.10


@pytest.mark.slow
def test_nlfff_lbfgs_recovers_low_lou_64():
    """The same case at 64³ (≈10 min): direction and energy nearly exact, free energy within 15 %."""
    mag, ex = sff.low_lou_magnetogram(nx=64, ny=64, nz=64)
    st = sff.run_study(mag, 64, nlfff_method="lbfgs", preprocess=False, verbose=False)
    nd = int(st.nlfff.meta.get("nd", 4))
    assert compare_vector_fields(ex, st.nlfff, margin=nd)["C_vec"] > 0.99
    assert abs(energy(st.nlfff, nd) / energy(ex, nd) - 1) < 0.02
    ratio = energy_decomposition(st.nlfff)["E_free"] / energy_decomposition(ex)["E_free"]
    assert abs(ratio - 1) < 0.15, ratio


def test_preprocess_reduces_force():
    mag, _ = sff.low_lou_magnetogram(noise=50, nx=32, ny=32, nz=20)
    pp = sff.preprocess(mag, verbose=False)
    assert sff.aly_criteria(pp)["eps_force"] < sff.aly_criteria(mag)["eps_force"]


def test_reference_potential_shares_base_with_nlfff(lowlou_study):
    # E − E_pot is a free energy only if the potential and the NLFFF share B_z at the base (R13)
    st = lowlou_study
    src = st.mag_pp if st.mag_pp is not None else st.mag
    ref = sff.reference_potential(st)
    assert np.abs(ref.bz[:, :, 0] - src.bz).sum() / np.abs(src.bz).sum() < 1e-3
    assert np.abs(st.nlfff.bz[:, :, 0] - src.bz).sum() / np.abs(src.bz).sum() < 1e-3


def test_base_and_volume_free_energy_agree(lowlou_study):
    # With the right reference potential, the two free energies must be similar
    st = lowlou_study
    e_base = energy(st.nlfff) - energy(sff.reference_potential(st))
    e_vol = energy_decomposition(st.nlfff)["E_free"]
    assert abs(e_base / e_vol - 1) < 0.1, e_base / e_vol


def test_report_generation(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from solarff.report.latex import find_pdflatex
    mag, exact = sff.low_lou_magnetogram(nx=24, ny=24, nz=16)
    st = sff.run_study(mag, nz=16, exact=exact, preprocess=True,
                       nlfff_kw={"max_iter": 150}, n_seeds=30, verbose=False)
    out = sff.generate_report(st, str(tmp_path / "rep"), compile=find_pdflatex() is not None,
                              verbose=False)
    for lang in ("es", "en"):
        tex = open(out["tex"][lang], encoding="utf-8").read()
        assert r"\section" in tex and "<<" not in tex
        if find_pdflatex():
            assert (tmp_path / "rep" / ("informe_es.pdf" if lang == "es" else "report_en.pdf")).exists()
    for f in ("methods_es.tex", "methods_en.tex", "table_quality.tex", "references.bib", "results.json"):
        assert (tmp_path / "rep" / "paper" / f).exists()


def test_monte_carlo_checkpoint_resume(tmp_path):
    """Resuming an interrupted Monte Carlo gives the same result as without interruption."""
    import pickle
    mag, _ = sff.low_lou_magnetogram(noise=30, nx=20, ny=20, nz=12)
    st = sff.run_study(mag, nz=12, verbose=False, nlfff_kw={"max_iter": 80})
    full = sff.monte_carlo(st, n=3, noise=30, verbose=False)
    ck = str(tmp_path / "mc.pkl")
    sff.monte_carlo(st, n=3, noise=30, verbose=False, checkpoint=ck)
    s = pickle.load(open(ck, "rb"))
    s["done"] = 1
    s["values"] = {k: v[:1] for k, v in s["values"].items()}
    s["good"][1:] = False
    pickle.dump(s, open(ck, "wb"))
    res = sff.monte_carlo(st, n=3, noise=30, verbose=False, checkpoint=ck)
    assert np.allclose(full["values"]["E_free"], res["values"]["E_free"])
    assert np.array_equal(full["stability"], res["stability"])
