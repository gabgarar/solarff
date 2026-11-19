"""Performance metrics, design of the confidence model, uncertainty budget and citations."""
import numpy as np
import pytest

import solarff as sff
from solarff.calibration import bootstrap_ci, brier, ece
from solarff.confidence import auc


def test_potential_departure_is_zero_for_the_potential_field_and_grows_with_the_currents(lowlou):
    """The departure from the potential line: nothing when the model is the potential field itself, and more with currents."""
    mag, exact = lowlou
    pot = sff.potential_field(mag, nz=20)
    seeds = sff.seeds_from_magnetogram(mag, n=40, method="random", polarity="both", threshold=0.05, rng=1)
    d0 = sff.potential_departure(pot, sff.trace_field_lines(pot, seeds), mag)
    d1 = sff.potential_departure(exact, sff.trace_field_lines(exact, seeds), mag)
    assert d0.shape == (len(seeds),) and np.nanmax(d0) < 1e-9
    assert np.nanmedian(d1) > 0.02 and np.nanmin(d1) >= 0
    assert sff.potential_departure(pot, [], mag).shape == (0,)


def test_rank_lines_orders_the_lines_and_gives_no_confidence(lowlou):
    """The ranking fitted on MHD simulations: an order from the most to the least reliable line, without a threshold."""
    from solarff.confidence import CALIBRATION_MHD, CALIBRATION_METRICS_MHD, calibration_metrics
    mag, exact = lowlou
    lines = sff.trace_field_lines(exact, sff.seeds_from_magnetogram(mag, n=40, method="random", polarity="both", threshold=0.05, rng=1))
    r = sff.rank_lines(exact, lines, mag)
    n, ok = len(lines), np.isfinite(r["score"])
    assert r["score"].shape == (n,) and sorted(r["order"].tolist()) == list(range(n)) and ok.sum() > n // 2
    assert np.all(np.diff(r["score"][r["order"][:ok.sum()]]) >= 0)             # from the most to the least reliable
    assert np.nanmin(r["percentile"]) == 0 and np.nanmax(r["percentile"]) == 1 and np.isnan(r["percentile"][~ok]).all()
    assert np.allclose(r["indicators"]["departure"], sff.potential_departure(exact, lines, mag), equal_nan=True)
    # the departure is the heaviest term, and a line that departs more gets a worse place
    k = CALIBRATION_MHD["features"].index("departure")
    assert int(np.argmax(np.abs(CALIBRATION_MHD["coef"]))) == k and CALIBRATION_MHD["coef"][k] > 0
    assert CALIBRATION_MHD["threshold"] is None and calibration_metrics(CALIBRATION_MHD) is CALIBRATION_METRICS_MHD
    with pytest.raises(ValueError, match="rank_lines"):
        sff.assess_lines(exact, lines, mag=mag, calibration=CALIBRATION_MHD)


def test_auc_ece_brier_match_reference_implementations():
    from sklearn.metrics import brier_score_loss, roc_auc_score
    rng = np.random.default_rng(3)
    p = rng.uniform(size=4000)
    y = rng.uniform(size=4000) < p ** 1.3
    p[:50] = 0.5  # ties
    assert np.isclose(auc(p, y), roc_auc_score(y, p), atol=1e-12)
    assert np.isclose(brier(p, y), brier_score_loss(y, p), atol=1e-12)
    idx = np.minimum((p * 10).astype(int), 9)  # ECE by hand: 10 equal-width bins
    manual = sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(10) if (idx == b).any())
    assert np.isclose(ece(p, y), manual, atol=1e-12)


def test_calibration_metrics_and_bootstrap():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)  # perfectly calibrated
    assert ece(p, y) < 0.02
    assert abs(brier(p, y) - 1 / 6) < 0.01  # E[p(1-p)] with uniform p
    assert ece(np.clip(p + 0.2, 0, 1), y) > 0.1
    lo, hi = bootstrap_ci(p, y, brier, n_boot=100)
    assert lo < brier(p, y) < hi
    g = np.repeat(np.arange(20), 1000)
    lo2, hi2 = bootstrap_ci(p, y, brier, groups=g, n_boot=100)
    assert lo2 < brier(p, y) < hi2


def test_length_sq_and_calibration_selection():
    from solarff.confidence import (CALIBRATION_METRICS_MC, DEFAULT_CALIBRATION, LENGTH_LOG_CENTER, _design,
                                    confidence, mc_gain_significant, select_calibration)
    L = np.array([10 ** LENGTH_LOG_CENTER, 0.01, 1.0])
    ind = {"sin_jb": np.full(3, 0.1), "alpha_spread": np.full(3, 0.1), "buffer": np.zeros(3),
           "apex": np.full(3, 0.2), "length": L, "length_h": L / 0.49}  # a box of the suite: height = 0.49 diagonals
    X = _design(ind, ["length", "length_sq"])
    assert abs(X[0, 1]) < 1e-12 and X[1, 1] > 0 and X[2, 1] > 0  # U centred on the typical length
    c = confidence(ind, DEFAULT_CALIBRATION)
    assert c[0] > c[1] and c[0] > c[2]  # the very short and the very long ones, less reliable
    expected_mc = CALIBRATION_METRICS_MC["auc_gain_mc_ci"][0] > 0
    assert mc_gain_significant() == expected_mc
    assert ("mc_instability" in select_calibration(("mc_instability",))["features"]) == expected_mc


def test_published_model_has_physical_signs():
    from solarff.confidence import CALIBRATION_V1, DEFAULT_CALIBRATION
    assert DEFAULT_CALIBRATION["features"] == ["alpha_spread", "buffer", "length", "length_sq", "length_h"]
    assert CALIBRATION_V1["features"] == ["alpha_spread", "buffer", "length", "length_sq"]  # the model of version 1.1
    for cal in (DEFAULT_CALIBRATION, CALIBRATION_V1):
        assert all(b > 0 for b in cal["coef"])  # more indicator → more likely to be wrong


def test_length_in_box_heights_lowers_the_confidence_in_flat_boxes():
    import pytest
    from solarff.confidence import CALIBRATION_V1, DEFAULT_CALIBRATION, confidence, length_in_heights
    ind = {"alpha_spread": np.full(2, 0.1), "buffer": np.zeros(2), "length": np.full(2, 0.3)}
    with pytest.raises(KeyError):      # the default model needs the length in box heights
        confidence(ind, DEFAULT_CALIBRATION)
    # the same line (same length relative to the horizontal diagonal) in a box of the suite and in a flat one
    ind["length_h"] = np.array([length_in_heights(0.3, (64, 64, 45)), length_in_heights(0.3, (64, 64, 16))])
    c = confidence(ind, DEFAULT_CALIBRATION)
    assert c[1] < c[0]                                                   # in the flat box it reaches the top
    assert np.ptp(confidence(ind, CALIBRATION_V1)) == 0                  # version 1.1 does not see the difference


def test_uncertainty_budget():
    from solarff.budget import magnitude, uncertainty_budget
    mc = {"n": 10, "error_scale": 1.0, "summary": {
        "E_free": {"nominal": 1.0, "p16": 0.98, "p50": 1.05, "p84": 1.10},
        "E_free_vol": {"nominal": 2.0, "p16": 1.98, "p50": 2.0, "p84": 2.02},
        "H_V": {"nominal": 3.0, "p16": 2.9, "p50": 3.0, "p84": 3.1}}}
    methods = {"spread": {}, "excluded": ["gr_p"], "models": {"nlfff": {"E_free": 1.0, "E_free_vol": 1.8}}}
    rob = {"size": 40, "cases": ["a"], "parameters": {
        "default": {"efree_err": -0.1, "efree_vol_err": 0.0, "hv_err": 0.0},
        "mu4=0.1": {"efree_err": -0.9, "efree_vol_err": -0.3, "hv_err": 0.1},
        "eps=0.1": {"efree_err": -0.05, "efree_vol_err": 0.02, "hv_err": 0.0}}}
    bias = {"size": 40, "refs": {"f": {"model_clean": {"E_free": 0.9, "E_free_vol": 1.1, "H_V": 1.0},
                                       "exact": {"E_free": 1.0, "E_free_vol": 1.0, "H_V": 1.0}}}}
    B = uncertainty_budget({"mc": mc, "methods": methods}, robustness=rob, bias=bias)
    rows = {r["key"]: r for r in B["rows"]}
    assert abs(rows["noise_random"]["values"]["E_free"][1] - 0.06 / 1.05) < 1e-12
    assert abs(rows["noise_bias"]["values"]["E_free"][1] - 0.05) < 1e-12
    assert rows["preprocessing"]["values"]["E_free"] == (-0.8, 0.0)
    assert abs(rows["nlfff_params"]["values"]["E_free"][1] - 0.05) < 1e-12
    assert abs(magnitude(rows["method_exact"]["values"]["E_free_vol"]) - 0.1) < 1e-12
    assert not rows["method_gr"]["detail"]["estimated"]
    assert B["dominant"]["E_free"] == "preprocessing"
    assert abs(B["definition"]["ratio"] - 1.8) < 1e-12


def test_citations_resolve_to_author_year():
    import re
    from solarff.report.texts import BIBTEX, REFERENCES, resolve_citations
    keys = [k for k, _ in REFERENCES]
    assert sorted(keys) == sorted(re.findall(r"@\w+\{(\w+),", BIBTEX))
    out = resolve_citations(r"x \citep{Guo2017} y \citep[GAM;][]{HastieTibshirani1986} \citet{Brier1950}")
    assert "(Guo et al. 2017)" in out and "(GAM; Hastie" in out and "Brier (1950)" in out
    for k in keys:  # every key can be cited
        assert "cite" not in resolve_citations(r"\citep{%s}" % k)
