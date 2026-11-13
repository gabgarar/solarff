"""The published confidence figures are reproduced from the saved data.

They use the calibration suite (outside git); skipped if absent.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.needs_data


def test_published_coefficients_are_reproduced(calibration_suite):
    from solarff.calibration import USED_FEATURES, stack
    from solarff.confidence import DEFAULT_CALIBRATION, _fit_logistic
    X, y, _ = stack(calibration_suite, USED_FEATURES)
    cal = _fit_logistic(X, y.astype(float))
    assert np.isclose(cal["intercept"], DEFAULT_CALIBRATION["intercept"], atol=2e-3)
    assert np.allclose(cal["coef"], DEFAULT_CALIBRATION["coef"], atol=2e-3)
    assert np.allclose(cal["mu"], DEFAULT_CALIBRATION["mu"], atol=1e-6)


def test_published_auc_ece_are_reproduced(calibration_suite):
    from solarff.calibration import USED_FEATURES, cross_validate
    from solarff.confidence import CALIBRATION_METRICS
    cv = cross_validate(calibration_suite, USED_FEATURES, models=["logistic"], group="field", n_boot=10,
                        verbose=False)
    m = cv["models"]["logistic"]
    assert abs(m["auc"] - CALIBRATION_METRICS["auc_combined"]) < 1e-3, m["auc"]
    assert abs(m["ece"] - CALIBRATION_METRICS["ece"]) < 1e-3, m["ece"]
    assert CALIBRATION_METRICS["n_cases"] == len(calibration_suite) == 162


def test_version_1_1_model_is_reproduced(calibration_suite_v1):
    from solarff.calibration import V1_FEATURES, cross_validate, stack
    from solarff.confidence import CALIBRATION_METRICS_V1, CALIBRATION_V1, _fit_logistic
    X, y, _ = stack(calibration_suite_v1, V1_FEATURES)
    cal = _fit_logistic(X, y.astype(float))
    assert np.isclose(cal["intercept"], CALIBRATION_V1["intercept"], atol=2e-3)
    assert np.allclose(cal["coef"], CALIBRATION_V1["coef"], atol=2e-3)
    cv = cross_validate(calibration_suite_v1, V1_FEATURES, models=["logistic"], group="field", n_boot=10, verbose=False)
    assert abs(cv["models"]["logistic"]["auc"] - CALIBRATION_METRICS_V1["auc_combined"]) < 1e-3
    assert CALIBRATION_METRICS_V1["n_cases"] == len(calibration_suite_v1) == 108
