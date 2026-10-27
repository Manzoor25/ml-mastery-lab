"""Tiny checks for the closed forms and the logistic gradient. No full data run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "foundations_linear_logistic",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sigmoid_is_finite_at_extremes():
    main = load_main()
    z = np.array([-1000.0, 0.0, 1000.0])
    out = main.sigmoid(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[1], 0.5)
    assert out[0] < 1e-6
    assert out[2] > 1.0 - 1e-6


def test_logistic_gradient_matches_finite_difference():
    main = load_main()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4))
    y = (x @ np.array([0.4, -0.2, 0.1, 0.3]) + rng.normal(scale=0.5, size=30) > 0).astype(float)
    coef = rng.normal(size=4)
    intercept = 0.2
    lam = 0.05
    analytic, analytic_i = main.logistic_gradient(x, y, coef, intercept, lam)
    numeric, numeric_i = main.finite_difference_logistic(x, y, coef, intercept, lam)
    np.testing.assert_allclose(analytic, numeric, atol=1e-5)
    np.testing.assert_allclose(analytic_i, numeric_i, atol=1e-5)


def test_ols_recovers_a_noiseless_line():
    main = load_main()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3))
    design = main.add_intercept(x)
    weights = np.array([0.3, 0.5, -1.0, 0.25])
    found = main.ols_closed_form(design, design @ weights)
    np.testing.assert_allclose(found, weights, atol=1e-8)


def test_ridge_matches_sklearn_on_a_small_problem():
    main = load_main()
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 3))
    y = x @ np.array([1.0, -0.5, 0.25]) + rng.normal(scale=0.1, size=50)
    found = main.ridge_closed_form(x, y, 1.5)
    library = Ridge(alpha=1.5).fit(x, y)
    np.testing.assert_allclose(found[:-1], library.coef_, atol=1e-8)
    np.testing.assert_allclose(found[-1], library.intercept_, atol=1e-8)


def test_newton_matches_sklearn_penalty():
    main = load_main()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 4))
    y = (x[:, 0] + 0.3 * x[:, 1] + rng.normal(scale=0.8, size=80) > 0).astype(float)
    lam = main.sklearn_l2_strength(len(y), 1.0)
    fitted = main.newton_logistic(x, y, lam)
    library = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=200,
        tol=1e-12,
    ).fit(x, y.astype(int))
    np.testing.assert_allclose(fitted["coef"], library.coef_[0], atol=1e-5)
    np.testing.assert_allclose(fitted["intercept"], library.intercept_[0], atol=1e-5)


def test_scale_uses_train_moments_only():
    main = load_main()
    x_train = np.array([[0.0, 1.0], [2.0, 3.0]])
    x_test = np.array([[8.0, 9.0]])
    train_s, test_s, _, _ = main.scale_from_train(x_train, x_test)
    np.testing.assert_allclose(train_s.mean(axis=0), 0.0, atol=1e-12)
    assert float(test_s.mean()) != 0.0


def test_readme_quotes_committed_metrics():
    metrics = (PROJECT / "results" / "metrics.json").read_text(encoding="utf-8")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    linkedin = (PROJECT / "LINKEDIN.md").read_text(encoding="utf-8")
    payload = json.loads(metrics)
    ols = str(payload["housing"]["ols_holdout_mse"])
    logistic = str(payload["cancer"]["newton_holdout_log_loss"])
    separated = str(payload["cancer"]["unregularized_holdout_log_loss"])
    assert ols in readme
    assert logistic in readme
    assert logistic in linkedin
    assert separated in linkedin
