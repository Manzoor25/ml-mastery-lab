"""Tiny checks for the bias-variance helpers. No wine download and no full run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "foundations_bias_variance",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_decomposition_matches_a_hand_calculation():
    main = load_main()
    truth = np.zeros(4)
    predictions = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0, 3.0],
        ]
    )
    parts = main.decomposition(predictions, truth)
    assert parts["bias2"] == 1.5
    assert parts["variance"] == 1.0


def test_polynomial_recovers_a_noiseless_quadratic():
    main = load_main()
    x = np.linspace(0.0, 1.0, 20)
    y = 1.5 - 2.0 * x + 0.5 * x**2
    predicted = main.predict_polynomial(x, y, x, 2)
    np.testing.assert_allclose(predicted, y, atol=1e-8)


def test_knn_with_all_neighbors_is_the_mean():
    main = load_main()
    x = np.array([0.0, 0.2, 0.8])
    y = np.array([1.0, 3.0, 5.0])
    predicted = main.predict_knn(x, y, np.array([0.1, 10.0]), n_neighbors=3)
    np.testing.assert_allclose(predicted, [3.0, 3.0])


def test_mean_ignores_the_rows_being_scored():
    main = load_main()
    predicted = main.predict_mean(np.array([1.0, 3.0]), 2)
    np.testing.assert_allclose(predicted, [2.0, 2.0])


def test_linear_scaler_uses_train_rows_only():
    main = load_main()
    x_train = np.array([[0.0], [2.0]])
    y_train = np.array([0.0, 2.0])
    predicted = main.fit_predict_table(x_train, y_train, np.array([[10.0]]), "linear", 1)
    np.testing.assert_allclose(predicted, [10.0])


def test_readme_quotes_committed_metrics():
    metrics_path = PROJECT / "results" / "metrics.json"
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    linkedin = (PROJECT / "LINKEDIN.md").read_text(encoding="utf-8")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    wine = metrics["wine"]
    selected = wine["selected_by_mean_inner_val_mse"]
    assert selected in readme
    assert f"{wine['selected_holdout_mse']:.4f}" in readme
    assert f"{wine['selected_holdout_mse']:.4f}" in linkedin
    mean_holdout = next(run["holdout_mse"] for run in wine["runs"] if run["name"] == "mean")
    assert f"{mean_holdout:.4f}" in readme
    assert f"{mean_holdout:.4f}" in linkedin
