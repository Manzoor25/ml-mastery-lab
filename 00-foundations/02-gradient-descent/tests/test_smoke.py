"""Tiny checks for the gradient and step-size helpers. No full wine run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "foundations_gradient_descent",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gradient_matches_finite_difference():
    main = load_main()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(25, 3))
    y = rng.normal(size=25)
    w = rng.normal(size=3)
    analytic = main.grad_mse(x, y, w)
    numeric = main.finite_difference_grad(x, y, w)
    np.testing.assert_allclose(analytic, numeric, atol=1e-5)


def test_closed_form_recovers_a_noiseless_line():
    main = load_main()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3))
    w = np.array([0.5, -1.0, 0.25])
    found = main.closed_form(x, x @ w)
    np.testing.assert_allclose(found, w, atol=1e-8)


def test_safe_step_decreases_loss():
    main = load_main()
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 2))
    y = x @ np.array([1.0, -0.5]) + rng.normal(scale=0.1, size=30)
    _small, smooth = main.hessian_spectrum(x)
    fitted = main.run_optimizer(
        x,
        y,
        np.zeros(2),
        "batch",
        8,
        1.0 / smooth,
        np.random.default_rng(3),
    )
    assert fitted["history"][-1] < fitted["history"][0]
    assert fitted["exploded"] is False


def test_too_large_step_explodes():
    main = load_main()
    x = np.ones((8, 1))
    y = np.zeros(8)
    fitted = main.run_optimizer(
        x,
        y,
        np.array([1.0]),
        "batch",
        30,
        1.25,
        np.random.default_rng(0),
    )
    assert fitted["exploded"] is True


def test_scale_uses_train_moments_only():
    main = load_main()
    x_train = np.array([[0.0, 1.0], [2.0, 3.0]])
    x_test = np.array([[8.0, 9.0]])
    train_s, test_s = main.scale_from_train(x_train, x_test)
    np.testing.assert_allclose(train_s.mean(axis=0), 0.0, atol=1e-12)
    assert float(test_s.mean()) != 0.0


def test_readme_quotes_committed_metrics():
    metrics = json.loads((PROJECT / "results" / "metrics.json").read_text(encoding="utf-8"))
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    linkedin = (PROJECT / "LINKEDIN.md").read_text(encoding="utf-8")
    wine = {run["name"]: run for run in metrics["wine_runs"]}
    surface = {run["name"]: run for run in metrics["surface_runs"]}
    quoted = [
        f"{metrics['mean_baseline_test_mse']:.4f}",
        f"{metrics['closed_form_test_mse']:.4f}",
        f"{metrics['closed_form_train_mse']:.4f}",
        str(wine["batch_gd_too_large"]["epochs_completed"]),
        f"{wine['adam_paper_default']['final_test_mse']:.4f}",
        str(surface["sgd"]["epochs_to_1pct"]),
    ]
    for snippet in quoted:
        assert snippet in readme, snippet
    assert f"{metrics['closed_form_test_mse']:.4f}" in linkedin
    assert f"{metrics['mean_baseline_test_mse']:.4f}" in linkedin
