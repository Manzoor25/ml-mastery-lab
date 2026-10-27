"""Tiny checks for the linear algebra and resampling helpers. No full run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "foundations_linalg_probability_stats",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_standardize_centers_and_scales():
    main = load_main()
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(200, 3))
    z = main.standardize(x)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=0, ddof=1), 1.0, atol=1e-10)


def test_pca_puts_mass_on_the_high_variance_axis():
    main = load_main()
    rng = np.random.default_rng(1)
    x = np.column_stack(
        [
            rng.normal(scale=8.0, size=500),
            rng.normal(scale=0.5, size=500),
        ]
    )
    centered = main.center(x)
    evals, evecs = main.pca_eigh(centered)
    fractions = main.variance_fractions(evals)
    assert fractions[0] > 0.95
    recon = main.reconstruct(centered, evecs, k=1)
    assert main.column_r2(centered[:, 0], recon[:, 0]) > 0.95


def test_percentile_ci_matches_quantiles():
    main = load_main()
    samples = np.arange(1, 101, dtype=float)
    low, high = main.percentile_ci(samples, alpha=0.05)
    expected = np.quantile(samples, [0.025, 0.975])
    np.testing.assert_allclose([low, high], expected)


def test_bootstrap_is_reproducible():
    main = load_main()
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 0.5, 1.0])
    first = main.bootstrap_mean_diff(a, b, 30, np.random.default_rng(7))
    second = main.bootstrap_mean_diff(a, b, 30, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert main.mean_diff(a, b) == 2.0


def test_permutation_pvalue_is_inside_unit_interval():
    main = load_main()
    rng = np.random.default_rng(3)
    a = rng.normal(size=20)
    b = rng.normal(size=18)
    pvalue = main.permutation_pvalue(a, b, n_perm=40, rng=np.random.default_rng(4))
    assert 0.0 < pvalue <= 1.0


def test_readme_quotes_committed_metrics():
    metrics = json.loads((PROJECT / "results" / "metrics.json").read_text(encoding="utf-8"))
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    linkedin = (PROJECT / "LINKEDIN.md").read_text(encoding="utf-8")
    quoted = [
        f"{metrics['unscaled_pc1_variance_fraction']:.4f}",
        f"{metrics['scaled_top2_variance_fraction']:.4f}",
        f"{metrics['alcohol_mean_diff']:.4f}",
        f"{metrics['alcohol_welch_pvalue']:.6e}",
        f"{metrics['null_family_wise_error_rate']:.3f}",
    ]
    for snippet in quoted:
        assert snippet in readme, snippet
    assert f"{metrics['null_family_wise_error_rate']:.3f}" in linkedin
