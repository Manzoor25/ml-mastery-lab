"""Tiny checks for the cleanup helpers. No Adult download and no full fit."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "eda_messy_cleanup",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_income_period_is_not_part_of_the_class():
    main = load_main()
    assert main.normalize_income(" >50K.") == ">50K"
    assert main.normalize_income("<=50K") == "<=50K"


def test_iqr_fence_collapses_when_the_quartiles_are_zero():
    main = load_main()
    fences = main.iqr_fences(np.array([0.0, 0.0, 0.0, 0.0, 10.0]))
    assert fences["low"] == 0.0
    assert fences["high"] == 0.0


def test_mcar_mask_is_reproducible():
    main = load_main()
    first = main.mcar_mask(4000, 0.25, 42)
    second = main.mcar_mask(4000, 0.25, 42)
    assert np.array_equal(first, second)
    assert 0.2 < float(first.mean()) < 0.3


def test_mnar_mask_hits_positive_values_more_often():
    main = load_main()
    values = np.concatenate([np.zeros(3000), np.full(3000, 5.0)])
    mask = main.mnar_mask(values, 42)
    assert float(mask[3000:].mean()) > float(mask[:3000].mean())


def test_majority_class_is_the_frequent_label():
    main = load_main()
    baseline = main.constant_baseline(np.array([0, 0, 0, 1]), np.array([0, 1]))
    assert baseline["majority_class"] == 0
    assert baseline["majority_class_accuracy"] == 0.5


def test_mean_imputer_does_not_use_the_holdout():
    train = np.array([[1.0], [np.nan], [3.0]])
    holdout = np.array([[100.0], [np.nan]])
    fitted = SimpleImputer(strategy="mean").fit(train)
    leaked = SimpleImputer(strategy="mean").fit(np.vstack([train, holdout]))
    assert fitted.statistics_[0] == np.nanmean(train)
    assert not np.isclose(fitted.statistics_[0], leaked.statistics_[0])


def test_readme_quotes_committed_metrics():
    metrics = (PROJECT / "results" / "metrics.json").read_text(encoding="utf-8")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    linkedin = (PROJECT / "LINKEDIN.md").read_text(encoding="utf-8")
    missing = [number for number in re.findall(r"`([0-9eE.+-]+)`", readme) if number not in metrics]
    assert missing == []
    assert "0.42049919201950064" in readme
    assert "0.42049919201950064" in linkedin
    assert "6846.322714681441" in readme
    payload = json.loads(metrics)
    assert payload["official_split"]["n_raw_label_overlap"] == 0
