"""Tiny transform checks. No concrete download and no model fit."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "numeric_transforms_concrete",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_selection_uses_cv_rmse_not_the_holdout():
    main = load_main()
    scores = {
        "pretty_test": {"cv_rmse_mean": 5.0, "n_features": 2, "test_rmse": 0.1},
        "cv_winner": {"cv_rmse_mean": 1.0, "n_features": 9, "test_rmse": 9.0},
        "tie_many": {"cv_rmse_mean": 1.0, "n_features": 4, "test_rmse": 0.01},
    }
    assert main.select_experiment(scores) == "tie_many"


def test_log1p_screen_ignores_a_mild_column():
    main = load_main()
    mild = np.arange(1, 13, dtype=float)
    skewed = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 30], dtype=float)
    matrix = np.column_stack([mild, skewed])
    transformer = main.Log1pSkewed(threshold=1.0).fit(matrix)
    assert transformer.columns_.tolist() == [1]
    transformed = transformer.transform(matrix)
    assert transformed[0, 0] == pytest.approx(1.0)
    assert transformed[-1, 1] == pytest.approx(np.log1p(30))


def test_water_cement_uses_the_ingredient_columns():
    main = load_main()
    row = np.array([[10.0, 0.0, 0.0, 4.0]], dtype=float)
    transformed = main.WaterCement().fit(row).transform(row)
    assert transformed[0, -1] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="cement must be positive"):
        main.WaterCement().fit(np.array([[0.0, 0.0, 0.0, 4.0]]))


def test_unshifted_box_cox_rejects_a_zero():
    main = load_main()
    values = np.array([[1.0], [0.0], [2.0], [3.0]])
    error = main.unshifted_box_cox_error(values)
    assert error is not None
    assert "strictly positive" in error


def test_shifted_box_cox_shifts_only_the_nonpositive_column():
    main = load_main()
    positive = np.linspace(1.0, 4.0, 12)
    zeros = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 1.0, 0.0, 2.0, 0.0, 1.0, 4.0])
    matrix = np.column_stack([zeros, positive])
    transformer = main.ShiftedBoxCox(shift=1.0).fit(matrix)
    assert transformer.shift_.tolist() == [1.0, 0.0]
    assert transformer.transform(matrix).shape == matrix.shape


def test_coarse_age_bins_follow_the_registered_edges():
    main = load_main()
    ages = [1.0, 7.0, 8.0, 365.0]
    rows = np.zeros((len(ages), 8), dtype=float)
    rows[:, main.AGE] = ages
    transformed = main.CoarseAgeBins().fit(rows).transform(rows)
    assert transformed.shape == (4, 11)
    assert transformed[0, 7:].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert transformed[1, 7:].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert transformed[2, 7:].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert transformed[3, 7:].tolist() == [0.0, 0.0, 0.0, 1.0]
    outside = rows.copy()
    outside[0, main.AGE] = 366.0
    with pytest.raises(ValueError, match="outside"):
        main.CoarseAgeBins().fit(rows).transform(outside)


def test_recipe_groups_ignore_age():
    main = load_main()
    frame = pd.DataFrame(
        {
            "Cement": [1.0, 1.0, 2.0],
            "Blast Furnace Slag": [0.0, 0.0, 0.0],
            "Fly Ash": [0.0, 0.0, 0.0],
            "Water": [1.0, 1.0, 1.0],
            "Superplasticizer": [0.0, 0.0, 0.0],
            "Coarse Aggregate": [1.0, 1.0, 1.0],
            "Fine Aggregate": [1.0, 1.0, 1.0],
            "Age": [1, 28, 1],
        }
    )
    groups = main.recipe_groups(frame)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_reported_model_was_not_chosen_on_the_holdout():
    payload = json.loads((PROJECT / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["protocol"]["reported_model"] == "poly3"
    assert payload["protocol"]["selection_uses_test_scores"] is False
    assert payload["split"]["n_groups_in_both"] == 0
    assert payload["transforms"]["log1p_columns"] == ["Age"]
    assert "strictly positive" in payload["transforms"]["unshifted_box_cox_error"]
    interval = payload["bootstrap"]["log1p_rmse_minus_selected"]
    assert interval["gap_p2_5"] < 0 < interval["gap_p97_5"]
    mean_gap = payload["bootstrap"]["training_mean_rmse_minus_selected"]
    assert mean_gap["gap_p2_5"] > 0


def test_readme_quotes_committed_metrics():
    metrics = (PROJECT / "results" / "metrics.json").read_text(encoding="utf-8")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    linkedin = (PROJECT / "LINKEDIN.md").read_text(encoding="utf-8")
    missing = [number for number in re.findall(r"`([0-9eE.+-]+)`", readme) if number not in metrics]
    assert missing == []
    for number in ("6.190033630528194", "15.664861218083027", "-0.06188102255652603"):
        assert number in readme
        assert number in linkedin
