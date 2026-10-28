"""Datetime-feature checks. No traffic download and no model fit."""

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
        "traffic_datetime_features",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reported_model_is_chosen_on_training_folds_only():
    main = load_main()
    names = [spec["name"] for spec in main.experiment_specs()]
    assert names == list(main.EXPERIMENT_ORDER)
    eligible = {spec["name"] for spec in main.experiment_specs() if spec["eligible"]}
    assert main.LEAKS.isdisjoint(eligible)
    assert main.CONTROLS.isdisjoint(eligible)
    chosen = main.select_model(
        [
            {"name": "seasonal_naive_24", "eligible": True, "cv_rmse": 1.0, "n_features": 1},
            {"name": "two_harmonic_plus_lags", "eligible": True, "cv_rmse": 2.0, "n_features": 4},
            {"name": "leaky_rolling", "eligible": False, "cv_rmse": 0.1, "n_features": 4},
        ]
    )
    assert chosen == "seasonal_naive_24"


def test_holdout_is_the_later_fraction():
    main = load_main()
    splits = main.make_splits(100, train_end_fraction=0.8)
    assert splits["train"].max() < splits["holdout"].min()
    assert len(splits["train"]) == 80
    assert len(splits["holdout"]) == 20


def test_hour_cycle_puts_midnight_next_to_eleven():
    main = load_main()
    hour = np.array([0, 11, 23])
    sine, cosine = main.hour_cycle(hour)
    late = np.array([sine[2], cosine[2]])
    midnight = np.array([sine[0], cosine[0]])
    midday = np.array([sine[1], cosine[1]])
    assert np.linalg.norm(late - midnight) < np.linalg.norm(late - midday)


def test_time_lag_does_not_cross_a_missing_hour_or_read_the_future():
    main = load_main()
    index = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 02:00:00"])
    values = np.array([10.0, 20.0])
    lagged = main.time_lag(index, values, 1)
    assert np.isnan(lagged[1])
    with pytest.raises(ValueError, match="future"):
        main.time_lag(index, values, -1)


def test_leaky_rolling_includes_the_current_hour_and_row_shift_ignores_the_gap():
    main = load_main()
    index = pd.date_range("2020-01-01", periods=3, freq="h")
    values = np.array([1.0, 2.0, 3.0])
    leaky = main.leaky_rolling(index, values, 24)
    past_mean, past_count = main.past_rolling(index, values, 24)
    assert leaky[-1] == pytest.approx(2.0)
    assert past_mean[-1] == pytest.approx(1.5)
    assert past_count[-1] == pytest.approx(2.0)
    gappy = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 02:00:00"])
    assert np.isnan(main.time_lag(gappy, values[:2], 1)[1])
    assert main.row_shift(values[:2], 1)[1] == pytest.approx(1.0)


def test_holiday_flag_uses_the_stamped_date_and_not_the_volume():
    main = load_main()
    index = pd.date_range("2020-12-25", periods=3, freq="h")
    names = np.array(["Christmas Day", pd.NA, pd.NA], dtype=object)
    flag = main.holiday_date_flag(index, names)
    assert flag.tolist() == [1.0, 1.0, 1.0]
    next_day = pd.date_range("2020-12-26", periods=1, freq="h")
    assert main.holiday_date_flag(next_day, np.array([pd.NA], dtype=object)).tolist() == [0.0]


def test_readme_quotes_committed_metrics():
    metrics_path = PROJECT / "results" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("metrics.json is written by main.py")
    metrics = metrics_path.read_text(encoding="utf-8")
    payload = json.loads(metrics)
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    linkedin = (PROJECT / "LINKEDIN.md").read_text(encoding="utf-8")
    missing = [number for number in re.findall(r"`([0-9eE.+-]+)`", readme) if number not in metrics]
    assert missing == []
    reported = payload["protocol"]["reported_model"]
    holdout = str(payload["experiments"][reported]["test"]["rmse"])
    baseline = str(payload["experiments"]["training_mean"]["test"]["rmse"])
    leaked = str(payload["experiments"]["leaky_rolling"]["test"]["rmse"])
    assert holdout in readme
    assert holdout in linkedin
    assert baseline in readme
    assert leaked in readme
    assert leaked in linkedin
    assert payload["protocol"]["selection_uses_test_scores"] is False
    assert payload["experiments"]["leaky_rolling"]["eligible"] is False
