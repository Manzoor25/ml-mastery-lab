"""Encoder checks. No hospital download and no model fit."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "categorical_encoding_readmission",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_selection_ignores_the_leaky_score_and_the_holdout():
    main = load_main()
    scores = {
        "pretty_test": {
            "eligible": True,
            "cv_log_loss_mean": 0.40,
            "n_features": 3,
            "test": {"log_loss": 0.01},
        },
        "leaky": {
            "eligible": False,
            "cv_log_loss_mean": 0.01,
            "n_features": 1,
            "test": {"log_loss": 0.50},
        },
        "cv_winner": {
            "eligible": True,
            "cv_log_loss_mean": 0.20,
            "n_features": 9,
            "test": {"log_loss": 0.90},
        },
        "tie_smaller": {
            "eligible": True,
            "cv_log_loss_mean": 0.20,
            "n_features": 4,
            "test": {"log_loss": 0.90},
        },
    }
    assert main.select_experiment(scores) == "tie_smaller"


def test_insample_target_encoding_copies_a_singleton_label():
    main = load_main()
    tokens = np.array([["only"], ["a"], ["a"]], dtype=object)
    target = np.array([1.0, 0.0, 0.0])
    leaked = main.GroupTargetEncoder(smoothing=0.0).fit(tokens, target).transform(tokens)
    assert leaked[0, 0] == pytest.approx(1.0)
    oof = main.GroupTargetEncoder(smoothing=0.0).fit_transform_oof(
        tokens,
        target,
        groups=np.array([0, 1, 2]),
        n_splits=3,
    )
    assert oof[0, 0] == pytest.approx(0.0)


def test_smoothed_target_encoding_shrinks_a_singleton():
    main = load_main()
    tokens = np.array([["only"], ["a"], ["a"]], dtype=object)
    target = np.array([1.0, 0.0, 0.0])
    encoded = main.GroupTargetEncoder(smoothing=20.0).fit(tokens, target).transform(tokens)
    assert encoded[0, 0] == pytest.approx((1.0 + 20.0 / 3.0) / 21.0)
    assert encoded[0, 0] != pytest.approx(1.0)


def test_frequency_encoder_maps_an_unseen_level_to_zero():
    main = load_main()
    train = np.array([["a"], ["a"], ["b"], ["b"]], dtype=object)
    encoded = (
        main.FrequencyEncoder().fit(train).transform(np.array([["a"], ["missing"]], dtype=object))
    )
    assert encoded[0, 0] == pytest.approx(0.5)
    assert encoded[1, 0] == pytest.approx(0.0)


def test_age_code_uses_the_bracket_not_the_batch():
    main = load_main()
    young = main.age_codes(np.array(["[0-10)", "[10-20)"]))
    missing_youngest = main.age_codes(np.array(["[10-20)"]))
    assert young[0, 0] == pytest.approx(0.0)
    assert missing_youngest[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unexpected age"):
        main.age_codes(np.array(["40-50"]))


def test_excluded_columns_stay_out_of_the_model():
    main = load_main()
    assert "race" in main.EXCLUDED
    assert "gender" in main.EXCLUDED
    assert "discharge_disposition_id" in main.EXCLUDED
    assert set(main.HIGH_CARD).isdisjoint(main.EXCLUDED)
    assert "diag_1" in main.HIGH_CARD
    assert "target_insample_m0" in main.INELIGIBLE


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
    holdout = str(payload["experiments"][reported]["test"]["log_loss"])
    baseline = str(payload["experiments"]["training_rate"]["test"]["log_loss"])
    assert holdout in readme
    assert holdout in linkedin
    assert baseline in readme
    assert payload["protocol"]["selection_uses_test_scores"] is False
    assert payload["split"]["n_patients_in_both"] == 0
    assert payload["experiments"]["target_insample_m0"]["eligible"] is False
