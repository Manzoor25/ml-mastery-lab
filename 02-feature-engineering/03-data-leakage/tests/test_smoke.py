"""Leakage-protocol checks. No bank download and no model fit."""

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
        "bank_marketing_leakage",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reported_protocol_excludes_duration_and_does_not_select_on_test():
    main = load_main()
    assert main.REPORTED == "honest"
    assert main.LEAK_COLUMN not in main.HONEST_COLUMNS
    assert "duration" not in main.NUMERIC
    names = [spec["name"] for spec in main.experiment_specs()]
    assert names == list(main.EXPERIMENT_ORDER)
    eligible = {spec["name"] for spec in main.experiment_specs() if spec["eligible"]}
    assert eligible == {"training_rate", "honest"}
    assert main.LEAKS.isdisjoint(eligible)


def test_heldout_calls_sit_between_the_past_and_the_later_tail():
    main = load_main()
    splits = main.make_splits(45211, seed=42)
    main.assert_split_order(splits, 45211)
    assert splits["past"].max() < splits["eval"].min()
    assert splits["eval"].max() < splits["future"].min()
    assert set(splits["eval"]).isdisjoint(splits["same_period"])
    assert len(splits["extra_same"]) == len(splits["same_period"])
    assert len(splits["extra_future"]) == len(splits["future"])
    assert set(splits["extra_same"]).issubset(set(splits["past"]))


def test_year_walk_wraps_forward_and_rejects_a_backward_month():
    main = load_main()
    assert main.walk_years(["may", "jun", "oct", "jan"]) == [2008, 2008, 2008, 2009]
    with pytest.raises(ValueError, match="not forward"):
        main.walk_years(["dec", "nov"])


def test_target_encoder_copies_a_label_only_when_that_row_was_fit():
    main = load_main()
    keys = np.array(["apr", "may", "may"])
    target = np.array([1.0, 0.0, 0.0])
    leaked = main.TargetEncoder(smoothing=0.0).fit(keys, target).transform(keys)
    assert leaked[0, 0] == pytest.approx(1.0)
    unseen = main.TargetEncoder(smoothing=0.0).fit(keys[1:], target[1:]).transform(keys[:1])
    assert unseen[0, 0] == pytest.approx(0.0)


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
    duration = str(payload["experiments"]["duration_leak"]["test"]["log_loss"])
    assert holdout in readme
    assert holdout in linkedin
    assert baseline in readme
    assert duration in readme
    assert duration in linkedin
    assert payload["protocol"]["selection_uses_test_scores"] is False
    assert payload["experiments"]["duration_leak"]["eligible"] is False
    assert payload["shuffled_split"]["comparable_to_heldout"] is False
