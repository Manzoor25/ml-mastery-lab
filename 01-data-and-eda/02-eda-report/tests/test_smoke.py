"""Tiny checks for the EDA helpers. No zip download and no full fit."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "eda_report_bike",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_identity_residual_is_zero_when_the_parts_sum_to_the_total():
    main = load_main()
    residual = main.identity_residual(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([4.0, 6.0]),
    )
    assert np.all(residual == 0.0)


def test_psi_is_zero_for_matching_counts_and_positive_when_mass_moves():
    main = load_main()
    matched = main.psi_from_counts(np.array([10.0, 30.0, 60.0]), np.array([10.0, 30.0, 60.0]))
    shifted = main.psi_from_counts(np.array([10.0, 30.0, 60.0]), np.array([60.0, 30.0, 10.0]))
    assert matched == 0.0
    assert shifted > 0.0


def test_reported_model_is_not_the_lower_leaky_score():
    payload = json.loads((PROJECT / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["protocol"]["reported_model"] == "honest"
    runs = {row["name"]: row for row in payload["time_split_runs"]}
    assert runs["leaky_casual"]["rmse"] < runs["honest"]["rmse"]
    assert payload["inventory"]["identity_max_abs_residual"] == 0.0
    assert payload["random_split"]["n_year_1_in_train"] > 0


def test_readme_quotes_committed_metrics():
    metrics = (PROJECT / "results" / "metrics.json").read_text(encoding="utf-8")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    linkedin = (PROJECT / "LINKEDIN.md").read_text(encoding="utf-8")
    missing = [number for number in re.findall(r"`([0-9eE.+-]+)`", readme) if number not in metrics]
    assert missing == []
    assert "158.0887211276591" in readme
    assert "158.0887211276591" in linkedin
    assert "110.56838688362778" in readme
    payload = json.loads(metrics)
    assert payload["data"]["page_instance_claim"] != payload["inventory"]["n_rows"]
