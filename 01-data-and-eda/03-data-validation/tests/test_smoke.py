"""Tiny checks for the schema helpers. No Adult download and no model fit."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "data_validation_adult",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_a_period_is_not_the_training_income_contract():
    main = load_main()
    allowed = {"<=50K", ">50K"}
    assert main.income_allowed("<=50K", allowed)
    assert not main.income_allowed("<=50K.", allowed)
    assert not main.income_allowed(" <=50K", allowed)


def test_padding_is_visible_before_any_model_runs():
    main = load_main()
    assert main.padded(" Private")
    assert not main.padded("Private")
    assert not main.padded("39")


def test_donor_parser_keeps_levels_and_drops_continuous_fields():
    main = load_main()
    text = "workclass: Private, Self-emp-not-inc.\nage: continuous.\n"
    levels = main.parse_donor_levels(text)
    assert levels["workclass"] == ["Private", "Self-emp-not-inc"]
    assert "age" not in levels


def test_tiny_schema_rejects_the_period():
    main = load_main()
    result = main.validate_income_frame(pd.DataFrame({"income": ["<=50K", ">50K."]}))
    assert result["passed"] is False
    assert result["n_failed_rows"] == 1


def test_unknown_category_can_fail_the_schema_without_a_resolved_score_gap():
    payload = json.loads((PROJECT / "results" / "metrics.json").read_text(encoding="utf-8"))
    unknown = next(
        row
        for row in payload["gates"]
        if row["extract"] == "unknown_workclass" and row["schema"] == "reported"
    )
    assert unknown["passed"] is False
    assert payload["paired_log_loss_gaps"]["unknown_workclass"]["excludes_zero"] is False


def test_readme_quotes_committed_metrics():
    metrics = (PROJECT / "results" / "metrics.json").read_text(encoding="utf-8")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    linkedin = (PROJECT / "LINKEDIN.md").read_text(encoding="utf-8")
    missing = [number for number in re.findall(r"`([0-9eE.+-]+)`", readme) if number not in metrics]
    assert missing == []
    assert "0.31833658576426743" in readme
    assert "0.425486700706032" in readme
    assert "0.31833658576426743" in linkedin
    assert "0.425486700706032" in linkedin
