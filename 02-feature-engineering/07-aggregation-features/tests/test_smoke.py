"""Aggregation-feature checks. No retail download and no model fit."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "retail_aggregation_features",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_experiment_list_is_locked_and_the_leaks_are_ineligible():
    main = load_main()
    names = [spec["name"] for spec in main.experiment_specs()]
    assert names == list(main.EXPERIMENT_ORDER)
    eligible = {spec["name"] for spec in main.experiment_specs() if spec["eligible"]}
    assert main.LEAKS.isdisjoint(eligible)
    assert "leaky_block" in main.LEAKS
    assert "leaky_label_spend" in main.LEAKS
    assert main.SHEET_NAME == "Year 2009-2010"
    assert main.FORBIDDEN_SHEET == "Year 2010-2011"
    assert main.LABEL_WINDOW_DAYS == 90


def test_selection_uses_development_rmse_not_the_test_score():
    main = load_main()
    chosen = main.select_model(
        [
            {
                "name": "invoice_count",
                "eligible": True,
                "dev_rmse": 0.4,
                "n_features": 1,
                "test_rmse": 0.01,
            },
            {
                "name": "monetary_sum",
                "eligible": True,
                "dev_rmse": 0.5,
                "n_features": 1,
                "test_rmse": 0.1,
            },
            {
                "name": "leaky_label_spend",
                "eligible": False,
                "dev_rmse": 0.01,
                "n_features": 1,
                "test_rmse": 0.01,
            },
        ]
    )
    assert chosen == "invoice_count"


def test_selection_breaks_a_tie_by_fewer_features_then_name():
    main = load_main()
    chosen = main.select_model(
        [
            {"name": "honest_block", "eligible": True, "dev_rmse": 0.4, "n_features": 6},
            {"name": "monetary_sum", "eligible": True, "dev_rmse": 0.4, "n_features": 1},
            {"name": "invoice_count", "eligible": True, "dev_rmse": 0.4, "n_features": 1},
        ]
    )
    assert chosen == "invoice_count"


def test_worst_eligible_is_not_the_reported_model():
    main = load_main()
    failed = main.worst_eligible(
        [
            {"name": "training_mean", "eligible": True, "dev_rmse": 9.0, "n_features": 0},
            {"name": "monetary_sum", "eligible": True, "dev_rmse": 1.0, "n_features": 1},
            {"name": "leaky_label_spend", "eligible": False, "dev_rmse": 20.0, "n_features": 1},
        ],
        "monetary_sum",
    )
    assert failed == "training_mean"


def test_cutoff_is_midnight_ninety_days_before_the_last_day():
    main = load_main()
    cutoff = main.cutoff_from_max(pd.Timestamp("2010-12-09 20:01:00"))
    assert cutoff == pd.Timestamp("2010-09-10 00:00:00")


def test_purchase_rule_drops_blank_customers_cancellations_and_nonpositive_amounts():
    main = load_main()
    frame = pd.DataFrame(
        {
            "Invoice": [1.0, "C2", 3.0, 4.0, 5.0],
            "StockCode": ["A", "B", "C", "D", "E"],
            "Description": ["item"] * 5,
            "Quantity": [1.0, 1.0, -1.0, 1.0, 2.0],
            "InvoiceDate": pd.to_datetime(["2010-01-01"] * 5),
            "Price": [2.0, 2.0, 2.0, 0.0, 3.0],
            "Customer ID": [10.0, 10.0, 10.0, 10.0, ""],
            "Country": ["United Kingdom"] * 5,
        }
    )
    lines, audit = main.prepare_lines(frame)
    assert list(lines["customer_id"]) == ["10"]
    assert lines["line_amount"].iloc[0] == pytest.approx(2.0)
    assert audit["purchase_rows"] == 1
    assert audit["cancellation_prefix_rows"] == 1
    assert audit["blank_customer_rows"] == 1


def test_honest_sum_excludes_the_label_window_and_the_leaky_sum_includes_it():
    main = load_main()
    lines = pd.DataFrame(
        {
            "customer_id": ["1", "1", "1", "2"],
            "invoice": ["a", "a", "b", "c"],
            "stock_code": ["s", "s", "t", "u"],
            "quantity": [1.0, 1.0, 2.0, 1.0],
            "price": [5.0, 5.0, 4.0, 3.0],
            "invoice_date": pd.to_datetime(
                ["2010-09-01", "2010-09-02", "2010-10-01", "2010-10-02"]
            ),
            "line_amount": [5.0, 5.0, 8.0, 3.0],
        }
    )
    cutoff = pd.Timestamp("2010-09-10")
    table = main.build_customer_table(lines, cutoff, pd.Timestamp("2010-12-09 20:01:00"))
    assert list(table.index) == ["1"]
    row = table.loc["1"]
    assert row["monetary_sum"] == pytest.approx(10.0)
    assert row["invoice_count"] == pytest.approx(1.0)
    assert row["label_spend"] == pytest.approx(8.0)
    assert row["leaky_monetary_sum"] == pytest.approx(18.0)
    assert row["leaky_invoice_count"] == pytest.approx(2.0)
    assert row["recency_days"] == pytest.approx(8.0)
    assert row["leaky_recency_days"] == pytest.approx(69.83402777777778)
    assert main.spanning_invoice_count(lines, cutoff) == 0


def test_customer_split_has_no_overlap():
    main = load_main()
    train, dev, test = main.split_customers([str(i) for i in range(100)])
    assert set(train).isdisjoint(dev)
    assert set(train).isdisjoint(test)
    assert set(dev).isdisjoint(test)
    assert len(train) + len(dev) + len(test) == 100
    assert len(test) == 20
    assert len(dev) == 20
    assert len(train) == 60


def test_readme_quotes_committed_metrics():
    metrics_path = PROJECT / "results" / "metrics.json"
    readme_path = PROJECT / "README.md"
    linkedin_path = PROJECT / "LINKEDIN.md"
    if not metrics_path.exists() or not readme_path.exists() or not linkedin_path.exists():
        pytest.skip("metrics and writeups are written after main.py")
    metrics = metrics_path.read_text(encoding="utf-8")
    payload = json.loads(metrics)
    readme = readme_path.read_text(encoding="utf-8")
    linkedin = linkedin_path.read_text(encoding="utf-8")
    missing = [number for number in re.findall(r"`([0-9eE.+-]+)`", readme) if number not in metrics]
    assert missing == []
    reported = payload["protocol"]["reported_model"]
    holdout = str(payload["experiments"][reported]["test"]["rmse"])
    baseline = str(payload["experiments"]["training_mean"]["test"]["rmse"])
    leaked = str(payload["experiments"]["leaky_label_spend"]["test"]["rmse"])
    assert holdout in readme
    assert holdout in linkedin
    assert baseline in readme
    assert leaked in readme
    assert leaked in linkedin
    assert payload["protocol"]["selection_uses_test_scores"] is False
    assert payload["experiments"]["leaky_label_spend"]["eligible"] is False
    assert payload["experiments"]["leaky_block"]["eligible"] is False
    words = re.findall(r"[A-Za-z0-9']+", linkedin)
    assert 150 <= len(words) <= 250
