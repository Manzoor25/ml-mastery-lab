"""Tiny SQL checks. No retail download and no model fit."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import duckdb
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
CUTOFF = "2011-09-01 00:00:00"
CUTOFF_DATE = "2011-09-01"


def load_main():
    spec = importlib.util.spec_from_file_location(
        "sql_for_ml_retail",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "invoice_ts": pd.to_datetime(
                [
                    "2011-08-01 10:00",
                    "2011-08-15 10:00",
                    "2011-10-01 10:00",
                    "2011-07-01 10:00",
                    "2011-10-02 10:00",
                    "2011-08-02 10:00",
                    "2011-08-31 23:59",
                    "2011-09-01 00:01",
                ]
            ),
            "invoice_no": ["A", "C1", "B", "D", "E", "F", "T1", "T1"],
            "stock_code": ["P1", "P1", "P2", "P1", "P1", "P9", "P3", "P4"],
            "description": ["Gift"] * 8,
            "quantity": [2, -1, 5, 1, 1, 9, 1, 4],
            "unit_price": [10.0, 10.0, 10.0, 4.0, 4.0, 9.0, 3.0, 3.0],
            "customer_id": pd.Series([1, 1, 1, 2, 3, pd.NA, 4, 4], dtype="Int64"),
            "country": ["United Kingdom", "France", "United Kingdom", "France"] * 2,
        }
    )


def connected(main):
    con = duckdb.connect()
    main.register_lines(con, tiny_frame())
    main.build_tables(con)
    return con


def test_future_lines_stay_out_of_the_honest_join():
    main = load_main()
    con = connected(main)
    honest = main.purchase_features(con, CUTOFF, CUTOFF, CUTOFF_DATE)
    leaky = main.purchase_features(con, CUTOFF, "2011-12-31 00:00:00", "2011-12-09")
    labels = main.repeat_labels(con, CUTOFF)
    assert set(honest["customer_id"]) == {1, 2, 4}
    assert set(leaky["customer_id"]) == {1, 2, 4}
    customer_1 = honest.loc[honest["customer_id"] == 1].iloc[0]
    assert customer_1["monetary"] == pytest.approx(20.0)
    assert customer_1["n_orders"] == 1
    assert customer_1["n_cancel_invoices"] == 1
    assert customer_1["recency_days"] == 31
    customer_4 = honest.loc[honest["customer_id"] == 4].iloc[0]
    assert customer_4["monetary"] == pytest.approx(3.0)
    assert customer_4["n_lines"] == 1
    leaky_1 = leaky.loc[leaky["customer_id"] == 1].iloc[0]
    leaky_4 = leaky.loc[leaky["customer_id"] == 4].iloc[0]
    assert leaky_1["monetary"] == pytest.approx(70.0)
    assert leaky_1["n_orders"] == 2
    assert leaky_4["monetary"] == pytest.approx(15.0)
    assert leaky_4["n_lines"] == 2
    got = labels.set_index("customer_id")["y"].to_dict()
    assert got[1] == 1
    assert got[2] == 0
    assert got[3] == 1
    assert got[4] == 1


def test_all_time_product_stat_includes_a_future_purchase():
    main = load_main()
    con = connected(main)
    leaked = main.product_mean_quantity(con, CUTOFF, CUTOFF, all_time=True)
    historical = main.product_mean_quantity(con, CUTOFF, CUTOFF, all_time=False)
    leaked_1 = float(leaked.loc[leaked["customer_id"] == 1, "product_mean_qty"].iloc[0])
    historical_1 = float(historical.loc[historical["customer_id"] == 1, "product_mean_qty"].iloc[0])
    assert leaked_1 == pytest.approx(4 / 3)
    assert historical_1 == pytest.approx(1.5)
    assert 3 not in set(leaked["customer_id"])


def test_the_better_score_is_not_the_reported_model():
    payload = json.loads((PROJECT / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["protocol"]["reported_model"] == "honest_history"
    assert payload["protocol"]["selection_uses_test_scores"] is False
    leaky_loss = payload["scores"]["leaky_window"]["log_loss"]
    honest_loss = payload["scores"]["honest_history"]["log_loss"]
    assert leaky_loss < honest_loss
    increased = payload["population"]["n_orders_increased_by_leaky_window"]
    assert increased == payload["population"]["n_positive"]


def test_readme_quotes_committed_metrics():
    metrics = (PROJECT / "results" / "metrics.json").read_text(encoding="utf-8")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    linkedin = (PROJECT / "LINKEDIN.md").read_text(encoding="utf-8")
    missing = [number for number in re.findall(r"`([0-9eE.+-]+)`", readme) if number not in metrics]
    assert missing == []
    assert "0.6063879510172124" in readme
    assert "0.03506553353837828" in readme
    assert "0.6063879510172124" in linkedin
    assert "0.03506553353837828" in linkedin
    assert "duckdb==1.4.4" in (ROOT / "requirements.txt").read_text(encoding="utf-8")
