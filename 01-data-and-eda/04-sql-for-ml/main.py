"""Build a repeat-purchase label from invoice lines, then keep the date filter.

The reported features are aggregates of purchase lines before 2011-09-01.
A query that extends that bound through the extract, and a product average
computed with no date bound, are scored and not selected. The reported
feature list was fixed before those scores were read.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.2
INVOICE_SPLIT_SEED = 42
CUTOFF = "2011-09-01 00:00:00"
CUTOFF_DATE = "2011-09-01"
DATE_FORMAT = "%m/%d/%Y %H:%M"
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000
LOG_LOSS_CLIP = 1e-12
DECISION_THRESHOLD = 0.5
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 7
DATASET_DOI = "10.24432/C5BW33"
DATA_URL = "https://archive.ics.uci.edu/static/public/352/data.csv"
API_URL = "https://archive.ics.uci.edu/dataset/352/online+retail"
API_NUM_INSTANCES = 541909
API_NUM_FEATURES = 6
API_HAS_MISSING_VALUES = "no"
API_DATE_TEXT = "01/12/2010 and 09/12/2011"
API_CANCELLATION_TEXT = "letter 'c'"
API_INVOICE_TEXT = "a 6-digit integral number uniquely assigned to each transaction"
API_STOCK_TEXT = "a 5-digit integral number uniquely assigned to each distinct product"
API_CUSTOMER_TEXT = "a 5-digit integral number uniquely assigned to each customer"
API_COUNTRY_TEXT = "the name of the country where each customer resides"
RECENCY_BINS = (
    (1, 30, "1-30 days"),
    (31, 90, "31-90 days"),
    (91, 180, "91-180 days"),
    (181, None, "181 days or more"),
)
HONEST_FEATURES = [
    "n_orders",
    "n_lines",
    "n_products",
    "monetary",
    "recency_days",
    "n_cancel_invoices",
    "mean_unit_price",
]
LINE_COLUMNS = [
    "invoice_ts",
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "unit_price",
    "customer_id",
    "country",
]

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CSV_PATH = DATA / "online_retail.csv"


def purchase_sql(alias: str = "") -> str:
    if alias not in {"", "lines", "order_lines"}:
        raise ValueError(f"unexpected SQL alias: {alias}")
    prefix = f"{alias}." if alias else ""
    return (
        f"{prefix}quantity > 0 AND {prefix}unit_price > 0 "
        f"AND upper({prefix}invoice_no) NOT LIKE 'C%'"
    )


def cancel_sql(alias: str = "") -> str:
    if alias not in {"", "lines", "order_lines"}:
        raise ValueError(f"unexpected SQL alias: {alias}")
    prefix = f"{alias}." if alias else ""
    return f"upper({prefix}invoice_no) LIKE 'C%'"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "ml-mastery-lab/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        dest.write_bytes(response.read())


def python_value(value):
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def row_dict(con: duckdb.DuckDBPyConnection, sql: str, params: dict | None = None) -> dict:
    cursor = con.execute(sql, params or {})
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("query returned no row")
    names = [item[0] for item in cursor.description]
    return {name: python_value(value) for name, value in zip(names, row, strict=True)}


def register_lines(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> None:
    missing = [column for column in LINE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing line columns: {missing}")
    con.register("line_input", frame.loc[:, LINE_COLUMNS])
    con.execute(
        """
        CREATE OR REPLACE TABLE lines AS
        SELECT
          CAST(invoice_ts AS TIMESTAMP) AS invoice_ts,
          CAST(invoice_no AS VARCHAR) AS invoice_no,
          CAST(stock_code AS VARCHAR) AS stock_code,
          CAST(description AS VARCHAR) AS description,
          CAST(quantity AS DOUBLE) AS quantity,
          CAST(unit_price AS DOUBLE) AS unit_price,
          CAST(customer_id AS BIGINT) AS customer_id,
          CAST(country AS VARCHAR) AS country
        FROM line_input
        """
    )


def load_extract(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    con.execute(
        "CREATE OR REPLACE TABLE raw_lines AS SELECT * FROM read_csv_auto($path)",
        {"path": str(path)},
    )
    columns = [row[0] for row in con.execute("DESCRIBE raw_lines").fetchall()]
    expected = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]
    missing = [name for name in expected if name not in columns]
    if missing:
        raise ValueError(f"extract is missing columns: {missing}")
    con.execute(
        """
        CREATE OR REPLACE TABLE staged AS
        SELECT
          try_strptime(CAST(InvoiceDate AS VARCHAR), $format) AS invoice_ts,
          CAST(InvoiceDate AS VARCHAR) AS invoice_date_text,
          CAST(InvoiceNo AS VARCHAR) AS invoice_no,
          CAST(StockCode AS VARCHAR) AS stock_code,
          CAST(Description AS VARCHAR) AS description,
          CAST(Quantity AS DOUBLE) AS quantity,
          CAST(UnitPrice AS DOUBLE) AS unit_price,
          try_cast(CAST(CustomerID AS VARCHAR) AS DOUBLE) AS customer_id_num,
          CAST(CustomerID AS VARCHAR) AS customer_id_text,
          CAST(Country AS VARCHAR) AS country
        FROM raw_lines
        """,
        {"format": DATE_FORMAT},
    )
    checks = row_dict(
        con,
        """
        SELECT
          count(*) AS n_rows,
          count(*) - count(invoice_ts) AS n_unparsed_dates,
          sum(
            CASE
              WHEN customer_id_text IS NOT NULL
               AND customer_id_text <> ''
               AND customer_id_num IS NULL THEN 1
              ELSE 0
            END
          ) AS n_non_numeric_customer,
          sum(
            CASE
              WHEN customer_id_num IS NOT NULL
               AND customer_id_num <> trunc(customer_id_num) THEN 1
              ELSE 0
            END
          ) AS n_fractional_customer
        FROM staged
        """,
    )
    if checks["n_unparsed_dates"]:
        raise RuntimeError(f"{checks['n_unparsed_dates']} dates did not match {DATE_FORMAT}")
    if checks["n_non_numeric_customer"] or checks["n_fractional_customer"]:
        raise RuntimeError(f"customer ids are not whole numbers: {checks}")
    con.execute(
        """
        CREATE OR REPLACE TABLE lines AS
        SELECT
          invoice_ts,
          invoice_date_text,
          invoice_no,
          stock_code,
          description,
          quantity,
          unit_price,
          CASE
            WHEN customer_id_num IS NULL THEN NULL
            ELSE CAST(customer_id_num AS BIGINT)
          END AS customer_id,
          country
        FROM staged
        """
    )


def build_tables(con: duckdb.DuckDBPyConnection) -> None:
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "lines" not in tables:
        raise RuntimeError("lines table is missing")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE order_lines AS
        SELECT
          invoice_no,
          row_number() OVER (
            ORDER BY invoice_no, invoice_ts, stock_code, quantity, unit_price, country
          ) AS line_id,
          stock_code,
          description,
          quantity,
          unit_price,
          invoice_ts,
          customer_id,
          country,
          ({purchase_sql()}) AS is_purchase,
          ({cancel_sql()}) AS is_cancellation
        FROM lines
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE orders AS
        SELECT
          invoice_no,
          count(DISTINCT customer_id) AS n_customers,
          min(customer_id) AS customer_id,
          count(DISTINCT invoice_ts) AS n_timestamps,
          count(DISTINCT country) AS n_countries,
          min(invoice_ts) AS ts_min,
          max(invoice_ts) AS ts_max,
          date_diff('minute', min(invoice_ts), max(invoice_ts)) AS span_minutes,
          count(*) AS n_lines
        FROM lines
        GROUP BY invoice_no
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE customers AS
        SELECT DISTINCT customer_id
        FROM lines
        WHERE customer_id IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE products AS
        SELECT
          stock_code,
          count(DISTINCT description) AS n_descriptions,
          count(*) AS n_lines
        FROM lines
        GROUP BY stock_code
        """
    )


def purchase_features(
    con: duckdb.DuckDBPyConnection,
    eligible_before: str,
    feature_before: str,
    recency_anchor: str,
) -> pd.DataFrame:
    """Aggregate purchase lines before feature_before for customers eligible earlier.

    The date predicate is on order_lines.invoice_ts. orders.ts_min is not used.
    """
    params = {
        "eligible_before": eligible_before,
        "feature_before": feature_before,
        "recency_anchor": recency_anchor,
    }
    purchases = con.execute(
        """
        WITH eligible AS (
          SELECT DISTINCT customers.customer_id
          FROM customers
          JOIN orders ON orders.customer_id = customers.customer_id
          JOIN order_lines ON order_lines.invoice_no = orders.invoice_no
          WHERE order_lines.is_purchase
            AND order_lines.invoice_ts < CAST($eligible_before AS TIMESTAMP)
        )
        SELECT
          eligible.customer_id,
          count(DISTINCT orders.invoice_no) AS n_orders,
          count(order_lines.line_id) AS n_lines,
          count(DISTINCT products.stock_code) AS n_products,
          CAST(sum(order_lines.quantity * order_lines.unit_price) AS DOUBLE) AS monetary,
          date_diff(
            'day',
            CAST(max(order_lines.invoice_ts) AS DATE),
            CAST($recency_anchor AS DATE)
          ) AS recency_days,
          avg(order_lines.unit_price) AS mean_unit_price
        FROM eligible
        JOIN orders ON orders.customer_id = eligible.customer_id
        JOIN order_lines ON order_lines.invoice_no = orders.invoice_no
        JOIN products ON products.stock_code = order_lines.stock_code
        WHERE order_lines.is_purchase
          AND order_lines.invoice_ts < CAST($feature_before AS TIMESTAMP)
        GROUP BY eligible.customer_id
        """,
        params,
    ).fetchdf()
    cancels = con.execute(
        """
        SELECT
          customers.customer_id,
          count(DISTINCT orders.invoice_no) AS n_cancel_invoices
        FROM customers
        JOIN orders ON orders.customer_id = customers.customer_id
        JOIN order_lines ON order_lines.invoice_no = orders.invoice_no
        WHERE order_lines.is_cancellation
          AND order_lines.invoice_ts < CAST($feature_before AS TIMESTAMP)
        GROUP BY customers.customer_id
        """,
        {"feature_before": feature_before},
    ).fetchdf()
    countries = con.execute(
        """
        SELECT customer_id, country AS latest_country
        FROM (
          SELECT
            orders.customer_id,
            order_lines.country,
            row_number() OVER (
              PARTITION BY orders.customer_id
              ORDER BY order_lines.invoice_ts DESC, orders.invoice_no DESC, order_lines.line_id DESC
            ) AS rn
          FROM orders
          JOIN order_lines ON order_lines.invoice_no = orders.invoice_no
          WHERE orders.customer_id IS NOT NULL
            AND order_lines.is_purchase
            AND order_lines.invoice_ts < CAST($feature_before AS TIMESTAMP)
        )
        WHERE rn = 1
        """,
        {"feature_before": feature_before},
    ).fetchdf()
    frame = purchases.merge(cancels, on="customer_id", how="left")
    frame = frame.merge(countries, on="customer_id", how="left")
    frame["n_cancel_invoices"] = frame["n_cancel_invoices"].fillna(0)
    frame["customer_id"] = frame["customer_id"].astype(int)
    for column in ("n_orders", "n_lines", "n_products", "recency_days", "n_cancel_invoices"):
        frame[column] = frame[column].astype(int)
    return frame


def signed_monetary(
    con: duckdb.DuckDBPyConnection, eligible_before: str, feature_before: str
) -> pd.DataFrame:
    frame = con.execute(
        """
        WITH eligible AS (
          SELECT DISTINCT customers.customer_id
          FROM customers
          JOIN orders ON orders.customer_id = customers.customer_id
          JOIN order_lines ON order_lines.invoice_no = orders.invoice_no
          WHERE order_lines.is_purchase
            AND order_lines.invoice_ts < CAST($eligible_before AS TIMESTAMP)
        )
        SELECT
          eligible.customer_id,
          CAST(sum(order_lines.quantity * order_lines.unit_price) AS DOUBLE) AS signed_monetary
        FROM eligible
        JOIN orders ON orders.customer_id = eligible.customer_id
        JOIN order_lines ON order_lines.invoice_no = orders.invoice_no
        WHERE order_lines.invoice_ts < CAST($feature_before AS TIMESTAMP)
        GROUP BY eligible.customer_id
        """,
        {"eligible_before": eligible_before, "feature_before": feature_before},
    ).fetchdf()
    frame["customer_id"] = frame["customer_id"].astype(int)
    return frame


def product_mean_quantity(
    con: duckdb.DuckDBPyConnection,
    eligible_before: str,
    feature_before: str,
    *,
    all_time: bool,
) -> pd.DataFrame:
    stat_filter = "" if all_time else "AND invoice_ts < CAST($feature_before AS TIMESTAMP)"
    frame = con.execute(
        f"""
        WITH product_stats AS (
          SELECT stock_code, avg(quantity) AS mean_qty
          FROM order_lines
          WHERE is_purchase
            {stat_filter}
          GROUP BY stock_code
        ),
        eligible AS (
          SELECT DISTINCT customers.customer_id
          FROM customers
          JOIN orders ON orders.customer_id = customers.customer_id
          JOIN order_lines ON order_lines.invoice_no = orders.invoice_no
          WHERE order_lines.is_purchase
            AND order_lines.invoice_ts < CAST($eligible_before AS TIMESTAMP)
        )
        SELECT
          eligible.customer_id,
          avg(product_stats.mean_qty) AS product_mean_qty
        FROM eligible
        JOIN orders ON orders.customer_id = eligible.customer_id
        JOIN order_lines ON order_lines.invoice_no = orders.invoice_no
        JOIN product_stats ON product_stats.stock_code = order_lines.stock_code
        WHERE order_lines.is_purchase
          AND order_lines.invoice_ts < CAST($feature_before AS TIMESTAMP)
        GROUP BY eligible.customer_id
        """,
        {"eligible_before": eligible_before, "feature_before": feature_before},
    ).fetchdf()
    frame["customer_id"] = frame["customer_id"].astype(int)
    return frame


def repeat_labels(con: duckdb.DuckDBPyConnection, cutoff: str) -> pd.DataFrame:
    frame = con.execute(
        """
        SELECT
          orders.customer_id,
          max(
            CASE
              WHEN order_lines.invoice_ts >= CAST($cutoff AS TIMESTAMP)
               AND order_lines.is_purchase THEN 1
              ELSE 0
            END
          ) AS y
        FROM orders
        JOIN order_lines ON order_lines.invoice_no = orders.invoice_no
        WHERE orders.customer_id IS NOT NULL
        GROUP BY orders.customer_id
        """,
        {"cutoff": cutoff},
    ).fetchdf()
    frame["customer_id"] = frame["customer_id"].astype(int)
    frame["y"] = frame["y"].astype(int)
    return frame


def file_inventory(con: duckdb.DuckDBPyConnection) -> dict:
    inventory = row_dict(
        con,
        f"""
        SELECT
          count(*) AS n_rows,
          count(DISTINCT invoice_no) AS n_invoices,
          count(DISTINCT stock_code) AS n_products,
          count(DISTINCT customer_id) AS n_customers,
          count(DISTINCT country) AS n_countries,
          count(*) - count(customer_id) AS n_missing_customer_id,
          count(*) - count(description) AS n_missing_description,
          count(*) - count(invoice_ts) AS n_unparsed_dates,
          count(*) - count(invoice_no) AS n_missing_invoice,
          count(*) - count(stock_code) AS n_missing_stock,
          count(*) - count(country) AS n_missing_country,
          min(invoice_ts) AS parsed_min,
          max(invoice_ts) AS parsed_max,
          min(invoice_date_text) AS lexicographic_min,
          max(invoice_date_text) AS lexicographic_max,
          sum(CASE WHEN {cancel_sql()} THEN 1 ELSE 0 END) AS n_cancel_prefix_rows,
          sum(CASE WHEN invoice_no LIKE 'c%' THEN 1 ELSE 0 END) AS n_lower_c_prefix_rows,
          sum(CASE WHEN quantity < 0 THEN 1 ELSE 0 END) AS n_negative_quantity_rows,
          sum(CASE WHEN unit_price < 0 THEN 1 ELSE 0 END) AS n_negative_price_rows,
          sum(CASE WHEN unit_price = 0 THEN 1 ELSE 0 END) AS n_zero_price_rows,
          sum(CASE WHEN {purchase_sql()} THEN 1 ELSE 0 END) AS n_purchase_rows,
          sum(
            CASE
              WHEN {cancel_sql()} AND quantity > 0 THEN 1
              ELSE 0
            END
          ) AS n_cancel_prefix_rows_with_positive_quantity,
          sum(
            CASE
              WHEN quantity < 0 AND NOT ({cancel_sql()}) THEN 1
              ELSE 0
            END
          ) AS n_negative_quantity_rows_without_cancel_prefix,
          sum(
            CASE
              WHEN length(stock_code) = 5 AND try_cast(stock_code AS INTEGER) IS NOT NULL THEN 0
              ELSE 1
            END
          ) AS n_stock_rows_not_five_digit,
          sum(
            CASE
              WHEN country <> trim(country) THEN 1
              ELSE 0
            END
          ) AS n_padded_country_rows
        FROM lines
        """,
    )
    keys = row_dict(
        con,
        """
        SELECT
          count(*) AS n_invoices,
          sum(CASE WHEN n_customers > 1 THEN 1 ELSE 0 END) AS n_multi_customer,
          sum(CASE WHEN n_timestamps > 1 THEN 1 ELSE 0 END) AS n_multi_timestamp,
          sum(CASE WHEN n_countries > 1 THEN 1 ELSE 0 END) AS n_multi_country,
          min(span_minutes) FILTER (WHERE n_timestamps > 1) AS multi_timestamp_min_span_minutes,
          max(span_minutes) FILTER (WHERE n_timestamps > 1) AS multi_timestamp_max_span_minutes,
          sum(
            CASE
              WHEN ts_min < CAST($cutoff AS TIMESTAMP)
               AND ts_max >= CAST($cutoff AS TIMESTAMP) THEN 1
              ELSE 0
            END
          ) AS n_invoices_spanning_cutoff
        FROM orders
        """,
        {"cutoff": CUTOFF},
    )
    products = row_dict(
        con,
        """
        SELECT
          count(*) AS n_products,
          sum(CASE WHEN n_descriptions > 1 THEN 1 ELSE 0 END)
            AS n_products_with_multiple_descriptions,
          max(n_descriptions) AS max_descriptions_for_one_code,
          sum(
            CASE
              WHEN length(stock_code) = 5 AND try_cast(stock_code AS INTEGER) IS NOT NULL THEN 0
              ELSE 1
            END
          ) AS n_stock_codes_not_five_digit
        FROM products
        """,
    )
    duplicates = row_dict(
        con,
        """
        SELECT coalesce(sum(n_rows - 1), 0) AS n_exact_duplicate_rows
        FROM (
          SELECT count(*) AS n_rows
          FROM lines
          GROUP BY
            invoice_no, stock_code, description, quantity, unit_price,
            invoice_ts, customer_id, country
          HAVING count(*) > 1
        )
        """,
    )
    null_customer = row_dict(
        con,
        """
        SELECT
          count(*) AS n_rows,
          count(DISTINCT invoice_no) AS n_invoices,
          count(DISTINCT country) AS n_countries,
          count(DISTINCT stock_code) AS n_products
        FROM lines
        WHERE customer_id IS NULL
        """,
    )
    windows = row_dict(
        con,
        """
        SELECT
          sum(
            CASE
              WHEN is_purchase AND invoice_ts < CAST($cutoff AS TIMESTAMP) THEN 1
              ELSE 0
            END
          ) AS n_purchase_rows_before_cutoff,
          sum(
            CASE
              WHEN is_purchase AND invoice_ts >= CAST($cutoff AS TIMESTAMP) THEN 1
              ELSE 0
            END
          ) AS n_purchase_rows_at_or_after_cutoff
        FROM order_lines
        """,
        {"cutoff": CUTOFF},
    )
    return {
        "file": inventory,
        "orders": keys,
        "products": products,
        "exact_duplicate_rows": duplicates["n_exact_duplicate_rows"],
        "null_customer_rows": null_customer,
        "windows": windows,
    }


def multi_country_customers(con: duckdb.DuckDBPyConnection, where_sql: str, params: dict) -> dict:
    return row_dict(
        con,
        f"""
        SELECT
          count(*) AS n_customers,
          sum(CASE WHEN n_countries > 1 THEN 1 ELSE 0 END) AS n_multi_country
        FROM (
          SELECT customer_id, count(DISTINCT country) AS n_countries
          FROM lines
          WHERE customer_id IS NOT NULL
            AND {where_sql}
          GROUP BY customer_id
        )
        """,
        params,
    )


def invoice_overlap(
    invoice_ids: list[str],
    customer_by_invoice: pd.DataFrame,
    eligible_ids: set[int],
    seed: int,
    test_size: float,
) -> dict:
    ordered = list(invoice_ids)
    if ordered != sorted(ordered):
        raise RuntimeError("invoice ids must be sorted before the split draw")
    rng = np.random.default_rng(seed)
    n_test = int(round(test_size * len(ordered)))
    chosen = set(rng.choice(np.asarray(ordered), size=n_test, replace=False))
    sides = customer_by_invoice.copy()
    sides["side"] = np.where(sides["invoice_no"].isin(chosen), "test", "train")
    both = sides.groupby("customer_id")["side"].nunique()
    both_ids = {int(customer_id) for customer_id, count in both.items() if count > 1}
    return {
        "seed": seed,
        "test_size": test_size,
        "n_invoices": len(ordered),
        "n_test_invoices": n_test,
        "n_customers": int(sides["customer_id"].nunique()),
        "n_customers_in_both": len(both_ids),
        "n_eligible": len(eligible_ids),
        "n_eligible_in_both": len(both_ids & eligible_ids),
    }


def pointwise_log_loss(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(proba, dtype=float), LOG_LOSS_CLIP, 1.0 - LOG_LOSS_CLIP)
    y_true = np.asarray(y_true, dtype=float)
    return -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))


def paired_mean_interval(difference: np.ndarray, n_resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        means[index] = rng.choice(difference, size=len(difference), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(np.mean(difference)),
        "interval_low": float(low),
        "interval_high": float(high),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def logistic() -> LogisticRegression:
    return LogisticRegression(
        C=LOGISTIC_C,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=SEED,
    )


def score_predictions(y_true: np.ndarray, proba: np.ndarray, n_iter: int | None) -> dict:
    scores = {
        "n_scored": int(len(y_true)),
        "log_loss": float(np.mean(pointwise_log_loss(y_true, proba))),
        "accuracy_at_0_5": float(np.mean((proba >= DECISION_THRESHOLD) == y_true)),
        "n_iter": n_iter,
    }
    if np.unique(y_true).size < 2 or np.unique(proba).size < 2:
        scores["roc_auc"] = None
    else:
        scores["roc_auc"] = float(roc_auc_score(y_true, proba))
    return scores


def fit_numeric(
    train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame, test_y: np.ndarray
) -> tuple[dict, np.ndarray]:
    model = Pipeline([("scale", StandardScaler()), ("model", logistic())])
    model.fit(train_x, train_y)
    n_iter = int(np.max(model.named_steps["model"].n_iter_))
    if n_iter >= LOGISTIC_MAX_ITER:
        raise RuntimeError(f"logistic regression hit {LOGISTIC_MAX_ITER} iterations")
    proba = model.predict_proba(test_x)[:, 1]
    return score_predictions(test_y, proba, n_iter), proba


def fit_country(
    train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame, test_y: np.ndarray
) -> tuple[dict, np.ndarray]:
    model = Pipeline(
        [
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("model", logistic()),
        ]
    )
    model.fit(train_x, train_y)
    n_iter = int(np.max(model.named_steps["model"].n_iter_))
    if n_iter >= LOGISTIC_MAX_ITER:
        raise RuntimeError(f"country model hit {LOGISTIC_MAX_ITER} iterations")
    proba = model.predict_proba(test_x)[:, 1]
    return score_predictions(test_y, proba, n_iter), proba


def aligned(frame: pd.DataFrame, customer_ids: np.ndarray) -> pd.DataFrame:
    indexed = frame.set_index("customer_id")
    if indexed.index.has_duplicates:
        raise RuntimeError("feature frame has duplicate customers")
    located = indexed.loc[customer_ids]
    if len(located) != len(customer_ids):
        raise RuntimeError("feature frame is missing customers from the split")
    return located


def recency_bin_rows(recency: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    assigned = np.zeros(len(recency), dtype=bool)
    for low, high, name in RECENCY_BINS:
        if high is None:
            mask = recency >= low
        else:
            mask = (recency >= low) & (recency <= high)
        assigned |= mask
        count = int(mask.sum())
        rows.append(
            {
                "bin": name,
                "low": low,
                "high": high,
                "n_customers": count,
                "n_positive": int(labels[mask].sum()) if count else 0,
                "repeat_rate": float(labels[mask].mean()) if count else None,
            }
        )
    if not assigned.all():
        raise RuntimeError("a training customer fell outside the recency bins")
    return rows


def plot_log_loss(path: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.0), constrained_layout=True)
    labels = [str(row["label"]) for row in rows]
    values = [float(row["log_loss"]) for row in rows]
    colors = ["#7a7a7a", "#8aa0b4", "#3174a1", "#c23b22", "#d17a22", "#6b4c9a", "#2f7d4a"]
    positions = np.arange(len(rows))
    ax.bar(positions, values, color=colors[: len(rows)])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Log loss")
    ax.set_title("Test log loss. A lower bar can still be the wrong query.")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_recency(path: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    labels = [str(row["bin"]) for row in rows]
    rates = [0.0 if row["repeat_rate"] is None else float(row["repeat_rate"]) for row in rows]
    positions = np.arange(len(rows))
    ax.bar(positions, rates, color="#3174a1")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Repeat rate")
    ax.set_xlabel("Days from last pre-cutoff purchase to the cutoff")
    ax.set_title("Training fold only. Not used to choose the model.")
    for position, row, rate in zip(positions, rows, rates, strict=True):
        ax.text(position, rate + 0.02, f"n={row['n_customers']}", ha="center", fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def feature_summary(frame: pd.DataFrame) -> dict:
    summary = {}
    for column in HONEST_FEATURES:
        values = frame[column].to_numpy(dtype=float)
        summary[column] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    return summary


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, datetime | date):
        return python_value(value)
    if isinstance(value, Decimal):
        return python_value(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def require_one_customer_per_invoice(inventory: dict) -> None:
    if inventory["orders"]["n_multi_customer"]:
        raise RuntimeError("an invoice has more than one customer id")
    if inventory["file"]["n_unparsed_dates"]:
        raise RuntimeError("unparsed dates remained after loading")
    if inventory["file"]["n_missing_invoice"] or inventory["file"]["n_missing_stock"]:
        raise RuntimeError("an invoice or stock code is missing")


def main() -> None:
    started = time.perf_counter()
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    download(DATA_URL, CSV_PATH)

    con = duckdb.connect()
    load_extract(con, CSV_PATH)
    build_tables(con)
    inventory = file_inventory(con)
    require_one_customer_per_invoice(inventory)
    parsed_min = datetime.strptime(inventory["file"]["parsed_min"], "%Y-%m-%d %H:%M:%S")
    parsed_max = datetime.strptime(inventory["file"]["parsed_max"], "%Y-%m-%d %H:%M:%S")
    cutoff_dt = datetime.strptime(CUTOFF, "%Y-%m-%d %H:%M:%S")
    if not parsed_min < cutoff_dt <= parsed_max:
        raise RuntimeError("cutoff is outside the parsed invoice span")

    leaky_before = (parsed_max + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
    leaky_anchor = parsed_max.date().isoformat()
    honest = purchase_features(con, CUTOFF, CUTOFF, CUTOFF_DATE)
    leaky = purchase_features(con, CUTOFF, leaky_before, leaky_anchor)
    signed = signed_monetary(con, CUTOFF, CUTOFF)
    product_leak = product_mean_quantity(con, CUTOFF, CUTOFF, all_time=True)
    labels = repeat_labels(con, CUTOFF)
    if set(honest["customer_id"]) != set(leaky["customer_id"]):
        raise RuntimeError("the leaky query changed who was eligible")
    if set(honest["customer_id"]) != set(signed["customer_id"]):
        raise RuntimeError("signed monetary dropped an eligible customer")
    if set(honest["customer_id"]) != set(product_leak["customer_id"]):
        raise RuntimeError("product statistic dropped an eligible customer")
    if honest["latest_country"].isna().any():
        raise RuntimeError("an eligible customer has no latest country")
    if (honest["recency_days"] < 1).any():
        raise RuntimeError("honest recency is below one day")

    labeled = honest.merge(labels, on="customer_id", how="left")
    if labeled["y"].isna().any():
        raise RuntimeError("an eligible customer has no label")
    comparison = honest.merge(leaky, on="customer_id", suffixes=("_honest", "_leaky"))
    if (comparison["n_orders_leaky"] < comparison["n_orders_honest"]).any():
        raise RuntimeError("a wider window removed an order")
    if (comparison["monetary_leaky"] < comparison["monetary_honest"]).any():
        raise RuntimeError("a wider window reduced monetary")
    n_orders_increased = int((comparison["n_orders_leaky"] > comparison["n_orders_honest"]).sum())
    n_positive = int(labeled["y"].sum())
    if inventory["orders"]["n_invoices_spanning_cutoff"] == 0 and n_orders_increased != n_positive:
        raise RuntimeError("order-count leak does not match the positive count")

    country_counts = (
        labeled["latest_country"].value_counts().rename_axis("country").reset_index(name="n")
    )
    modal_country = str(country_counts.iloc[0]["country"])
    modal_n = int(country_counts.iloc[0]["n"])
    history_countries = multi_country_customers(con, "TRUE", {})
    purchase_countries = multi_country_customers(
        con,
        f"{purchase_sql()} AND invoice_ts < CAST($cutoff AS TIMESTAMP)",
        {"cutoff": CUTOFF},
    )
    signed_gap = labeled.merge(signed, on="customer_id")
    n_signed_differs = int((signed_gap["signed_monetary"] != signed_gap["monetary"]).sum())

    customer_ids = labeled["customer_id"].to_numpy()
    y = labeled["y"].to_numpy(dtype=int)
    train_ids, test_ids, y_train, y_test = train_test_split(
        customer_ids,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    train_rate = float(np.mean(y_train))
    constant_proba = np.full(len(y_test), train_rate, dtype=float)
    scores = {
        "constant_rate": score_predictions(y_test, constant_proba, None),
    }
    scores["constant_rate"]["train_positive_rate"] = train_rate
    probabilities = {"constant_rate": constant_proba}

    train_honest = aligned(labeled, train_ids)
    test_honest = aligned(labeled, test_ids)
    if not np.array_equal(test_honest["y"].to_numpy(dtype=int), y_test):
        raise RuntimeError("test labels are not aligned")
    honest_scores, honest_proba = fit_numeric(
        train_honest[HONEST_FEATURES],
        y_train,
        test_honest[HONEST_FEATURES],
        y_test,
    )
    scores["honest_history"] = honest_scores
    probabilities["honest_history"] = honest_proba
    recency_scores, recency_proba = fit_numeric(
        train_honest[["recency_days"]],
        y_train,
        test_honest[["recency_days"]],
        y_test,
    )
    scores["recency_only"] = recency_scores
    probabilities["recency_only"] = recency_proba

    leaky_aligned_train = aligned(leaky, train_ids)
    leaky_aligned_test = aligned(leaky, test_ids)
    leaky_scores, leaky_proba = fit_numeric(
        leaky_aligned_train[HONEST_FEATURES],
        y_train,
        leaky_aligned_test[HONEST_FEATURES],
        y_test,
    )
    scores["leaky_window"] = leaky_scores
    probabilities["leaky_window"] = leaky_proba

    product_train = aligned(product_leak, train_ids)
    product_test = aligned(product_leak, test_ids)
    product_scores, product_proba = fit_numeric(
        product_train[["product_mean_qty"]],
        y_train,
        product_test[["product_mean_qty"]],
        y_test,
    )
    scores["product_stat_leak"] = product_scores
    probabilities["product_stat_leak"] = product_proba

    signed_train = aligned(signed, train_ids)
    signed_test = aligned(signed, test_ids)
    signed_scores, signed_proba = fit_numeric(
        signed_train[["signed_monetary"]],
        y_train,
        signed_test[["signed_monetary"]],
        y_test,
    )
    scores["signed_monetary"] = signed_scores
    probabilities["signed_monetary"] = signed_proba

    country_scores, country_proba = fit_country(
        train_honest[["latest_country"]],
        y_train,
        test_honest[["latest_country"]],
        y_test,
    )
    scores["latest_country"] = country_scores
    probabilities["latest_country"] = country_proba
    train_countries = set(train_honest["latest_country"])
    n_unseen_country = int((~test_honest["latest_country"].isin(train_countries)).sum())

    gaps = {}
    for name, reference in (
        ("recency_only_minus_constant", ("recency_only", "constant_rate")),
        ("honest_history_minus_constant", ("honest_history", "constant_rate")),
        ("honest_history_minus_recency_only", ("honest_history", "recency_only")),
        ("leaky_window_minus_honest_history", ("leaky_window", "honest_history")),
        ("product_stat_leak_minus_honest_history", ("product_stat_leak", "honest_history")),
        ("signed_monetary_minus_honest_history", ("signed_monetary", "honest_history")),
        ("latest_country_minus_honest_history", ("latest_country", "honest_history")),
    ):
        first, second = reference
        gaps[name] = paired_mean_interval(
            pointwise_log_loss(y_test, probabilities[first])
            - pointwise_log_loss(y_test, probabilities[second]),
            BOOTSTRAP_RESAMPLES,
            BOOTSTRAP_SEED,
        )

    eligible_ids = set(labeled["customer_id"].astype(int))
    invoice_customers = con.execute(
        """
        SELECT invoice_no, customer_id
        FROM orders
        WHERE customer_id IS NOT NULL
        ORDER BY invoice_no
        """
    ).fetchdf()
    invoice_customers["customer_id"] = invoice_customers["customer_id"].astype(int)
    history_invoices = con.execute(
        """
        SELECT invoice_no
        FROM orders
        WHERE customer_id IS NOT NULL
          AND ts_min < CAST($cutoff AS TIMESTAMP)
        ORDER BY invoice_no
        """,
        {"cutoff": CUTOFF},
    ).fetchdf()
    all_ids = sorted(invoice_customers["invoice_no"].astype(str).tolist())
    history_ids = sorted(history_invoices["invoice_no"].astype(str).tolist())
    invoice_customers["invoice_no"] = invoice_customers["invoice_no"].astype(str)
    history_customers = invoice_customers[invoice_customers["invoice_no"].isin(history_ids)]
    overlaps = {
        "all_customer_invoices": invoice_overlap(
            all_ids, invoice_customers, eligible_ids, INVOICE_SPLIT_SEED, TEST_SIZE
        ),
        "history_invoices": invoice_overlap(
            history_ids, history_customers, eligible_ids, INVOICE_SPLIT_SEED, TEST_SIZE
        ),
    }
    training_bins = recency_bin_rows(
        train_honest["recency_days"].to_numpy(dtype=int),
        train_honest["y"].to_numpy(dtype=int),
    )
    n_eligible = int(len(labeled))
    metrics = {
        "duckdb_version": duckdb.__version__,
        "data": {
            "doi": DATASET_DOI,
            "source_url": DATA_URL,
            "api_url": API_URL,
            "creator": "Daqing Chen",
            "license": "CC BY 4.0",
            "license_note": (
                "The API record has no license field. The donation policy says approved "
                "datasets are CC BY 4.0."
            ),
            "api_num_instances": API_NUM_INSTANCES,
            "api_num_features": API_NUM_FEATURES,
            "api_has_missing_values": API_HAS_MISSING_VALUES,
            "api_customer_id_missing_values": "no",
            "api_description_missing_values": "no",
            "api_date_text": API_DATE_TEXT,
            "api_cancellation_text": API_CANCELLATION_TEXT,
            "api_invoice_text": API_INVOICE_TEXT,
            "api_stock_text": API_STOCK_TEXT,
            "api_customer_text": API_CUSTOMER_TEXT,
            "api_country_text": API_COUNTRY_TEXT,
            "api_intro_paper_opened": False,
            "n_file_columns": 8,
        },
        "protocol": {
            "cutoff": CUTOFF,
            "cutoff_date": CUTOFF_DATE,
            "date_format": DATE_FORMAT,
            "purchase_rule": purchase_sql(),
            "label": "any purchase line at or after the cutoff",
            "eligible": "non-null customer id and a purchase line before the cutoff",
            "reported_model": "honest_history",
            "reported_features": HONEST_FEATURES,
            "selection_uses_test_scores": False,
            "leaky_feature_before": leaky_before,
            "leaky_recency_anchor": leaky_anchor,
            "split": "customer",
            "test_size": TEST_SIZE,
            "seed": SEED,
            "invoice_split_seed": INVOICE_SPLIT_SEED,
            "logistic_c": LOGISTIC_C,
            "logistic_max_iter": LOGISTIC_MAX_ITER,
            "log_loss_clip": LOG_LOSS_CLIP,
            "decision_threshold": DECISION_THRESHOLD,
            "threshold_was_tuned": False,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "gap_definition": "pointwise log loss of the first model minus the second",
            "not_selected": [
                "leaky_window",
                "product_stat_leak",
                "signed_monetary",
                "latest_country",
            ],
        },
        "inventory": inventory,
        "population": {
            "n_eligible": n_eligible,
            "n_positive": n_positive,
            "positive_rate": float(n_positive / n_eligible),
            "n_train": int(len(train_ids)),
            "n_test": int(len(test_ids)),
            "train_positive_rate": train_rate,
            "test_positive_rate": float(np.mean(y_test)),
            "n_orders_increased_by_leaky_window": n_orders_increased,
            "n_customers_signed_monetary_differs": n_signed_differs,
            "modal_latest_country": modal_country,
            "n_modal_latest_country": modal_n,
            "modal_latest_country_share": float(modal_n / n_eligible),
            "n_test_rows_with_country_absent_from_train": n_unseen_country,
            "country_counts": country_counts.to_dict(orient="records"),
            "multi_country_all_lines": history_countries,
            "multi_country_purchase_before_cutoff": purchase_countries,
        },
        "feature_summary": feature_summary(labeled),
        "invoice_split": overlaps,
        "training_recency_bins": training_bins,
        "scores": scores,
        "paired_log_loss_gaps": gaps,
    }
    metrics["elapsed_seconds"] = time.perf_counter() - started
    payload = json_ready(metrics)
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_log_loss(
        RESULTS / "log_loss.png",
        [
            {"label": "Training rate", "log_loss": scores["constant_rate"]["log_loss"]},
            {"label": "Recency only", "log_loss": scores["recency_only"]["log_loss"]},
            {"label": "Honest history", "log_loss": scores["honest_history"]["log_loss"]},
            {"label": "Leaky window", "log_loss": scores["leaky_window"]["log_loss"]},
            {"label": "Product stat leak", "log_loss": scores["product_stat_leak"]["log_loss"]},
            {"label": "Signed monetary", "log_loss": scores["signed_monetary"]["log_loss"]},
            {"label": "Latest country", "log_loss": scores["latest_country"]["log_loss"]},
        ],
    )
    plot_recency(RESULTS / "repeat_by_recency.png", training_bins)
    print(
        json.dumps(
            {
                "elapsed_seconds": payload["elapsed_seconds"],
                "n_eligible": payload["population"]["n_eligible"],
                "positive_rate": payload["population"]["positive_rate"],
                "honest_log_loss": payload["scores"]["honest_history"]["log_loss"],
                "constant_log_loss": payload["scores"]["constant_rate"]["log_loss"],
                "leaky_log_loss": payload["scores"]["leaky_window"]["log_loss"],
                "history_eligible_in_both": payload["invoice_split"]["history_invoices"][
                    "n_eligible_in_both"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
