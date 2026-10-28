"""Customer-level aggregations from a purchase log.

The 90-day label window, the purchase rule, and the experiment list are fixed
before any ridge model is fit. Development RMSE chooses the eligible recipe.
The holdout customers are not used to add, drop, or replace an aggregation.
A sum through the end of the sheet, and the label itself, are scored and are
not eligible.

This is a teaching model on a UK gift retailer's invoice lines. It is not a
credit score and it is not a pricing tool.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
BOOTSTRAP_SEED = 7
BOOTSTRAP_RESAMPLES = 1000
LABEL_WINDOW_DAYS = 90
TEST_SIZE = 0.2
DEV_SIZE_OF_REST = 0.25
RIDGE_ALPHA = 1.0
LABEL_QUANTILE = 0.99
SHEET_NAME = "Year 2009-2010"
FORBIDDEN_SHEET = "Year 2010-2011"
DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATASET_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
ZIP_NAME = "online+retail+ii.zip"
XLSX_NAME = "online_retail_II.xlsx"
REQUIRED_COLUMNS = (
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
)
AGGREGATE_COLUMNS = (
    "invoice_count",
    "monetary_sum",
    "recency_days",
    "nunique_stock",
    "mean_line_amount",
    "mean_quantity",
)
EXPERIMENT_ORDER = (
    "training_mean",
    "invoice_count",
    "monetary_sum",
    "recency_days",
    "rfm",
    "basket",
    "honest_block",
    "leaky_block",
    "leaky_label_spend",
)
LEAKS = frozenset({"leaky_block", "leaky_label_spend"})
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "training_mean": (),
    "invoice_count": ("invoice_count",),
    "monetary_sum": ("monetary_sum",),
    "recency_days": ("recency_days",),
    "rfm": ("recency_days", "invoice_count", "monetary_sum"),
    "basket": ("nunique_stock", "mean_line_amount", "mean_quantity"),
    "honest_block": (
        "recency_days",
        "invoice_count",
        "monetary_sum",
        "nunique_stock",
        "mean_line_amount",
        "mean_quantity",
    ),
    "leaky_block": (
        "leaky_recency_days",
        "leaky_invoice_count",
        "leaky_monetary_sum",
        "leaky_nunique_stock",
        "leaky_mean_line_amount",
        "leaky_mean_quantity",
    ),
    "leaky_label_spend": ("label_spend",),
}

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
RESULTS = PROJECT / "results"


def experiment_specs() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "eligible": name not in LEAKS,
            "features": FEATURE_SETS[name],
        }
        for name in EXPERIMENT_ORDER
    ]


def select_model(rows: list[dict[str, object]]) -> str:
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(key=lambda row: (row["dev_rmse"], row["n_features"], row["name"]))
    return str(eligible[0]["name"])


def worst_eligible(rows: list[dict[str, object]], reported: str) -> str:
    candidates = [row for row in rows if row["eligible"] and row["name"] != reported]
    candidates.sort(key=lambda row: (-row["dev_rmse"], -row["n_features"], row["name"]))
    return str(candidates[0]["name"])


def cutoff_from_max(
    max_timestamp: pd.Timestamp, window_days: int = LABEL_WINDOW_DAYS
) -> pd.Timestamp:
    stamp = pd.Timestamp(max_timestamp)
    return stamp.floor("D") - pd.Timedelta(days=window_days)


def customer_token(value: object) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "" or not text.isdigit():
            return None
        return str(int(text))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        number = float(value)
        if number != int(number):
            return None
        return str(int(number))
    return None


def invoice_key(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return str(value).strip().upper()


def stock_key(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


def is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def prepare_lines(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    invoice = frame["Invoice"].map(invoice_key)
    customer = frame["Customer ID"].map(customer_token)
    quantity = pd.to_numeric(frame["Quantity"], errors="coerce")
    price = pd.to_numeric(frame["Price"], errors="coerce")
    when = pd.to_datetime(frame["InvoiceDate"])
    stock = frame["StockCode"].map(stock_key)
    raw_customer = frame["Customer ID"]
    purchase = customer.notna() & ~invoice.str.startswith("C") & quantity.gt(0) & price.gt(0)
    audit = {
        "rows": int(len(frame)),
        "blank_customer_rows": int(raw_customer.map(is_blank).sum()),
        "unparsed_customer_rows": int((customer.isna() & ~raw_customer.map(is_blank)).sum()),
        "cancellation_prefix_rows": int(invoice.str.startswith("C").sum()),
        "negative_quantity_rows": int(quantity.lt(0).sum()),
        "zero_quantity_rows": int(quantity.eq(0).sum()),
        "nonpositive_price_rows": int(price.le(0).sum()),
        "negative_price_rows": int(price.lt(0).sum()),
        "cancellation_with_positive_quantity": int(
            (invoice.str.startswith("C") & quantity.gt(0)).sum()
        ),
        "negative_quantity_without_cancellation_prefix": int(
            (quantity.lt(0) & ~invoice.str.startswith("C")).sum()
        ),
        "purchase_rows": int(purchase.sum()),
    }
    lines = pd.DataFrame(
        {
            "customer_id": customer[purchase].to_numpy(),
            "invoice": invoice[purchase].to_numpy(),
            "stock_code": stock[purchase].to_numpy(),
            "quantity": quantity[purchase].to_numpy(dtype=float),
            "price": price[purchase].to_numpy(dtype=float),
            "invoice_date": when[purchase].to_numpy(),
        }
    )
    lines["line_amount"] = lines["quantity"] * lines["price"]
    return lines, audit


def aggregate(lines: pd.DataFrame, recency_anchor: pd.Timestamp) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(columns=list(AGGREGATE_COLUMNS))
    grouped = lines.groupby("customer_id", sort=True)
    last_purchase = grouped["invoice_date"].max()
    frame = pd.DataFrame(
        {
            "invoice_count": grouped["invoice"].nunique().astype(float),
            "monetary_sum": grouped["line_amount"].sum().astype(float),
            "nunique_stock": grouped["stock_code"].nunique().astype(float),
            "mean_line_amount": grouped["line_amount"].mean().astype(float),
            "mean_quantity": grouped["quantity"].mean().astype(float),
        }
    )
    anchor = pd.Timestamp(recency_anchor)
    frame["recency_days"] = (anchor - last_purchase).dt.total_seconds() / 86400.0
    return frame.loc[:, list(AGGREGATE_COLUMNS)]


def build_customer_table(
    lines: pd.DataFrame,
    cutoff: pd.Timestamp,
    sheet_end: pd.Timestamp,
) -> pd.DataFrame:
    history = lines.loc[lines["invoice_date"] < cutoff].copy()
    future = lines.loc[lines["invoice_date"] >= cutoff].copy()
    if history.empty:
        raise ValueError("no purchase lines before the cutoff")
    honest = aggregate(history, cutoff)
    leaky = aggregate(lines, sheet_end).reindex(honest.index)
    if leaky.isna().any().any():
        raise ValueError("eligible customer missing from the full-sheet aggregation")
    label = future.groupby("customer_id")["line_amount"].sum().reindex(honest.index).fillna(0.0)
    table = honest.copy()
    table["label_spend"] = label.astype(float)
    for column in AGGREGATE_COLUMNS:
        table[f"leaky_{column}"] = leaky[column].astype(float)
    return table


def spanning_invoice_count(lines: pd.DataFrame, cutoff: pd.Timestamp) -> int:
    bounds = lines.groupby("invoice")["invoice_date"].agg(["min", "max"])
    return int(((bounds["min"] < cutoff) & (bounds["max"] >= cutoff)).sum())


def split_customers(customer_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.array(sorted(set(customer_ids), key=int))
    rest, test = train_test_split(ordered, test_size=TEST_SIZE, random_state=SEED)
    train, dev = train_test_split(rest, test_size=DEV_SIZE_OF_REST, random_state=SEED)
    return train, dev, test


def design_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=float)
    return frame.loc[:, list(columns)].to_numpy(dtype=float)


def predict_folds(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_x.shape[1] == 0:
        value = float(np.mean(train_y))
        return (
            np.full(len(train_y), value),
            np.full(len(dev_x), value),
            np.full(len(test_x), value),
        )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    model = Ridge(alpha=RIDGE_ALPHA, solver="cholesky")
    train_ready = scaler.fit_transform(imputer.fit_transform(train_x))
    model.fit(train_ready, train_y)
    dev_ready = scaler.transform(imputer.transform(dev_x))
    test_ready = scaler.transform(imputer.transform(test_x))
    return model.predict(train_ready), model.predict(dev_ready), model.predict(test_ready)


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def rmse_gap_interval(
    y_true: np.ndarray,
    other_pred: np.ndarray,
    reported_pred: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | bool]:
    point = float(
        np.sqrt(mean_squared_error(y_true, other_pred))
        - np.sqrt(mean_squared_error(y_true, reported_pred))
    )
    draws = rng.integers(0, len(y_true), size=(BOOTSTRAP_RESAMPLES, len(y_true)))
    sampled_y = y_true[draws]
    other_rmse = np.sqrt(np.mean((sampled_y - other_pred[draws]) ** 2, axis=1))
    reported_rmse = np.sqrt(np.mean((sampled_y - reported_pred[draws]) ** 2, axis=1))
    gaps = other_rmse - reported_rmse
    low, high = np.quantile(gaps, [0.025, 0.975])
    return {
        "point_gap": point,
        "interval_low": float(low),
        "interval_high": float(high),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_workbook() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / ZIP_NAME
    if not zip_path.exists():
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "ml-mastery-lab"})
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            zip_path.open("wb") as handle,
        ):
            handle.write(response.read())
    actual = sha256(zip_path)
    if actual != DATASET_SHA256:
        raise RuntimeError(f"zip sha256 mismatch: {actual}")
    xlsx_path = DATA_DIR / XLSX_NAME
    if not xlsx_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extract(XLSX_NAME, DATA_DIR)
    return xlsx_path


def read_sheet(path: Path) -> pd.DataFrame:
    from python_calamine import CalamineWorkbook

    workbook = CalamineWorkbook.from_path(str(path))
    if FORBIDDEN_SHEET not in workbook.sheet_names or SHEET_NAME not in workbook.sheet_names:
        raise RuntimeError(f"unexpected sheets: {workbook.sheet_names}")
    rows = workbook.get_sheet_by_name(SHEET_NAME).to_python()
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    if list(frame.columns) != list(REQUIRED_COLUMNS):
        raise RuntimeError(f"unexpected columns: {list(frame.columns)}")
    return frame


def file_audit(frame: pd.DataFrame) -> dict[str, float | int]:
    country = frame["Country"].astype(str)
    description = frame["Description"].map(
        lambda value: "" if value is None else str(value).strip()
    )
    uk_rows = int((country == "United Kingdom").sum())
    return {
        "countries": int(frame["Country"].nunique()),
        "united_kingdom_rows": uk_rows,
        "united_kingdom_row_share": float(uk_rows / len(frame)),
        "blank_description_rows": int((description == "").sum()),
    }


def plot_rmse(path: Path, experiments: dict[str, dict[str, object]], reported: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    names = list(EXPERIMENT_ORDER)
    x = np.arange(len(names))
    heights = [experiments[name]["test"]["rmse"] for name in names]
    bars = ax.bar(x, heights, color="#4C78A8")
    for bar, name in zip(bars, names, strict=True):
        if name in LEAKS:
            bar.set_hatch("//")
            bar.set_facecolor("#F58518")
    bars[names.index(reported)].set_facecolor("#54A24B")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Holdout RMSE, sterling spend in the label window, log scale")
    ax.set_xlabel("Aggregation recipe")
    ax.set_title("Same held-out customers. Log scale. Hatched bars are not eligible.")
    handles = [
        Patch(facecolor="#54A24B", label="Selected on development RMSE"),
        Patch(facecolor="#4C78A8", label="Eligible, not selected"),
        Patch(facecolor="#F58518", hatch="//", edgecolor="black", label="Not eligible"),
    ]
    ax.legend(handles=handles)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_monetary(path: Path, table: pd.DataFrame, holdout: np.ndarray) -> None:
    held = table.loc[holdout]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)
    pairs = (
        (axes[0], "monetary_sum", "History only, known at the cutoff"),
        (axes[1], "leaky_monetary_sum", "Includes the label window. Not eligible."),
    )
    for axis, column, title in pairs:
        axis.scatter(
            np.log1p(held[column].to_numpy(dtype=float)),
            np.log1p(held["label_spend"].to_numpy(dtype=float)),
            s=12,
            alpha=0.45,
            c="#4C78A8",
            linewidths=0,
        )
        axis.set_title(title)
        axis.set_xlabel("log1p of aggregated spend")
        axis.set_ylabel("log1p of label-window spend")
    fig.suptitle("Holdout customers. This plot did not choose the model.")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(
    path: Path, actual: np.ndarray, predicted: np.ndarray, model_name: str
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(actual, predicted, s=14, alpha=0.45, c="#4C78A8", linewidths=0)
    limit = float(max(np.max(np.abs(actual)), np.max(np.abs(predicted)), 1.0))
    ax.plot([-limit, limit], [-limit, limit], color="#666666", linewidth=1)
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_title(f"Holdout predicted spend, {model_name}. Not used to choose the model.")
    ax.set_xlabel("Actual label-window spend, sterling, symlog")
    ax.set_ylabel("Predicted spend, sterling, symlog")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def to_builtin(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite metric")
    return value


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    workbook = ensure_workbook()
    raw = read_sheet(workbook)
    lines, purchase_audit = prepare_lines(raw)
    sheet_end = pd.Timestamp(lines["invoice_date"].max())
    cutoff = cutoff_from_max(sheet_end)
    table = build_customer_table(lines, cutoff, sheet_end)
    identity_gap = (
        table["leaky_monetary_sum"] - table["monetary_sum"] - table["label_spend"]
    ).abs()
    train_ids, dev_ids, test_ids = split_customers(table.index.astype(str).tolist())
    folds = {"train": table.loc[train_ids], "dev": table.loc[dev_ids], "test": table.loc[test_ids]}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    experiments: dict[str, dict[str, object]] = {}
    selection_rows: list[dict[str, object]] = []
    for spec in experiment_specs():
        name = str(spec["name"])
        features = tuple(spec["features"])
        train_pred, dev_pred, test_pred = predict_folds(
            design_matrix(folds["train"], features),
            folds["train"]["label_spend"].to_numpy(dtype=float),
            design_matrix(folds["dev"], features),
            design_matrix(folds["test"], features),
        )
        predictions[name] = {"train": train_pred, "dev": dev_pred, "test": test_pred}
        scored = {
            fold: regression_scores(
                folds[fold]["label_spend"].to_numpy(dtype=float),
                predictions[name][fold],
            )
            for fold in ("train", "dev", "test")
        }
        experiments[name] = {
            "eligible": bool(spec["eligible"]),
            "n_features": len(features),
            "features": list(features),
            "train": scored["train"],
            "dev": scored["dev"],
            "test": scored["test"],
        }
        selection_rows.append(
            {
                "name": name,
                "eligible": bool(spec["eligible"]),
                "dev_rmse": scored["dev"]["rmse"],
                "n_features": len(features),
                "test_rmse": scored["test"]["rmse"],
            }
        )
    reported = select_model(selection_rows)
    failed = worst_eligible(selection_rows, reported)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    test_y = folds["test"]["label_spend"].to_numpy(dtype=float)
    train_y = folds["train"]["label_spend"].to_numpy(dtype=float)
    comparisons = {
        name: rmse_gap_interval(
            test_y, predictions[name]["test"], predictions[reported]["test"], rng
        )
        for name in EXPERIMENT_ORDER
        if name != reported
    }
    train_label_p99 = float(np.quantile(train_y, LABEL_QUANTILE))
    above = test_y > train_label_p99
    residual = test_y - predictions[reported]["test"]
    sse = residual**2
    payload = {
        "dataset": {
            "name": "Online Retail II",
            "uci_id": 502,
            "doi": "10.24432/C5CG6D",
            "license": "CC BY 4.0",
            "citation": "Chen, D. (2012). Online Retail II. UCI Machine Learning Repository.",
            "url": DATASET_URL,
            "zip_sha256": DATASET_SHA256,
            "xlsx_sha256": sha256(workbook),
            "sheet": SHEET_NAME,
            "sheet_not_read": FORBIDDEN_SHEET,
            "rows": purchase_audit["rows"],
        },
        "protocol": {
            "question": "which history aggregation predicts label-window spend",
            "not_the_sql_project": "UCI 352 repeat purchase is a different extract and target",
            "label_window_days": LABEL_WINDOW_DAYS,
            "cutoff": cutoff.isoformat(),
            "sheet_end": sheet_end.isoformat(),
            "cutoff_rule": "floor sheet max timestamp to midnight, then subtract 90 days",
            "purchase_rule": (
                "parsed customer, quantity > 0, price > 0, invoice does not start with C"
            ),
            "eligible_rule": "at least one purchase line before the cutoff",
            "label": "sum of purchase line amount at or after the cutoff, else 0",
            "split": "sorted customer ids, seed 42, test_size 0.2, then 0.25 of the rest",
            "estimator": "Ridge alpha 1.0, cholesky, median imputer and scaler fit on train only",
            "ridge_alpha": RIDGE_ALPHA,
            "selection_metric": "development_rmse",
            "selection_tie_break": "fewer features, then name",
            "selection_uses_test_scores": False,
            "reported_model": reported,
            "failed_eligible_model": failed,
            "failed_rule": "highest development RMSE among eligible models that were not selected",
            "gap_definition": "other_test_rmse - reported_test_rmse",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "target_transform": "none",
        },
        "audit": {
            **purchase_audit,
            **file_audit(raw),
            "purchase_invoices": int(lines["invoice"].nunique()),
            "purchase_customers": int(lines["customer_id"].nunique()),
            "history_purchase_rows": int((lines["invoice_date"] < cutoff).sum()),
            "label_purchase_rows": int((lines["invoice_date"] >= cutoff).sum()),
            "spanning_invoices": spanning_invoice_count(lines, cutoff),
            "eligible_customers": int(len(table)),
            "label_positive_customers": int((table["label_spend"] > 0).sum()),
            "train_customers": int(len(train_ids)),
            "dev_customers": int(len(dev_ids)),
            "test_customers": int(len(test_ids)),
            "train_label_mean": float(np.mean(train_y)),
            "dev_label_mean": float(folds["dev"]["label_spend"].mean()),
            "test_label_mean": float(np.mean(test_y)),
            "train_label_p99": train_label_p99,
            "holdout_customers_above_train_label_p99": int(above.sum()),
            "reported_holdout_sse_share_above_train_label_p99": float(sse[above].sum() / sse.sum()),
            "leaky_monetary_minus_honest_minus_label_max_abs": float(identity_gap.max()),
            "customers_whose_leaky_monetary_exceeds_honest": int(
                (table["leaky_monetary_sum"] > table["monetary_sum"]).sum()
            ),
            "customers_only_in_label_window": int(
                len(set(lines["customer_id"]) - set(table.index))
            ),
            "reported_holdout_negative_predictions": int(np.sum(predictions[reported]["test"] < 0)),
        },
        "experiments": experiments,
        "comparisons": comparisons,
    }
    metrics_path = RESULTS / "metrics.json"
    metrics_path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n", encoding="utf-8")
    plot_rmse(RESULTS / "rmse_by_recipe.png", experiments, reported)
    plot_monetary(RESULTS / "monetary_vs_label.png", table, test_ids)
    plot_predictions(
        RESULTS / "holdout_pred_vs_actual.png",
        test_y,
        predictions[reported]["test"],
        reported,
    )
    print(f"reported {reported}")
    print(f"failed {failed}")
    print(f"test rmse {experiments[reported]['test']['rmse']}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
