"""Describe Capital Bikeshare hours before treating a correlation as a cause.

The 2011 rows are the training period. The 2012 rows are the next period a
stakeholder would ask about. A random holdout is reported separately and is
not the same question. Casual and registered are parts of the target, not
features. The reported model was chosen by that exclusion, not by the holdout.
"""

from __future__ import annotations

import json
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
RIDGE_ALPHA = 1.0
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 7
PSI_EPSILON = 1e-6
ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
REPOSITORY_URL = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
DATASET_DOI = "10.24432/C5W894"
PAGE_INSTANCE_CLAIM = 17389
PAGE_SEASON_CLAIM = "1:winter, 2:spring, 3:summer, 4:fall"
PAGE_TEMP_CLAIM = "(t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)"
PAGE_ATEMP_CLAIM = "(t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)"

EXPECTED_COLUMNS = [
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]
HONEST_NUMERIC = ["temp", "hum", "windspeed"]
HONEST_CATEGORICAL = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
TARGET = "cnt"

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def download_zip(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    request = urllib.request.Request(ZIP_URL, headers={"User-Agent": "ml-mastery-lab/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def find_named(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob("*") if path.is_file() and path.name.lower() == name]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name}, found {matches}")
    return matches[0]


def load_hour_frame() -> tuple[pd.DataFrame, str]:
    download_zip(DATA / "bike-sharing.zip")
    extracted = DATA / "extracted"
    if not extracted.exists():
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(DATA / "bike-sharing.zip") as archive:
            archive.extractall(extracted)
    frame = pd.read_csv(find_named(extracted, "hour.csv"))
    readme = find_named(extracted, "readme.txt").read_text(encoding="utf-8")
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected columns: {list(frame.columns)}")
    frame["dteday"] = pd.to_datetime(frame["dteday"])
    return frame, readme


def readme_line(readme: str, prefix: str) -> str:
    matches = [line.strip() for line in readme.splitlines() if line.strip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one readme line starting {prefix!r}, found {matches}")
    return matches[0]


def identity_residual(casual: np.ndarray, registered: np.ndarray, total: np.ndarray) -> np.ndarray:
    """casual + registered - cnt. An accounting identity is zero, not a cause."""
    return (
        np.asarray(casual, dtype=float)
        + np.asarray(registered, dtype=float)
        - np.asarray(total, dtype=float)
    )


def psi_from_counts(
    expected: np.ndarray, actual: np.ndarray, epsilon: float = PSI_EPSILON
) -> float:
    """Population stability index. Empty bins are clipped, then both sides are renormalized."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.shape != actual.shape or expected.ndim != 1:
        raise ValueError("PSI wants two equal count vectors")
    if np.any(expected < 0) or np.any(actual < 0):
        raise ValueError("counts cannot be negative")
    expected_share = expected / expected.sum()
    actual_share = actual / actual.sum()
    expected_share = np.clip(expected_share, epsilon, None)
    actual_share = np.clip(actual_share, epsilon, None)
    expected_share = expected_share / expected_share.sum()
    actual_share = actual_share / actual_share.sum()
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def quantile_psi(train: np.ndarray, test: np.ndarray, n_bins: int = 10) -> dict[str, float | int]:
    edges = np.quantile(train, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError("not enough unique quantile edges for PSI")
    train_bins = np.digitize(train, edges[1:-1], right=False)
    test_bins = np.digitize(test, edges[1:-1], right=False)
    n_used = len(edges) - 1
    train_counts = np.bincount(train_bins, minlength=n_used)
    test_counts = np.bincount(test_bins, minlength=n_used)
    return {
        "n_bins": int(n_used),
        "n_empty_train_bins": int(np.sum(train_counts == 0)),
        "n_empty_test_bins": int(np.sum(test_counts == 0)),
        "psi": psi_from_counts(train_counts, test_counts),
    }


def category_psi(train: np.ndarray, test: np.ndarray) -> dict[str, float | int]:
    levels = np.union1d(train, test)
    train_counts = np.array([np.sum(train == level) for level in levels])
    test_counts = np.array([np.sum(test == level) for level in levels])
    return {
        "n_bins": int(len(levels)),
        "n_empty_train_bins": int(np.sum(train_counts == 0)),
        "n_empty_test_bins": int(np.sum(test_counts == 0)),
        "psi": psi_from_counts(train_counts, test_counts),
    }


def distribution(values: np.ndarray) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    return {
        "n": int(clean.size),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)),
        "minimum": float(np.min(clean)),
        "maximum": float(np.max(clean)),
    }


def ks_record(train: np.ndarray, test: np.ndarray) -> dict[str, float]:
    result = stats.ks_2samp(train, test, method="auto")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def regression_scores(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mse = float(mean_squared_error(y_true, prediction))
    center = float(np.mean(y_true))
    total = float(np.mean((y_true - center) ** 2))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2_against_scored_mean": float(1.0 - mse / total),
        "mse": mse,
    }


def pointwise_squared_error(y_true: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    error = np.asarray(prediction, dtype=float) - np.asarray(y_true, dtype=float)
    return error**2


def bootstrap_diff(
    loss_a: np.ndarray, loss_b: np.ndarray, seed: int
) -> dict[str, float | int | bool]:
    """Mean of loss_a minus loss_b. A positive value means b has the lower loss."""
    observed = float(np.mean(loss_a - loss_b))
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(loss_a), size=(BOOTSTRAP_RESAMPLES, len(loss_a)))
    diffs = np.mean(loss_a[indexes] - loss_b[indexes], axis=1)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return {
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "seed": seed,
        "mean_diff": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def make_ridge(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("a model needs at least one column")
    return Pipeline(
        [
            ("columns", ColumnTransformer(transformers)),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[dict[str, float], np.ndarray, Pipeline]:
    model = make_ridge(numeric, categorical)
    columns = numeric + categorical
    model.fit(train[columns], train[TARGET].to_numpy(dtype=float))
    prediction = model.predict(test[columns])
    scores = regression_scores(test[TARGET].to_numpy(dtype=float), prediction)
    scores["n_train"] = int(len(train))
    scores["n_scored"] = int(len(test))
    return scores, prediction, model


def scaled_numeric_coefficients(model: Pipeline, numeric: list[str]) -> dict[str, float]:
    ridge = model.named_steps["ridge"]
    return {
        name: float(coef) for name, coef in zip(numeric, ridge.coef_[: len(numeric)], strict=True)
    }


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def month_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts = frame.groupby(["season", "mnth"], dropna=False).size()
    out: dict[str, dict[str, int]] = {}
    for (season, month), count in counts.items():
        out.setdefault(str(int(season)), {})[str(int(month))] = int(count)
    return out


def weekday_names(frame: pd.DataFrame) -> dict[str, list[str]]:
    names = frame.groupby("weekday")["dteday"].apply(
        lambda values: sorted(pd.to_datetime(values).dt.day_name().unique())
    )
    return {str(int(key)): list(value) for key, value in names.items()}


def workingday_disagreements(frame: pd.DataFrame) -> int:
    weekend = frame["dteday"].dt.day_name().isin(["Saturday", "Sunday"])
    expected = (~weekend & frame["holiday"].eq(0)).astype(int)
    return int((expected.to_numpy() != frame["workingday"].to_numpy()).sum())


def missing_hours(frame: pd.DataFrame) -> dict[str, int]:
    start = frame["dteday"].min()
    end = frame["dteday"].max()
    days = pd.date_range(start, end, freq="D")
    observed = set(
        zip(frame["dteday"].dt.strftime("%Y-%m-%d"), frame["hr"].astype(int), strict=True)
    )
    missing = 0
    days_missing = 0
    for day in days:
        stamp = day.strftime("%Y-%m-%d")
        absent = sum((stamp, hour) not in observed for hour in range(24))
        missing += absent
        days_missing += int(absent > 0)
    return {
        "n_calendar_days": int(len(days)),
        "n_expected_hours": int(len(days) * 24),
        "n_observed_hours": int(len(frame)),
        "n_missing_hours": int(missing),
        "n_days_with_a_missing_hour": int(days_missing),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def association_table(frame: pd.DataFrame) -> list[dict[str, float | str | int]]:
    rows = [
        {
            "scope": "all_rows",
            "n": int(len(frame)),
            "temp_cnt_pearson": pearson(frame["temp"], frame["cnt"]),
            "casual_cnt_pearson": pearson(frame["casual"], frame["cnt"]),
            "temp_atemp_pearson": pearson(frame["temp"], frame["atemp"]),
        }
    ]
    for year, group in frame.groupby("yr"):
        rows.append(
            {
                "scope": f"year_{int(year)}",
                "n": int(len(group)),
                "temp_cnt_pearson": pearson(group["temp"], group["cnt"]),
                "casual_cnt_pearson": pearson(group["casual"], group["cnt"]),
                "temp_atemp_pearson": pearson(group["temp"], group["atemp"]),
            }
        )
    for season, group in frame.groupby("season"):
        rows.append(
            {
                "scope": f"season_{int(season)}",
                "n": int(len(group)),
                "temp_cnt_pearson": pearson(group["temp"], group["cnt"]),
                "casual_cnt_pearson": pearson(group["casual"], group["cnt"]),
                "temp_atemp_pearson": pearson(group["temp"], group["atemp"]),
            }
        )
    return rows


def plot_histograms(path: Path, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bins = np.linspace(frame["cnt"].min(), frame["cnt"].max(), 40)
    for year, color, label in ((0, "#3174a1", "2011"), (1, "#c23b22", "2012")):
        values = frame.loc[frame["yr"] == year, "cnt"]
        ax.hist(values, bins=bins, color=color, alpha=0.55, label=label)
    ax.set_xlabel("Rentals in the hour")
    ax.set_ylabel("Hours")
    ax.set_title("Hourly rental counts by year")
    ax.legend()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_hour_profile(path: Path, profile: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for year, color, label in ((0, "#3174a1", "2011"), (1, "#c23b22", "2012")):
        rows = profile.loc[profile["yr"] == year].sort_values("hr")
        ax.plot(rows["hr"], rows["mean_cnt"], color=color, marker="o", label=label)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean rentals")
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, profile["mean_cnt"].max() * 1.08)
    ax.set_title("Same commuting shape. The level is the shift.")
    ax.legend()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_temp(path: Path, frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    hexbin = axes[0].hexbin(frame["temp"], frame["cnt"], gridsize=30, cmap="Blues", mincnt=1)
    axes[0].set_xlabel("Normalized temperature, file units")
    axes[0].set_ylabel("Rentals in the hour")
    axes[0].set_title("Temperature and rentals, all hours")
    colorbar = fig.colorbar(hexbin, ax=axes[0])
    colorbar.set_label("Hours in the bin")
    edges = np.quantile(frame.loc[frame["yr"] == 0, "temp"], np.linspace(0.0, 1.0, 11))
    edges = np.unique(edges)
    for year, color, label in ((0, "#3174a1", "2011"), (1, "#c23b22", "2012")):
        group = frame.loc[frame["yr"] == year]
        bins = np.digitize(group["temp"], edges[1:-1], right=False)
        means = [float(group.loc[bins == index, "cnt"].mean()) for index in range(len(edges) - 1)]
        centers = (edges[:-1] + edges[1:]) / 2.0
        axes[1].plot(centers, means, color=color, marker="o", label=label)
    axes[1].set_xlabel("2011 temperature-bin center")
    axes[1].set_ylabel("Mean rentals")
    axes[1].set_title("Same bins. The year gap remains.")
    axes[1].legend()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_correlation(path: Path, frame: pd.DataFrame) -> None:
    columns = ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    corr = frame[columns].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    image = ax.imshow(corr.to_numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for row in range(len(columns)):
        for col in range(len(columns)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Pearson correlation. Casual and registered are parts of cnt.")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_models(path: Path, runs: list[dict]) -> None:
    ordered = sorted(runs, key=lambda row: row["rmse"])
    colors = ["#c23b22" if row["name"] == "leaky_casual" else "#3174a1" for row in ordered]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.barh([row["name"] for row in ordered], [row["rmse"] for row in ordered], color=colors)
    ax.set_xlabel("RMSE on 2012 hours")
    ax.set_title("2012 holdout. The red bar uses part of the target.")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_seasons(path: Path, frame: pd.DataFrame) -> None:
    table = frame.groupby(["season", "mnth"]).size().unstack(fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    image = ax.imshow(table.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(column) for column in table.columns])
    ax.set_yticks(range(table.shape[0]), [f"season {index}" for index in table.index])
    ax.set_xlabel("Month number")
    ax.set_ylabel("Season code")
    ax.set_title("Season codes share boundary months. Names are not in the file.")
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            ax.text(col, row, str(int(table.iloc[row, col])), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Hours")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def inventory_from(frame: pd.DataFrame, readme: str) -> dict:
    residual = identity_residual(frame["casual"], frame["registered"], frame["cnt"])
    duplicate_hours = int(frame.duplicated(["dteday", "hr"]).sum())
    months = month_counts(frame)
    shared = []
    for month in range(1, 13):
        seasons = sorted(int(season) for season, values in months.items() if str(month) in values)
        if len(seasons) > 1:
            shared.append({"month": month, "seasons": seasons})
    return {
        "n_rows": int(len(frame)),
        "n_missing_cells": int(frame.isna().sum().sum()),
        "date_min": frame["dteday"].min().strftime("%Y-%m-%d"),
        "date_max": frame["dteday"].max().strftime("%Y-%m-%d"),
        "n_year_0": int((frame["yr"] == 0).sum()),
        "n_year_1": int((frame["yr"] == 1).sum()),
        "identity_max_abs_residual": float(np.max(np.abs(residual))),
        "identity_n_nonzero": int(np.sum(residual != 0)),
        "n_duplicate_date_hour": duplicate_hours,
        "missing_hours": missing_hours(frame),
        "months_by_season": months,
        "months_in_more_than_one_season": shared,
        "weekday_names": weekday_names(frame),
        "n_workingday_definition_disagreements": workingday_disagreements(frame),
        "weathersit_counts": {
            str(int(key)): int(value)
            for key, value in frame["weathersit"].value_counts().sort_index().items()
        },
        "readme_season_line": readme_line(readme, "- season"),
        "readme_temp_line": readme_line(readme, "- temp"),
        "readme_atemp_line": readme_line(readme, "- atemp"),
        "readme_citation_doi": "10.1007/s13748-013-0040-3",
    }


def shift_from(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    columns = ["cnt", "temp", "atemp", "hum", "windspeed", "casual", "registered"]
    by_year = {
        column: {"year_0": distribution(train[column]), "year_1": distribution(test[column])}
        for column in columns
    }
    ks = {
        column: ks_record(train[column].to_numpy(), test[column].to_numpy())
        for column in ["cnt", "temp", "hum", "windspeed", "hr"]
    }
    profile = (
        pd.concat([train, test], ignore_index=True)
        .groupby(["yr", "hr"], as_index=False)["cnt"]
        .mean()
        .rename(columns={"cnt": "mean_cnt"})
    )
    means = profile.pivot(index="hr", columns="yr", values="mean_cnt")
    casual_share = {
        "year_0": float(train["casual"].sum() / train["cnt"].sum()),
        "year_1": float(test["casual"].sum() / test["cnt"].sum()),
    }
    return {
        "by_year": by_year,
        "ks": ks,
        "cnt_psi": quantile_psi(
            train["cnt"].to_numpy(dtype=float), test["cnt"].to_numpy(dtype=float)
        ),
        "temp_psi": quantile_psi(
            train["temp"].to_numpy(dtype=float), test["temp"].to_numpy(dtype=float)
        ),
        "hour_psi": category_psi(train["hr"].to_numpy(), test["hr"].to_numpy()),
        "weathersit_psi": category_psi(
            train["weathersit"].to_numpy(), test["weathersit"].to_numpy()
        ),
        "hour_profile_pearson": pearson(means[0].to_numpy(), means[1].to_numpy()),
        "hour_profile": [
            {
                "hr": int(hour),
                "year_0": float(means.loc[hour, 0]),
                "year_1": float(means.loc[hour, 1]),
            }
            for hour in means.index
        ],
        "casual_share_of_cnt": casual_share,
        "profile_frame": profile,
    }


def random_split_report(frame: pd.DataFrame, test_size: float) -> dict:
    index = np.arange(len(frame))
    train_index, test_index = train_test_split(index, test_size=test_size, random_state=SEED)
    train = frame.iloc[train_index]
    test = frame.iloc[test_index]
    scores, _, _ = fit_and_score(train, test, HONEST_NUMERIC, HONEST_CATEGORICAL)
    scores["name"] = "honest_on_random_holdout"
    scores["n_year_1_in_train"] = int((train["yr"] == 1).sum())
    scores["n_year_1_in_holdout"] = int((test["yr"] == 1).sum())
    scores["share_year_1_in_train"] = float(np.mean(train["yr"] == 1))
    return scores


def main() -> None:
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame, readme = load_hour_frame()
    if "10.1007/s13748-013-0040-3" not in readme:
        raise ValueError("zip readme no longer contains the cited DOI")
    inventory = inventory_from(frame, readme)
    train = frame.loc[frame["yr"] == 0].copy()
    test = frame.loc[frame["yr"] == 1].copy()
    shift = shift_from(train, test)
    profile = shift.pop("profile_frame")
    association = association_table(frame)

    runs = []
    predictions = {}
    baseline_prediction = np.full(len(test), float(train[TARGET].mean()))
    baseline = regression_scores(test[TARGET].to_numpy(dtype=float), baseline_prediction)
    baseline.update({"name": "train_mean", "n_train": int(len(train)), "n_scored": int(len(test))})
    runs.append(baseline)
    predictions["train_mean"] = baseline_prediction

    specifications = {
        "honest": (HONEST_NUMERIC, HONEST_CATEGORICAL),
        "temp_only": (["temp"], []),
        "honest_plus_atemp": (["temp", "atemp", "hum", "windspeed"], HONEST_CATEGORICAL),
        "leaky_casual": (["casual", *HONEST_NUMERIC], HONEST_CATEGORICAL),
        "honest_plus_yr": (["yr", *HONEST_NUMERIC], HONEST_CATEGORICAL),
    }
    coefficients = {}
    for name, (numeric, categorical) in specifications.items():
        scores, prediction, model = fit_and_score(train, test, numeric, categorical)
        scores["name"] = name
        runs.append(scores)
        predictions[name] = prediction
        coefficients[name] = scaled_numeric_coefficients(model, numeric)

    random_scores = random_split_report(frame, len(test) / len(frame))
    comparisons = [
        bootstrap_diff(
            pointwise_squared_error(test[TARGET], predictions["train_mean"]),
            pointwise_squared_error(test[TARGET], predictions["honest"]),
            BOOTSTRAP_SEED,
        ),
        bootstrap_diff(
            pointwise_squared_error(test[TARGET], predictions["honest"]),
            pointwise_squared_error(test[TARGET], predictions["leaky_casual"]),
            BOOTSTRAP_SEED + 1,
        ),
        bootstrap_diff(
            pointwise_squared_error(test[TARGET], predictions["temp_only"]),
            pointwise_squared_error(test[TARGET], predictions["honest"]),
            BOOTSTRAP_SEED + 2,
        ),
    ]
    comparison_names = [
        "train_mean_minus_honest",
        "honest_minus_leaky_casual",
        "temp_only_minus_honest",
    ]
    for name, comparison in zip(comparison_names, comparisons, strict=True):
        comparison["name"] = name

    plot_histograms(RESULTS / "cnt_by_year.png", frame)
    plot_hour_profile(RESULTS / "hour_profile.png", profile)
    plot_temp(RESULTS / "temp_vs_cnt.png", frame)
    plot_correlation(RESULTS / "correlation.png", frame)
    plot_models(
        RESULTS / "model_rmse.png",
        [row for row in runs if row["name"] != "honest_plus_yr"],
    )
    plot_seasons(RESULTS / "season_months.png", frame)

    metrics = {
        "elapsed_seconds": time.perf_counter() - started,
        "data": {
            "name": "Bike Sharing",
            "uci_id": 275,
            "doi": DATASET_DOI,
            "repository_url": REPOSITORY_URL,
            "zip_url": ZIP_URL,
            "creators": ["Hadi Fanaee-T"],
            "license": "CC BY 4.0",
            "paper_doi_from_zip_readme": "10.1007/s13748-013-0040-3",
            "page_instance_claim": PAGE_INSTANCE_CLAIM,
            "page_season_claim": PAGE_SEASON_CLAIM,
            "page_temp_claim": PAGE_TEMP_CLAIM,
            "page_atemp_claim": PAGE_ATEMP_CLAIM,
        },
        "protocol": {
            "seed": SEED,
            "ridge_alpha": RIDGE_ALPHA,
            "time_split": "yr == 0 train, yr == 1 test",
            "reported_model": "honest",
            "excluded_from_honest_model": [
                "instant",
                "dteday",
                "yr",
                "atemp",
                "casual",
                "registered",
            ],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "psi_epsilon": PSI_EPSILON,
            "r2_definition": "1 - mse / mean squared deviation from the scored-period mean",
        },
        "inventory": inventory,
        "shift": shift,
        "association": association,
        "coefficients": coefficients,
        "time_split_runs": runs,
        "random_split": random_scores,
        "bootstrap": comparisons,
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "elapsed_seconds": metrics["elapsed_seconds"],
                "honest_rmse": next(row["rmse"] for row in runs if row["name"] == "honest"),
                "baseline_rmse": baseline["rmse"],
            }
        )
    )


if __name__ == "__main__":
    main()
