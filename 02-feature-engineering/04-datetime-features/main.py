"""Calendar, lag, and rolling features for hourly traffic volume.

The reported model is the eligible experiment with the lowest training-period
time-series CV RMSE. Leaky fits are scored on the same held-out hours and are
not eligible. The holdout is not used to add, drop, or replace a feature.

This is a teaching model on a public sensor extract. It is not a traffic tool.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TRAIN_END_FRACTION = 0.8
RIDGE_ALPHA = 1.0
N_SPLITS = 4
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 7
ROLL_WINDOW = 24
LAGS = (1, 24, 168)
WEEK_PLOT_HOURS = 168
TEMP_INVALID_BELOW = 200.0
RAIN_INVALID_ABOVE = 100.0
DATASET_ID = 492
DATASET_DOI = "10.24432/C5X60B"
ZIP_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
API_URL = "https://archive.ics.uci.edu/api/dataset?id=492"
API_CSV_URL = "https://archive.ics.uci.edu/static/public/492/data.csv"
ZIP_MEMBER = "Metro_Interstate_Traffic_Volume.csv.gz"

RAW_CALENDAR = ("hour", "dow", "month", "holiday_date")
ONE_HARMONIC = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "holiday_date",
)
TWO_HARMONIC = (
    "hour_sin",
    "hour_cos",
    "hour_sin2",
    "hour_cos2",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "holiday_date",
)
VOLUME_LAGS = (
    "lag_1",
    "lag_24",
    "lag_168",
    "lag_1_missing",
    "lag_24_missing",
    "lag_168_missing",
    "roll_mean_24",
    "roll_count_24",
    "roll_mean_missing",
)
WEATHER_LAGS = (
    "temp_lag1",
    "rain_lag1",
    "snow_lag1",
    "clouds_lag1",
    "temp_lag1_missing",
    "rain_lag1_missing",
    "snow_lag1_missing",
    "clouds_lag1_missing",
)
SAME_WEATHER = (
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "temp_missing",
    "rain_missing",
    "snow_missing",
    "clouds_missing",
)
LEAKY_LAGS = (
    "lag_1",
    "lag_24",
    "lag_168",
    "lag_1_missing",
    "lag_24_missing",
    "lag_168_missing",
    "leaky_roll_mean",
    "leaky_roll_count",
    "leaky_roll_missing",
)
ROW_LAGS = (
    "row_lag_1",
    "row_lag_24",
    "row_lag_168",
    "row_lag_1_missing",
    "row_lag_24_missing",
    "row_lag_168_missing",
    "row_roll_mean",
    "row_roll_count",
    "row_roll_missing",
)
LEAKS = frozenset({"same_hour_weather", "leaky_rolling", "row_shift_lags"})
CONTROLS = frozenset({"post_gap_refit"})
EXPERIMENT_ORDER = (
    "training_mean",
    "seasonal_naive_24",
    "raw_calendar",
    "one_harmonic",
    "two_harmonic",
    "one_hot_hour",
    "lags_only",
    "two_harmonic_plus_lags",
    "same_hour_weather",
    "leaky_rolling",
    "row_shift_lags",
    "post_gap_refit",
)
LABELS = {
    "training_mean": "training mean",
    "seasonal_naive_24": "yesterday, same hour",
    "raw_calendar": "raw calendar",
    "one_harmonic": "one harmonic",
    "two_harmonic": "two harmonics",
    "one_hot_hour": "one-hot hour",
    "lags_only": "lags only",
    "two_harmonic_plus_lags": "harmonics plus lags",
    "same_hour_weather": "same-hour weather",
    "leaky_rolling": "rolling includes now",
    "row_shift_lags": "row shift lags",
    "post_gap_refit": "refit after the gap",
}

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def hour_cycle(hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * np.asarray(hour, dtype=float) / 24.0
    return np.sin(angle), np.cos(angle)


def two_hour_cycle(hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = 4.0 * np.pi * np.asarray(hour, dtype=float) / 24.0
    return np.sin(angle), np.cos(angle)


def dow_cycle(dow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * np.asarray(dow, dtype=float) / 7.0
    return np.sin(angle), np.cos(angle)


def month_cycle(month: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * (np.asarray(month, dtype=float) % 12.0) / 12.0
    return np.sin(angle), np.cos(angle)


def time_lag(index: pd.DatetimeIndex, values: np.ndarray, hours: int) -> np.ndarray:
    if hours <= 0:
        raise ValueError("time lag must look at an earlier hour, not the future")
    series = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(index))
    lookup = pd.DatetimeIndex(index) - pd.Timedelta(hours=hours)
    return series.reindex(lookup).to_numpy(dtype=float)


def clock_window(
    index: pd.DatetimeIndex,
    values: np.ndarray,
    window: int,
    include_current: bool,
) -> tuple[np.ndarray, np.ndarray]:
    series = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(index))
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    source = series.reindex(full_index)
    if not include_current:
        source = source.shift(1)
    mean = source.rolling(window, min_periods=1).mean()
    count = source.rolling(window, min_periods=1).count()
    return (
        mean.reindex(series.index).to_numpy(dtype=float),
        count.reindex(series.index).to_numpy(dtype=float),
    )


def past_rolling(
    index: pd.DatetimeIndex,
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    return clock_window(index, values, window, include_current=False)


def leaky_rolling(index: pd.DatetimeIndex, values: np.ndarray, window: int) -> np.ndarray:
    mean, _count = clock_window(index, values, window, include_current=True)
    return mean


def row_shift(values: np.ndarray, periods: int) -> np.ndarray:
    if periods <= 0:
        raise ValueError("row shift must look at an earlier row, not the future")
    return pd.Series(np.asarray(values, dtype=float)).shift(periods).to_numpy(dtype=float)


def row_past_rolling(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    past = pd.Series(np.asarray(values, dtype=float)).shift(1)
    mean = past.rolling(window, min_periods=1).mean()
    count = past.rolling(window, min_periods=1).count()
    return mean.to_numpy(dtype=float), count.to_numpy(dtype=float)


def holiday_date_flag(index: pd.DatetimeIndex, names: np.ndarray) -> np.ndarray:
    stamped = pd.Series(names, index=pd.DatetimeIndex(index)).dropna()
    if stamped.empty:
        return np.zeros(len(index), dtype=float)
    holiday_days = set(pd.DatetimeIndex(stamped.index).date)
    return np.array(
        [1.0 if stamp.date() in holiday_days else 0.0 for stamp in pd.DatetimeIndex(index)],
        dtype=float,
    )


def make_splits(n: int, train_end_fraction: float = TRAIN_END_FRACTION) -> dict[str, np.ndarray]:
    cut = int(n * train_end_fraction)
    if cut < 1 or n - cut < 1:
        raise ValueError("train fraction does not leave a later holdout")
    return {"train": np.arange(0, cut), "holdout": np.arange(cut, n)}


def experiment_specs() -> list[dict]:
    ridge = "ridge"
    return [
        {"name": "training_mean", "kind": "rate", "eligible": True, "n_features": 0},
        {"name": "seasonal_naive_24", "kind": "naive", "eligible": True, "n_features": 1},
        {
            "name": "raw_calendar",
            "kind": ridge,
            "eligible": True,
            "columns": RAW_CALENDAR,
            "categorical": (),
        },
        {
            "name": "one_harmonic",
            "kind": ridge,
            "eligible": True,
            "columns": ONE_HARMONIC,
            "categorical": (),
        },
        {
            "name": "two_harmonic",
            "kind": ridge,
            "eligible": True,
            "columns": TWO_HARMONIC,
            "categorical": (),
        },
        {
            "name": "one_hot_hour",
            "kind": ridge,
            "eligible": True,
            "columns": ("holiday_date",),
            "categorical": ("hour", "dow", "month"),
        },
        {
            "name": "lags_only",
            "kind": ridge,
            "eligible": True,
            "columns": VOLUME_LAGS,
            "categorical": (),
        },
        {
            "name": "two_harmonic_plus_lags",
            "kind": ridge,
            "eligible": True,
            "columns": TWO_HARMONIC + VOLUME_LAGS + WEATHER_LAGS,
            "categorical": (),
        },
        {
            "name": "same_hour_weather",
            "kind": ridge,
            "eligible": False,
            "columns": TWO_HARMONIC + VOLUME_LAGS + WEATHER_LAGS + SAME_WEATHER,
            "categorical": (),
        },
        {
            "name": "leaky_rolling",
            "kind": ridge,
            "eligible": False,
            "columns": TWO_HARMONIC + LEAKY_LAGS + WEATHER_LAGS,
            "categorical": (),
        },
        {
            "name": "row_shift_lags",
            "kind": ridge,
            "eligible": False,
            "columns": TWO_HARMONIC + ROW_LAGS + WEATHER_LAGS,
            "categorical": (),
        },
        {
            "name": "post_gap_refit",
            "kind": ridge,
            "eligible": False,
            "columns": TWO_HARMONIC + VOLUME_LAGS + WEATHER_LAGS,
            "categorical": (),
            "train": "post_gap",
        },
    ]


def select_model(rows: list[dict]) -> str:
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no eligible model")
    eligible.sort(key=lambda row: (float(row["cv_rmse"]), int(row["n_features"]), str(row["name"])))
    return str(eligible[0]["name"])


def missing_indicator(values: np.ndarray) -> np.ndarray:
    return np.isnan(values).astype(float)


def impute_scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.array(train, dtype=float, copy=True)
    test = np.array(test, dtype=float, copy=True)
    medians = np.zeros(train.shape[1], dtype=float)
    for column in range(train.shape[1]):
        observed = train[:, column]
        observed = observed[np.isfinite(observed)]
        medians[column] = (
            float(observed.mean()) if observed.size == 0 else float(np.median(observed))
        )
    for matrix in (train, test):
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix[missing] = np.take(medians, np.where(missing)[1])
    scaler = StandardScaler()
    return scaler.fit_transform(train), scaler.transform(test)


def design(
    frame: pd.DataFrame,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    columns: tuple[str, ...],
    categorical: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    fit_num, eval_num = impute_scale(
        frame.iloc[fit_idx][list(columns)].to_numpy(dtype=float),
        frame.iloc[eval_idx][list(columns)].to_numpy(dtype=float),
    )
    if not categorical:
        return fit_num, eval_num
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    fit_cat = frame.iloc[fit_idx][list(categorical)].astype(str)
    eval_cat = frame.iloc[eval_idx][list(categorical)].astype(str)
    encoder.fit(fit_cat)
    return (
        np.hstack([fit_num, encoder.transform(fit_cat)]),
        np.hstack([eval_num, encoder.transform(eval_cat)]),
    )


def score_prediction(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError("score received a non-finite value")
    residual = y - pred
    mse = float(np.mean(residual**2))
    centered = y - float(y.mean())
    total = float(np.mean(centered**2))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - mse / total) if total > 0.0 else None,
        "mean_predicted": float(pred.mean()),
        "mean_actual": float(y.mean()),
    }


def predict_rate(frame: pd.DataFrame, fit_idx: np.ndarray, eval_idx: np.ndarray) -> np.ndarray:
    rate = float(frame.iloc[fit_idx]["traffic_volume"].mean())
    return np.full(len(eval_idx), rate, dtype=float)


def predict_naive(frame: pd.DataFrame, fit_idx: np.ndarray, eval_idx: np.ndarray) -> np.ndarray:
    fallback = float(frame.iloc[fit_idx]["traffic_volume"].mean())
    lagged = frame.iloc[eval_idx]["lag_24"].to_numpy(dtype=float)
    return np.where(np.isnan(lagged), fallback, lagged)


def predict_ridge(
    frame: pd.DataFrame,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    spec: dict,
) -> tuple[np.ndarray, int]:
    train_x, eval_x = design(
        frame,
        fit_idx,
        eval_idx,
        tuple(spec["columns"]),
        tuple(spec["categorical"]),
    )
    model = Ridge(alpha=RIDGE_ALPHA, solver="cholesky")
    model.fit(train_x, frame.iloc[fit_idx]["traffic_volume"].to_numpy(dtype=float))
    return model.predict(eval_x).astype(float), int(train_x.shape[1])


def fold_rmse(
    frame: pd.DataFrame, train_idx: np.ndarray, spec: dict
) -> tuple[float, float, list[float]]:
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    folds: list[float] = []
    for train_pos, valid_pos in splitter.split(train_idx):
        fit_idx = train_idx[train_pos]
        valid_idx = train_idx[valid_pos]
        if spec["kind"] == "rate":
            pred = predict_rate(frame, fit_idx, valid_idx)
        elif spec["kind"] == "naive":
            pred = predict_naive(frame, fit_idx, valid_idx)
        else:
            pred, _ = predict_ridge(frame, fit_idx, valid_idx, spec)
        y = frame.iloc[valid_idx]["traffic_volume"].to_numpy(dtype=float)
        folds.append(score_prediction(y, pred)["rmse"])
    return float(np.mean(folds)), float(np.std(folds)), folds


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def cluster_bootstrap_gap(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
) -> dict[str, float | int | str]:
    buckets: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters.tolist()):
        buckets.setdefault(str(cluster), []).append(index)
    keys = list(buckets)
    row_lists = [np.array(buckets[key], dtype=int) for key in keys]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gaps = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for draw in range(BOOTSTRAP_RESAMPLES):
        chosen = rng.integers(0, len(keys), size=len(keys))
        rows = np.concatenate([row_lists[index] for index in chosen])
        gaps[draw] = rmse(y[rows], reference[rows]) - rmse(y[rows], candidate[rows])
    return {
        "gap_mean": float(gaps.mean()),
        "gap_p2_5": float(np.quantile(gaps, 0.025)),
        "gap_p97_5": float(np.quantile(gaps, 0.975)),
        "fraction_positive": float(np.mean(gaps > 0)),
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "n_clusters": len(keys),
        "seed": BOOTSTRAP_SEED,
        "unit": "local date within the holdout",
        "definition": "reference RMSE minus candidate RMSE",
    }


def plot_hour_profile(path: Path, rows: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    axis.plot(
        [row["hour"] for row in rows],
        [row["mean_volume"] for row in rows],
        color="#4C78A8",
        marker="o",
    )
    axis.set_xticks(range(0, 24, 2))
    axis.set_xlabel("Hour of day, local time as stored in the file")
    axis.set_ylabel("Mean traffic volume")
    axis.set_title("Training hours only. The day is not a single wave.")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_rmse(path: Path, scores: dict[str, dict], reported: str) -> None:
    names = list(EXPERIMENT_ORDER)
    values = [scores[name]["test"]["rmse"] for name in names]
    colors = []
    for name in names:
        if name == reported:
            colors.append("#54A24B")
        elif name in LEAKS:
            colors.append("#E45756")
        elif name == "training_mean":
            colors.append("#9D9D9D")
        elif name == "seasonal_naive_24":
            colors.append("#F58518")
        else:
            colors.append("#4C78A8")
    fig, axis = plt.subplots(figsize=(10.2, 6.8), constrained_layout=True)
    bars = axis.barh([LABELS[name] for name in names], values, color=colors)
    for bar, name in zip(bars, names, strict=True):
        if name in LEAKS:
            bar.set_hatch("//")
    axis.invert_yaxis()
    axis.set_xlabel("RMSE on the held-out hours")
    axis.set_ylabel("Experiment")
    axis.set_title("Same held-out hours. Lower is better. Hatched bars are not a forecast.")
    axis.legend(
        handles=[
            Patch(facecolor="#54A24B", label="Reported protocol"),
            Patch(facecolor="#9D9D9D", label="Training mean"),
            Patch(facecolor="#F58518", label="Seasonal naive"),
            Patch(facecolor="#E45756", hatch="//", label="Not a forecast"),
            Patch(facecolor="#4C78A8", label="Other fit"),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
    )
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_week(
    path: Path,
    index: pd.DatetimeIndex,
    actual: np.ndarray,
    curves: dict[str, np.ndarray],
    reported_name: str,
) -> None:
    fig, axis = plt.subplots(figsize=(10.6, 4.8), constrained_layout=True)
    axis.plot(index, actual, color="#333333", linewidth=1.4, label="actual")
    colors = {"reported": "#54A24B", "seasonal_naive_24": "#F58518", "training_mean": "#9D9D9D"}
    for name, values in curves.items():
        label = LABELS[reported_name] if name == "reported" else LABELS[name]
        axis.plot(index, values, color=colors[name], linewidth=1.2, label=label)
    axis.xaxis.set_major_locator(mdates.DayLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axis.set_xlabel("Held-out hour. First 168 hours, not a week chosen after the plot.")
    axis.set_ylabel("Traffic volume")
    axis.set_title("First 168 held-out hours")
    axis.legend(frameon=False, ncol=3)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_residual(path: Path, rows: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    axis.bar([row["hour"] for row in rows], [row["mean_residual"] for row in rows], color="#4C78A8")
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_xticks(range(0, 24, 2))
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Mean actual minus predicted")
    axis.set_title("Held-out residual of the reported model")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def py(value):
    if isinstance(value, dict):
        return {str(key): py(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    return value


def normalize_holiday(series: pd.Series) -> pd.Series:
    return series.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})


def read_official() -> pd.DataFrame:
    download(ZIP_URL, DATA / "traffic.zip")
    with zipfile.ZipFile(DATA / "traffic.zip") as archive:
        if archive.namelist() != [ZIP_MEMBER]:
            raise ValueError(f"unexpected zip members: {archive.namelist()}")
        with gzip.open(archive.open(ZIP_MEMBER), "rt", encoding="utf-8") as handle:
            frame = pd.read_csv(handle, keep_default_na=False)
    return frame


def assert_api_match(official: pd.DataFrame, api: dict) -> dict[str, bool | int | str | None]:
    download(API_CSV_URL, DATA / "uci_data.csv")
    exported = pd.read_csv(DATA / "uci_data.csv", keep_default_na=False)
    if list(exported.columns) != list(official.columns):
        raise ValueError("UCI API export columns do not match the official zip")
    if len(exported) != len(official):
        raise ValueError("UCI API export row count does not match the official zip")
    for column in official.columns:
        if column == "holiday":
            continue
        if not official[column].equals(exported[column]):
            raise ValueError(f"API export differs on {column}")
    holiday_match = normalize_holiday(official["holiday"]).equals(
        normalize_holiday(exported["holiday"])
    )
    if not holiday_match:
        raise ValueError(
            "API holiday column does not match the official zip after token normalization"
        )
    if int(api["num_instances"]) != len(official) or int(api["uci_id"]) != DATASET_ID:
        raise ValueError("UCI API metadata does not match the official zip")
    if api.get("dataset_doi") != DATASET_DOI:
        raise ValueError("UCI API DOI does not match the dataset page")
    return {
        "api_matches_zip": True,
        "api_license_field": api.get("license"),
        "api_has_missing_values": api.get("has_missing_values"),
        "api_num_instances": int(api["num_instances"]),
        "intro_paper": api.get("intro_paper"),
    }


def screen_invalid(frame: pd.DataFrame) -> dict[str, int]:
    temp_bad = frame["temp"] < TEMP_INVALID_BELOW
    rain_bad = frame["rain_1h"] > RAIN_INVALID_ABOVE
    counts = {
        "temp_rows_below_200": int(temp_bad.sum()),
        "temp_hours_below_200": int(frame.loc[temp_bad, "date_time"].nunique()),
        "rain_rows_above_100": int(rain_bad.sum()),
        "rain_hours_above_100": int(frame.loc[rain_bad, "date_time"].nunique()),
    }
    frame.loc[temp_bad, "temp"] = np.nan
    frame.loc[rain_bad, "rain_1h"] = np.nan
    return counts


def spread_audit(frame: pd.DataFrame) -> dict[str, float | int]:
    grouped = frame.groupby("date_time", sort=False)
    audit: dict[str, float | int] = {}
    for column, label in (
        ("temp", "temp"),
        ("rain_1h", "rain"),
        ("snow_1h", "snow"),
        ("clouds_all", "clouds"),
    ):
        spread = grouped[column].max() - grouped[column].min()
        audit[f"timestamps_{label}_disagrees"] = int((spread > 0).sum())
        audit[f"max_{label}_spread"] = float(spread.max())
    volume_levels = grouped["traffic_volume"].nunique()
    if int((volume_levels > 1).sum()) != 0:
        raise ValueError("traffic volume disagrees within a timestamp")
    return audit


def collapse(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    grouped = frame.groupby("date_time", sort=True)
    collapsed = grouped.agg(
        traffic_volume=("traffic_volume", "first"),
        temp=("temp", "mean"),
        rain_1h=("rain_1h", "mean"),
        snow_1h=("snow_1h", "mean"),
        clouds_all=("clouds_all", "mean"),
        n_rows=("traffic_volume", "size"),
        n_descriptions=("weather_description", "nunique"),
    )
    named = frame.loc[frame["holiday"].notna(), ["date_time", "holiday"]].drop_duplicates()
    if named["date_time"].duplicated().any():
        raise ValueError("holiday name disagrees within a timestamp")
    holiday = named.set_index("date_time")["holiday"]
    collapsed["holiday"] = holiday.reindex(collapsed.index)
    audit = {
        "max_descriptions_in_one_hour": int(collapsed["n_descriptions"].max()),
        "max_rows_in_one_hour": int(collapsed["n_rows"].max()),
        "hours_temp_missing_after_screen": int(collapsed["temp"].isna().sum()),
        "hours_rain_missing_after_screen": int(collapsed["rain_1h"].isna().sum()),
    }
    return collapsed, audit


def maximum_gap(index: pd.DatetimeIndex) -> dict[str, float | str]:
    delta = index.to_series().diff().dt.total_seconds().to_numpy() / 3600.0
    position = int(np.nanargmax(delta))
    return {
        "gap_hours": float(delta[position]),
        "gap_start": index[position - 1].isoformat(sep=" "),
        "gap_end": index[position].isoformat(sep=" "),
        "gap_end_position": position,
    }


def build_frame(collapsed: pd.DataFrame) -> pd.DataFrame:
    frame = collapsed.copy()
    index = pd.DatetimeIndex(frame.index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("collapsed hours are not a sorted unique timeline")
    volume = frame["traffic_volume"].to_numpy(dtype=float)
    frame["hour"] = index.hour.astype(int)
    frame["dow"] = index.dayofweek.astype(int)
    frame["month"] = index.month.astype(int)
    frame["holiday_date"] = holiday_date_flag(index, frame["holiday"].to_numpy())
    frame["hour_sin"], frame["hour_cos"] = hour_cycle(frame["hour"].to_numpy())
    frame["hour_sin2"], frame["hour_cos2"] = two_hour_cycle(frame["hour"].to_numpy())
    frame["dow_sin"], frame["dow_cos"] = dow_cycle(frame["dow"].to_numpy())
    frame["month_sin"], frame["month_cos"] = month_cycle(frame["month"].to_numpy())
    for hours in LAGS:
        lagged = time_lag(index, volume, hours)
        frame[f"lag_{hours}"] = lagged
        frame[f"lag_{hours}_missing"] = missing_indicator(lagged)
    roll_mean, roll_count = past_rolling(index, volume, ROLL_WINDOW)
    frame["roll_mean_24"] = roll_mean
    frame["roll_count_24"] = roll_count
    frame["roll_mean_missing"] = missing_indicator(roll_mean)
    leaky, leaky_count = clock_window(index, volume, ROLL_WINDOW, include_current=True)
    frame["leaky_roll_mean"] = leaky
    frame["leaky_roll_count"] = leaky_count
    frame["leaky_roll_missing"] = missing_indicator(leaky)
    for hours, name in ((1, "row_lag_1"), (24, "row_lag_24"), (168, "row_lag_168")):
        shifted = row_shift(volume, hours)
        frame[name] = shifted
        frame[f"{name}_missing"] = missing_indicator(shifted)
    row_mean, row_count = row_past_rolling(volume, ROLL_WINDOW)
    frame["row_roll_mean"] = row_mean
    frame["row_roll_count"] = row_count
    frame["row_roll_missing"] = missing_indicator(row_mean)
    for source, target in (
        ("temp", "temp_lag1"),
        ("rain_1h", "rain_lag1"),
        ("snow_1h", "snow_lag1"),
        ("clouds_all", "clouds_lag1"),
    ):
        lagged = time_lag(index, frame[source].to_numpy(dtype=float), 1)
        frame[target] = lagged
        frame[f"{target}_missing"] = missing_indicator(lagged)
    for source, target in (
        ("temp", "temp_missing"),
        ("rain_1h", "rain_missing"),
        ("snow_1h", "snow_missing"),
        ("clouds_all", "clouds_missing"),
    ):
        frame[target] = missing_indicator(frame[source].to_numpy(dtype=float))
    return frame


def profile(frame: pd.DataFrame, idx: np.ndarray, key: str) -> list[dict[str, float | int]]:
    rows = []
    subset = frame.iloc[idx]
    for value, group in subset.groupby(key, sort=True):
        rows.append(
            {
                key: int(value),
                "n": int(len(group)),
                "mean_volume": float(group["traffic_volume"].mean()),
            }
        )
    return rows


def residual_by_hour(
    frame: pd.DataFrame, idx: np.ndarray, pred: np.ndarray
) -> list[dict[str, float | int]]:
    subset = frame.iloc[idx][["hour", "traffic_volume"]].copy()
    subset["pred"] = pred
    subset["residual"] = subset["traffic_volume"] - subset["pred"]
    rows = []
    for hour, group in subset.groupby("hour", sort=True):
        rows.append(
            {
                "hour": int(hour),
                "n": int(len(group)),
                "mean_actual": float(group["traffic_volume"].mean()),
                "mean_predicted": float(group["pred"].mean()),
                "mean_residual": float(group["residual"].mean()),
            }
        )
    return rows


def load_frame() -> tuple[pd.DataFrame, dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    download(API_URL, DATA / "uci_api.json")
    official = read_official()
    api = json.loads((DATA / "uci_api.json").read_text(encoding="utf-8"))["data"]
    match = assert_api_match(official, api)
    required = {
        "holiday",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "weather_description",
        "date_time",
        "traffic_volume",
    }
    if set(official.columns) != required:
        raise ValueError(f"unexpected columns: {list(official.columns)}")
    none_count = int((official["holiday"] == "None").sum())
    if none_count == 0:
        raise ValueError(
            "official holiday column has no None token; the parser may have dropped it"
        )
    named = official.loc[~official["holiday"].isin(["None", ""]), "holiday"]
    official["date_time"] = pd.to_datetime(official["date_time"])
    if not official["date_time"].is_monotonic_increasing:
        raise ValueError("official file is not sorted by date_time")
    for column in ("temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"):
        official[column] = pd.to_numeric(official[column], errors="raise")
    raw_spreads = spread_audit(official)
    screened = screen_invalid(official)
    official["holiday"] = normalize_holiday(official["holiday"])
    collapsed, collapse_audit = collapse(official)
    frame = build_frame(collapsed)
    if not set(frame["hour"]).issuperset(range(24)):
        raise ValueError("an hour of day is missing from the collapsed file")
    audit = {
        "raw_rows": int(len(official)),
        "unique_timestamps": int(len(frame)),
        "duplicate_timestamp_rows": int(len(official) - len(frame)),
        "holiday_none_token": none_count,
        "holiday_named_rows": int(len(named)),
        "holiday_named_counts": {
            str(key): int(value) for key, value in named.value_counts().items()
        },
        "holiday_stamp_hours": {
            str(int(hour)): int(count)
            for hour, count in official.loc[official["holiday"].notna(), "date_time"]
            .dt.hour.value_counts()
            .items()
        },
        "start": pd.DatetimeIndex(frame.index).min().isoformat(sep=" "),
        "end": pd.DatetimeIndex(frame.index).max().isoformat(sep=" "),
        "volume_min": float(frame["traffic_volume"].min()),
        "volume_max": float(frame["traffic_volume"].max()),
        "volume_mean": float(frame["traffic_volume"].mean()),
        "volume_median": float(frame["traffic_volume"].median()),
        **raw_spreads,
        **screened,
        **collapse_audit,
        **match,
    }
    return frame, audit


def gap_table(index: pd.DatetimeIndex) -> dict[str, int | list[str]]:
    delta = index.to_series().diff().dt.total_seconds().to_numpy() / 3600.0
    finite = np.isfinite(delta)
    two_hour = finite & (delta == 2.0)
    months = np.asarray(index.month)
    march = two_hour & (months == 3)
    november = two_hour & (months == 11)
    return {
        "gaps_gt_1": int(np.sum(finite & (delta > 1))),
        "gaps_ge_24": int(np.sum(finite & (delta >= 24))),
        "gaps_eq_2": int(two_hour.sum()),
        "gaps_eq_2_in_march": int(march.sum()),
        "gaps_eq_2_in_november": int(november.sum()),
        "two_hour_gap_ends_march": [stamp.isoformat(sep=" ") for stamp in index[march]],
        "two_hour_gap_ends_november": [stamp.isoformat(sep=" ") for stamp in index[november]],
    }


def main() -> None:
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame, audit = load_frame()
    index = pd.DatetimeIndex(frame.index)
    full_index = pd.date_range(index.min(), index.max(), freq="h")
    audit["expected_hours"] = int(len(full_index))
    audit["missing_hours"] = int(len(full_index) - len(index))
    audit.update(gap_table(index))
    gap = maximum_gap(index)
    audit.update(gap)
    splits = make_splits(len(frame))
    if index[splits["holdout"][0]] < pd.Timestamp(str(gap["gap_end"])):
        raise ValueError("holdout starts inside or before the maximum gap")
    specs = experiment_specs()
    if tuple(spec["name"] for spec in specs) != EXPERIMENT_ORDER:
        raise ValueError("experiment order drifted from the registered list")
    y = frame["traffic_volume"].to_numpy(dtype=float)
    train_idx = splits["train"]
    holdout_idx = splits["holdout"]
    post_gap_idx = train_idx[index[train_idx] >= pd.Timestamp(str(gap["gap_end"]))]
    if len(post_gap_idx) < N_SPLITS + 1:
        raise ValueError("post-gap training block is too small")

    cv_rows: list[dict] = []
    for spec in specs:
        if not spec["eligible"]:
            continue
        print(f"cv {spec['name']}", flush=True)
        cv_mean, cv_std, folds = fold_rmse(frame, train_idx, spec)
        width = (
            int(spec["n_features"])
            if spec["kind"] != "ridge"
            else design(
                frame,
                train_idx,
                train_idx,
                tuple(spec["columns"]),
                tuple(spec["categorical"]),
            )[0].shape[1]
        )
        cv_rows.append(
            {
                "name": spec["name"],
                "eligible": True,
                "cv_rmse": cv_mean,
                "cv_std": cv_std,
                "cv_folds": folds,
                "n_features": int(width),
            }
        )
        print(f"  cv_rmse={cv_mean:.4f}", flush=True)
    reported = select_model(cv_rows)
    if reported in LEAKS or reported in CONTROLS:
        raise ValueError("selection returned an ineligible model")
    print(f"reported {reported}", flush=True)

    predictions: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    cv_by_name = {row["name"]: row for row in cv_rows}
    for spec in specs:
        print(f"running {spec['name']}", flush=True)
        fit_idx = post_gap_idx if spec.get("train") == "post_gap" else train_idx
        if spec["kind"] == "rate":
            pred = predict_rate(frame, fit_idx, holdout_idx)
            width = 0
        elif spec["kind"] == "naive":
            pred = predict_naive(frame, fit_idx, holdout_idx)
            width = 1
        else:
            pred, width = predict_ridge(frame, fit_idx, holdout_idx, spec)
        predictions[spec["name"]] = pred
        cv_row = cv_by_name.get(spec["name"])
        results[spec["name"]] = {
            "eligible": bool(spec["eligible"]),
            "kind": spec["kind"],
            "train_rows": "post_gap" if spec.get("train") == "post_gap" else "before_cut",
            "n_features": int(width),
            "cv_rmse": None if cv_row is None else cv_row["cv_rmse"],
            "cv_std": None if cv_row is None else cv_row["cv_std"],
            "cv_folds": None if cv_row is None else cv_row["cv_folds"],
            "test": score_prediction(y[holdout_idx], pred),
        }
        print(f"  rmse={results[spec['name']]['test']['rmse']:.4f}", flush=True)

    both_present = np.isfinite(frame["lag_1"].to_numpy()) & np.isfinite(
        frame["row_lag_1"].to_numpy()
    )
    lag_disagree = both_present & ~np.isclose(
        frame["lag_1"].to_numpy(dtype=float),
        frame["row_lag_1"].to_numpy(dtype=float),
    )
    lag_missing_row_present = np.isnan(frame["lag_1"].to_numpy()) & np.isfinite(
        frame["row_lag_1"].to_numpy()
    )
    clusters = index[holdout_idx].strftime("%Y-%m-%d")
    pairs = (
        ("training_mean", reported),
        ("seasonal_naive_24", reported),
        ("one_harmonic", "two_harmonic"),
        ("raw_calendar", "one_hot_hour"),
        ("two_harmonic_plus_lags", "leaky_rolling"),
        ("two_harmonic_plus_lags", "same_hour_weather"),
        ("two_harmonic_plus_lags", "row_shift_lags"),
        ("two_harmonic_plus_lags", "post_gap_refit"),
    )
    bootstrap = {}
    for reference, candidate in pairs:
        if reference == candidate:
            continue
        bootstrap[f"{reference}_minus_{candidate}"] = cluster_bootstrap_gap(
            y[holdout_idx],
            predictions[reference],
            predictions[candidate],
            clusters,
        )
    hour_rows = profile(frame, train_idx, "hour")
    weekday_rows = profile(frame, train_idx, "dow")
    residual_rows = residual_by_hour(frame, holdout_idx, predictions[reported])
    plot_hour_profile(RESULTS / "hour_profile.png", hour_rows)
    plot_rmse(RESULTS / "rmse.png", results, reported)
    week = holdout_idx[:WEEK_PLOT_HOURS]
    week_curves = {"reported": predictions[reported][:WEEK_PLOT_HOURS]}
    if reported != "seasonal_naive_24":
        week_curves["seasonal_naive_24"] = predictions["seasonal_naive_24"][:WEEK_PLOT_HOURS]
    else:
        week_curves["training_mean"] = predictions["training_mean"][:WEEK_PLOT_HOURS]
    plot_week(RESULTS / "holdout_week.png", index[week], y[week], week_curves, reported)
    plot_residual(RESULTS / "residual_by_hour.png", residual_rows)

    payload = {
        "dataset": {
            "name": "Metro Interstate Traffic Volume",
            "uci_id": DATASET_ID,
            "doi": DATASET_DOI,
            "creator": "John Hogue",
            "page_license": "CC BY 4.0",
            "time_description": "Hour of the data collected in local CST time",
            "target": "traffic_volume",
            "target_description": "Hourly I-94 ATR 301 reported westbound traffic volume",
        },
        "audit": audit,
        "protocol": {
            "train_end_fraction": TRAIN_END_FRACTION,
            "seed": SEED,
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_solver": "cholesky",
            "n_splits": N_SPLITS,
            "roll_window": ROLL_WINDOW,
            "lags": list(LAGS),
            "temp_invalid_below": TEMP_INVALID_BELOW,
            "rain_invalid_above": RAIN_INVALID_ABOVE,
            "week_plot_hours": WEEK_PLOT_HOURS,
            "reported_model": reported,
            "selection_rule": "lowest mean time-series CV RMSE, then fewer features, then name",
            "selection_uses_test_scores": False,
            "ineligible": sorted(LEAKS | CONTROLS),
            "dropped_before_fit": ["weather_main", "weather_description"],
            "holiday_feature": "date is flagged when any hour on that date has a holiday name",
            "dow_encoding": "pandas Monday=0 through Sunday=6",
            "month_cycle": "month % 12, so December is adjacent to January",
        },
        "split": {
            "train_end": int(len(train_idx)),
            "train_start": index[train_idx[0]].isoformat(sep=" "),
            "train_last": index[train_idx[-1]].isoformat(sep=" "),
            "holdout_start": index[holdout_idx[0]].isoformat(sep=" "),
            "holdout_end": index[holdout_idx[-1]].isoformat(sep=" "),
            "n_train": int(len(train_idx)),
            "n_holdout": int(len(holdout_idx)),
            "n_post_gap_train": int(len(post_gap_idx)),
            "holdout_lag_1_missing": int(np.isnan(frame.iloc[holdout_idx]["lag_1"]).sum()),
            "holdout_lag_24_missing": int(np.isnan(frame.iloc[holdout_idx]["lag_24"]).sum()),
            "holdout_lag_168_missing": int(np.isnan(frame.iloc[holdout_idx]["lag_168"]).sum()),
            "train_row_shift_differs": int(lag_disagree[train_idx].sum()),
            "holdout_row_shift_differs": int(lag_disagree[holdout_idx].sum()),
            "train_lag_1_missing_but_row_present": int(lag_missing_row_present[train_idx].sum()),
            "holdout_lag_1_missing_but_row_present": int(
                lag_missing_row_present[holdout_idx].sum()
            ),
            "n_volume_zero": int((frame["traffic_volume"] == 0).sum()),
            "week_plot_start": index[week[0]].isoformat(sep=" "),
            "week_plot_end": index[week[-1]].isoformat(sep=" "),
        },
        "hour_profile_train": hour_rows,
        "weekday_profile_train": weekday_rows,
        "residual_by_hour": residual_rows,
        "experiments": results,
        "bootstrap": bootstrap,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(py(payload), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {RESULTS / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
