"""Clean a real census extract, then measure how the cleanup changes the score.

The UCI Adult files are used as published. Question marks, padded fields, the
test-file label period, duplicate rows, and the repeated capital-gain maximum
are counted from those files. They are not simulated.

A second split injects MCAR, MAR, and MNAR masks into complete rows so the
original values are known. Imputers are fit on the training rows only. The
holdout is scored after the methods are chosen. It is not used to pick a method.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.2
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000
LOG_LOSS_CLIP = 1e-12
MCAR_RATE = 0.25
MAR_AGE_CUTOFF = 50.0
MAR_HIGH_RATE = 0.60
MAR_LOW_RATE = 0.05
MNAR_POSITIVE_RATE = 0.80
MNAR_ZERO_RATE = 0.05
MASK_SEED = 42
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 7
IQR_MULTIPLIER = 1.5

TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
DATASET_DOI = "10.24432/C5XW20"
REPOSITORY_URL = "https://archive.ics.uci.edu/dataset/2/adult"
LICENSE_URL = "https://archive.ics.uci.edu/contribute/donation"

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
MODEL_NUMERIC = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
MODEL_CATEGORICAL = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "native_country",
]
INJECTION_COLUMNS = [
    "age",
    "education_num",
    "hours_per_week",
    "capital_gain",
    "sex_male",
]
HOURS_INDEX = INJECTION_COLUMNS.index("hours_per_week")
CAPITAL_INDEX = INJECTION_COLUMNS.index("capital_gain")
INCOME_LABELS = {">50K", "<=50K"}

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "ml-mastery-lab/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def normalize_income(value: str) -> str:
    """Strip padding and one donor-added trailing period from an income label."""
    text = value.strip()
    if text.endswith("."):
        text = text[:-1]
    if text not in INCOME_LABELS:
        raise ValueError(f"unexpected income label: {value!r}")
    return text


def iqr_fences(values: np.ndarray, multiplier: float = IQR_MULTIPLIER) -> dict[str, float]:
    """Tukey fences from the 25th and 75th percentiles. A zero-inflated column collapses."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    q1, q3 = np.percentile(clean, [25, 75])
    span = float(q3 - q1)
    return {
        "q1": float(q1),
        "q3": float(q3),
        "span": span,
        "low": float(q1 - multiplier * span),
        "high": float(q3 + multiplier * span),
    }


def mcar_mask(n: int, rate: float, seed: int) -> np.ndarray:
    """Missingness that ignores the row. The values are not an argument on purpose."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(rate)
    return np.random.default_rng(seed).random(n) < rate


def mar_mask(age: np.ndarray, seed: int) -> np.ndarray:
    """Missingness that depends on observed age, not on the field being hidden."""
    rate = np.where(np.asarray(age, dtype=float) >= MAR_AGE_CUTOFF, MAR_HIGH_RATE, MAR_LOW_RATE)
    return np.random.default_rng(seed).random(len(rate)) < rate


def mnar_mask(values: np.ndarray, seed: int) -> np.ndarray:
    """Missingness that depends on the value being hidden. Zero and positive differ."""
    rate = np.where(
        np.asarray(values, dtype=float) > 0.0,
        MNAR_POSITIVE_RATE,
        MNAR_ZERO_RATE,
    )
    return np.random.default_rng(seed).random(len(rate)) < rate


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


def read_adult_lines(path: Path, skip_preamble: bool) -> tuple[list[str], str | None]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preamble = None
    if skip_preamble:
        preamble = lines[0]
        if "Cross validator" not in preamble:
            raise ValueError(f"unexpected test preamble: {preamble!r}")
        lines = lines[1:]
    return lines, preamble


def frames_from_lines(lines: list[str], source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for line_number, line in enumerate(lines, start=1):
        parts = line.split(",")
        if len(parts) != len(COLUMNS):
            raise ValueError(f"{source} line {line_number} has {len(parts)} fields")
        rows.append(parts)
    raw = pd.DataFrame(rows, columns=COLUMNS)
    stripped = raw.map(lambda value: value.strip())
    return raw, stripped


def leading_space_count(raw: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for column in COLUMNS:
        counts[column] = int(raw[column].str.startswith(" ").sum())
    return counts


def income_inventory(raw_income: pd.Series) -> dict[str, int | list[str]]:
    stripped = raw_income.str.strip()
    normalized = stripped.map(normalize_income)
    unexpected = sorted(set(stripped.str.replace(r"\.$", "", regex=True)) - INCOME_LABELS)
    return {
        "n": int(len(raw_income)),
        "n_with_leading_space": int(raw_income.str.startswith(" ").sum()),
        "n_with_trailing_period_after_strip": int(stripped.str.endswith(".").sum()),
        "n_positive_normalized": int((normalized == ">50K").sum()),
        "unexpected_labels": unexpected,
    }


def to_model_frame(stripped: pd.DataFrame) -> pd.DataFrame:
    frame = stripped.copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["income"] = frame["income"].map(normalize_income)
    for column in COLUMNS:
        if column in NUMERIC_COLUMNS or column == "income":
            continue
        frame[column] = frame[column].replace("?", np.nan)
    return frame


def question_mark_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        column: int(frame[column].isna().sum())
        for column in ["workclass", "occupation", "native_country", *NUMERIC_COLUMNS]
    }


def duplicate_inventory(frame: pd.DataFrame) -> dict[str, int]:
    feature_columns = [column for column in COLUMNS if column != "income"]
    grouped = frame.groupby(feature_columns, dropna=False)["income"].nunique()
    return {
        "n_exact_duplicate_rows": int(frame.duplicated().sum()),
        "n_feature_duplicate_rows": int(frame.duplicated(feature_columns).sum()),
        "n_feature_keys_with_conflicting_income": int((grouped > 1).sum()),
    }


def category_alignment(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, list[str] | int]:
    train_only = []
    test_only = []
    for column in MODEL_CATEGORICAL:
        train_levels = set(train[column].dropna().unique())
        test_levels = set(test[column].dropna().unique())
        train_only.extend(f"{column}={level}" for level in sorted(train_levels - test_levels))
        test_only.extend(f"{column}={level}" for level in sorted(test_levels - train_levels))
    return {
        "n_train_only": len(train_only),
        "n_test_only": len(test_only),
        "train_only": train_only,
        "test_only": test_only,
    }


def education_is_one_to_one(frame: pd.DataFrame) -> dict[str, int]:
    mapping = frame.groupby("education", dropna=False)["education_num"].nunique()
    reverse = frame.groupby("education_num", dropna=False)["education"].nunique()
    return {
        "max_education_num_per_education": int(mapping.max()),
        "max_education_per_education_num": int(reverse.max()),
    }


def repeated_maximum(values: pd.Series) -> dict[str, float | int]:
    maximum = float(values.max())
    return {
        "maximum": maximum,
        "n_at_maximum": int((values == maximum).sum()),
        "n_positive": int((values > 0).sum()),
        "n_zero": int((values == 0).sum()),
    }


def pointwise_log_loss(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(proba, dtype=float), LOG_LOSS_CLIP, 1.0 - LOG_LOSS_CLIP)
    y_true = np.asarray(y_true, dtype=float)
    return -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))


def classification_scores(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    losses = pointwise_log_loss(y_true, proba)
    return {
        "log_loss": float(np.mean(losses)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy_at_0_5": float(np.mean((np.asarray(proba) >= 0.5) == np.asarray(y_true))),
    }


def constant_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float | int | None]:
    rate = float(np.mean(y_train))
    proba = np.full(len(y_test), rate)
    majority = int(rate >= 0.5)
    return {
        "train_positive_rate": rate,
        "test_positive_rate": float(np.mean(y_test)),
        "n_train": int(len(y_train)),
        "n_scored": int(len(y_test)),
        "log_loss": float(np.mean(pointwise_log_loss(y_test, proba))),
        "majority_class": majority,
        "majority_class_accuracy": float(np.mean(np.full(len(y_test), majority) == y_test)),
        "roc_auc": None,
    }


def make_logistic() -> LogisticRegression:
    return LogisticRegression(
        C=LOGISTIC_C,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=SEED,
    )


def fit_matrix(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, int]:
    scaler = StandardScaler()
    model = make_logistic()
    model.fit(scaler.fit_transform(x_train), y_train)
    proba = model.predict_proba(scaler.transform(x_test))[:, 1]
    n_iter = int(np.max(model.n_iter_))
    if n_iter >= LOGISTIC_MAX_ITER:
        raise RuntimeError(f"logistic regression hit {LOGISTIC_MAX_ITER} iterations")
    return proba, n_iter


def transformer_for(numeric: list[str], categorical: list[str], method: str) -> ColumnTransformer:
    if method == "mean_mode":
        numeric_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        return ColumnTransformer(
            [
                ("numeric", numeric_pipe, numeric),
                ("categorical", categorical_pipe, categorical),
            ]
        )
    if method == "indicator":
        numeric_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        return ColumnTransformer(
            [
                ("numeric", numeric_pipe, numeric),
                (
                    "numeric_missing",
                    MissingIndicator(features="missing-only"),
                    numeric,
                ),
                ("categorical", categorical_pipe, categorical),
                (
                    "categorical_missing",
                    MissingIndicator(features="missing-only"),
                    categorical,
                ),
            ]
        )
    if method == "level":
        return ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                ),
            ]
        )
    raise ValueError(method)


def score_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    method: str,
) -> dict[str, float | int]:
    model = Pipeline(
        [
            ("columns", transformer_for(numeric, categorical, method)),
            ("logistic", make_logistic()),
        ]
    )
    model.fit(train[numeric + categorical], y_train)
    proba = model.predict_proba(test[numeric + categorical])[:, 1]
    n_iter = int(np.max(model.named_steps["logistic"].n_iter_))
    if n_iter >= LOGISTIC_MAX_ITER:
        raise RuntimeError(f"{method} logistic regression hit {LOGISTIC_MAX_ITER} iterations")
    scores = classification_scores(y_test, proba)
    scores["n_train"] = int(len(train))
    scores["n_scored"] = int(len(test))
    scores["n_iter"] = n_iter
    scores["proba"] = proba
    return scores


def without_proba(scores: dict) -> dict:
    return {key: value for key, value in scores.items() if key != "proba"}


def bootstrap_diff(loss_a: np.ndarray, loss_b: np.ndarray, seed: int) -> dict[str, float]:
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


def mask_copy(matrix: np.ndarray, column: int, mask: np.ndarray) -> np.ndarray:
    copied = matrix.copy()
    copied[mask, column] = np.nan
    return copied


def fit_imputer(method: str) -> SimpleImputer | IterativeImputer:
    if method == "mean":
        return SimpleImputer(strategy="mean")
    if method == "zero":
        return SimpleImputer(strategy="constant", fill_value=0.0)
    if method == "indicator":
        return SimpleImputer(strategy="mean", add_indicator=True)
    if method == "iterative":
        return IterativeImputer(max_iter=25, random_state=SEED)
    raise ValueError(method)


def reconstruction(
    true_values: np.ndarray, imputed: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    truth = true_values[mask]
    filled = imputed[mask]
    absolute = np.abs(filled - truth)
    return {
        "n_masked": int(mask.sum()),
        "mae": float(np.mean(absolute)),
        "median_ae": float(np.median(absolute)),
        "true_masked_mean": float(np.mean(truth)),
        "imputed_masked_mean": float(np.mean(filled)),
    }


def plot_missingness(path: Path, counts: dict[str, int], co_missing: int) -> None:
    labels = list(counts)
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#3174a1")
    ax.set_ylabel("Rows where the stripped field is ?")
    ax.set_xlabel("Column")
    ax.set_title(f"Real missing tokens. Workclass and occupation are both ? in {co_missing} rows.")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_capital_gain(
    path: Path, values: np.ndarray, fences: dict[str, float], maximum: float
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    groups = {
        "zero": int(np.sum(values == 0)),
        "positive below cap": int(np.sum((values > 0) & (values < maximum))),
        "at the maximum": int(np.sum(values == maximum)),
    }
    bars = axes[0].bar(list(groups), list(groups.values()), color=["#8c8c8c", "#3174a1", "#c23b22"])
    for bar, count in zip(bars, groups.values(), strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_ylabel("Training rows")
    axes[0].set_title("Capital gain is mostly zero")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_ylim(0, max(groups.values()) * 1.12)
    positive = values[(values > 0) & (values < maximum)]
    axes[1].hist(positive, bins=30, color="#3174a1")
    if len(positive) and fences["low"] <= positive.max() and fences["high"] >= positive.min():
        axes[1].axvline(fences["high"], color="#c23b22", linestyle="--", label="IQR high fence")
        axes[1].legend()
    else:
        axes[1].text(
            0.02,
            0.95,
            "IQR high fence is left of this panel",
            transform=axes[1].transAxes,
            va="top",
        )
    axes[1].set_xlabel("Capital gain, excluding zero and the maximum")
    axes[1].set_ylabel("Rows")
    axes[1].set_title("Positive gains below the repeated maximum")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_mae(path: Path, rows: list[dict]) -> None:
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    panels = (
        (["mcar_hours", "mar_hours"], "Hours per week"),
        (["mnar_capital_gain"], "Capital gain"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    for ax, (mechanisms, title) in zip(axes, panels, strict=True):
        x = np.arange(len(mechanisms))
        width = 0.18
        for index, method in enumerate(methods):
            heights = [
                next(
                    row["mae"]
                    for row in rows
                    if row["mechanism"] == mechanism and row["method"] == method
                )
                for mechanism in mechanisms
            ]
            ax.bar(x + (index - 1.5) * width, heights, width=width, label=method)
        ax.set_xticks(x, mechanisms, rotation=15, ha="right")
        ax.set_ylabel("MAE on masked holdout cells")
        ax.set_title(title)
    axes[0].legend()
    fig.suptitle("Imputation error where the original value is known")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_mnar(path: Path, truth: np.ndarray, mean_fill: float, zero_fill: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(np.log10(np.asarray(truth, dtype=float) + 1.0), bins=30, color="#3174a1")
    ax.axvline(np.log10(mean_fill + 1.0), color="#e39c2b", linestyle="--", label="mean fill")
    ax.axvline(np.log10(zero_fill + 1.0), color="#c23b22", linestyle="--", label="zero fill")
    ax.set_xlabel("log10(true capital gain + 1) on MNAR-masked holdout cells")
    ax.set_ylabel("Cells")
    ax.set_title("Log scale. Both constant fills stay at the left.")
    ax.legend()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_official(path: Path, runs: list[dict]) -> None:
    ordered = sorted(runs, key=lambda row: row["log_loss"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.barh([row["name"] for row in ordered], [row["log_loss"] for row in ordered], color="#3174a1")
    ax.set_xlabel("Log loss on the published test file")
    ax.set_title("Published test file. Bars are training choices or encodings.")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_injection_excess(path: Path, models: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    panels = (
        (axes[0], "narrow", "Narrow: damaged column only"),
        (axes[1], "wide", "Wide: the other columns stay visible"),
    )
    oracle = {
        (row["mechanism"], row["width"]): row["log_loss"]
        for row in models
        if row["method"] == "oracle"
    }
    for ax, width, title in panels:
        rows = [row for row in models if row["width"] == width and row["method"] != "oracle"]
        mechanisms = []
        methods = []
        for row in rows:
            if row["mechanism"] not in mechanisms:
                mechanisms.append(row["mechanism"])
            if row["method"] not in methods:
                methods.append(row["method"])
        x = np.arange(len(mechanisms))
        bar_width = 0.18
        for index, method in enumerate(methods):
            heights = []
            for mechanism in mechanisms:
                loss = next(
                    row["log_loss"]
                    for row in rows
                    if row["mechanism"] == mechanism and row["method"] == method
                )
                heights.append(loss - oracle[(mechanism, width)])
            ax.bar(
                x + (index - (len(methods) - 1) / 2) * bar_width,
                heights,
                width=bar_width,
                label=method,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, mechanisms, rotation=15, ha="right")
        ax.set_ylabel("Holdout log loss minus oracle")
        ax.set_title(title)
    axes[0].legend()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_sources() -> dict:
    download(TRAIN_URL, DATA / "adult.data")
    download(TEST_URL, DATA / "adult.test")
    train_lines, _ = read_adult_lines(DATA / "adult.data", skip_preamble=False)
    test_lines, preamble = read_adult_lines(DATA / "adult.test", skip_preamble=True)
    train_raw, train_stripped = frames_from_lines(train_lines, "adult.data")
    test_raw, test_stripped = frames_from_lines(test_lines, "adult.test")
    train = to_model_frame(train_stripped)
    test = to_model_frame(test_stripped)
    if train[NUMERIC_COLUMNS].isna().any().any() or test[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError("a numeric field was not numeric after stripping")
    return {
        "preamble": preamble,
        "train_raw": train_raw,
        "test_raw": test_raw,
        "train_stripped": train_stripped,
        "test_stripped": test_stripped,
        "train": train,
        "test": test,
    }


def official_experiments(sources: dict) -> dict:
    train = sources["train"]
    test = sources["test"]
    y_train = (train["income"] == ">50K").to_numpy(dtype=int)
    y_test = (test["income"] == ">50K").to_numpy(dtype=int)
    baseline = constant_baseline(y_train, y_test)
    train_labels = set(sources["train_raw"]["income"].unique())
    test_labels = set(sources["test_raw"]["income"].unique())
    label_overlap = sorted(train_labels & test_labels)

    runs = []
    stored_losses = {}

    def add_run(name: str, scores: dict) -> None:
        public = without_proba(scores)
        public["name"] = name
        runs.append(public)
        stored_losses[name] = pointwise_log_loss(y_test, scores["proba"])

    level_train = sources["train_stripped"].copy()
    level_test = sources["test_stripped"].copy()
    for frame in (level_train, level_test):
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column])
    add_run(
        "question_mark_as_level",
        score_frames(
            level_train,
            level_test,
            y_train,
            y_test,
            MODEL_NUMERIC,
            MODEL_CATEGORICAL,
            "level",
        ),
    )
    add_run(
        "mean_mode",
        score_frames(train, test, y_train, y_test, MODEL_NUMERIC, MODEL_CATEGORICAL, "mean_mode"),
    )
    add_run(
        "missing_indicator",
        score_frames(train, test, y_train, y_test, MODEL_NUMERIC, MODEL_CATEGORICAL, "indicator"),
    )

    complete_columns = ["workclass", "occupation", "native_country"]
    complete_train = train.loc[train[complete_columns].notna().all(axis=1)].copy()
    complete_scores = score_frames(
        complete_train,
        test,
        (complete_train["income"] == ">50K").to_numpy(dtype=int),
        y_test,
        MODEL_NUMERIC,
        MODEL_CATEGORICAL,
        "mean_mode",
    )
    add_run("complete_case_train", complete_scores)

    feature_columns = [column for column in COLUMNS if column != "income"]
    deduped = train.drop_duplicates(feature_columns, keep="first")
    add_run(
        "drop_feature_duplicates",
        score_frames(
            deduped,
            test,
            (deduped["income"] == ">50K").to_numpy(dtype=int),
            y_test,
            MODEL_NUMERIC,
            MODEL_CATEGORICAL,
            "mean_mode",
        ),
    )

    maximum = float(train["capital_gain"].max())
    capped = train.loc[train["capital_gain"] != maximum].copy()
    add_run(
        "drop_repeated_maximum_train",
        score_frames(
            capped,
            test,
            (capped["income"] == ">50K").to_numpy(dtype=int),
            y_test,
            MODEL_NUMERIC,
            MODEL_CATEGORICAL,
            "mean_mode",
        ),
    )

    fences = iqr_fences(train["capital_gain"].to_numpy(dtype=float))
    outside = (train["capital_gain"] < fences["low"]) | (train["capital_gain"] > fences["high"])
    iqr_train = train.loc[~outside].copy()
    add_run(
        "iqr_drop_capital_gain_train",
        score_frames(
            iqr_train,
            test,
            (iqr_train["income"] == ">50K").to_numpy(dtype=int),
            y_test,
            MODEL_NUMERIC,
            MODEL_CATEGORICAL,
            "mean_mode",
        ),
    )

    log_train = train.copy()
    log_test = test.copy()
    log_train["capital_gain"] = np.log1p(log_train["capital_gain"])
    log_test["capital_gain"] = np.log1p(log_test["capital_gain"])
    add_run(
        "log1p_capital_gain",
        score_frames(
            log_train,
            log_test,
            y_train,
            y_test,
            MODEL_NUMERIC,
            MODEL_CATEGORICAL,
            "mean_mode",
        ),
    )
    add_run(
        "with_fnlwgt",
        score_frames(
            train,
            test,
            y_train,
            y_test,
            [*MODEL_NUMERIC, "fnlwgt"],
            MODEL_CATEGORICAL,
            "mean_mode",
        ),
    )

    padded_test = test.copy()
    for column in MODEL_CATEGORICAL:
        padded_test[column] = sources["test_raw"][column].to_numpy()
    add_run(
        "train_stripped_test_padded",
        score_frames(
            train,
            padded_test,
            y_train,
            y_test,
            MODEL_NUMERIC,
            MODEL_CATEGORICAL,
            "mean_mode",
        ),
    )

    unknown_cells = 0
    categorical_cells = 0
    for column in MODEL_CATEGORICAL:
        train_levels = set(train[column].dropna().unique())
        raw_values = sources["test_raw"][column]
        categorical_cells += int(len(raw_values))
        unknown_cells += int((~raw_values.isin(train_levels)).sum())

    occupation_missing = train["occupation"].isna()
    comparisons = [
        bootstrap_diff(
            stored_losses["mean_mode"],
            stored_losses["missing_indicator"],
            BOOTSTRAP_SEED,
        ),
        bootstrap_diff(
            stored_losses["mean_mode"],
            stored_losses["question_mark_as_level"],
            BOOTSTRAP_SEED + 1,
        ),
        bootstrap_diff(
            stored_losses["train_stripped_test_padded"],
            stored_losses["mean_mode"],
            BOOTSTRAP_SEED + 2,
        ),
        bootstrap_diff(
            stored_losses["iqr_drop_capital_gain_train"],
            stored_losses["mean_mode"],
            BOOTSTRAP_SEED + 3,
        ),
    ]
    comparison_names = [
        "mean_mode_minus_missing_indicator",
        "mean_mode_minus_question_mark_as_level",
        "padded_test_minus_mean_mode",
        "iqr_drop_minus_mean_mode",
    ]
    for name, comparison in zip(comparison_names, comparisons, strict=True):
        comparison["name"] = name

    return {
        "baseline": baseline,
        "raw_label_overlap": label_overlap,
        "n_raw_label_overlap": len(label_overlap),
        "n_train_rows_dropped_by_iqr": int(outside.sum()),
        "n_train_rows_at_capital_gain_maximum": int((train["capital_gain"] == maximum).sum()),
        "n_padded_test_categorical_cells": categorical_cells,
        "n_padded_test_cells_unknown_to_stripped_train": unknown_cells,
        "occupation_missing_positive_rate": float(y_train[occupation_missing.to_numpy()].mean()),
        "occupation_observed_positive_rate": float(y_train[~occupation_missing.to_numpy()].mean()),
        "n_occupation_missing": int(occupation_missing.sum()),
        "runs": runs,
        "bootstrap": comparisons,
    }


def injection_matrix(frame: pd.DataFrame) -> np.ndarray:
    sex = frame["sex"].to_numpy()
    if set(pd.unique(sex)) - {"Male", "Female"}:
        raise ValueError(f"unexpected sex values: {sorted(set(pd.unique(sex)))}")
    return np.column_stack(
        [
            frame["age"].to_numpy(dtype=float),
            frame["education_num"].to_numpy(dtype=float),
            frame["hours_per_week"].to_numpy(dtype=float),
            frame["capital_gain"].to_numpy(dtype=float),
            (sex == "Male").astype(float),
        ]
    )


def run_injection(train: pd.DataFrame) -> tuple[dict, dict]:
    complete_columns = [
        "workclass",
        "occupation",
        "native_country",
        "age",
        "education_num",
        "hours_per_week",
        "capital_gain",
        "sex",
    ]
    complete = train.loc[train[complete_columns].notna().all(axis=1)].copy()
    y_all = (complete["income"] == ">50K").to_numpy(dtype=int)
    index = np.arange(len(complete))
    train_index, test_index = train_test_split(
        index,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_all,
    )
    matrix = injection_matrix(complete)
    x_train = matrix[train_index]
    x_test = matrix[test_index]
    y_train = y_all[train_index]
    y_test = y_all[test_index]
    mechanisms = {
        "mcar_hours": (
            HOURS_INDEX,
            mcar_mask(len(x_train), MCAR_RATE, MASK_SEED),
            mcar_mask(len(x_test), MCAR_RATE, MASK_SEED + 1),
        ),
        "mar_hours": (
            HOURS_INDEX,
            mar_mask(x_train[:, 0], MASK_SEED),
            mar_mask(x_test[:, 0], MASK_SEED + 1),
        ),
        "mnar_capital_gain": (
            CAPITAL_INDEX,
            mnar_mask(x_train[:, CAPITAL_INDEX], MASK_SEED),
            mnar_mask(x_test[:, CAPITAL_INDEX], MASK_SEED + 1),
        ),
    }
    reconstructions = []
    models = []
    losses = {}
    mnar_plot = None
    for mechanism, (column, train_mask, test_mask) in mechanisms.items():
        masked_train = mask_copy(x_train, column, train_mask)
        masked_test = mask_copy(x_test, column, test_mask)
        if mechanism == "mcar_hours":
            imputer = SimpleImputer(strategy="mean").fit(masked_train)
            train_mean = float(np.nanmean(masked_train[:, column]))
            if not np.allclose(imputer.statistics_[column], train_mean):
                raise RuntimeError("mean imputer did not match the training observed mean")
            leaked = SimpleImputer(strategy="mean").fit(np.vstack([masked_train, masked_test]))
            if np.allclose(leaked.statistics_[column], train_mean):
                raise RuntimeError(
                    "train and train-plus-holdout means matched; leakage check is vacuous"
                )
        fitted = {}
        for method in ("mean", "zero", "indicator", "iterative"):
            imputer = fit_imputer(method)
            fitted[method] = (
                np.asarray(imputer.fit_transform(masked_train), dtype=float),
                np.asarray(imputer.transform(masked_test), dtype=float),
            )
            recon = reconstruction(x_test[:, column], fitted[method][1][:, column], test_mask)
            recon.update(
                {
                    "mechanism": mechanism,
                    "method": method,
                    "train_missing_rate": float(np.mean(train_mask)),
                    "holdout_missing_rate": float(np.mean(test_mask)),
                }
            )
            reconstructions.append(recon)
        mnar_plot = _capture_mnar(mechanism, x_test, test_mask, column, fitted, mnar_plot)
        for width in ("narrow", "wide"):
            oracle_columns = [column] if width == "narrow" else list(range(x_train.shape[1]))
            proba, n_iter = fit_matrix(
                x_train[:, oracle_columns],
                y_train,
                x_test[:, oracle_columns],
            )
            _store_model(models, losses, y_test, proba, n_iter, mechanism, "oracle", width, y_train)
            for method, (filled_train, filled_test) in fitted.items():
                selected_train, selected_test = _select_width(
                    width, column, method, filled_train, filled_test
                )
                proba, n_iter = fit_matrix(selected_train, y_train, selected_test)
                _store_model(
                    models, losses, y_test, proba, n_iter, mechanism, method, width, y_train
                )
    contrasts = []
    for offset, (name, left, right) in enumerate(
        (
            (
                "mcar_narrow_mean_minus_oracle",
                ("mcar_hours", "mean", "narrow"),
                ("mcar_hours", "oracle", "narrow"),
            ),
            (
                "mar_narrow_mean_minus_oracle",
                ("mar_hours", "mean", "narrow"),
                ("mar_hours", "oracle", "narrow"),
            ),
            (
                "mnar_narrow_mean_minus_oracle",
                ("mnar_capital_gain", "mean", "narrow"),
                ("mnar_capital_gain", "oracle", "narrow"),
            ),
            (
                "mnar_narrow_zero_minus_mean",
                ("mnar_capital_gain", "zero", "narrow"),
                ("mnar_capital_gain", "mean", "narrow"),
            ),
            (
                "mnar_narrow_mean_minus_indicator",
                ("mnar_capital_gain", "mean", "narrow"),
                ("mnar_capital_gain", "indicator", "narrow"),
            ),
            (
                "mnar_wide_mean_minus_indicator",
                ("mnar_capital_gain", "mean", "wide"),
                ("mnar_capital_gain", "indicator", "wide"),
            ),
        )
    ):
        comparison = bootstrap_diff(losses[left], losses[right], BOOTSTRAP_SEED + offset)
        comparison["name"] = name
        contrasts.append(comparison)
    result = {
        "n_complete_rows": int(len(complete)),
        "n_train_rows_before_complete_case": int(len(train)),
        "complete_positive_rate": float(np.mean(y_all)),
        "full_train_positive_rate": float(np.mean(train["income"] == ">50K")),
        "n_injection_train": int(len(y_train)),
        "n_injection_holdout": int(len(y_test)),
        "baseline": constant_baseline(y_train, y_test),
        "reconstructions": reconstructions,
        "models": models,
        "bootstrap": contrasts,
    }
    return result, mnar_plot


def _capture_mnar(mechanism, x_test, test_mask, column, fitted, current):
    if mechanism != "mnar_capital_gain":
        return current
    return {
        "truth": x_test[test_mask, column],
        "mean_fill": float(np.mean(fitted["mean"][1][test_mask, column])),
        "zero_fill": float(np.mean(fitted["zero"][1][test_mask, column])),
    }


def _select_width(width, column, method, filled_train, filled_test):
    if width == "wide":
        return filled_train, filled_test
    selected_train = filled_train[:, [column]]
    selected_test = filled_test[:, [column]]
    if method == "indicator":
        selected_train = np.column_stack([selected_train, filled_train[:, -1]])
        selected_test = np.column_stack([selected_test, filled_test[:, -1]])
    return selected_train, selected_test


def _store_model(models, losses, y_test, proba, n_iter, mechanism, method, width, y_train):
    scores = classification_scores(y_test, proba)
    models.append(
        {
            "mechanism": mechanism,
            "method": method,
            "width": width,
            "n_train": int(len(y_train)),
            "n_scored": int(len(y_test)),
            "n_iter": n_iter,
            **scores,
        }
    )
    losses[(mechanism, method, width)] = pointwise_log_loss(y_test, proba)


def inventory_from(sources: dict) -> dict:
    train = sources["train"]
    test = sources["test"]
    combined = pd.concat([train, test], ignore_index=True)
    work_missing = train["workclass"].isna()
    occupation_missing = train["occupation"].isna()
    capital = repeated_maximum(train["capital_gain"])
    hours = repeated_maximum(train["hours_per_week"])
    return {
        "test_preamble": sources["preamble"],
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_leading_spaces": leading_space_count(sources["train_raw"]),
        "test_leading_spaces": leading_space_count(sources["test_raw"]),
        "train_income": income_inventory(sources["train_raw"]["income"]),
        "test_income": income_inventory(sources["test_raw"]["income"]),
        "train_question_marks": question_mark_counts(train),
        "test_question_marks": question_mark_counts(test),
        "train_workclass_and_occupation_missing": int((work_missing & occupation_missing).sum()),
        "train_workclass_missing_occupation_observed": int(
            (work_missing & ~occupation_missing).sum()
        ),
        "train_occupation_missing_workclass_observed": int(
            (~work_missing & occupation_missing).sum()
        ),
        "train_duplicates": duplicate_inventory(train),
        "test_duplicates": duplicate_inventory(test),
        "combined_duplicates": duplicate_inventory(combined),
        "category_alignment": category_alignment(train, test),
        "education_mapping": education_is_one_to_one(train),
        "train_capital_gain": capital,
        "train_capital_gain_iqr": iqr_fences(train["capital_gain"].to_numpy(dtype=float)),
        "train_hours": hours,
        "train_hours_iqr": iqr_fences(train["hours_per_week"].to_numpy(dtype=float)),
        "train_hours_min": float(train["hours_per_week"].min()),
        "train_hours_max": float(train["hours_per_week"].max()),
        "n_train_hours_outside_iqr": int(
            (
                (
                    train["hours_per_week"]
                    < iqr_fences(train["hours_per_week"].to_numpy(dtype=float))["low"]
                )
                | (
                    train["hours_per_week"]
                    > iqr_fences(train["hours_per_week"].to_numpy(dtype=float))["high"]
                )
            ).sum()
        ),
        "n_holand_netherlands_train": int((train["native_country"] == "Holand-Netherlands").sum()),
        "n_holand_netherlands_test": int((test["native_country"] == "Holand-Netherlands").sum()),
    }


def main() -> None:
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    inventory = inventory_from(sources)
    official = official_experiments(sources)
    injection, mnar_plot = run_injection(sources["train"])
    plot_missingness(
        RESULTS / "missingness.png",
        {
            column: inventory["train_question_marks"][column]
            for column in ("workclass", "occupation", "native_country")
        },
        inventory["train_workclass_and_occupation_missing"],
    )
    plot_capital_gain(
        RESULTS / "capital_gain.png",
        sources["train"]["capital_gain"].to_numpy(dtype=float),
        inventory["train_capital_gain_iqr"],
        float(inventory["train_capital_gain"]["maximum"]),
    )
    plot_mae(RESULTS / "imputation_mae.png", injection["reconstructions"])
    plot_injection_excess(RESULTS / "injection_log_loss.png", injection["models"])
    plot_official(RESULTS / "official_log_loss.png", official["runs"])
    plot_mnar(
        RESULTS / "mnar_capital_gain.png",
        mnar_plot["truth"],
        mnar_plot["mean_fill"],
        mnar_plot["zero_fill"],
    )
    metrics = {
        "elapsed_seconds": time.perf_counter() - started,
        "data": {
            "name": "Adult",
            "uci_id": 2,
            "doi": DATASET_DOI,
            "repository_url": REPOSITORY_URL,
            "train_url": TRAIN_URL,
            "test_url": TEST_URL,
            "creators": ["Barry Becker", "Ronny Kohavi"],
            "license": "CC BY 4.0",
            "license_source": LICENSE_URL,
            "names_file_train_rows": 32561,
            "names_file_test_rows": 16281,
            "names_file_complete_train_rows": 30162,
            "names_file_duplicate_or_conflicting": 6,
        },
        "protocol": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "logistic_c": LOGISTIC_C,
            "logistic_max_iter": LOGISTIC_MAX_ITER,
            "log_loss_clip": LOG_LOSS_CLIP,
            "decision_threshold": 0.5,
            "mcar_rate": MCAR_RATE,
            "mar_age_cutoff": MAR_AGE_CUTOFF,
            "mar_high_rate": MAR_HIGH_RATE,
            "mar_low_rate": MAR_LOW_RATE,
            "mnar_positive_rate": MNAR_POSITIVE_RATE,
            "mnar_zero_rate": MNAR_ZERO_RATE,
            "mask_seed": MASK_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "iqr_multiplier": IQR_MULTIPLIER,
            "excluded_features": ["education", "race", "fnlwgt"],
            "fnlwgt_excluded_reason": "sampling weight, not an attribute of the person",
            "race_excluded_reason": "protected attribute; this project does not audit fairness",
        },
        "inventory": inventory,
        "official_split": {key: value for key, value in official.items()},
        "injection": injection,
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"elapsed_seconds": metrics["elapsed_seconds"], "n_official": len(official["runs"])}
        )
    )


if __name__ == "__main__":
    main()
