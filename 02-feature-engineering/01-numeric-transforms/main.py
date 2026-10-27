"""Numeric transforms for a linear model of concrete strength.

The reported pipeline is the one with the lowest group-CV RMSE. The holdout
score is recorded after that choice. It is not used to add, drop, or replace
a transform.
"""

from __future__ import annotations

import json
import re
import time
import traceback
import urllib.request
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    PolynomialFeatures,
    PowerTransformer,
    StandardScaler,
)

SEED = 42
TEST_SIZE = 0.2
RIDGE_ALPHA = 1.0
CV_SPLITS = 5
SKEW_THRESHOLD = 1.0
BOXCOX_SHIFT = 1.0
QUANTILE_BINS = 5
AGE_BIN_EDGES = (0.0, 7.0, 28.0, 90.0, 365.0)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 7
DATASET_ID = 165
DATASET_DOI = "10.24432/C5PK67"
DATA_URL = "https://archive.ics.uci.edu/static/public/165/data.csv"
API_URL = "https://archive.ics.uci.edu/api/dataset?id=165"
CEMENT = 0
WATER = 3
AGE = 7
FEATURE_COLUMNS = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
]
RECIPE_COLUMNS = FEATURE_COLUMNS[:-1]
TARGET = "Concrete compressive strength"
EXPERIMENT_ORDER = (
    "training_mean",
    "raw_unscaled",
    "scaled",
    "log1p_skewed",
    "yeo_johnson",
    "box_cox_shifted",
    "coarse_age_bins",
    "quantile_bins",
    "water_cement",
    "pairwise",
    "poly2",
    "poly2_unscaled",
    "poly3",
    "domain_ratio",
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CSV_PATH = DATA / "concrete.csv"
API_PATH = DATA / "uci_api.json"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "ml-mastery-lab/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        dest.write_bytes(response.read())


def python_number(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [python_number(item) for item in value.tolist()]
    return value


def rmse(y_true, y_pred) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def recipe_groups(frame: pd.DataFrame) -> np.ndarray:
    """Group ids from the seven ingredients. Age is not part of the recipe."""
    missing = [column for column in RECIPE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing recipe columns: {missing}")
    values = frame.loc[:, RECIPE_COLUMNS].to_numpy(dtype=float)
    codes, _ = pd.factorize(pd.Series(map(tuple, values)), sort=True)
    return np.asarray(codes, dtype=int)


def select_experiment(scores: dict[str, dict]) -> str:
    """Lowest group-CV RMSE. Ties break toward fewer features, then the name."""
    if not scores:
        raise ValueError("no scored experiments")
    return min(
        scores,
        key=lambda name: (scores[name]["cv_rmse_mean"], scores[name]["n_features"], name),
    )


class Log1pSkewed(BaseEstimator, TransformerMixin):
    """log1p columns whose training skew exceeds the threshold and whose min is nonnegative."""

    def __init__(self, threshold: float = SKEW_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.n_features_in_ = array.shape[1]
        skews = []
        chosen = []
        for index in range(array.shape[1]):
            column = array[:, index]
            skew = float(stats.skew(column, bias=False, nan_policy="omit"))
            skews.append(skew)
            if np.isfinite(skew) and skew > self.threshold and float(np.min(column)) >= 0:
                chosen.append(index)
        self.skew_ = np.asarray(skews, dtype=float)
        self.columns_ = np.asarray(chosen, dtype=int)
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float).copy()
        for index in self.columns_:
            array[:, index] = np.log1p(array[:, index])
        return array


class ShiftedBoxCox(BaseEstimator, TransformerMixin):
    """Box-Cox after a train-chosen shift on columns that are not strictly positive."""

    def __init__(self, shift: float = BOXCOX_SHIFT):
        self.shift = shift

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.n_features_in_ = array.shape[1]
        minimum = array.min(axis=0)
        self.shift_ = np.where(minimum <= 0, self.shift, 0.0).astype(float)
        self.transformer_ = PowerTransformer(method="box-cox", standardize=True)
        self.transformer_.fit(array + self.shift_)
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float)
        return self.transformer_.transform(array + self.shift_)


class WaterCement(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.n_features_in_ = array.shape[1]
        if np.any(array[:, CEMENT] <= 0):
            raise ValueError("cement must be positive to form the water-cement ratio")
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float)
        ratio = array[:, WATER] / array[:, CEMENT]
        return np.column_stack([array, ratio])


class DomainFeatures(BaseEstimator, TransformerMixin):
    """Keep the ingredients, replace age with log1p(age), and append water/cement."""

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.n_features_in_ = array.shape[1]
        if np.any(array[:, CEMENT] <= 0):
            raise ValueError("cement must be positive to form the water-cement ratio")
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float)
        ratio = array[:, WATER] / array[:, CEMENT]
        logged_age = np.log1p(array[:, AGE])
        others = np.delete(array, AGE, axis=1)
        return np.column_stack([others, logged_age, ratio])


class CoarseAgeBins(BaseEstimator, TransformerMixin):
    def __init__(self, edges: tuple[float, ...] = AGE_BIN_EDGES):
        self.edges = edges

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.n_features_in_ = array.shape[1]
        self.edges_ = np.asarray(self.edges, dtype=float)
        if self.edges_.size < 2 or np.any(np.diff(self.edges_) <= 0):
            raise ValueError("age bin edges must be strictly increasing")
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float)
        age = array[:, AGE]
        if np.any(age < self.edges_[0]) or np.any(age > self.edges_[-1]):
            raise ValueError("age fell outside the pre-registered bin edges")
        codes = np.digitize(age, self.edges_[1:-1], right=True)
        dummies = np.eye(self.edges_.size - 1, dtype=float)[codes]
        others = np.delete(array, AGE, axis=1)
        return np.hstack([others, dummies])


def unshifted_box_cox_error(values: np.ndarray) -> str | None:
    try:
        PowerTransformer(method="box-cox", standardize=True).fit(np.asarray(values, dtype=float))
    except ValueError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def ridge() -> Ridge:
    return Ridge(alpha=RIDGE_ALPHA, random_state=SEED)


def scaled_ridge(*steps: tuple[str, BaseEstimator]) -> Pipeline:
    return Pipeline([*steps, ("scale", StandardScaler()), ("model", ridge())])


def build_estimators() -> dict[str, BaseEstimator]:
    return {
        "raw_unscaled": Pipeline([("model", ridge())]),
        "scaled": scaled_ridge(),
        "log1p_skewed": scaled_ridge(("log1p", Log1pSkewed())),
        "yeo_johnson": Pipeline(
            [
                ("yeo", PowerTransformer(method="yeo-johnson", standardize=True)),
                ("model", ridge()),
            ]
        ),
        "box_cox_shifted": Pipeline([("boxcox", ShiftedBoxCox()), ("model", ridge())]),
        "coarse_age_bins": scaled_ridge(("age_bins", CoarseAgeBins())),
        "quantile_bins": scaled_ridge(
            (
                "bins",
                KBinsDiscretizer(
                    n_bins=QUANTILE_BINS,
                    encode="onehot-dense",
                    strategy="quantile",
                    subsample=None,
                ),
            )
        ),
        "water_cement": scaled_ridge(("ratio", WaterCement())),
        "pairwise": scaled_ridge(
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
        ),
        "poly2": scaled_ridge(("poly", PolynomialFeatures(degree=2, include_bias=False))),
        "poly2_unscaled": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", ridge()),
            ]
        ),
        "poly3": scaled_ridge(("poly", PolynomialFeatures(degree=3, include_bias=False))),
        "domain_ratio": scaled_ridge(("domain", DomainFeatures())),
    }


def output_width(estimator, x_train: np.ndarray) -> int:
    if not hasattr(estimator, "steps"):
        return 0
    data = x_train[:1]
    for _, step in estimator.steps[:-1]:
        data = step.transform(data)
    return int(np.asarray(data).shape[1])


def score_split(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def training_mean_scores(x_train, y_train, groups_train, x_test, y_test) -> dict:
    folds = GroupKFold(n_splits=CV_SPLITS)
    cv_rmse = []
    cv_mae = []
    train_rmse = []
    for inner_train, inner_valid in folds.split(x_train, y_train, groups_train):
        mean = float(np.mean(y_train[inner_train]))
        cv_rmse.append(rmse(y_train[inner_valid], np.full(inner_valid.size, mean)))
        cv_mae.append(
            float(mean_absolute_error(y_train[inner_valid], np.full(inner_valid.size, mean)))
        )
        train_rmse.append(rmse(y_train[inner_train], np.full(inner_train.size, mean)))
    full_mean = float(np.mean(y_train))
    test_pred = np.full(y_test.size, full_mean)
    train_pred = np.full(y_train.size, full_mean)
    return {
        "n_features": 0,
        "cv_rmse_mean": float(np.mean(cv_rmse)),
        "cv_rmse_std": float(np.std(cv_rmse, ddof=1)),
        "cv_mae_mean": float(np.mean(cv_mae)),
        "cv_train_rmse_mean": float(np.mean(train_rmse)),
        "train": score_split(y_train, train_pred),
        "test": score_split(y_test, test_pred),
        "test_predictions": test_pred,
    }


def fit_experiment(name, estimator, x_train, y_train, groups_train, x_test, y_test) -> dict:
    folds = GroupKFold(n_splits=CV_SPLITS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scored, fitted, train_pred, test_pred = _fit_scored_experiment(
            name, estimator, x_train, y_train, groups_train, x_test, y_test, folds
        )
    if not np.isfinite(train_pred).all() or not np.isfinite(test_pred).all():
        raise FloatingPointError(f"{name} produced a non-finite prediction")
    cv_rmse = -scored["test_rmse"]
    messages = [str(item.message) for item in caught]
    return {
        "n_features": output_width(fitted, x_train),
        "cv_rmse_mean": float(np.mean(cv_rmse)),
        "cv_rmse_std": float(np.std(cv_rmse, ddof=1)),
        "cv_mae_mean": float(np.mean(-scored["test_mae"])),
        "cv_train_rmse_mean": float(np.mean(-scored["train_rmse"])),
        "train": score_split(y_train, train_pred),
        "test": score_split(y_test, test_pred),
        "test_predictions": test_pred,
        "fitted": fitted,
        "warnings": messages,
    }


def _fit_scored_experiment(name, estimator, x_train, y_train, groups_train, x_test, y_test, folds):
    scored = cross_validate(
        estimator,
        x_train,
        y_train,
        groups=groups_train,
        cv=folds,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        },
        return_train_score=True,
        n_jobs=1,
        error_score="raise",
    )
    fitted = clone(estimator).fit(x_train, y_train)
    train_pred = fitted.predict(x_train)
    test_pred = fitted.predict(x_test)
    return scored, fitted, train_pred, test_pred


def warning_summary(messages: list[str]) -> dict:
    indexes = sorted(
        {int(match) for message in messages for match in re.findall(r"feature (\d+)", message)}
    )
    return {
        "n_warnings": len(messages),
        "unique_messages": sorted(set(messages)),
        "feature_indexes": indexes,
        "feature_names": [FEATURE_COLUMNS[index] for index in indexes],
    }


def bootstrap_gap(y_true, baseline_pred, challenger_pred) -> dict[str, float | int]:
    """Positive gap means the challenger has the lower resampled RMSE."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gaps = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    n_rows = y_true.size
    for index in range(BOOTSTRAP_RESAMPLES):
        chosen = rng.integers(0, n_rows, n_rows)
        gaps[index] = rmse(y_true[chosen], baseline_pred[chosen]) - rmse(
            y_true[chosen], challenger_pred[chosen]
        )
    return {
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "gap_mean": float(np.mean(gaps)),
        "gap_p2_5": float(np.quantile(gaps, 0.025)),
        "gap_p97_5": float(np.quantile(gaps, 0.975)),
        "fraction_challenger_better": float(np.mean(gaps > 0)),
    }


def column_stats(values: np.ndarray, train_values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_zero": int(np.sum(values == 0)),
        "n_negative": int(np.sum(values < 0)),
        "n_non_integer": int(np.sum(values != np.trunc(values))),
        "skew_all": float(stats.skew(values, bias=False)),
        "skew_train": float(stats.skew(train_values, bias=False)),
    }


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = [*FEATURE_COLUMNS, TARGET]
    if list(frame.columns) != expected:
        raise ValueError(f"unexpected columns: {list(frame.columns)}")
    if frame[expected].isna().any().any():
        raise ValueError("the extract has a missing value in a modeled column")
    return frame


def load_api(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    if data.get("uci_id") != DATASET_ID:
        raise ValueError(f"API uci_id was {data.get('uci_id')}, expected {DATASET_ID}")
    return data


def conflicting_strength(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby(FEATURE_COLUMNS, dropna=False)[TARGET]
    counts = grouped.nunique()
    sizes = grouped.size()
    ranges = grouped.max() - grouped.min()
    conflicting = counts[counts > 1]
    return {
        "n_feature_keys": int(counts.size),
        "n_feature_duplicate_rows": int(frame.duplicated(FEATURE_COLUMNS).sum()),
        "n_exact_duplicate_rows": int(frame.duplicated().sum()),
        "n_feature_keys_with_conflicting_strength": int(conflicting.size),
        "max_distinct_strengths_for_one_key": int(counts.max()),
        "max_within_key_strength_range": float(ranges.loc[conflicting.index].max())
        if len(conflicting)
        else 0.0,
        "n_rows_in_conflicting_keys": int(sizes.loc[conflicting.index].sum())
        if len(conflicting)
        else 0,
    }


def public_score(result: dict) -> dict:
    return {
        "n_features": result["n_features"],
        "cv_rmse_mean": result["cv_rmse_mean"],
        "cv_rmse_std": result["cv_rmse_std"],
        "cv_mae_mean": result["cv_mae_mean"],
        "cv_train_rmse_mean": result["cv_train_rmse_mean"],
        "train": result["train"],
        "test": result["test"],
    }


def plot_age(path: Path, raw_age: np.ndarray, logged_age: np.ndarray, yeo_age: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    panels = (
        (raw_age, "Age (days)", "Raw age"),
        (logged_age, "log1p(age)", "log1p age"),
        (yeo_age, "Standardized Yeo-Johnson age", "Yeo-Johnson age"),
    )
    for axis, (values, xlabel, title) in zip(axes, panels, strict=True):
        axis.hist(values, bins=30, color="#3174a1", edgecolor="white")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Training rows")
        axis.set_title(title)
    fig.suptitle("Age on the training mixes")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cv(path: Path, scores: dict[str, dict], selected: str) -> None:
    names = [name for name in EXPERIMENT_ORDER if name in scores]
    values = [scores[name]["cv_rmse_mean"] for name in names]
    fig, axis = plt.subplots(figsize=(8.5, 6.4), constrained_layout=True)
    colors = ["#3174a1" if name == selected else "#8aa0b4" for name in names]
    labels = [f"{name} (selected)" if name == selected else name for name in names]
    axis.barh(labels, values, color=colors)
    axis.invert_yaxis()
    axis.set_xlabel("Group CV RMSE (MPa)")
    axis.set_ylabel("Experiment")
    axis.set_title("Selection uses group CV RMSE, not the holdout")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
    axis.scatter(y_true, y_pred, s=18, alpha=0.75, color="#3174a1", linewidths=0)
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    axis.plot([low, high], [low, high], color="#333333", linewidth=1)
    axis.set_xlabel("Measured strength (MPa)")
    axis.set_ylabel("Predicted strength (MPa)")
    axis.set_title(f"Holdout predictions, {name}")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fitted_transform(fitted: BaseEstimator, name: str):
    if not hasattr(fitted, "named_steps") or name not in fitted.named_steps:
        return None
    return fitted.named_steps[name]


def lambda_map(values: np.ndarray) -> dict[str, float]:
    return {column: float(value) for column, value in zip(FEATURE_COLUMNS, values, strict=True)}


def main() -> None:
    started = time.perf_counter()
    download(DATA_URL, CSV_PATH)
    download(API_URL, API_PATH)
    frame = load_frame(CSV_PATH)
    api = load_api(API_PATH)
    groups = recipe_groups(frame)
    features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    target = frame[TARGET].to_numpy(dtype=float)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_index, test_index = next(splitter.split(features, target, groups))
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]
    groups_train, groups_test = groups[train_index], groups[test_index]
    overlap = int(len(set(groups_train).intersection(groups_test)))
    if overlap:
        raise RuntimeError(f"group split put {overlap} recipes on both sides")

    results = {
        "training_mean": training_mean_scores(x_train, y_train, groups_train, x_test, y_test)
    }
    failures = {}
    for name, estimator in build_estimators().items():
        try:
            results[name] = fit_experiment(
                name, estimator, x_train, y_train, groups_train, x_test, y_test
            )
        except Exception as exc:  # noqa: BLE001
            failures[name] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
    missing = [name for name in EXPERIMENT_ORDER if name not in results and name not in failures]
    if missing:
        raise RuntimeError(f"experiments neither scored nor failed: {missing}")
    selected = select_experiment({name: public_score(results[name]) for name in results})
    test_winner = min(results, key=lambda name: (results[name]["test"]["rmse"], name))
    selected_fit = results[selected].get("fitted")
    log_transformer = fitted_transform(selected_fit, "log1p") if selected_fit is not None else None
    if log_transformer is None and "log1p_skewed" in results:
        log_transformer = fitted_transform(results["log1p_skewed"]["fitted"], "log1p")
    yeo = (
        fitted_transform(results["yeo_johnson"]["fitted"], "yeo")
        if "yeo_johnson" in results
        else None
    )
    box = (
        fitted_transform(results["box_cox_shifted"]["fitted"], "boxcox")
        if "box_cox_shifted" in results
        else None
    )

    row_train, row_test = train_test_split(
        np.arange(target.size), test_size=TEST_SIZE, random_state=SEED
    )
    row_overlap = np.isin(groups[row_test], groups[row_train])
    random_scaled = clone(build_estimators()["scaled"]).fit(features[row_train], target[row_train])
    random_selected_name = selected if selected != "training_mean" else "scaled"
    random_selected = clone(build_estimators()[random_selected_name]).fit(
        features[row_train], target[row_train]
    )

    scaler = StandardScaler().fit(x_train)
    population_std = x_train.std(axis=0, ddof=0)
    sample_std = x_train.std(axis=0, ddof=1)
    api_types = {item["name"]: item.get("type") for item in api.get("variables", [])}
    api_units = {item["name"]: item.get("units") for item in api.get("variables", [])}
    columns = {
        column: column_stats(features[:, index], x_train[:, index])
        for index, column in enumerate(FEATURE_COLUMNS)
    }
    for column in columns:
        columns[column]["api_type"] = api_types.get(column)
        columns[column]["api_units"] = api_units.get(column)
    conflicts = conflicting_strength(frame)
    recipe_sizes = pd.Series(groups).value_counts()
    mean_pred = results["training_mean"]["test_predictions"]
    selected_pred = results[selected]["test_predictions"]
    scaled_pred = results["scaled"]["test_predictions"] if "scaled" in results else None

    RESULTS.mkdir(parents=True, exist_ok=True)
    plot_age(
        RESULTS / "age_transform.png",
        x_train[:, AGE],
        np.log1p(x_train[:, AGE]),
        yeo.transform(x_train)[:, AGE] if yeo is not None else np.log1p(x_train[:, AGE]),
    )
    plot_cv(
        RESULTS / "cv_rmse.png", {name: public_score(results[name]) for name in results}, selected
    )
    plot_predictions(RESULTS / "pred_vs_actual.png", y_test, selected_pred, selected)

    payload = {
        "dataset": {
            "name": api.get("name"),
            "uci_id": DATASET_ID,
            "doi": DATASET_DOI,
            "api_dataset_doi": api.get("dataset_doi"),
            "api_license_field": api.get("license"),
            "api_num_instances": api.get("num_instances"),
            "api_num_features": api.get("num_features"),
            "api_has_missing_values": api.get("has_missing_values"),
            "api_intro_paper_doi": (api.get("intro_paper") or {}).get("DOI"),
            "api_intro_paper_title": (api.get("intro_paper") or {}).get("title"),
            "n_rows": int(frame.shape[0]),
            "n_columns": int(frame.shape[1]),
            "n_features": len(FEATURE_COLUMNS),
            "n_missing_modeled_cells": int(frame.isna().sum().sum()),
            "n_recipe_groups": int(pd.Series(groups).nunique()),
            "n_recipes_with_one_row": int((recipe_sizes == 1).sum()),
            "rows_per_recipe_min": int(recipe_sizes.min()),
            "rows_per_recipe_max": int(recipe_sizes.max()),
            "n_age_at_most_28": int(np.sum(features[:, AGE] <= 28)),
            "target_min": float(np.min(target)),
            "target_max": float(np.max(target)),
            "target_mean": float(np.mean(target)),
            "columns": columns,
            **conflicts,
        },
        "protocol": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "test_size_unit": "groups",
            "ridge_alpha": RIDGE_ALPHA,
            "cv_splits": CV_SPLITS,
            "cv": "GroupKFold",
            "skew_threshold": SKEW_THRESHOLD,
            "skew_bias": False,
            "boxcox_shift": BOXCOX_SHIFT,
            "age_bin_edges": list(AGE_BIN_EDGES),
            "quantile_bins": QUANTILE_BINS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "selection_metric": "cv_rmse_mean",
            "selection_uses_test_scores": False,
            "reported_model": selected,
            "experiment_order": list(EXPERIMENT_ORDER),
        },
        "split": {
            "n_train_rows": int(train_index.size),
            "n_test_rows": int(test_index.size),
            "n_train_groups": int(pd.Series(groups_train).nunique()),
            "n_test_groups": int(pd.Series(groups_test).nunique()),
            "n_groups_in_both": overlap,
            "train_target_mean": float(np.mean(y_train)),
            "scaler_matches_population_std": bool(np.allclose(scaler.scale_, population_std)),
            "scaler_matches_sample_std": bool(np.allclose(scaler.scale_, sample_std)),
        },
        "scores": {
            name: public_score(results[name]) for name in EXPERIMENT_ORDER if name in results
        },
        "failures": failures,
        "selection": {
            "reported_model": selected,
            "test_rmse_winner": test_winner,
            "test_winner_was_selected": test_winner == selected,
        },
        "transforms": {
            "unshifted_box_cox_error": unshifted_box_cox_error(x_train),
            "log1p_columns": [
                FEATURE_COLUMNS[index] for index in getattr(log_transformer, "columns_", [])
            ],
            "log1p_skew_train": {
                column: float(value)
                for column, value in zip(
                    FEATURE_COLUMNS,
                    getattr(log_transformer, "skew_", []),
                    strict=False,
                )
            },
            "yeo_johnson_lambdas": lambda_map(yeo.lambdas_) if yeo is not None else None,
            "box_cox_shifted_lambdas": (
                lambda_map(box.transformer_.lambdas_) if box is not None else None
            ),
            "box_cox_shifts": lambda_map(box.shift_) if box is not None else None,
        },
        "random_row_split": {
            "n_train_rows": int(row_train.size),
            "n_test_rows": int(row_test.size),
            "n_test_rows_whose_recipe_is_in_train": int(row_overlap.sum()),
            "n_test_recipes_also_in_train": int(
                len(set(groups[row_test]).intersection(groups[row_train]))
            ),
            "scaled_test": score_split(target[row_test], random_scaled.predict(features[row_test])),
            "selected_pipeline_test": score_split(
                target[row_test], random_selected.predict(features[row_test])
            ),
            "selected_pipeline": random_selected_name,
            "used_for_selection": False,
        },
        "bootstrap": {
            "training_mean_rmse_minus_selected": bootstrap_gap(y_test, mean_pred, selected_pred),
            "scaled_rmse_minus_selected": (
                bootstrap_gap(y_test, scaled_pred, selected_pred)
                if scaled_pred is not None
                else None
            ),
            "log1p_rmse_minus_selected": bootstrap_gap(
                y_test, results["log1p_skewed"]["test_predictions"], selected_pred
            ),
        },
        "gaps": {
            "training_mean_test_rmse_minus_selected": results["training_mean"]["test"]["rmse"]
            - results[selected]["test"]["rmse"],
            "scaled_test_rmse_minus_selected": results["scaled"]["test"]["rmse"]
            - results[selected]["test"]["rmse"],
            "log1p_test_rmse_minus_selected": results["log1p_skewed"]["test"]["rmse"]
            - results[selected]["test"]["rmse"],
            "poly3_cv_rmse_minus_log1p": results["poly3"]["cv_rmse_mean"]
            - results["log1p_skewed"]["cv_rmse_mean"],
            "scaled_cv_rmse_minus_raw": results["scaled"]["cv_rmse_mean"]
            - results["raw_unscaled"]["cv_rmse_mean"],
            "domain_cv_rmse_minus_log1p": results["domain_ratio"]["cv_rmse_mean"]
            - results["log1p_skewed"]["cv_rmse_mean"],
        },
        "quantile_bin_warnings": warning_summary(results["quantile_bins"].get("warnings", [])),
        "elapsed_seconds": time.perf_counter() - started,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
            "scipy": scipy.__version__,
        },
    }
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reported_model": selected, "failures": list(failures)}, indent=2))


if __name__ == "__main__":
    main()
