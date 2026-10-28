"""Three leakage bugs on the same held-out bank calls.

The reported model is the honest protocol, fixed before any score is computed.
Leaky fits are recorded and are not eligible. The held-out calls are not used
to add, drop, or replace a feature.

This is a teaching model on a public research extract. It is not a campaign tool.
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
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
PAST_FRACTION = 0.6
WINDOW_END_FRACTION = 0.8
SHUFFLED_TEST_SIZE = 0.2
LOGISTIC_C = 1.0
MAX_ITER = 500
CLIP = 1e-6
SMOOTHING = 20.0
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 7
START_YEAR = 2008
EXPECTED_ROWS = 45211
DATASET_ID = 222
DATASET_DOI = "10.24432/C5K306"
ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
API_URL = "https://archive.ics.uci.edu/api/dataset?id=222"
API_CSV_URL = "https://archive.ics.uci.edu/static/public/222/data.csv"
ORDER_SENTENCE = "ordered by date (from May 2008 to November 2010)"
LICENSE_SENTENCE = "This dataset is public available for research."
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
DURATION_EDGES = (0, 60, 120, 180, 300, 600, 1200, np.inf)
MISSING = "unknown"

CATEGORICAL = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "poutcome",
)
NUMERIC = ("age", "balance", "campaign", "pdays", "previous", "never_contacted")
HONEST_COLUMNS = CATEGORICAL + NUMERIC
LEAK_COLUMN = "duration"
REPORTED = "honest"
LEAKS = frozenset(
    {
        "duration_leak",
        "same_period_rows",
        "future_rows",
        "target_before_split",
        "scaler_before_split",
    }
)
CONTROLS = frozenset(
    {
        "same_period_size_control",
        "future_size_control",
        "target_train_only",
    }
)
EXPERIMENT_ORDER = (
    "training_rate",
    "honest",
    "duration_leak",
    "same_period_rows",
    "same_period_size_control",
    "future_rows",
    "future_size_control",
    "target_train_only",
    "target_before_split",
    "scaler_before_split",
)
LABELS = {
    "training_rate": "training rate",
    "honest": "honest",
    "duration_leak": "duration leak",
    "same_period_rows": "same-period rows",
    "same_period_size_control": "extra past rows",
    "future_rows": "later rows",
    "future_size_control": "extra past, later size",
    "target_train_only": "month mean, train only",
    "target_before_split": "month mean before split",
    "scaler_before_split": "scaler before split",
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


def walk_years(run_months: list[str], start_year: int = START_YEAR) -> list[int]:
    years: list[int] = []
    year = start_year
    previous: str | None = None
    for name in run_months:
        if previous is not None:
            step = (MONTHS.index(name) - MONTHS.index(previous)) % 12
            if step == 0 or step >= 7:
                raise ValueError(f"month run is not forward: {previous} -> {name}")
            if MONTHS.index(name) < MONTHS.index(previous):
                year += 1
        years.append(year)
        previous = name
    return years


def month_runs(months: np.ndarray) -> list[dict[str, int | str]]:
    runs: list[dict[str, int | str]] = []
    start = 0
    for index in range(1, len(months) + 1):
        if index == len(months) or months[index] != months[start]:
            runs.append({"month": str(months[start]), "start": start, "end": index})
            start = index
    years = walk_years([str(run["month"]) for run in runs])
    for run, year in zip(runs, years, strict=True):
        run["year"] = year
        run["label"] = f"{year}-{run['month']}"
    return runs


def make_splits(n: int, seed: int = SEED) -> dict[str, np.ndarray]:
    past_end = int(n * PAST_FRACTION)
    window_end = int(n * WINDOW_END_FRACTION)
    if past_end < 1 or window_end - past_end < 2 or n - window_end < 1:
        raise ValueError("split fractions do not leave a past, a window, and a later tail")
    past = np.arange(0, past_end)
    window = np.arange(past_end, window_end)
    future = np.arange(window_end, n)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(window)
    n_eval = len(window) // 2
    eval_idx = np.sort(perm[:n_eval])
    same_period = np.sort(perm[n_eval:])
    extra_same = rng.choice(past, size=len(same_period), replace=True)
    extra_future = rng.choice(past, size=len(future), replace=True)
    return {
        "past": past,
        "window": window,
        "future": future,
        "eval": eval_idx,
        "same_period": same_period,
        "extra_same": extra_same,
        "extra_future": extra_future,
    }


def assert_split_order(splits: dict[str, np.ndarray], n: int) -> None:
    past, eval_idx, same_period, future = (
        splits["past"],
        splits["eval"],
        splits["same_period"],
        splits["future"],
    )
    if past.max() >= eval_idx.min() or eval_idx.max() >= future.min():
        raise ValueError("held-out calls are not strictly between the past and the later tail")
    if set(eval_idx).intersection(same_period) or set(eval_idx).intersection(past):
        raise ValueError("held-out calls overlap the training blocks")
    window = np.concatenate([eval_idx, same_period])
    if set(window) != set(splits["window"]) or len(window) != len(splits["window"]):
        raise ValueError("the middle window was not partitioned")
    covered = np.concatenate([past, splits["window"], future])
    if not np.array_equal(np.sort(covered), np.arange(n)):
        raise ValueError("the time blocks do not cover the file")


def row_set(name: str, splits: dict[str, np.ndarray]) -> np.ndarray:
    if name == "past":
        return splits["past"]
    if name == "past_plus_same_period":
        return np.concatenate([splits["past"], splits["same_period"]])
    if name == "past_plus_resampled_same":
        return np.concatenate([splits["past"], splits["extra_same"]])
    if name == "past_plus_future":
        return np.concatenate([splits["past"], splits["future"]])
    if name == "past_plus_resampled_future":
        return np.concatenate([splits["past"], splits["extra_future"]])
    if name == "past_and_eval":
        return np.concatenate([splits["past"], splits["eval"]])
    raise ValueError(name)


def experiment_specs() -> list[dict[str, str | bool | None]]:
    return [
        {
            "name": "training_rate",
            "kind": "rate",
            "train": "past",
            "preprocess": "past",
            "target_fit": None,
            "duration": False,
            "eligible": True,
        },
        {
            "name": "honest",
            "kind": "model",
            "train": "past",
            "preprocess": "past",
            "target_fit": None,
            "duration": False,
            "eligible": True,
        },
        {
            "name": "duration_leak",
            "kind": "model",
            "train": "past",
            "preprocess": "past",
            "target_fit": None,
            "duration": True,
            "eligible": False,
        },
        {
            "name": "same_period_rows",
            "kind": "model",
            "train": "past_plus_same_period",
            "preprocess": "train",
            "target_fit": None,
            "duration": False,
            "eligible": False,
        },
        {
            "name": "same_period_size_control",
            "kind": "model",
            "train": "past_plus_resampled_same",
            "preprocess": "train",
            "target_fit": None,
            "duration": False,
            "eligible": False,
        },
        {
            "name": "future_rows",
            "kind": "model",
            "train": "past_plus_future",
            "preprocess": "train",
            "target_fit": None,
            "duration": False,
            "eligible": False,
        },
        {
            "name": "future_size_control",
            "kind": "model",
            "train": "past_plus_resampled_future",
            "preprocess": "train",
            "target_fit": None,
            "duration": False,
            "eligible": False,
        },
        {
            "name": "target_train_only",
            "kind": "model",
            "train": "past",
            "preprocess": "past",
            "target_fit": "past",
            "duration": False,
            "eligible": False,
        },
        {
            "name": "target_before_split",
            "kind": "model",
            "train": "past",
            "preprocess": "past",
            "target_fit": "past_and_eval",
            "duration": False,
            "eligible": False,
        },
        {
            "name": "scaler_before_split",
            "kind": "model",
            "train": "past",
            "preprocess": "past_and_eval",
            "target_fit": None,
            "duration": False,
            "eligible": False,
        },
    ]


class TargetEncoder:
    """Smoothed category mean. Unseen categories get the mean of the fit rows."""

    def __init__(self, smoothing: float) -> None:
        self.smoothing = smoothing

    def fit(self, keys: np.ndarray, y: np.ndarray) -> TargetEncoder:
        target = np.asarray(y, dtype=float)
        self.global_mean_ = float(target.mean())
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for key, value in zip(keys.tolist(), target.tolist(), strict=True):
            token = str(key)
            sums[token] = sums.get(token, 0.0) + float(value)
            counts[token] = counts.get(token, 0) + 1
        smoothing = self.smoothing
        global_mean = self.global_mean_
        self.map_ = {
            key: (sums[key] + smoothing * global_mean) / (counts[key] + smoothing) for key in counts
        }
        return self

    def transform(self, keys: np.ndarray) -> np.ndarray:
        encoded = [self.map_.get(str(key), self.global_mean_) for key in keys.tolist()]
        return np.asarray(encoded, dtype=float).reshape(-1, 1)


def levels(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("Int64").astype("string").fillna(MISSING).astype(str).to_numpy()
    return series.astype("string").fillna(MISSING).astype(str).to_numpy()


def categorical_tokens(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([levels(frame[column]) for column in CATEGORICAL])


def numeric_frame(frame: pd.DataFrame, include_duration: bool) -> np.ndarray:
    columns = list(NUMERIC)
    if include_duration:
        columns.append(LEAK_COLUMN)
    return frame[columns].to_numpy(dtype=float)


def stack(parts: list[np.ndarray | object]) -> object:
    from scipy import sparse

    blocks = []
    for part in parts:
        if sparse.issparse(part):
            blocks.append(part.tocsr())
        else:
            blocks.append(sparse.csr_matrix(np.asarray(part, dtype=float)))
    return sparse.hstack(blocks, format="csr")


def clip_proba(proba: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(proba, dtype=float), CLIP, 1.0 - CLIP)


def score_proba(y: np.ndarray, proba: np.ndarray) -> dict[str, float | None]:
    proba = clip_proba(proba)
    scored: dict[str, float | None] = {
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "average_precision": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "accuracy_at_0_5": float(accuracy_score(y, proba >= 0.5)),
        "mean_predicted": float(proba.mean()),
        "positive_rate": float(np.mean(y)),
        "roc_auc": None,
    }
    if len(np.unique(y)) > 1 and np.unique(proba).size > 1:
        scored["roc_auc"] = float(roc_auc_score(y, proba))
    return scored


def logistic() -> LogisticRegression:
    return LogisticRegression(
        C=LOGISTIC_C,
        solver="liblinear",
        max_iter=MAX_ITER,
        random_state=SEED,
    )


def design_matrices(
    frame: pd.DataFrame,
    y: np.ndarray,
    fit_idx: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    include_duration: bool,
    target_fit_idx: np.ndarray | None,
) -> tuple[object, object]:
    fit_frame = frame.iloc[fit_idx]
    train_frame = frame.iloc[train_idx]
    eval_frame = frame.iloc[eval_idx]
    scaler = StandardScaler()
    scaler.fit(numeric_frame(fit_frame, include_duration))
    numeric_train = scaler.transform(numeric_frame(train_frame, include_duration))
    numeric_eval = scaler.transform(numeric_frame(eval_frame, include_duration))
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    encoder.fit(categorical_tokens(fit_frame))
    categorical_train = encoder.transform(categorical_tokens(train_frame))
    categorical_eval = encoder.transform(categorical_tokens(eval_frame))
    train_parts: list[object] = [numeric_train, categorical_train]
    eval_parts: list[object] = [numeric_eval, categorical_eval]
    if target_fit_idx is not None:
        target_encoder = TargetEncoder(SMOOTHING).fit(
            levels(frame.iloc[target_fit_idx]["month"]),
            y[target_fit_idx],
        )
        train_parts.append(target_encoder.transform(levels(train_frame["month"])))
        eval_parts.append(target_encoder.transform(levels(eval_frame["month"])))
    return stack(train_parts), stack(eval_parts)


def fit_experiment(
    frame: pd.DataFrame,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
    spec: dict,
) -> tuple[np.ndarray, int, int]:
    eval_idx = splits["eval"]
    if spec["kind"] == "rate":
        rate = float(y[splits["past"]].mean())
        return np.full(len(eval_idx), rate, dtype=float), 0, 0
    train_idx = row_set(str(spec["train"]), splits)
    preprocess_name = "past" if spec["preprocess"] == "past" else str(spec["train"])
    if spec["preprocess"] == "past_and_eval":
        preprocess_name = "past_and_eval"
    fit_idx = row_set(preprocess_name, splits)
    target_fit = None if spec["target_fit"] is None else row_set(str(spec["target_fit"]), splits)
    train_matrix, eval_matrix = design_matrices(
        frame,
        y,
        fit_idx,
        train_idx,
        eval_idx,
        include_duration=bool(spec["duration"]),
        target_fit_idx=target_fit,
    )
    model = logistic()
    model.fit(train_matrix, y[train_idx])
    n_iter = int(np.max(model.n_iter_))
    if n_iter >= MAX_ITER:
        raise RuntimeError(f"{spec['name']} did not converge in {MAX_ITER} iterations")
    return clip_proba(model.predict_proba(eval_matrix)[:, 1]), int(train_matrix.shape[1]), n_iter


def duration_bins(duration: np.ndarray, y: np.ndarray) -> list[dict[str, float | int | str | None]]:
    binned = pd.Series(pd.cut(duration, DURATION_EDGES, right=True, include_lowest=True))
    rows: list[dict[str, float | int | str | None]] = []
    for interval in binned.cat.categories:
        mask = (binned == interval).to_numpy()
        chosen = y[mask]
        rows.append(
            {
                "bin": str(interval),
                "n": int(mask.sum()),
                "n_yes": int(chosen.sum()) if mask.any() else 0,
                "yes_rate": float(chosen.mean()) if mask.any() else None,
            }
        )
    return rows


def class_duration(duration: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    yes = y == 1
    return {
        "n_yes": int(yes.sum()),
        "n_no": int((~yes).sum()),
        "mean_yes": float(duration[yes].mean()),
        "mean_no": float(duration[~yes].mean()),
        "median_yes": float(np.median(duration[yes])),
        "median_no": float(np.median(duration[~yes])),
    }


def reliability(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if right == 1.0:
            mask = (proba >= left) & (proba <= right)
        else:
            mask = (proba >= left) & (proba < right)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "n": int(mask.sum()),
                "mean_predicted": float(proba[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return rows


def date_ids(runs: list[dict], day: np.ndarray) -> np.ndarray:
    ids = np.empty(len(day), dtype=object)
    for run_index, run in enumerate(runs):
        ids[int(run["start"]) : int(run["end"])] = [
            f"{run['label']}-{int(value):02d}-run{run_index}"
            for value in day[int(run["start"]) : int(run["end"])]
        ]
    if pd.isna(ids).any():
        raise ValueError("a row has no date id")
    return ids


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
        gaps[draw] = log_loss(y[rows], clip_proba(reference[rows]), labels=[0, 1]) - log_loss(
            y[rows], clip_proba(candidate[rows]), labels=[0, 1]
        )
    return {
        "gap_mean": float(gaps.mean()),
        "gap_p2_5": float(np.quantile(gaps, 0.025)),
        "gap_p97_5": float(np.quantile(gaps, 0.975)),
        "fraction_positive": float(np.mean(gaps > 0)),
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "n_clusters": len(keys),
        "seed": BOOTSTRAP_SEED,
        "unit": "contact date within the held-out half",
        "definition": "reference log loss minus candidate log loss",
    }


def plot_log_loss(path: Path, scores: dict[str, dict]) -> None:
    names = list(EXPERIMENT_ORDER)
    losses = [scores[name]["test"]["log_loss"] for name in names]
    colors = []
    for name in names:
        if name == REPORTED:
            colors.append("#54A24B")
        elif name in LEAKS:
            colors.append("#E45756")
        elif name == "training_rate":
            colors.append("#9D9D9D")
        else:
            colors.append("#4C78A8")
    fig, axis = plt.subplots(figsize=(10.2, 6.6), constrained_layout=True)
    labels = [LABELS[name] for name in names]
    bars = axis.barh(labels, losses, color=colors)
    for bar, name in zip(bars, names, strict=True):
        if name in LEAKS:
            bar.set_hatch("//")
    axis.invert_yaxis()
    axis.set_xlabel("Log loss on the held-out calls")
    axis.set_ylabel("Experiment")
    axis.set_title(
        "Same held-out calls. Lower is better. Hatched bars use unavailable information."
    )
    axis.legend(
        handles=[
            Patch(facecolor="#54A24B", label="Reported protocol"),
            Patch(facecolor="#9D9D9D", label="Training-rate baseline"),
            Patch(facecolor="#E45756", hatch="//", label="Leak"),
            Patch(facecolor="#4C78A8", label="Control"),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
    )
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _cut_position(index: int, runs: list[dict]) -> float:
    for position, run in enumerate(runs):
        start, end = int(run["start"]), int(run["end"])
        if start <= index < end:
            return position + (index - start) / int(run["n"])
    return float(len(runs))


def plot_time(path: Path, runs: list[dict], past_end: int, window_end: int) -> None:
    fig, axis = plt.subplots(figsize=(12.6, 5.6), constrained_layout=True)
    positions = np.arange(len(runs))
    rates = [float(run["yes_rate"]) for run in runs]
    sizes = [max(18, int(run["n"]) / 25) for run in runs]
    axis.plot(positions, rates, color="#4C78A8", linewidth=1.2)
    axis.scatter(positions, rates, s=sizes, color="#4C78A8")
    axis.axvline(
        _cut_position(past_end, runs),
        color="#54A24B",
        linestyle="--",
        linewidth=1.2,
        label="End of past block",
    )
    axis.axvline(
        _cut_position(window_end, runs),
        color="#E45756",
        linestyle="--",
        linewidth=1.2,
        label="End of middle window",
    )
    axis.set_xticks(positions, [str(run["label"]) for run in runs], rotation=70, ha="right")
    axis.set_ylabel("Subscription rate")
    axis.set_xlabel("Campaign month in file order. Point size is the number of calls.")
    axis.set_title("Subscription rate by campaign month. Vertical lines are the row-order cuts.")
    axis.legend(frameon=False)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_duration(path: Path, rows: list[dict]) -> None:
    labels = [str(row["bin"]) for row in rows]
    rates = [0.0 if row["yes_rate"] is None else float(row["yes_rate"]) for row in rows]
    fig, axis = plt.subplots(figsize=(9.4, 5.0), constrained_layout=True)
    axis.bar(labels, rates, color="#4C78A8")
    axis.set_xlabel("Call duration in seconds, earlier calls only")
    axis.set_ylabel("Subscription rate")
    axis.set_title("Longer calls subscribed more often. Duration is known only after the call.")
    axis.tick_params(axis="x", rotation=30)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_reliability(path: Path, honest_rows: list[dict], duration_rows: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(5.8, 5.4), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#333333", linewidth=1, label="perfect")
    axis.scatter(
        [row["mean_predicted"] for row in honest_rows],
        [row["observed_rate"] for row in honest_rows],
        s=[max(20, row["n"] / 8) for row in honest_rows],
        color="#54A24B",
        label="honest",
    )
    axis.scatter(
        [row["mean_predicted"] for row in duration_rows],
        [row["observed_rate"] for row in duration_rows],
        s=[max(20, row["n"] / 8) for row in duration_rows],
        color="#E45756",
        label="duration leak",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed subscription rate")
    axis.set_title("Held-out reliability")
    axis.legend(frameon=False)
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
    return value


def assert_api_match(full: pd.DataFrame, api: pd.DataFrame) -> None:
    if len(full) != len(api) or list(api.columns)[9] != "day_of_week":
        raise ValueError("UCI API export does not match the expected bank table")
    for column in ("age", "balance", "duration", "campaign", "pdays", "previous", "month", "y"):
        if not full[column].equals(api[column]):
            raise ValueError(f"API export differs on {column}")
    if not full["day"].equals(api["day_of_week"]):
        raise ValueError("API day_of_week is not the documented day column")
    for column in (
        "job",
        "education",
        "contact",
        "poutcome",
        "marital",
        "default",
        "housing",
        "loan",
    ):
        exported = api[column].astype("string").fillna("unknown").astype(str)
        if not full[column].astype(str).equals(exported):
            raise ValueError(f"API missing values do not match unknown in {column}")


def load_frame() -> tuple[pd.DataFrame, dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    download(ZIP_URL, DATA / "bank.zip")
    download(API_URL, DATA / "uci_api.json")
    download(API_CSV_URL, DATA / "uci_data.csv")
    with zipfile.ZipFile(DATA / "bank.zip") as archive:
        names = archive.read("bank-names.txt").decode("latin-1")
        if ORDER_SENTENCE not in names or LICENSE_SENTENCE not in names:
            raise ValueError("bank-names.txt does not contain the expected order and license lines")
        full = pd.read_csv(archive.open("bank-full.csv"), sep=";")
    api_frame = pd.read_csv(DATA / "uci_data.csv")
    assert_api_match(full, api_frame)
    api = json.loads((DATA / "uci_api.json").read_text(encoding="utf-8"))["data"]
    if int(api["num_instances"]) != len(full) or int(api["uci_id"]) != DATASET_ID:
        raise ValueError("UCI API metadata does not match bank-full.csv")
    if len(full) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(full)}")
    required = set(CATEGORICAL) | set(NUMERIC) | {LEAK_COLUMN, "y"}
    required.discard("never_contacted")
    if not required.issubset(full.columns):
        raise ValueError(f"missing columns: {sorted(required - set(full.columns))}")
    return full, api


def prepare(full: pd.DataFrame) -> pd.DataFrame:
    frame = full.copy()
    if frame["y"].isna().any() or not set(frame["y"].unique()).issubset({"yes", "no"}):
        raise ValueError("target must be yes or no")
    frame["never_contacted"] = (frame["pdays"] == -1).astype(float)
    return frame


def run_table(runs: list[dict], y: np.ndarray) -> list[dict]:
    rows = []
    for run in runs:
        start, end = int(run["start"]), int(run["end"])
        target = y[start:end]
        rows.append(
            {
                "label": run["label"],
                "year": int(run["year"]),
                "month": run["month"],
                "start": start,
                "end": end,
                "n": end - start,
                "n_yes": int(target.sum()),
                "yes_rate": float(target.mean()),
            }
        )
    return rows


def block_summary(
    y: np.ndarray, idx: np.ndarray, months: np.ndarray
) -> dict[str, float | int | str]:
    target = y[idx]
    return {
        "n": int(len(idx)),
        "n_yes": int(target.sum()),
        "yes_rate": float(target.mean()),
        "first_month": str(months[idx[0]]),
        "last_month": str(months[idx[-1]]),
    }


def main() -> None:
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw, api = load_frame()
    frame = prepare(raw)
    y = frame["y"].eq("yes").to_numpy(dtype=int)
    months = frame["month"].astype(str).str.lower().to_numpy()
    if not np.array_equal(months, frame["month"].to_numpy()):
        raise ValueError("month codes are not lowercase")
    runs = month_runs(months)
    if runs[0]["label"] != "2008-may" or runs[-1]["label"] != "2010-nov":
        raise ValueError("year walk does not match the names-file endpoints")
    splits = make_splits(len(frame), SEED)
    assert_split_order(splits, len(frame))
    specs = experiment_specs()
    if tuple(spec["name"] for spec in specs) != EXPERIMENT_ORDER:
        raise ValueError("experiment order drifted from the registered list")
    if REPORTED != "honest" or LEAK_COLUMN in HONEST_COLUMNS:
        raise ValueError("reported protocol changed")

    probabilities: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    for spec in specs:
        print(f"running {spec['name']}", flush=True)
        proba, width, n_iter = fit_experiment(frame, y, splits, spec)
        probabilities[str(spec["name"])] = proba
        results[str(spec["name"])] = {
            "eligible": bool(spec["eligible"]),
            "kind": spec["kind"],
            "train_rows": spec["train"],
            "preprocess_rows": spec["preprocess"],
            "target_fit_rows": spec["target_fit"],
            "uses_duration": bool(spec["duration"]),
            "n_features": width,
            "max_n_iter": n_iter,
            "test": score_proba(y[splits["eval"]], proba),
        }
        print(f"  log_loss={results[str(spec['name'])]['test']['log_loss']:.6f}", flush=True)

    shuffled_train, shuffled_test = train_test_split(
        np.arange(len(frame)),
        test_size=SHUFFLED_TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )
    shuffled_splits = {
        "past": np.sort(shuffled_train),
        "eval": np.sort(shuffled_test),
        "same_period": np.array([], dtype=int),
        "future": np.array([], dtype=int),
        "extra_same": np.array([], dtype=int),
        "extra_future": np.array([], dtype=int),
        "window": np.array([], dtype=int),
    }
    shuffled_proba, shuffled_width, shuffled_iter = fit_experiment(
        frame,
        y,
        shuffled_splits,
        next(spec for spec in specs if spec["name"] == "honest"),
    )
    future_set = set(splits["future"].tolist())
    window_set = set(splits["window"].tolist())
    past_set = set(splits["past"].tolist())

    eval_idx = splits["eval"]
    past_months = set(months[splits["past"]].tolist())
    eval_months = months[eval_idx]
    clusters = date_ids(runs, frame["day"].to_numpy())[eval_idx]
    run_rows = run_table(runs, y)
    duration = frame[LEAK_COLUMN].to_numpy(dtype=float)
    zero = duration == 0
    honest_reliability = reliability(y[eval_idx], probabilities["honest"])
    duration_reliability = reliability(y[eval_idx], probabilities["duration_leak"])
    past_bins = duration_bins(duration[splits["past"]], y[splits["past"]])
    plot_log_loss(RESULTS / "log_loss.png", results)
    plot_time(
        RESULTS / "subscription_over_time.png",
        run_rows,
        len(splits["past"]),
        int(splits["window"][-1]) + 1,
    )
    plot_duration(RESULTS / "duration_bins.png", past_bins)
    plot_reliability(RESULTS / "reliability.png", honest_reliability, duration_reliability)

    def gap(reference: str, candidate: str) -> dict:
        return cluster_bootstrap_gap(
            y[eval_idx],
            probabilities[reference],
            probabilities[candidate],
            clusters,
        )

    payload = {
        "dataset": {
            "name": "Bank Marketing",
            "uci_id": DATASET_ID,
            "doi": DATASET_DOI,
            "rows": int(len(frame)),
            "columns": int(raw.shape[1]),
            "n_yes": int(y.sum()),
            "yes_rate": float(y.mean()),
            "file": "bank-full.csv",
            "separator": ";",
            "ordered_by_date_sentence": ORDER_SENTENCE,
            "names_file_license_sentence": LICENSE_SENTENCE,
            "api_license_field": api.get("license"),
            "api_duration_warning": next(
                var["description"].strip() for var in api["variables"] if var["name"] == "duration"
            ),
            "api_day_column_name": "day_of_week",
            "api_day_description": next(
                var["description"].strip()
                for var in api["variables"]
                if var["name"] == "day_of_week"
            ),
            "day_min": int(frame["day"].min()),
            "day_max": int(frame["day"].max()),
            "day_nunique": int(frame["day"].nunique()),
            "intro_paper_doi": "10.1016/j.dss.2014.03.001",
        },
        "protocol": {
            "positive_label": "yes",
            "time_order": "bank-full.csv file order",
            "year_assignment": "walk forward from May 2008; a lower month index starts a new year",
            "past_fraction": PAST_FRACTION,
            "window_end_fraction": WINDOW_END_FRACTION,
            "eval_fraction_of_window": 0.5,
            "seed": SEED,
            "logistic_c": LOGISTIC_C,
            "solver": "liblinear",
            "max_iter": MAX_ITER,
            "class_weight": None,
            "probability_clip": CLIP,
            "smoothing": SMOOTHING,
            "target_key": "month",
            "excluded_from_honest": [LEAK_COLUMN],
            "honest_columns": list(HONEST_COLUMNS),
            "reported_model": REPORTED,
            "selection_uses_test_scores": False,
            "ineligible": sorted(LEAKS | CONTROLS),
            "campaign_use": False,
        },
        "split": {
            "past_end": int(len(splits["past"])),
            "window_end": int(splits["window"][-1]) + 1,
            "past": block_summary(y, splits["past"], months),
            "window": block_summary(y, splits["window"], months),
            "eval": block_summary(y, eval_idx, months),
            "same_period": block_summary(y, splits["same_period"], months),
            "future": block_summary(y, splits["future"], months),
            "n_eval_rows_with_month_absent_from_past": int(
                np.isin(eval_months, list(past_months), invert=True).sum()
            ),
            "n_eval_rows_with_month_seen_in_past": int(
                np.isin(eval_months, list(past_months)).sum()
            ),
            "past_months": sorted(past_months),
            "eval_is_a_random_half_of_the_middle_window": True,
        },
        "duration": {
            "n_zero": int(zero.sum()),
            "n_yes_when_zero": int(y[zero].sum()),
            "past": class_duration(duration[splits["past"]], y[splits["past"]]),
            "eval": class_duration(duration[eval_idx], y[eval_idx]),
            "past_bins": past_bins,
            "used_in_reported_model": False,
        },
        "month_runs": run_rows,
        "experiments": results,
        "shuffled_split": {
            "comparable_to_heldout": False,
            "used_for_selection": False,
            "test_size": SHUFFLED_TEST_SIZE,
            "seed": SEED,
            "n_features": shuffled_width,
            "max_n_iter": shuffled_iter,
            "n_test_rows": int(len(shuffled_test)),
            "n_test_rows_from_past": int(sum(index in past_set for index in shuffled_test)),
            "n_test_rows_from_window": int(sum(index in window_set for index in shuffled_test)),
            "n_test_rows_from_future": int(sum(index in future_set for index in shuffled_test)),
            "n_train_rows_from_future": int(sum(index in future_set for index in shuffled_train)),
            "test": score_proba(y[shuffled_test], shuffled_proba),
        },
        "reliability": {"honest": honest_reliability, "duration_leak": duration_reliability},
        "bootstrap": {
            "training_rate_minus_honest": gap("training_rate", "honest"),
            "honest_minus_duration_leak": gap("honest", "duration_leak"),
            "honest_minus_same_period_rows": gap("honest", "same_period_rows"),
            "same_period_size_control_minus_same_period_rows": gap(
                "same_period_size_control", "same_period_rows"
            ),
            "honest_minus_future_rows": gap("honest", "future_rows"),
            "future_size_control_minus_future_rows": gap("future_size_control", "future_rows"),
            "target_train_only_minus_target_before_split": gap(
                "target_train_only", "target_before_split"
            ),
            "honest_minus_scaler_before_split": gap("honest", "scaler_before_split"),
            "honest_minus_target_train_only": gap("honest", "target_train_only"),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(py(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {RESULTS / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
