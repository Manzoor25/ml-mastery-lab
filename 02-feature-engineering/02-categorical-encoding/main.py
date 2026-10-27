"""Categorical encodings for 30-day diabetic readmission.

The reported model is the eligible experiment with the lowest group-CV log loss.
A target encoding fit on the same rows it scores is recorded and is not eligible.
The holdout is not used to add, drop, or replace an encoding.

This is a teaching model on a public research extract. It is not a clinical tool.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
LOGISTIC_C = 1.0
MAX_ITER = 200
CLIP = 1e-6
MIN_FREQUENCY = 20
SMOOTHING = 20.0
HASH_WIDE = 16
HASH_NARROW = 4
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 7
DATASET_ID = 296
DATASET_DOI = "10.24432/C5230J"
DATA_URL = "https://archive.ics.uci.edu/static/public/296/data.csv"
API_URL = "https://archive.ics.uci.edu/api/dataset?id=296"
EXPECTED_ROWS = 101766
EARLY = "<30"
AGE_RE = re.compile(r"\[(\d+)-(\d+)\)")
MISSING = "missing"

NUMERIC = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)
LOW_CARD = (
    "age",
    "admission_type_id",
    "admission_source_id",
    "payer_code",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
)
HIGH_CARD = ("diag_1", "diag_2", "diag_3", "medical_specialty")
EXCLUDED = (
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "weight",
    "discharge_disposition_id",
    "examide",
    "citoglipton",
)
EXPERIMENT_ORDER = (
    "training_rate",
    "low_card_only",
    "age_ordinal",
    "one_hot",
    "one_hot_min_frequency",
    "ordinal_codes",
    "frequency",
    "hash_16",
    "hash_4",
    "target_oof_m20",
    "target_oof_m0",
    "target_insample_m0",
)
INELIGIBLE = frozenset({"target_insample_m0"})

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def select_experiment(scores: dict[str, dict]) -> str:
    eligible = {name: row for name, row in scores.items() if row["eligible"]}
    if not eligible:
        raise ValueError("no eligible experiment")
    return min(
        eligible,
        key=lambda name: (
            eligible[name]["cv_log_loss_mean"],
            eligible[name]["n_features"],
            name,
        ),
    )


def as_tokens(series: pd.Series) -> np.ndarray:
    return series.astype("string").fillna(MISSING).astype(str).to_numpy()


def column_tokens(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([as_tokens(frame[column]) for column in columns])


def age_bounds(token: str) -> tuple[int, int]:
    match = AGE_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"unexpected age bracket: {token}")
    lower = int(match.group(1))
    upper = int(match.group(2))
    if upper != lower + 10:
        raise ValueError(f"age bracket is not a 10-year bin: {token}")
    return lower, upper


def age_codes(tokens: np.ndarray) -> np.ndarray:
    coded = [age_bounds(token)[0] / 10 for token in tokens]
    return np.asarray(coded, dtype=float).reshape(-1, 1)


class GroupTargetEncoder:
    """Smoothed category means. Unseen categories get the training mean.

    in-sample transform uses every training row, including the row being encoded.
    fit_transform_oof holds out the row's patient group.
    """

    def __init__(self, smoothing: float) -> None:
        self.smoothing = smoothing

    def fit(self, tokens: np.ndarray, y: np.ndarray) -> GroupTargetEncoder:
        y = np.asarray(y, dtype=float)
        self.global_mean_ = float(y.mean())
        self.maps_ = [self._means(tokens[:, j], y) for j in range(tokens.shape[1])]
        self.n_features_in_ = tokens.shape[1]
        return self

    def _means(self, column: np.ndarray, y: np.ndarray) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for value, target in zip(column.tolist(), y.tolist(), strict=True):
            key = str(value)
            sums[key] = sums.get(key, 0.0) + float(target)
            counts[key] = counts.get(key, 0) + 1
        smoothing = self.smoothing
        global_mean = self.global_mean_
        return {
            key: (sums[key] + smoothing * global_mean) / (counts[key] + smoothing) for key in counts
        }

    def transform(self, tokens: np.ndarray) -> np.ndarray:
        out = np.empty(tokens.shape, dtype=float)
        for j, mapping in enumerate(self.maps_):
            out[:, j] = [mapping.get(str(value), self.global_mean_) for value in tokens[:, j]]
        return out

    def fit_transform_oof(
        self,
        tokens: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = CV_SPLITS,
    ) -> np.ndarray:
        out = np.empty(tokens.shape, dtype=float)
        splitter = GroupKFold(n_splits=n_splits)
        for inner_train, inner_val in splitter.split(tokens, y, groups):
            encoder = GroupTargetEncoder(self.smoothing).fit(tokens[inner_train], y[inner_train])
            out[inner_val] = encoder.transform(tokens[inner_val])
        self.fit(tokens, y)
        return out


class FrequencyEncoder:
    def fit(self, tokens: np.ndarray, y: np.ndarray | None = None) -> FrequencyEncoder:
        del y
        n_rows = tokens.shape[0]
        self.maps_ = []
        for j in range(tokens.shape[1]):
            values, counts = np.unique(tokens[:, j], return_counts=True)
            self.maps_.append(
                {
                    str(value): float(count) / n_rows
                    for value, count in zip(values, counts, strict=True)
                }
            )
        return self

    def transform(self, tokens: np.ndarray) -> np.ndarray:
        out = np.zeros(tokens.shape, dtype=float)
        for j, mapping in enumerate(self.maps_):
            out[:, j] = [mapping.get(str(value), 0.0) for value in tokens[:, j]]
        return out


class ColumnHasher:
    def __init__(self, n_features: int) -> None:
        self.n_features = n_features

    def fit(self, tokens: np.ndarray, y: np.ndarray | None = None) -> ColumnHasher:
        del y
        self.n_columns_ = int(tokens.shape[1])
        return self

    def transform(self, tokens: np.ndarray) -> sparse.csr_matrix:
        parts = []
        for j in range(self.n_columns_):
            hasher = FeatureHasher(n_features=self.n_features, input_type="string")
            samples = [[f"c{j}={value}"] for value in tokens[:, j]]
            parts.append(hasher.transform(samples))
        return sparse.hstack(parts, format="csr")


def stack(parts: list[np.ndarray | sparse.spmatrix]) -> sparse.csr_matrix:
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
    scored = {
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "average_precision": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "accuracy_at_0_5": float(accuracy_score(y, proba >= 0.5)),
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


def low_card_tokens(frame: pd.DataFrame, drop_age: bool) -> np.ndarray:
    columns = tuple(column for column in LOW_CARD if not (drop_age and column == "age"))
    return column_tokens(frame, columns)


def scaled_numeric(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train), scaler.transform(other)


def high_card_views(
    kind: str,
    train_tokens: np.ndarray,
    other_tokens: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    smoothing: float = SMOOTHING,
    n_hash: int = HASH_WIDE,
    leaky_train: np.ndarray | None = None,
    leaky_other: np.ndarray | None = None,
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray | sparse.csr_matrix]:
    if kind == "none":
        empty_train = np.zeros((train_tokens.shape[0], 0))
        empty_other = np.zeros((other_tokens.shape[0], 0))
        return empty_train, empty_other
    if kind == "one_hot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
        return encoder.fit_transform(train_tokens), encoder.transform(other_tokens)
    if kind == "one_hot_min":
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float64,
            min_frequency=MIN_FREQUENCY,
        )
        return encoder.fit_transform(train_tokens), encoder.transform(other_tokens)
    if kind == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )
        coded_train = encoder.fit_transform(train_tokens)
        coded_other = encoder.transform(other_tokens)
        return scaled_numeric(coded_train, coded_other)
    if kind == "frequency":
        encoder = FrequencyEncoder().fit(train_tokens)
        return encoder.transform(train_tokens), encoder.transform(other_tokens)
    if kind == "hash":
        encoder = ColumnHasher(n_hash).fit(train_tokens)
        return encoder.transform(train_tokens), encoder.transform(other_tokens)
    if kind == "target_oof":
        encoder = GroupTargetEncoder(smoothing)
        train_view = encoder.fit_transform_oof(train_tokens, y_train, groups_train)
        return train_view, encoder.transform(other_tokens)
    if kind == "target_leaky":
        if leaky_train is None or leaky_other is None:
            raise ValueError("leaky target encoding needs a precomputed matrix")
        return leaky_train, leaky_other
    raise ValueError(f"unknown high-card kind: {kind}")


def design_matrices(
    train_frame: pd.DataFrame,
    other_frame: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    spec: dict,
    leaky_train: np.ndarray | None = None,
    leaky_other: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    drop_age = spec["kind"] == "age_ordinal"
    num_train, num_other = scaled_numeric(
        train_frame[list(NUMERIC)].to_numpy(dtype=float),
        other_frame[list(NUMERIC)].to_numpy(dtype=float),
    )
    low_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    low_train = low_encoder.fit_transform(low_card_tokens(train_frame, drop_age))
    low_other = low_encoder.transform(low_card_tokens(other_frame, drop_age))
    parts_train: list[np.ndarray | sparse.spmatrix] = [num_train, low_train]
    parts_other: list[np.ndarray | sparse.spmatrix] = [num_other, low_other]
    if drop_age:
        age_train = age_codes(as_tokens(train_frame["age"]))
        age_other = age_codes(as_tokens(other_frame["age"]))
        scaled_train, scaled_other = scaled_numeric(age_train, age_other)
        parts_train.append(scaled_train)
        parts_other.append(scaled_other)
    if spec["kind"] != "age_ordinal":
        high_train, high_other = high_card_views(
            spec["kind"],
            column_tokens(train_frame, HIGH_CARD),
            column_tokens(other_frame, HIGH_CARD),
            y_train,
            groups_train,
            smoothing=spec.get("smoothing", SMOOTHING),
            n_hash=spec.get("n_hash", HASH_WIDE),
            leaky_train=leaky_train,
            leaky_other=leaky_other,
        )
        parts_train.append(high_train)
        parts_other.append(high_other)
    return stack(parts_train), stack(parts_other)


def predict_rate(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    return np.full(n_rows, float(y_train.mean()), dtype=float)


def evaluate_split(
    train_frame: pd.DataFrame,
    other_frame: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    spec: dict,
    leaky_train: np.ndarray | None = None,
    leaky_other: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    if spec["kind"] == "rate":
        return clip_proba(predict_rate(y_train, len(other_frame))), 0
    if spec["kind"] == "target_leaky":
        if leaky_train is None or leaky_other is None:
            raise ValueError("leaky target encoding needs precomputed train and other blocks")
        if len(leaky_train) != len(train_frame) or len(leaky_other) != len(other_frame):
            raise ValueError("leaky target-encoding blocks do not match the row split")
    train_matrix, other_matrix = design_matrices(
        train_frame,
        other_frame,
        y_train,
        groups_train,
        spec,
        leaky_train=leaky_train,
        leaky_other=leaky_other,
    )
    model = logistic()
    model.fit(train_matrix, y_train)
    return clip_proba(model.predict_proba(other_matrix)[:, 1]), int(train_matrix.shape[1])


def cross_validate_experiment(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    spec: dict,
) -> dict[str, float | list[float] | int]:
    folds = list(GroupKFold(n_splits=CV_SPLITS).split(frame, y, groups))
    leaky_all = None
    if spec["kind"] == "target_leaky":
        leaky_all = (
            GroupTargetEncoder(spec["smoothing"])
            .fit(column_tokens(frame, HIGH_CARD), y)
            .transform(column_tokens(frame, HIGH_CARD))
        )
    losses = []
    aucs = []
    precisions = []
    positive_rates = []
    widths = []
    for train_idx, valid_idx in folds:
        proba, width = evaluate_split(
            frame.iloc[train_idx],
            frame.iloc[valid_idx],
            y[train_idx],
            groups[train_idx],
            spec,
            leaky_train=None if leaky_all is None else leaky_all[train_idx],
            leaky_other=None if leaky_all is None else leaky_all[valid_idx],
        )
        scored = score_proba(y[valid_idx], proba)
        losses.append(scored["log_loss"])
        aucs.append(scored["roc_auc"])
        precisions.append(scored["average_precision"])
        positive_rates.append(float(y[valid_idx].mean()))
        widths.append(width)
    finite_auc = [value for value in aucs if value is not None]
    return {
        "cv_log_loss_mean": float(np.mean(losses)),
        "cv_log_loss_std": float(np.std(losses, ddof=1)),
        "cv_log_loss_folds": [float(value) for value in losses],
        "cv_roc_auc_mean": float(np.mean(finite_auc)) if finite_auc else None,
        "cv_average_precision_mean": float(np.mean(precisions)),
        "cv_positive_rate_folds": positive_rates,
        "cv_n_features_max": int(max(widths)),
    }


def level_summary(tokens: np.ndarray, min_count: int) -> dict[str, int]:
    values, counts = np.unique(tokens, return_counts=True)
    rare = counts < min_count
    return {
        "n_levels": int(values.size),
        "n_levels_below_min_frequency": int(rare.sum()),
        "n_rows_in_levels_below_min_frequency": int(counts[rare].sum()) if rare.any() else 0,
        "n_singleton_levels": int((counts == 1).sum()),
    }


def hash_bins(tokens: np.ndarray, n_features: int) -> dict[str, int]:
    shared = 0
    occupied = 0
    for j in range(tokens.shape[1]):
        hasher = FeatureHasher(n_features=n_features, input_type="string")
        unique = [[f"c{j}={value}"] for value in np.unique(tokens[:, j])]
        counts = np.diff(hasher.transform(unique).tocsc().indptr)
        occupied += int((counts > 0).sum())
        shared += int((counts > 1).sum())
    return {
        "n_features_per_column": n_features,
        "n_occupied_bins": occupied,
        "n_bins_with_two_or_more_levels": shared,
    }


def disposition_table(frame: pd.DataFrame) -> list[dict[str, float | int]]:
    codes = frame["discharge_disposition_id"].to_numpy()
    early = frame["readmitted"].eq(EARLY).to_numpy()
    rows = []
    for code in np.unique(codes):
        mask = codes == code
        rows.append(
            {
                "discharge_disposition_id": int(code),
                "n": int(mask.sum()),
                "early_rate": float(early[mask].mean()),
            }
        )
    return sorted(rows, key=lambda row: (-row["n"], row["discharge_disposition_id"]))


def missing_counts(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, int]:
    return {column: int(frame[column].isna().sum()) for column in columns}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "ml-mastery-lab/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def load_frame() -> pd.DataFrame:
    download(DATA_URL, DATA / "diabetes.csv")
    download(API_URL, DATA / "uci_api.json")
    frame = pd.read_csv(DATA / "diabetes.csv", low_memory=False)
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(frame)}")
    required = set(NUMERIC) | set(LOW_CARD) | set(HIGH_CARD) | set(EXCLUDED) | {"readmitted"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for column in ("examide", "citoglipton"):
        if frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"{column} was expected to be constant")
    if frame["encounter_id"].duplicated().any():
        raise ValueError("encounter_id is not unique")
    return frame


def experiment_specs() -> list[dict]:
    return [
        {"name": "training_rate", "kind": "rate", "eligible": True},
        {"name": "low_card_only", "kind": "none", "eligible": True},
        {"name": "age_ordinal", "kind": "age_ordinal", "eligible": True},
        {"name": "one_hot", "kind": "one_hot", "eligible": True},
        {"name": "one_hot_min_frequency", "kind": "one_hot_min", "eligible": True},
        {"name": "ordinal_codes", "kind": "ordinal", "eligible": True},
        {"name": "frequency", "kind": "frequency", "eligible": True},
        {"name": "hash_16", "kind": "hash", "n_hash": HASH_WIDE, "eligible": True},
        {"name": "hash_4", "kind": "hash", "n_hash": HASH_NARROW, "eligible": True},
        {"name": "target_oof_m20", "kind": "target_oof", "smoothing": SMOOTHING, "eligible": True},
        {"name": "target_oof_m0", "kind": "target_oof", "smoothing": 0.0, "eligible": True},
        {
            "name": "target_insample_m0",
            "kind": "target_leaky",
            "smoothing": 0.0,
            "eligible": False,
        },
    ]


def patient_bootstrap_gap(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | int | str]:
    buckets: dict[int, list[int]] = {}
    for index, group in enumerate(groups.tolist()):
        buckets.setdefault(int(group), []).append(index)
    patients = np.array(list(buckets), dtype=int)
    row_lists = [np.array(buckets[patient], dtype=int) for patient in patients]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gaps = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for draw in range(BOOTSTRAP_RESAMPLES):
        chosen = rng.integers(0, len(patients), size=len(patients))
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
        "seed": BOOTSTRAP_SEED,
        "unit": "patient",
        "definition": "reference log loss minus candidate log loss",
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


def plot_cv(path: Path, scores: dict[str, dict], selected: str) -> None:
    names = [name for name in EXPERIMENT_ORDER if name in scores and name != "training_rate"]
    baseline = scores["training_rate"]["cv_log_loss_mean"]
    gains = [baseline - scores[name]["cv_log_loss_mean"] for name in names]
    errors = [scores[name]["cv_log_loss_std"] for name in names]
    colors = []
    for name in names:
        if name == selected:
            colors.append("#54A24B")
        elif name in INELIGIBLE:
            colors.append("#E45756")
        else:
            colors.append("#4C78A8")
    fig, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    bars = axis.bar(names, gains, yerr=errors, color=colors, capsize=3)
    for bar, name in zip(bars, names, strict=True):
        if name in INELIGIBLE:
            bar.set_hatch("//")
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_ylabel("CV log-loss improvement over the training rate")
    axis.set_xlabel("Encoding")
    axis.set_title(f"Patient-grouped CV. Selected: {selected}. Hatched bar was not eligible.")
    axis.tick_params(axis="x", rotation=55)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cardinality(path: Path, summaries: dict[str, dict]) -> None:
    names = list(HIGH_CARD)
    levels = [summaries[name]["n_levels"] for name in names]
    rare = [summaries[name]["n_levels_below_min_frequency"] for name in names]
    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    positions = np.arange(len(names))
    axis.bar(positions - 0.18, levels, width=0.36, label="Training levels", color="#4C78A8")
    axis.bar(
        positions + 0.18,
        rare,
        width=0.36,
        label=f"Levels with fewer than {MIN_FREQUENCY} rows",
        color="#F58518",
    )
    axis.set_xticks(positions, names)
    axis.set_ylabel("Distinct codes in the training patients")
    axis.set_xlabel("Column")
    axis.set_title("High-cardinality columns are mostly rare codes")
    axis.legend(frameon=False)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_reliability(path: Path, rows: list[dict], name: str) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#333333", linewidth=1, label="perfect")
    if rows:
        axis.scatter(
            [row["mean_predicted"] for row in rows],
            [row["observed_rate"] for row in rows],
            s=[max(20, row["n"] / 30) for row in rows],
            color="#4C78A8",
            label="holdout bins",
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed early-readmission rate")
    axis.set_title(f"Holdout reliability: {name}")
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


def main() -> None:
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = load_frame()
    y = raw["readmitted"].eq(EARLY).to_numpy(dtype=int)
    groups = raw["patient_nbr"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(raw, y, groups))
    train = raw.iloc[train_idx].reset_index(drop=True)
    test = raw.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]
    if set(groups_train).intersection(groups_test):
        raise ValueError("a patient is in both the train and test groups")

    specs = experiment_specs()
    if tuple(spec["name"] for spec in specs) != EXPERIMENT_ORDER:
        raise ValueError("experiment order drifted from the registered list")
    results: dict[str, dict] = {}
    holdout_proba: dict[str, np.ndarray] = {}
    for spec in specs:
        print(f"running {spec['name']}", flush=True)
        cv_scores = cross_validate_experiment(train, y_train, groups_train, spec)
        leaky_train = None
        leaky_test = None
        if spec["kind"] == "target_leaky":
            encoder = GroupTargetEncoder(spec["smoothing"]).fit(
                column_tokens(train, HIGH_CARD), y_train
            )
            leaky_train = encoder.transform(column_tokens(train, HIGH_CARD))
            leaky_test = encoder.transform(column_tokens(test, HIGH_CARD))
        proba, width = evaluate_split(
            train,
            test,
            y_train,
            groups_train,
            spec,
            leaky_train=leaky_train,
            leaky_other=leaky_test,
        )
        holdout_proba[spec["name"]] = proba
        results[spec["name"]] = {
            "eligible": spec["eligible"],
            "kind": spec["kind"],
            "n_features": width,
            **cv_scores,
            "test": score_proba(y_test, proba),
        }
        print(
            f"  cv={results[spec['name']]['cv_log_loss_mean']:.6f} "
            f"test={results[spec['name']]['test']['log_loss']:.6f}",
            flush=True,
        )

    selected = select_experiment(results)
    row_train, row_test = train_test_split(
        np.arange(len(raw)),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    random_overlap = len(set(groups[row_train]).intersection(groups[row_test]))
    selected_spec = next(spec for spec in specs if spec["name"] == selected)
    random_proba, _ = evaluate_split(
        raw.iloc[row_train],
        raw.iloc[row_test],
        y[row_train],
        groups[row_train],
        selected_spec,
    )
    train_tokens = column_tokens(train, HIGH_CARD)
    summaries = {
        column: level_summary(train_tokens[:, j], MIN_FREQUENCY)
        for j, column in enumerate(HIGH_CARD)
    }
    dispositions = disposition_table(raw)
    zero_dispositions = [row for row in dispositions if row["early_rate"] == 0.0]
    reliability_rows = reliability(y_test, holdout_proba[selected])
    rival_name = min(
        (name for name in results if name != selected and results[name]["eligible"]),
        key=lambda name: (
            results[name]["cv_log_loss_mean"],
            results[name]["n_features"],
            name,
        ),
    )
    plot_cv(RESULTS / "cv_log_loss.png", results, selected)
    plot_cardinality(RESULTS / "cardinality.png", summaries)
    plot_reliability(RESULTS / "reliability.png", reliability_rows, selected)

    payload = {
        "dataset": {
            "name": "Diabetes 130-US Hospitals for Years 1999-2008",
            "uci_id": DATASET_ID,
            "doi": DATASET_DOI,
            "rows": int(len(raw)),
            "columns": int(raw.shape[1]),
            "n_patients": int(pd.Series(groups).nunique()),
            "n_early_readmission": int(y.sum()),
            "early_rate": float(y.mean()),
            "license_note": (
                "UCI API returned no license field. The repository donation policy "
                "distributes approved datasets under CC BY 4.0."
            ),
        },
        "protocol": {
            "positive_label": EARLY,
            "group": "patient_nbr",
            "test_size": TEST_SIZE,
            "seed": SEED,
            "cv_splits": CV_SPLITS,
            "logistic_c": LOGISTIC_C,
            "solver": "liblinear",
            "max_iter": MAX_ITER,
            "class_weight": None,
            "probability_clip": CLIP,
            "min_frequency": MIN_FREQUENCY,
            "smoothing": SMOOTHING,
            "hash_features_per_column": {"hash_16": HASH_WIDE, "hash_4": HASH_NARROW},
            "excluded_columns": list(EXCLUDED),
            "high_card_columns": list(HIGH_CARD),
            "selection": "lowest group-CV log loss, then fewer features, then name",
            "selection_uses_test_scores": False,
            "ineligible": sorted(INELIGIBLE),
            "reported_model": selected,
            "clinical_use": False,
        },
        "split": {
            "n_train_rows": int(len(train)),
            "n_test_rows": int(len(test)),
            "n_train_patients": int(pd.Series(groups_train).nunique()),
            "n_test_patients": int(pd.Series(groups_test).nunique()),
            "n_patients_in_both": 0,
            "train_early_rate": float(y_train.mean()),
            "test_early_rate": float(y_test.mean()),
            "train_majority_accuracy": float(1.0 - y_train.mean()),
            "test_majority_accuracy": float(1.0 - y_test.mean()),
        },
        "excluded": {
            "weight_missing": int(raw["weight"].isna().sum()),
            "race_missing": int(raw["race"].isna().sum()),
            "gender_unknown_or_invalid": int(raw["gender"].eq("Unknown/Invalid").sum()),
            "gender_not_used": True,
            "race_not_used": True,
            "constant_columns": ["examide", "citoglipton"],
        },
        "missing_before_token": missing_counts(
            raw, HIGH_CARD + ("payer_code", "max_glu_serum", "A1Cresult", "weight")
        ),
        "high_card_train": summaries,
        "hash_bins_train": {
            "hash_16": hash_bins(train_tokens, HASH_WIDE),
            "hash_4": hash_bins(train_tokens, HASH_NARROW),
        },
        "discharge_disposition": {
            "dropped_before_fit": True,
            "rows": dispositions,
            "zero_early_rate_ids": [row["discharge_disposition_id"] for row in zero_dispositions],
            "n_rows_with_zero_early_rate_id": int(sum(row["n"] for row in zero_dispositions)),
        },
        "experiments": results,
        "random_row_split": {
            "used_for_selection": False,
            "n_train_rows": int(len(row_train)),
            "n_test_rows": int(len(row_test)),
            "n_patients_in_both": int(random_overlap),
            "selected_pipeline": selected,
            "test": score_proba(y[row_test], random_proba),
        },
        "bootstrap": {
            "training_rate_minus_selected": patient_bootstrap_gap(
                y_test,
                holdout_proba["training_rate"],
                holdout_proba[selected],
                groups_test,
            ),
            "low_card_only_minus_selected": patient_bootstrap_gap(
                y_test,
                holdout_proba["low_card_only"],
                holdout_proba[selected],
                groups_test,
            ),
            "rival_minus_selected": patient_bootstrap_gap(
                y_test,
                holdout_proba[rival_name],
                holdout_proba[selected],
                groups_test,
            ),
            "rival": rival_name,
        },
        "reliability": reliability_rows,
        "gaps": {
            "training_rate_cv_minus_selected": results["training_rate"]["cv_log_loss_mean"]
            - results[selected]["cv_log_loss_mean"],
            "low_card_cv_minus_selected": results["low_card_only"]["cv_log_loss_mean"]
            - results[selected]["cv_log_loss_mean"],
            "leaky_cv_minus_leaky_test": results["target_insample_m0"]["cv_log_loss_mean"]
            - results["target_insample_m0"]["test"]["log_loss"],
            "oof_m0_cv_minus_oof_m0_test": results["target_oof_m0"]["cv_log_loss_mean"]
            - results["target_oof_m0"]["test"]["log_loss"],
            "oof_m20_cv_minus_oof_m20_test": results["target_oof_m20"]["cv_log_loss_mean"]
            - results["target_oof_m20"]["test"]["log_loss"],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
            "scipy": scipy.__version__,
        },
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(py(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"reported_model": selected, "rival": rival_name}, indent=2), flush=True)


if __name__ == "__main__":
    main()
