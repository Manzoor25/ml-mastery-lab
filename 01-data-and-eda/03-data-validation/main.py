"""Fail a census extract before scoring it.

The reported schema is fixed from the donor note, the training file, and two
domain caps that the note does not state. Test scores are not used to edit it.
A null-only schema, a donor lower-bound schema, a question-mark ban, and a
training-maximum weight cap are comparison gates. They are not selected
because of the log loss they would have produced.
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
import pandera
import pandera.pandas as pa
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000
LOG_LOSS_CLIP = 1e-12
DONOR_AGE_GT = 16
DONOR_WEIGHT_GT = 1
DONOR_HOURS_GT = 0
AGE_CAP = 120
HOURS_CAP = 168
CORRUPTION_RATE = 0.05
CORRUPTION_SEED = 42
AGE_SHIFT = 100
UNKNOWN_LEVEL = "N/A"
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 7
DATASET_DOI = "10.24432/C5XW20"
API_MISSING_SYMBOL = "NaN"

TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"

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
CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
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
DONOR_FIELD = {
    "workclass": "workclass",
    "education": "education",
    "marital_status": "marital-status",
    "occupation": "occupation",
    "relationship": "relationship",
    "race": "race",
    "sex": "sex",
    "native_country": "native-country",
}

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


def padded(value: str) -> bool:
    return value.startswith(" ") or value.endswith(" ")


def income_allowed(value: str, allowed: set[str]) -> bool:
    return value in allowed


def parse_donor_levels(text: str) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}
    for line in text.splitlines():
        if ":" not in line or line.startswith("|"):
            continue
        name, rest = line.split(":", 1)
        if "continuous" in rest:
            continue
        items = [item.strip() for item in rest.split(",") if item.strip()]
        if items and items[-1].endswith("."):
            items[-1] = items[-1][:-1]
        levels[name.strip()] = items
    return levels


def class_line(text: str) -> str | None:
    for line in text.splitlines():
        if ">50K" in line and not line.startswith("|"):
            return line.strip()
    return None


def read_adult_lines(path: Path, skip_preamble: bool) -> tuple[list[str], str | None]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preamble = None
    if skip_preamble:
        preamble = lines[0]
        if "Cross validator" not in preamble:
            raise ValueError(f"unexpected test preamble: {preamble!r}")
        lines = lines[1:]
    return lines, preamble


def frame_from_lines(lines: list[str], source: str) -> pd.DataFrame:
    rows = []
    for line_number, line in enumerate(lines, start=1):
        parts = line.split(",")
        if len(parts) != len(COLUMNS):
            raise ValueError(f"{source} line {line_number} has {len(parts)} fields")
        rows.append(parts)
    return pd.DataFrame(rows, columns=COLUMNS)


def strip_frame(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.map(lambda value: value.strip())


def parse_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame.copy()
    for column in NUMERIC_COLUMNS:
        parsed[column] = pd.to_numeric(parsed[column], errors="raise")
    return parsed


def without_income_period(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["income"] = cleaned["income"].str.replace(r"\.$", "", regex=True)
    return cleaned


def scoring_labels(income: pd.Series) -> np.ndarray:
    text = income.str.strip().str.replace(r"\.$", "", regex=True)
    unexpected = sorted(set(text) - {"<=50K", ">50K"})
    if unexpected:
        raise ValueError(f"unexpected income labels: {unexpected}")
    return (text == ">50K").to_numpy(dtype=int)


def question_mark_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int((frame[column] == "?").sum()) for column in CATEGORICAL_COLUMNS}


def education_map_report(frame: pd.DataFrame) -> dict[str, object]:
    forward = frame.groupby("education")["education_num"].nunique()
    reverse = frame.groupby("education_num")["education"].nunique()
    one_to_one = bool(forward.max() == 1 and reverse.max() == 1)
    mapping = {}
    if one_to_one:
        mapping = {
            str(level): int(number)
            for level, number in frame.groupby("education")["education_num"].first().items()
        }
    return {
        "max_nums_per_education": int(forward.max()),
        "max_educations_per_num": int(reverse.max()),
        "is_one_to_one": one_to_one,
        "map": mapping,
    }


def corruption_index(n_rows: int, rate: float, seed: int) -> np.ndarray:
    size = int(round(rate * n_rows))
    if size <= 0 or size >= n_rows:
        raise ValueError(f"corruption size {size} is outside 1..{n_rows - 1}")
    return np.random.default_rng(seed).choice(n_rows, size=size, replace=False)


def pointwise_log_loss(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(proba, dtype=float), LOG_LOSS_CLIP, 1.0 - LOG_LOSS_CLIP)
    y_true = np.asarray(y_true, dtype=float)
    return -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))


def paired_mean_interval(
    difference: np.ndarray, n_resamples: int, seed: int
) -> dict[str, float | bool | int]:
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


def summarize_errors(exc: Exception) -> dict[str, object]:
    if not isinstance(exc, pa.errors.SchemaErrors):
        return {
            "passed": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "n_failure_cases": None,
            "n_failed_rows": None,
            "by_check": [],
            "example_values": [],
        }
    cases = exc.failure_cases
    grouped = cases.groupby(["column", "check"], dropna=False).size().reset_index(name="n")
    by_check = []
    for row in grouped.itertuples(index=False):
        column = None if pd.isna(row.column) else str(row.column)
        by_check.append({"column": column, "check": str(row.check), "n": int(row.n)})
    indexes = cases["index"].dropna() if "index" in cases.columns else pd.Series(dtype=object)
    examples = []
    if "failure_case" in cases.columns:
        examples = [
            str(value)[:80] for value in cases["failure_case"].dropna().astype(str).unique()[:5]
        ]
    return {
        "passed": False,
        "error_type": type(exc).__name__,
        "n_failure_cases": int(len(cases)),
        "n_failed_rows": int(indexes.nunique()) if len(indexes) else None,
        "by_check": by_check,
        "example_values": examples,
    }


def run_schema(schema: pa.DataFrameSchema, frame: pd.DataFrame) -> dict[str, object]:
    try:
        schema.validate(frame, lazy=True)
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as exc:
        return summarize_errors(exc)
    return {
        "passed": True,
        "error_type": None,
        "n_failure_cases": 0,
        "n_failed_rows": 0,
        "by_check": [],
        "example_values": [],
    }


def no_padding_schema() -> pa.DataFrameSchema:
    no_space = pa.Check(
        lambda series: ~series.map(padded),
        element_wise=False,
        name="no leading or trailing space",
    )
    return pa.DataFrameSchema(
        {column: pa.Column(str, no_space) for column in COLUMNS},
        strict=True,
    )


def nulls_only_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {column: pa.Column(nullable=False) for column in COLUMNS},
        strict=True,
    )


def donor_lower_bounds_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "age": pa.Column(checks=pa.Check.gt(DONOR_AGE_GT)),
            "fnlwgt": pa.Column(checks=pa.Check.gt(DONOR_WEIGHT_GT)),
            "hours_per_week": pa.Column(checks=pa.Check.gt(DONOR_HOURS_GT)),
        }
    )


def reported_schema(
    levels: dict[str, list[str]],
    question_columns: set[str],
    income_labels: set[str],
    education_map: dict[str, int],
) -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {
        "age": pa.Column(checks=[pa.Check.gt(DONOR_AGE_GT), pa.Check.le(AGE_CAP)]),
        "fnlwgt": pa.Column(checks=pa.Check.gt(DONOR_WEIGHT_GT)),
        "education_num": pa.Column(),
        "capital_gain": pa.Column(checks=pa.Check.ge(0)),
        "capital_loss": pa.Column(checks=pa.Check.ge(0)),
        "hours_per_week": pa.Column(checks=[pa.Check.gt(DONOR_HOURS_GT), pa.Check.le(HOURS_CAP)]),
        "income": pa.Column(checks=pa.Check.isin(sorted(income_labels))),
    }
    for column in CATEGORICAL_COLUMNS:
        allowed = list(levels[DONOR_FIELD[column]])
        if column in question_columns:
            allowed.append("?")
        columns[column] = pa.Column(checks=pa.Check.isin(allowed))
    checks = []
    if education_map:
        checks.append(
            pa.Check(
                lambda frame, mapping=education_map: (
                    frame["education"].map(mapping).eq(frame["education_num"])
                ),
                name="education_num matches training map",
            )
        )
    return pa.DataFrameSchema(columns, checks=checks, strict=True)


def no_question_mark_schema(
    levels: dict[str, list[str]], income_labels: set[str]
) -> pa.DataFrameSchema:
    return reported_schema(levels, set(), income_labels, {})


def train_max_weight_schema(train_max: float) -> pa.DataFrameSchema:
    return pa.DataFrameSchema({"fnlwgt": pa.Column(checks=pa.Check.le(train_max))})


def validate_income_frame(frame: pd.DataFrame) -> dict[str, object]:
    schema = pa.DataFrameSchema({"income": pa.Column(str, pa.Check.isin(["<=50K", ">50K"]))})
    return run_schema(schema, frame)


def make_model() -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), MODEL_NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                MODEL_CATEGORICAL,
            ),
        ]
    )
    classifier = LogisticRegression(
        C=LOGISTIC_C,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=SEED,
    )
    return Pipeline([("columns", transformer), ("model", classifier)])


def score_frame(
    model: Pipeline, frame: pd.DataFrame, y_true: np.ndarray
) -> dict[str, float | int | None]:
    features = frame[MODEL_NUMERIC + MODEL_CATEGORICAL]
    proba = model.predict_proba(features)[:, 1]
    n_iter = int(np.max(model.named_steps["model"].n_iter_))
    if n_iter >= LOGISTIC_MAX_ITER:
        raise RuntimeError(f"logistic regression hit {LOGISTIC_MAX_ITER} iterations")
    scores = {
        "n_scored": int(len(y_true)),
        "log_loss": float(np.mean(pointwise_log_loss(y_true, proba))),
        "accuracy_at_0_5": float(np.mean((proba >= 0.5) == y_true)),
        "n_iter": n_iter,
    }
    try:
        scores["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        scores["roc_auc"] = None
    return scores


def constant_scores(y_true: np.ndarray, rate: float) -> dict[str, float | int | None]:
    proba = np.full(len(y_true), rate, dtype=float)
    return {
        "n_scored": int(len(y_true)),
        "train_positive_rate": rate,
        "log_loss": float(np.mean(pointwise_log_loss(y_true, proba))),
        "accuracy_at_0_5": float(np.mean((proba >= 0.5) == y_true)),
        "roc_auc": None,
    }


def plot_gates(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    labels = [str(row["label"]) for row in rows]
    counts = [int(row["n_failed_rows"] or 0) for row in rows]
    colors = ["#c23b22" if count else "#3174a1" for count in counts]
    positions = np.arange(len(rows))
    ax.barh(positions, counts, color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Failed rows")
    ax.set_title("A pass is zero. The null check passes.")
    for position, count in zip(positions, counts, strict=True):
        ax.text(count, position, " pass" if count == 0 else f" {count}", va="center", fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_log_loss(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    labels = [str(row["label"]) for row in rows]
    values = [float(row["log_loss"]) for row in rows]
    colors = ["#7a7a7a", "#3174a1", "#c23b22", "#d17a22", "#6b4c9a"]
    positions = np.arange(len(rows))
    ax.bar(positions, values, color=colors[: len(rows)])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Log loss")
    ax.set_title("Log loss if a rejected file is scored")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    download(TRAIN_URL, DATA / "adult.data")
    download(TEST_URL, DATA / "adult.test")
    download(NAMES_URL, DATA / "adult.names")

    names_text = (DATA / "adult.names").read_text(encoding="utf-8")
    donor_levels = parse_donor_levels(names_text)
    missing_fields = [field for field in DONOR_FIELD.values() if field not in donor_levels]
    if missing_fields:
        raise ValueError(f"donor note is missing fields: {missing_fields}")

    train_lines, _ = read_adult_lines(DATA / "adult.data", skip_preamble=False)
    test_lines, preamble = read_adult_lines(DATA / "adult.test", skip_preamble=True)
    raw_train = frame_from_lines(train_lines, "adult.data")
    raw_test = frame_from_lines(test_lines, "adult.test")
    stripped_train = strip_frame(raw_train)
    stripped_test = strip_frame(raw_test)
    parsed_train = parse_numeric(stripped_train)
    parsed_test = parse_numeric(stripped_test)
    cleaned_test = without_income_period(parsed_test)

    income_labels = set(parsed_train["income"].unique())
    question_counts = question_mark_counts(parsed_train)
    question_columns = {column for column, count in question_counts.items() if count}
    education = education_map_report(parsed_train)
    education_map = education["map"] if education["is_one_to_one"] else {}

    changed = corruption_index(len(cleaned_test), CORRUPTION_RATE, CORRUPTION_SEED)
    age_corrupted = cleaned_test.copy()
    age_corrupted.loc[age_corrupted.index[changed], "age"] = (
        age_corrupted.loc[age_corrupted.index[changed], "age"] + AGE_SHIFT
    )
    category_corrupted = cleaned_test.copy()
    category_corrupted.loc[category_corrupted.index[changed], "workclass"] = UNKNOWN_LEVEL

    schemas = {
        "no_padding": no_padding_schema(),
        "nulls_only": nulls_only_schema(),
        "donor_lower_bounds": donor_lower_bounds_schema(),
        "reported": reported_schema(donor_levels, question_columns, income_labels, education_map),
        "no_question_mark": no_question_mark_schema(donor_levels, income_labels),
        "train_max_weight": train_max_weight_schema(float(parsed_train["fnlwgt"].max())),
    }
    extracts = {
        "raw_train": raw_train,
        "raw_test": raw_test,
        "stripped_train": parsed_train,
        "stripped_test": parsed_test,
        "cleaned_test": cleaned_test,
        "age_plus_100": age_corrupted,
        "unknown_workclass": category_corrupted,
    }
    gate_plan = [
        ("raw_train", "no_padding", "Raw train, no padding"),
        ("raw_test", "no_padding", "Raw test, no padding"),
        ("stripped_train", "reported", "Stripped train, reported"),
        ("stripped_test", "reported", "Stripped test, reported"),
        ("stripped_test", "nulls_only", "Stripped test, nulls only"),
        ("cleaned_test", "reported", "Cleaned test, reported"),
        ("cleaned_test", "nulls_only", "Cleaned test, nulls only"),
        ("age_plus_100", "donor_lower_bounds", "Age +100, lower bounds"),
        ("age_plus_100", "reported", "Age +100, reported"),
        ("unknown_workclass", "reported", "Unknown category, reported"),
        ("stripped_train", "no_question_mark", "Train, no question mark"),
        ("cleaned_test", "train_max_weight", "Cleaned test, train-max weight"),
    ]
    gates = []
    for extract_name, schema_name, label in gate_plan:
        result = run_schema(schemas[schema_name], extracts[extract_name])
        result["extract"] = extract_name
        result["schema"] = schema_name
        result["label"] = label
        gates.append(result)

    y_train = scoring_labels(parsed_train["income"])
    y_test = scoring_labels(raw_test["income"])
    if not np.array_equal(y_test, scoring_labels(cleaned_test["income"])):
        raise RuntimeError("label repair changed which rows are positive")
    model = make_model()
    model.fit(parsed_train[MODEL_NUMERIC + MODEL_CATEGORICAL], y_train)
    rate = float(np.mean(y_train))
    raw_features = raw_test.copy()
    for column in NUMERIC_COLUMNS:
        raw_features[column] = pd.to_numeric(raw_features[column], errors="raise")
    scored = {
        "constant_rate": constant_scores(y_test, rate),
        "cleaned_test": score_frame(model, cleaned_test, y_test),
        "padded_features": score_frame(model, raw_features, y_test),
        "age_plus_100": score_frame(model, age_corrupted, y_test),
        "unknown_workclass": score_frame(model, category_corrupted, y_test),
    }
    cleaned_loss = pointwise_log_loss(
        y_test,
        model.predict_proba(cleaned_test[MODEL_NUMERIC + MODEL_CATEGORICAL])[:, 1],
    )
    comparisons = {}
    for name in ("padded_features", "age_plus_100", "unknown_workclass"):
        frame = {
            "padded_features": raw_features,
            "age_plus_100": age_corrupted,
            "unknown_workclass": category_corrupted,
        }[name]
        loss = pointwise_log_loss(
            y_test,
            model.predict_proba(frame[MODEL_NUMERIC + MODEL_CATEGORICAL])[:, 1],
        )
        comparisons[name] = paired_mean_interval(
            loss - cleaned_loss, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
        )

    above_cap = int((age_corrupted["age"] > AGE_CAP).sum())
    still_within_cap = int(
        np.sum(age_corrupted.loc[age_corrupted.index[changed], "age"] <= AGE_CAP)
    )
    metrics = {
        "elapsed_seconds": time.perf_counter() - started,
        "pandera_version": pandera.__version__,
        "data": {
            "doi": DATASET_DOI,
            "api_missing_symbol": API_MISSING_SYMBOL,
            "names_missing_token": "?",
            "names_class_line": class_line(names_text),
            "test_preamble": preamble,
            "n_train": int(len(raw_train)),
            "n_test": int(len(raw_test)),
            "n_train_plus_test": int(len(raw_train) + len(raw_test)),
            "names_instance_claim": 48842,
            "names_train_claim": 32561,
            "names_test_claim": 16281,
        },
        "protocol": {
            "seed": SEED,
            "logistic_c": LOGISTIC_C,
            "logistic_max_iter": LOGISTIC_MAX_ITER,
            "log_loss_clip": LOG_LOSS_CLIP,
            "donor_age_gt": DONOR_AGE_GT,
            "donor_weight_gt": DONOR_WEIGHT_GT,
            "donor_hours_gt": DONOR_HOURS_GT,
            "age_cap": AGE_CAP,
            "hours_cap": HOURS_CAP,
            "age_cap_is_in_donor_note": False,
            "hours_cap_is_in_donor_note": False,
            "corruption_rate": CORRUPTION_RATE,
            "corruption_seed": CORRUPTION_SEED,
            "age_shift": AGE_SHIFT,
            "unknown_level": UNKNOWN_LEVEL,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "reported_model_features_exclude": ["fnlwgt", "education", "race"],
            "question_mark_allowed_because": "donor note maps Unknown to ?, and training uses it",
        },
        "inventory": {
            "income_labels_train": sorted(income_labels),
            "income_labels_stripped_test": sorted(parsed_test["income"].unique()),
            "question_mark_counts_train": question_counts,
            "education_map": education,
            "train_fnlwgt_max": float(parsed_train["fnlwgt"].max()),
            "test_fnlwgt_max": float(parsed_test["fnlwgt"].max()),
            "train_capital_gain_max": float(parsed_train["capital_gain"].max()),
            "test_capital_gain_max": float(parsed_test["capital_gain"].max()),
            "n_corrupted_rows": int(len(changed)),
            "n_age_rows_above_cap": above_cap,
            "n_corrupted_age_rows_still_within_cap": still_within_cap,
            "n_raw_train_padded_rows": int(raw_train.map(padded).any(axis=1).sum()),
            "n_raw_test_padded_rows": int(raw_test.map(padded).any(axis=1).sum()),
            "n_null_cells_raw_test": int(raw_test.isna().sum().sum()),
        },
        "gates": gates,
        "scores": scored,
        "paired_log_loss_gaps": comparisons,
    }
    payload = json_ready(metrics)
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_rows = [
        row
        for row in gates
        if (row["extract"], row["schema"])
        in {
            ("raw_test", "no_padding"),
            ("stripped_test", "reported"),
            ("stripped_test", "nulls_only"),
            ("cleaned_test", "reported"),
            ("age_plus_100", "donor_lower_bounds"),
            ("age_plus_100", "reported"),
            ("stripped_train", "no_question_mark"),
            ("cleaned_test", "train_max_weight"),
        }
    ]
    plot_gates(RESULTS / "schema_gates.png", plot_rows)
    plot_log_loss(
        RESULTS / "log_loss.png",
        [
            {"label": "Training rate", "log_loss": scored["constant_rate"]["log_loss"]},
            {"label": "Cleaned test", "log_loss": scored["cleaned_test"]["log_loss"]},
            {"label": "Padded features", "log_loss": scored["padded_features"]["log_loss"]},
            {"label": "Age +100", "log_loss": scored["age_plus_100"]["log_loss"]},
            {"label": "Unknown category", "log_loss": scored["unknown_workclass"]["log_loss"]},
        ],
    )
    print(
        json.dumps(
            {
                "elapsed_seconds": payload["elapsed_seconds"],
                "cleaned_log_loss": payload["scores"]["cleaned_test"]["log_loss"],
                "padded_log_loss": payload["scores"]["padded_features"]["log_loss"],
                "gates_failed": sum(not row["passed"] for row in gates),
            }
        )
    )


if __name__ == "__main__":
    main()
