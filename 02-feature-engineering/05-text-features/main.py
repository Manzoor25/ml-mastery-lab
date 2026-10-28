"""Bag-of-words, TF-IDF, n-grams, and a latent semantic embedding.

The experiment list, the Ekman label map, and the official train/dev/test
files are fixed before any classifier is fit. Development log loss chooses
the eligible recipe. The official test split is not used to add, drop, or
replace a feature. A vectorizer fit on the test text is scored and is not
eligible.

This is a teaching model on public comments. It is not a moderation tool.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

SEED = 42
BOOTSTRAP_SEED = 7
BOOTSTRAP_RESAMPLES = 1000
MAX_FEATURES = 10000
MIN_DF = 2
SVD_COMPONENTS = 100
SVD_ITER = 5
LOGISTIC_C = 1.0
MAX_ITER = 200
TOL = 0.001
TOKEN_PATTERN = r"(?u)\b\w\w+\b"
LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
EXPERIMENT_ORDER = (
    "training_rate",
    "bow_unigram",
    "tfidf_unigram",
    "tfidf_bigram",
    "tfidf_char",
    "lsa_100",
    "leaky_tfidf_fit_on_all",
)
LEAKS = frozenset({"leaky_tfidf_fit_on_all"})
EMOTION_NAMES = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)
EKMAN_GROUPS = {
    "anger": ("anger", "annoyance", "disapproval"),
    "disgust": ("disgust",),
    "fear": ("fear", "nervousness"),
    "joy": (
        "joy",
        "amusement",
        "approval",
        "excitement",
        "gratitude",
        "love",
        "optimism",
        "relief",
        "pride",
        "admiration",
        "desire",
        "caring",
    ),
    "sadness": ("sadness", "disappointment", "embarrassment", "grief", "remorse"),
    "surprise": ("surprise", "realization", "confusion", "curiosity"),
}
FILE_URLS = {
    "train.tsv": (
        "https://raw.githubusercontent.com/google-research/google-research/master/"
        "goemotions/data/train.tsv"
    ),
    "dev.tsv": (
        "https://raw.githubusercontent.com/google-research/google-research/master/"
        "goemotions/data/dev.tsv"
    ),
    "test.tsv": (
        "https://raw.githubusercontent.com/google-research/google-research/master/"
        "goemotions/data/test.tsv"
    ),
    "emotions.txt": (
        "https://raw.githubusercontent.com/google-research/google-research/master/"
        "goemotions/data/emotions.txt"
    ),
    "ekman_mapping.json": (
        "https://raw.githubusercontent.com/google-research/google-research/master/"
        "goemotions/data/ekman_mapping.json"
    ),
}
FILE_SHA256 = {
    "train.tsv": "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5",
    "dev.tsv": "575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62",
    "test.tsv": "0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4",
    "emotions.txt": "45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89",
    "ekman_mapping.json": "d6b1fea382917c1685c42e1324a0344106be9390f52a3850ce032656d103a328",
}
FINE_TO_COARSE = {"neutral": "neutral"}
for _coarse, _fines in EKMAN_GROUPS.items():
    for _fine in _fines:
        FINE_TO_COARSE[_fine] = _coarse

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"


def experiment_specs() -> list[dict[str, object]]:
    return [{"name": name, "eligible": name not in LEAKS} for name in EXPERIMENT_ORDER]


def select_model(rows: list[dict[str, object]]) -> str:
    """Lowest development log loss. Test scores are not an argument to the sort."""
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no eligible experiment")
    order = {name: index for index, name in enumerate(EXPERIMENT_ORDER)}
    eligible.sort(key=lambda row: (row["dev_log_loss"], order[str(row["name"])]))
    return str(eligible[0]["name"])


def coarse_label(label_ids: str, names: tuple[str, ...] = EMOTION_NAMES) -> str | None:
    groups = {FINE_TO_COARSE[names[int(token)]] for token in label_ids.split(",")}
    if len(groups) != 1:
        return None
    return next(iter(groups))


def fit_corpus(train: list[str], dev: list[str], test: list[str], leaky: bool) -> list[str]:
    if leaky:
        return list(train) + list(dev) + list(test)
    return list(train)


def unseen_text_mask(texts: list[str], blocked: set[str]) -> np.ndarray:
    return np.array([text not in blocked for text in texts], dtype=bool)


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ml-mastery-lab/text-features"})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        handle.write(response.read())


def ensure_inputs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in FILE_URLS.items():
        dest = DATA / name
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == FILE_SHA256[name]:
            continue
        if dest.exists():
            dest.unlink()
        _download(url, dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != FILE_SHA256[name]:
            raise RuntimeError(f"{name} hash {digest} does not match the pinned file")
    emotions = tuple((DATA / "emotions.txt").read_text(encoding="utf-8").splitlines())
    if emotions != EMOTION_NAMES:
        raise RuntimeError("emotions.txt does not match the pinned label order")
    mapping = json.loads((DATA / "ekman_mapping.json").read_text(encoding="utf-8"))
    expected = {key: list(value) for key, value in EKMAN_GROUPS.items()}
    if mapping != expected:
        raise RuntimeError("ekman_mapping.json does not match the pinned grouping")


def load_split(name: str) -> pd.DataFrame:
    frame = pd.read_csv(
        DATA / f"{name}.tsv",
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        dtype=str,
        keep_default_na=False,
    )
    frame["coarse"] = frame["label_ids"].map(coarse_label)
    frame["split"] = name
    return frame


def make_vectorizer(name: str) -> CountVectorizer | TfidfVectorizer:
    common = {
        "lowercase": True,
        "min_df": MIN_DF,
        "max_features": MAX_FEATURES,
        "dtype": np.float32,
    }
    if name == "bow_unigram":
        return CountVectorizer(ngram_range=(1, 1), token_pattern=TOKEN_PATTERN, **common)
    if name in {"tfidf_unigram", "lsa_100", "leaky_tfidf_fit_on_all"}:
        return TfidfVectorizer(ngram_range=(1, 1), token_pattern=TOKEN_PATTERN, **common)
    if name == "tfidf_bigram":
        return TfidfVectorizer(ngram_range=(1, 2), token_pattern=TOKEN_PATTERN, **common)
    if name == "tfidf_char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common)
    raise KeyError(name)


def make_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=LOGISTIC_C,
        solver="saga",
        max_iter=MAX_ITER,
        tol=TOL,
        fit_intercept=True,
        random_state=SEED,
    )


def align_proba(classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
    if set(classes) != set(LABELS):
        raise ValueError(f"classes {list(classes)} do not match the locked label list")
    index = {label: column for column, label in enumerate(classes)}
    ordered = np.column_stack([proba[:, index[label]] for label in LABELS]).astype(np.float64)
    totals = ordered.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("probability row sums to zero")
    return ordered / totals


def score_proba(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    predicted = np.array(LABELS, dtype=object)[np.argmax(proba, axis=1)]
    return {
        "log_loss": float(log_loss(y_true, proba, labels=list(LABELS))),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(
            f1_score(y_true, predicted, average="macro", labels=list(LABELS), zero_division=0)
        ),
    }


def class_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, dict[str, float]]:
    predicted = np.array(LABELS, dtype=object)[np.argmax(proba, axis=1)]
    rows = {}
    for label in LABELS:
        truth = y_true == label
        guess = predicted == label
        support = int(truth.sum())
        true_positive = int(np.logical_and(truth, guess).sum())
        precision = true_positive / int(guess.sum()) if int(guess.sum()) else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows[label] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return rows


def bootstrap_gap(
    y_true: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | bool]:
    n_rows = len(y_true)
    gaps = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for draw in range(BOOTSTRAP_RESAMPLES):
        chosen = rng.integers(0, n_rows, n_rows)
        ref_loss = log_loss(y_true[chosen], reference[chosen], labels=list(LABELS))
        cand_loss = log_loss(y_true[chosen], candidate[chosen], labels=list(LABELS))
        gaps[draw] = ref_loss - cand_loss
    low, high = np.quantile(gaps, [0.025, 0.975])
    point = float(
        log_loss(y_true, reference, labels=list(LABELS))
        - log_loss(y_true, candidate, labels=list(LABELS))
    )
    return {
        "point_gap": point,
        "interval_low": float(low),
        "interval_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def zero_rows(matrix: object) -> int:
    if hasattr(matrix, "getnnz"):
        return int((matrix.getnnz(axis=1) == 0).sum())
    return int(np.all(np.asarray(matrix) == 0, axis=1).sum())


def fit_experiment(
    name: str,
    train_text: list[str],
    dev_text: list[str],
    test_text: list[str],
    y_train: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if name == "training_rate":
            classifier = DummyClassifier(strategy="prior", random_state=SEED)
            classifier.fit(np.zeros((len(y_train), 1)), y_train)
            dev_proba = align_proba(
                classifier.classes_, classifier.predict_proba(np.zeros((len(y_dev), 1)))
            )
            test_proba = align_proba(
                classifier.classes_, classifier.predict_proba(np.zeros((len(y_test), 1)))
            )
            details = {
                "n_features": 0,
                "vectorizer_fit_documents": 0,
                "cap_bound": False,
                "train_zero_rows": 0,
                "explained_variance_ratio_sum": None,
                "vectorizer": None,
                "classifier": None,
            }
        else:
            vectorizer = make_vectorizer(name)
            corpus = fit_corpus(
                train_text,
                dev_text,
                test_text,
                leaky=name in LEAKS,
            )
            vectorizer.fit(corpus)
            train_matrix = vectorizer.transform(train_text)
            explained = None
            if name == "lsa_100":
                svd = TruncatedSVD(
                    n_components=SVD_COMPONENTS,
                    algorithm="randomized",
                    n_iter=SVD_ITER,
                    random_state=SEED,
                )
                train_matrix = svd.fit_transform(train_matrix)
                explained = float(svd.explained_variance_ratio_.sum())
                dev_matrix = svd.transform(vectorizer.transform(dev_text))
                test_matrix = svd.transform(vectorizer.transform(test_text))
                n_features = int(SVD_COMPONENTS)
            else:
                dev_matrix = vectorizer.transform(dev_text)
                test_matrix = vectorizer.transform(test_text)
                n_features = int(len(vectorizer.vocabulary_))
            classifier = make_classifier()
            classifier.fit(train_matrix, y_train)
            dev_proba = align_proba(classifier.classes_, classifier.predict_proba(dev_matrix))
            test_proba = align_proba(classifier.classes_, classifier.predict_proba(test_matrix))
            details = {
                "n_features": n_features,
                "vectorizer_fit_documents": len(corpus),
                "cap_bound": bool(name != "lsa_100" and n_features == MAX_FEATURES),
                "train_zero_rows": zero_rows(vectorizer.transform(train_text)),
                "explained_variance_ratio_sum": explained,
                "vectorizer": vectorizer,
                "classifier": classifier,
            }
    n_iter = getattr(classifier, "n_iter_", None)
    if n_iter is not None:
        n_iter = int(np.max(n_iter))
    return {
        "name": name,
        "eligible": name not in LEAKS,
        "dev_log_loss": score_proba(y_dev, dev_proba)["log_loss"],
        "dev": score_proba(y_dev, dev_proba),
        "test": score_proba(y_test, test_proba),
        "test_proba": test_proba,
        "convergence_warnings": sum(
            1 for item in caught if issubclass(item.category, ConvergenceWarning)
        ),
        "max_n_iter": n_iter,
        **details,
    }


def comparison_record(
    y_true: np.ndarray,
    stored: dict[str, np.ndarray],
    reference: str,
    candidate: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    gap = bootstrap_gap(y_true, stored[reference], stored[candidate], rng)
    return {
        "reference": reference,
        "candidate": candidate,
        "gap_definition": "reference_log_loss - candidate_log_loss",
        **gap,
    }


def plot_counts(path: Path, counts: dict[str, dict[str, int]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(LABELS))
    width = 0.25
    for offset, split, color in (
        (-width, "train", "#4C78A8"),
        (0, "dev", "#F58518"),
        (width, "test", "#54A24B"),
    ):
        total = sum(counts[split].values())
        heights = [counts[split][label] / total for label in LABELS]
        ax.bar(x + offset, heights, width, label=split, color=color)
    ax.set_xticks(x, LABELS, rotation=20, ha="right")
    ax.set_title("Share of kept comments after the official Ekman map")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Share of split")
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_log_loss(path: Path, experiments: dict[str, dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(EXPERIMENT_ORDER))
    width = 0.38
    dev_bars = ax.bar(
        x - width / 2,
        [experiments[name]["dev"]["log_loss"] for name in EXPERIMENT_ORDER],
        width,
        label="Development, used for selection",
        color="#4C78A8",
    )
    test_bars = ax.bar(
        x + width / 2,
        [experiments[name]["test"]["log_loss"] for name in EXPERIMENT_ORDER],
        width,
        label="Official test, not used for selection",
        color="#F58518",
    )
    for bars in (dev_bars, test_bars):
        for bar, name in zip(bars, EXPERIMENT_ORDER, strict=True):
            if name in LEAKS:
                bar.set_hatch("//")
    ax.set_xticks(x, EXPERIMENT_ORDER, rotation=25, ha="right")
    ax.set_title("Log loss by locked feature recipe. Hatched bars are not eligible.")
    ax.set_xlabel("Feature recipe")
    ax.set_ylabel("Multiclass log loss")
    handles, legend_labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="Not eligible"))
    legend_labels.append("Not eligible")
    ax.legend(handles, legend_labels)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(path: Path, y_true: np.ndarray, proba: np.ndarray, model_name: str) -> None:
    predicted = np.array(LABELS, dtype=object)[np.argmax(proba, axis=1)]
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    lookup = {label: index for index, label in enumerate(LABELS)}
    for truth, guess in zip(y_true, predicted, strict=True):
        matrix[lookup[str(truth)], lookup[str(guess)]] += 1
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("Labeled emotion")
    ax.set_title(f"Official test counts, {model_name}. Not used to choose the model.")
    fig.colorbar(image, ax=ax, label="Test comments")
    threshold = matrix.max() / 2
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_top_tokens(
    path: Path,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    title: str,
) -> None:
    names = vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(len(LABELS), 1, figsize=(8.5, 12), sharex=False)
    for ax, label in zip(axes, LABELS, strict=True):
        row = list(classifier.classes_).index(label)
        coef = classifier.coef_[row]
        chosen = np.argsort(coef)[-8:]
        ax.barh(names[chosen], coef[chosen], color="#4C78A8")
        ax.set_title(label)
        ax.set_xlabel("Logistic coefficient")
        ax.set_ylabel("Token")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def top_token_table(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
) -> dict[str, list[dict[str, object]]]:
    names = vectorizer.get_feature_names_out()
    table = {}
    for label in LABELS:
        row = list(classifier.classes_).index(label)
        coef = classifier.coef_[row]
        chosen = np.argsort(coef)[-8:][::-1]
        table[label] = [
            {"token": str(names[index]), "coefficient": float(coef[index])} for index in chosen
        ]
    return table


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def count_labels(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["coarse"].value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}


def main() -> None:
    ensure_inputs()
    raw = {name: load_split(name) for name in ("train", "dev", "test")}
    kept = {name: frame.dropna(subset=["coarse"]).copy() for name, frame in raw.items()}
    for frame in kept.values():
        frame["coarse"] = frame["coarse"].astype(str)
    missing = [label for label in LABELS if label not in set(kept["train"]["coarse"])]
    if missing:
        raise RuntimeError(f"training split is missing locked labels: {missing}")

    train_text = kept["train"]["text"].tolist()
    dev_text = kept["dev"]["text"].tolist()
    test_text = kept["test"]["text"].tolist()
    y_train = kept["train"]["coarse"].to_numpy()
    y_dev = kept["dev"]["coarse"].to_numpy()
    y_test = kept["test"]["coarse"].to_numpy()
    blocked = set(raw["train"]["text"]).union(raw["dev"]["text"])
    unseen = unseen_text_mask(test_text, blocked)

    fitted = []
    for name in EXPERIMENT_ORDER:
        print(f"fitting {name}", flush=True)
        fitted.append(fit_experiment(name, train_text, dev_text, test_text, y_train, y_dev, y_test))
    reported = select_model(fitted)
    by_name = {row["name"]: row for row in fitted}
    stored = {row["name"]: row["test_proba"] for row in fitted}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pairs = (
        ("training_rate", reported, "selected_vs_training_rate"),
        ("bow_unigram", "tfidf_unigram", "tfidf_unigram_vs_bow_unigram"),
        ("tfidf_unigram", "tfidf_bigram", "tfidf_bigram_vs_tfidf_unigram"),
        ("tfidf_unigram", "tfidf_char", "tfidf_char_vs_tfidf_unigram"),
        ("tfidf_unigram", "lsa_100", "lsa_100_vs_tfidf_unigram"),
        ("tfidf_unigram", "leaky_tfidf_fit_on_all", "leaky_tfidf_fit_on_all_vs_tfidf_unigram"),
    )
    comparisons = {
        key: comparison_record(y_test, stored, reference, candidate, rng)
        for reference, candidate, key in pairs
    }
    selected_unseen = score_proba(y_test[unseen], stored[reported][unseen])
    token_model = by_name["tfidf_unigram"]

    experiments = {}
    for row in fitted:
        experiments[row["name"]] = {
            "eligible": row["eligible"],
            "dev": row["dev"],
            "test": row["test"],
            "n_features": row["n_features"],
            "vectorizer_fit_documents": row["vectorizer_fit_documents"],
            "cap_bound": row["cap_bound"],
            "train_zero_rows": row["train_zero_rows"],
            "explained_variance_ratio_sum": row["explained_variance_ratio_sum"],
            "convergence_warnings": row["convergence_warnings"],
            "max_n_iter": row["max_n_iter"],
        }
    payload = {
        "dataset": {
            "name": "GoEmotions",
            "paper": "Demszky et al., ACL 2020",
            "doi": "10.18653/v1/2020.acl-main.372",
            "license": "apache-2.0",
            "license_source": "Hugging Face dataset card tag and the google-research repository",
            "files": (
                "google-research/google-research goemotions/data "
                "train, dev, test, emotions, ekman map"
            ),
            "target": "ekman_emotion",
            "target_description": (
                "Official Ekman grouping of the fine labels. Neutral stays neutral. "
                "A comment is kept only when every fine label maps to the same coarse emotion."
            ),
        },
        "protocol": {
            "reported_model": reported,
            "selection_metric": "dev_log_loss",
            "selection_uses_test_scores": False,
            "split": "official train, dev, and test files",
            "seed_used_for_split": False,
            "seed": SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "max_features": MAX_FEATURES,
            "min_df": MIN_DF,
            "svd_components": SVD_COMPONENTS,
            "logistic_c": LOGISTIC_C,
            "solver": "saga",
            "max_iter": MAX_ITER,
            "tol": TOL,
            "class_weight": None,
            "gap_definition": "reference_log_loss - candidate_log_loss",
            "positive_gap_means": "candidate has lower log loss",
        },
        "audit": {
            "raw_rows": {name: int(len(frame)) for name, frame in raw.items()},
            "kept_rows": {name: int(len(frame)) for name, frame in kept.items()},
            "dropped_cross_group": {
                name: int(frame["coarse"].isna().sum()) for name, frame in raw.items()
            },
            "kept_label_counts": {name: count_labels(frame) for name, frame in kept.items()},
            "id_overlap": {
                "train_dev": int(
                    len(set(raw["train"]["comment_id"]) & set(raw["dev"]["comment_id"]))
                ),
                "train_test": int(
                    len(set(raw["train"]["comment_id"]) & set(raw["test"]["comment_id"]))
                ),
                "dev_test": int(
                    len(set(raw["dev"]["comment_id"]) & set(raw["test"]["comment_id"]))
                ),
            },
            "exact_text_overlap_raw": {
                "train_dev": int(len(set(raw["train"]["text"]) & set(raw["dev"]["text"]))),
                "train_test": int(len(set(raw["train"]["text"]) & set(raw["test"]["text"]))),
                "dev_test": int(len(set(raw["dev"]["text"]) & set(raw["test"]["text"]))),
            },
            "test_rows_exact_text_in_train_or_dev": int((~unseen).sum()),
        },
        "experiments": experiments,
        "reported_test_class_metrics": class_metrics(y_test, stored[reported]),
        "tfidf_unigram_top_tokens": top_token_table(
            token_model["vectorizer"],
            token_model["classifier"],
        ),
        "comparisons": comparisons,
        "overlap_sensitivity": {
            "does_not_reselect": True,
            "excluded_test_rows": int((~unseen).sum()),
            "kept_test_rows": int(unseen.sum()),
            "test": selected_unseen,
        },
    }
    if reported not in {"training_rate", "lsa_100"}:
        reported_row = by_name[reported]
        payload["reported_top_features"] = top_token_table(
            reported_row["vectorizer"],
            reported_row["classifier"],
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(
        json.dumps(jsonable(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    plot_counts(RESULTS / "class_counts.png", payload["audit"]["kept_label_counts"])
    plot_log_loss(RESULTS / "log_loss.png", experiments)
    plot_confusion(RESULTS / "confusion.png", y_test, stored[reported], reported)
    plot_top_tokens(
        RESULTS / "top_tokens.png",
        token_model["vectorizer"],
        token_model["classifier"],
        "Largest positive coefficients of tfidf_unigram",
    )
    if reported not in {"training_rate", "lsa_100"}:
        plot_top_tokens(
            RESULTS / "reported_top_features.png",
            reported_row["vectorizer"],
            reported_row["classifier"],
            f"Largest positive coefficients of {reported}",
        )
    print(f"reported {reported}", flush=True)
    print(f"test log loss {experiments[reported]['test']['log_loss']}", flush=True)


if __name__ == "__main__":
    main()
