"""Bias-variance on a known curve, then on red wine quality.

The sine curve is simulated so the true function and the noise variance are
known. That is the only place this script separates bias, variance, and noise.
The wine run is a real table. It reports train, validation, and holdout error.
It does not invent a noise term for those rows.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import warnings
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from numpy.exceptions import RankWarning
from numpy.polynomial.polynomial import polyfit, polyval
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")

SEED = 42
SIGMA = 0.5
N_TRAIN = 20
N_REPS = 100
GRID_SIZE = 40
POLY_DEGREES = (1, 3, 5, 9, 15)
KNN_K = (1, 3, 7, 19)
HOLD_FRAC = 0.20
INNER_SPLITS = 30
INNER_VAL_FRAC = 0.25
N_BOOT = 40
TREE_DEPTHS = (1, 2, 3, 5, 8, None)
WINE_POLY_DEGREES = (2, 3, 4)
RIDGE_ALPHA = 1.0
CSV_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)
ZIP_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
WINE_COLUMNS = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
)
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA_PATH = ROOT.parents[1] / "data" / "winequality-red.csv"


def true_curve(x: np.ndarray) -> np.ndarray:
    """Population mean of the simulated response. Amplitude 1, period 1."""
    return np.sin(2.0 * np.pi * np.asarray(x, dtype=float))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(residual**2))


def decomposition(predictions: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Average bias squared and prediction variance over a fixed grid.

    ``predictions`` has shape (n_reps, n_points). Variance uses the sample
    variance across repetitions (ddof=1), then the mean over grid points.
    """
    if predictions.ndim != 2:
        raise ValueError("predictions must be (n_reps, n_points)")
    if predictions.shape[0] < 2:
        raise ValueError("need at least two fits to estimate variance")
    mean_pred = predictions.mean(axis=0)
    bias2 = float(np.mean((mean_pred - truth) ** 2))
    variance = float(np.mean(predictions.var(axis=0, ddof=1)))
    return {
        "bias2": bias2,
        "variance": variance,
        "mean_squared_distance_to_truth": bias2 + variance,
    }


def predict_polynomial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_out: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Ordinary least squares in the monomial basis, via numpy's polyfit."""
    coef = fit_polynomial(x_train, y_train, degree)
    return np.asarray(polyval(np.asarray(x_out, dtype=float), coef), dtype=float)


def fit_polynomial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    degree: int,
    rank_warnings: dict[int, int] | None = None,
) -> np.ndarray:
    """Fit one polynomial. Count fits that emit a RankWarning, if asked."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RankWarning)
        coef = polyfit(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float), degree)
    if rank_warnings is not None and any(issubclass(item.category, RankWarning) for item in caught):
        rank_warnings[degree] = rank_warnings.get(degree, 0) + 1
    return np.asarray(coef, dtype=float)


def predict_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_out: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(np.asarray(x_train, dtype=float).reshape(-1, 1), np.asarray(y_train, dtype=float))
    return np.asarray(
        model.predict(np.asarray(x_out, dtype=float).reshape(-1, 1)),
        dtype=float,
    )


def predict_mean(y_train: np.ndarray, n_out: int) -> np.ndarray:
    return np.full(n_out, float(np.mean(y_train)))


def fit_predict_table(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_out: np.ndarray,
    kind: str,
    param: int | None,
) -> np.ndarray:
    """Fit on the training rows only. ``param`` is degree, depth, or unused."""
    if kind == "mean":
        return predict_mean(y_train, len(x_out))
    if kind == "linear":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    elif kind == "poly":
        if param is None:
            raise ValueError("polynomial degree is required")
        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=param, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    elif kind == "ridge_poly":
        if param is None:
            raise ValueError("polynomial degree is required")
        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=param, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=RIDGE_ALPHA)),
            ]
        )
    elif kind == "tree":
        model = DecisionTreeRegressor(max_depth=param, random_state=0)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    model.fit(x_train, y_train)
    predicted = np.asarray(model.predict(x_out), dtype=float)
    if not np.all(np.isfinite(predicted)):
        raise RuntimeError(f"non-finite predictions from {kind} {param}")
    return predicted


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ml-mastery-lab/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def load_red_wine() -> pd.DataFrame:
    """Red wines only. The white-wine file is a separate table and is not used."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            DATA_PATH.write_bytes(_download_bytes(CSV_URL))
        except (urllib.error.URLError, TimeoutError, OSError):
            blob = _download_bytes(ZIP_URL)
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                name = next(
                    item for item in archive.namelist() if item.endswith("winequality-red.csv")
                )
                DATA_PATH.write_bytes(archive.read(name))
    frame = pd.read_csv(DATA_PATH, sep=";")
    if tuple(frame.columns) != WINE_COLUMNS:
        raise ValueError(f"unexpected columns: {list(frame.columns)}")
    if frame.isna().any().any():
        raise ValueError("unexpected missing values")
    return frame


def _sim_record(
    name: str,
    family: str,
    predictions: np.ndarray,
    train_mses: np.ndarray,
    truth: np.ndarray,
    noise_rng: np.random.Generator,
) -> dict[str, object]:
    parts = decomposition(predictions, truth)
    noise = noise_rng.normal(0.0, SIGMA, size=predictions.shape)
    noisy_mse = float(np.mean((truth + noise - predictions) ** 2))
    summed = parts["bias2"] + parts["variance"] + SIGMA**2
    return {
        "name": name,
        "family": family,
        "bias2": parts["bias2"],
        "variance": parts["variance"],
        "noise_var": float(SIGMA**2),
        "decomposition_sum": float(summed),
        "monte_carlo_noisy_mse": noisy_mse,
        "decomposition_gap": float(noisy_mse - summed),
        "mean_train_mse": float(np.mean(train_mses)),
        "max_abs_prediction": float(np.max(np.abs(predictions))),
    }


def simulate_curve() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    noise_rng = np.random.default_rng(SEED + 7)
    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    truth = true_curve(grid)
    draws: dict[str, np.ndarray] = {}
    train_store: dict[str, np.ndarray] = {}
    specs: list[tuple[str, str]] = [
        (f"poly_degree_{degree}", "polynomial") for degree in POLY_DEGREES
    ]
    specs.extend((f"knn_{k}", "knn") for k in KNN_K)
    for name, _family in specs:
        draws[name] = np.empty((N_REPS, GRID_SIZE))
        train_store[name] = np.empty(N_REPS)

    x_example = None
    y_example = None
    rank_warnings: dict[int, int] = {}
    for rep in range(N_REPS):
        x_train = rng.uniform(0.0, 1.0, size=N_TRAIN)
        y_train = true_curve(x_train) + rng.normal(0.0, SIGMA, size=N_TRAIN)
        if rep == 0:
            x_example = x_train.copy()
            y_example = y_train.copy()
        for degree in POLY_DEGREES:
            name = f"poly_degree_{degree}"
            coef = fit_polynomial(x_train, y_train, degree, rank_warnings)
            draws[name][rep] = polyval(grid, coef)
            train_store[name][rep] = mse(y_train, polyval(x_train, coef))
        for k in KNN_K:
            name = f"knn_{k}"
            pred_grid = predict_knn(x_train, y_train, grid, k)
            pred_train = predict_knn(x_train, y_train, x_train, k)
            draws[name][rep] = pred_grid
            train_store[name][rep] = mse(y_train, pred_train)

    runs = [
        _sim_record(name, family, draws[name], train_store[name], truth, noise_rng)
        for name, family in specs
    ]
    payload = {
        "function": "sin(2*pi*x)",
        "sigma": SIGMA,
        "noise_var": float(SIGMA**2),
        "n_train": N_TRAIN,
        "n_reps": N_REPS,
        "n_grid": GRID_SIZE,
        "x_distribution": "uniform(0, 1)",
        "grid": "linspace(0, 1, n_grid)",
        "variance_ddof": 1,
        "rank_warning_fits": {
            str(degree): count for degree, count in sorted(rank_warnings.items())
        },
        "runs": runs,
    }
    plot_arrays = {
        "grid": grid,
        "truth": truth,
        "x_example": np.asarray(x_example),
        "y_example": np.asarray(y_example),
        "draws": draws,
    }
    return payload, plot_arrays


def wine_model_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"name": "mean", "kind": "mean", "param": None, "family": "baseline"},
        {"name": "linear", "kind": "linear", "param": 1, "family": "linear"},
    ]
    for degree in WINE_POLY_DEGREES:
        specs.append(
            {
                "name": f"poly_degree_{degree}",
                "kind": "poly",
                "param": degree,
                "family": "polynomial",
            }
        )
    specs.append(
        {
            "name": "ridge_poly_degree_4_alpha_1",
            "kind": "ridge_poly",
            "param": 4,
            "family": "ridge",
        }
    )
    for depth in TREE_DEPTHS:
        label = "full" if depth is None else str(depth)
        specs.append(
            {
                "name": f"tree_depth_{label}",
                "kind": "tree",
                "param": depth,
                "family": "tree",
            }
        )
    return specs


def _split_ends(
    n_rows: int, frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n_rows)
    n_second = int(n_rows * frac)
    return order[n_second:], order[:n_second]


def expanded_width(n_features: int, degree: int) -> int:
    transformer = PolynomialFeatures(degree=degree, include_bias=False)
    return int(transformer.fit(np.zeros((1, n_features))).n_output_features_)


def wine_experiment(frame: pd.DataFrame) -> dict[str, object]:
    features = frame.drop(columns=["quality"]).to_numpy(dtype=float)
    target = frame["quality"].to_numpy(dtype=float)
    hold_rng = np.random.default_rng(SEED)
    inner_rng = np.random.default_rng(SEED + 1)
    boot_rng = np.random.default_rng(SEED + 2)
    pool_idx, hold_idx = _split_ends(len(target), HOLD_FRAC, hold_rng)
    x_pool, y_pool = features[pool_idx], target[pool_idx]
    x_hold, y_hold = features[hold_idx], target[hold_idx]
    specs = wine_model_specs()
    inner_train = {spec["name"]: [] for spec in specs}
    inner_val = {spec["name"]: [] for spec in specs}
    for _ in range(INNER_SPLITS):
        fit_idx, val_idx = _split_ends(len(y_pool), INNER_VAL_FRAC, inner_rng)
        for spec in specs:
            kind = str(spec["kind"])
            param = spec["param"] if spec["param"] is None else int(spec["param"])
            train_pred = fit_predict_table(
                x_pool[fit_idx], y_pool[fit_idx], x_pool[fit_idx], kind, param
            )
            val_pred = fit_predict_table(
                x_pool[fit_idx], y_pool[fit_idx], x_pool[val_idx], kind, param
            )
            inner_train[spec["name"]].append(mse(y_pool[fit_idx], train_pred))
            inner_val[spec["name"]].append(mse(y_pool[val_idx], val_pred))

    holdout_scores: dict[str, float] = {}
    holdout_pred: dict[str, np.ndarray] = {}
    for spec in specs:
        kind = str(spec["kind"])
        param = spec["param"] if spec["param"] is None else int(spec["param"])
        predicted = fit_predict_table(x_pool, y_pool, x_hold, kind, param)
        holdout_pred[str(spec["name"])] = predicted
        holdout_scores[str(spec["name"])] = mse(y_hold, predicted)

    selected = min(specs, key=lambda spec: float(np.mean(inner_val[spec["name"]])))
    peeked = min(specs, key=lambda spec: holdout_scores[str(spec["name"])])
    boot_names = ["mean", "linear", "tree_depth_1", "tree_depth_full", "poly_degree_4"]
    if selected["name"] not in boot_names:
        boot_names.append(str(selected["name"]))
    boot_var: dict[str, float] = {}
    boot_mse: dict[str, float] = {}
    for name in boot_names:
        spec = next(item for item in specs if item["name"] == name)
        kind = str(spec["kind"])
        param = spec["param"] if spec["param"] is None else int(spec["param"])
        preds = np.empty((N_BOOT, len(y_hold)))
        scores = np.empty(N_BOOT)
        for draw in range(N_BOOT):
            take = boot_rng.integers(0, len(y_pool), size=len(y_pool))
            predicted = fit_predict_table(x_pool[take], y_pool[take], x_hold, kind, param)
            preds[draw] = predicted
            scores[draw] = mse(y_hold, predicted)
        boot_var[name] = float(np.mean(preds.var(axis=0, ddof=1)))
        boot_mse[name] = float(np.mean(scores))

    runs = []
    for spec in specs:
        name = str(spec["name"])
        val_scores = np.asarray(inner_val[name], dtype=float)
        train_scores = np.asarray(inner_train[name], dtype=float)
        runs.append(
            {
                "name": name,
                "family": spec["family"],
                "param": spec["param"],
                "mean_inner_train_mse": float(train_scores.mean()),
                "mean_inner_val_mse": float(val_scores.mean()),
                "std_inner_val_mse": float(val_scores.std(ddof=1)),
                "holdout_mse": holdout_scores[name],
            }
        )
    feature_names = [column for column in frame.columns if column != "quality"]
    linear_model = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    linear_model.fit(x_pool, y_pool)
    coefficients = np.asarray(linear_model.named_steps["model"].coef_, dtype=float)
    largest = int(np.argmax(np.abs(coefficients)))
    return {
        "n_rows": int(len(frame)),
        "n_features": int(features.shape[1]),
        "feature_names": feature_names,
        "quality_min": float(target.min()),
        "quality_max": float(target.max()),
        "quality_mean": float(target.mean()),
        "n_unique_feature_rows": int(pd.DataFrame(features).drop_duplicates().shape[0]),
        "n_holdout": int(len(hold_idx)),
        "n_train_pool": int(len(pool_idx)),
        "hold_frac": HOLD_FRAC,
        "inner_splits": INNER_SPLITS,
        "inner_val_frac": INNER_VAL_FRAC,
        "n_bootstrap": N_BOOT,
        "ridge_alpha": RIDGE_ALPHA,
        "poly_widths": {
            f"degree_{degree}": expanded_width(features.shape[1], degree)
            for degree in WINE_POLY_DEGREES
        },
        "largest_abs_scaled_linear_coef": {
            "feature": feature_names[largest],
            "coefficient": float(coefficients[largest]),
        },
        "selected_by_mean_inner_val_mse": selected["name"],
        "selected_holdout_mse": holdout_scores[str(selected["name"])],
        "peeked_by_holdout_mse": peeked["name"],
        "peeked_holdout_mse": holdout_scores[str(peeked["name"])],
        "bootstrap_prediction_variance": boot_var,
        "bootstrap_mean_holdout_mse": boot_mse,
        "runs": runs,
    }


def _tick_labels(runs: list[dict[str, object]]) -> list[str]:
    return [str(run["name"]).replace("_", "\n") for run in runs]


def save_simulated_fits(bundle: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    grid = bundle["grid"]
    truth = bundle["truth"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for curve in bundle["draws"]["knn_1"][:20]:
        ax.plot(grid, curve, color="#56B4E9", alpha=0.28, linewidth=1.0)
    ax.plot(grid, truth, color="black", linewidth=2.0, label="True mean, sin(2*pi*x)")
    ax.plot(
        grid,
        bundle["draws"]["poly_degree_1"].mean(axis=0),
        color="#0072B2",
        linewidth=2.0,
        label="Mean degree-1 fit",
    )
    ax.plot(
        grid,
        bundle["draws"]["poly_degree_3"].mean(axis=0),
        color="#009E73",
        linewidth=2.0,
        label="Mean degree-3 fit",
    )
    ax.scatter(
        bundle["x_example"],
        bundle["y_example"],
        s=18,
        color="#D55E00",
        zorder=3,
        label="One training draw",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("Response")
    ax.set_title("1-NN paths disagree. A line cannot follow this curve.")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_tradeoff(runs: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    stable = [run for run in runs if float(run["monte_carlo_noisy_mse"]) < 10.0]
    omitted = [str(run["name"]) for run in runs if run not in stable]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    positions = np.arange(len(stable))
    axes[0].plot(
        positions,
        [float(run["mean_train_mse"]) for run in stable],
        marker="o",
        color="#0072B2",
        label="Mean train MSE",
    )
    axes[0].plot(
        positions,
        [float(run["monte_carlo_noisy_mse"]) for run in stable],
        marker="o",
        color="#D55E00",
        label="Fresh noisy-grid MSE",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(_tick_labels(stable), fontsize=7)
    axes[0].set_ylabel("Mean squared error")
    axes[0].set_xlabel("Model")
    axes[0].set_title("Fits that stay on this scale")
    axes[0].legend(frameon=False, fontsize=8)
    fresh = [float(run["monte_carlo_noisy_mse"]) for run in runs]
    axes[1].plot(np.arange(len(runs)), fresh, marker="o", color="#D55E00")
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(len(runs)))
    axes[1].set_xticklabels(_tick_labels(runs), fontsize=7)
    axes[1].set_ylabel("Fresh noisy-grid MSE")
    axes[1].set_xlabel("Model")
    axes[1].set_title("Same score, log scale, including " + ", ".join(omitted))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_decomposition(runs: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    shown = [run for run in runs if float(run["variance"]) < 10.0]
    omitted = [str(run["name"]) for run in runs if float(run["variance"]) >= 10.0]
    labels = _tick_labels(shown)
    bias = np.array([float(run["bias2"]) for run in shown])
    variance = np.array([float(run["variance"]) for run in shown])
    noise = np.array([float(run["noise_var"]) for run in shown])
    positions = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(positions, noise, color="#999999", label="Noise variance")
    ax.bar(positions, bias, bottom=noise, color="#0072B2", label="Bias squared")
    ax.bar(positions, variance, bottom=noise + bias, color="#E69F00", label="Variance")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean squared error, averaged over the grid")
    ax.set_xlabel("Model")
    ax.set_title("Known curve. Omitted from this axis: " + ", ".join(omitted))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_wine_complexity(runs: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    trees = [run for run in runs if run["family"] == "tree"]
    flexible = [
        run for run in runs if run["family"] in {"baseline", "linear", "polynomial", "ridge"}
    ]
    by_name = {str(run["name"]): run for run in runs}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    positions = np.arange(len(trees))
    axes[0].plot(
        positions,
        [float(run["mean_inner_train_mse"]) for run in trees],
        marker="o",
        color="#0072B2",
        label="Mean train MSE",
    )
    axes[0].errorbar(
        positions,
        [float(run["mean_inner_val_mse"]) for run in trees],
        yerr=[float(run["std_inner_val_mse"]) for run in trees],
        marker="o",
        color="#D55E00",
        capsize=3,
        label="Mean validation MSE ± 1 SD",
    )
    axes[0].axhline(
        float(by_name["linear"]["mean_inner_val_mse"]),
        color="#009E73",
        linestyle="--",
        label="Linear validation MSE",
    )
    axes[0].axhline(
        float(by_name["mean"]["mean_inner_val_mse"]),
        color="#666666",
        linestyle=":",
        label="Mean-baseline validation MSE",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(_tick_labels(trees), fontsize=7)
    axes[0].set_ylabel("Mean squared error, quality points squared")
    axes[0].set_xlabel("Tree")
    axes[0].set_title("Trees on 30 inner splits")
    axes[0].legend(frameon=False, fontsize=7)
    flex_pos = np.arange(len(flexible))
    axes[1].plot(
        flex_pos,
        [float(run["mean_inner_train_mse"]) for run in flexible],
        marker="o",
        color="#0072B2",
        label="Mean train MSE",
    )
    axes[1].plot(
        flex_pos,
        [float(run["mean_inner_val_mse"]) for run in flexible],
        marker="o",
        color="#D55E00",
        label="Mean validation MSE",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(flex_pos)
    axes[1].set_xticklabels(_tick_labels(flexible), fontsize=7)
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Mean squared error, log scale")
    axes[1].set_title("Polynomials, including the blow-up")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_wine_holdout(
    runs: list[dict[str, object]],
    selected: str,
    peeked: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    wanted = [
        "mean",
        "linear",
        "poly_degree_2",
        "ridge_poly_degree_4_alpha_1",
        "tree_depth_full",
    ]
    chosen = [next(run for run in runs if run["name"] == name) for name in wanted]
    omitted = next(run for run in runs if run["name"] == "poly_degree_4")
    labels = _tick_labels(chosen)
    values = [float(run["holdout_mse"]) for run in chosen]
    colors = []
    for run in chosen:
        if run["name"] == selected:
            colors.append("#009E73")
        elif run["name"] == peeked:
            colors.append("#E69F00")
        else:
            colors.append("#0072B2")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    positions = np.arange(len(chosen))
    ax.bar(positions, values, color=colors)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_ylabel("Holdout mean squared error")
    ax.set_xlabel("Model fit on the training pool")
    ax.set_title(f"Degree-4 holdout MSE {float(omitted['holdout_mse']):.4f} is off this axis")
    ax.legend(
        handles=[
            Patch(facecolor="#009E73", label="Chosen on validation"),
            Patch(facecolor="#E69F00", label="Best on this holdout"),
            Patch(facecolor="#0072B2", label="Not selected"),
        ],
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def analyze() -> dict[str, object]:
    started = time.perf_counter()
    simulated, plot_arrays = simulate_curve()
    frame = load_red_wine()
    wine = wine_experiment(frame)
    elapsed = time.perf_counter() - started
    RESULTS.mkdir(parents=True, exist_ok=True)
    save_simulated_fits(plot_arrays, RESULTS / "simulated_fits.png")
    save_tradeoff(simulated["runs"], RESULTS / "simulated_tradeoff.png")
    save_decomposition(simulated["runs"], RESULTS / "simulated_decomposition.png")
    save_wine_complexity(wine["runs"], RESULTS / "wine_complexity.png")
    save_wine_holdout(
        wine["runs"],
        str(wine["selected_by_mean_inner_val_mse"]),
        str(wine["peeked_by_holdout_mse"]),
        RESULTS / "wine_holdout.png",
    )
    metrics: dict[str, object] = {
        "seed": SEED,
        "elapsed_seconds": elapsed,
        "simulation": simulated,
        "wine": wine,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    metrics = analyze()
    wine = metrics["wine"]
    print(f"elapsed_seconds={metrics['elapsed_seconds']:.2f}")
    print(f"selected={wine['selected_by_mean_inner_val_mse']}")
    print(f"selected_holdout_mse={wine['selected_holdout_mse']}")
    print(f"peeked={wine['peeked_by_holdout_mse']}")
    for run in metrics["simulation"]["runs"]:
        print(
            f"sim {run['name']}: train={run['mean_train_mse']:.4f} "
            f"fresh={run['monte_carlo_noisy_mse']:.4f} "
            f"bias2={run['bias2']:.4f} var={run['variance']:.4f}"
        )
    for run in wine["runs"]:
        print(
            f"wine {run['name']}: train={run['mean_inner_train_mse']:.4f} "
            f"val={run['mean_inner_val_mse']:.4f} holdout={run['holdout_mse']:.4f}"
        )


if __name__ == "__main__":
    main()
