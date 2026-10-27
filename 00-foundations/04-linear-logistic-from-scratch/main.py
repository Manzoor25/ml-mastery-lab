"""Linear and logistic regression from scratch, checked against scikit-learn.

Housing prices use ordinary least squares, ridge, and batch gradient descent.
Breast-cancer diagnosis uses Newton and batch gradient descent on the same
penalized log loss that scikit-learn's lbfgs solver minimizes. The holdout
is scored after those choices. It is not used to pick a step size or a penalty.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
COLLINEAR_SEED = 45
TEST_SIZE = 0.2
RIDGE_ALPHA = 1.0
LOGISTIC_C = 1.0
GD_MAX_EPOCHS = 2000
GD_TRAIN_TOL = 1e-10
EXPLODE_MSE = 1_000_000.0
LOG_LOSS_CLIP = 1e-12
COLLINEAR_NOISE = 1e-4
NEWTON_MAX_ITER = 30
LIBRARY_COEF_TOL = 1e-5

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def scale_from_train(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize with the training mean and sample standard deviation."""
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0, ddof=1)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    return (x_train - mu) / sigma, (x_test - mu) / sigma, mu, sigma


def mse(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    resid = x @ w - y
    return float(np.mean(resid**2))


def grad_mse(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of mean squared error, including the intercept column."""
    resid = x @ w - y
    return (2.0 / len(y)) * (x.T @ resid)


def lipschitz_mse(x: np.ndarray) -> float:
    """Largest eigenvalue of the MSE Hessian, (2/n) X^T X."""
    hessian = (2.0 / len(x)) * (x.T @ x)
    return float(np.linalg.eigvalsh(hessian)[-1])


def ols_closed_form(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    weights, *_ = np.linalg.lstsq(x, y, rcond=None)
    return np.asarray(weights, dtype=float)


def ridge_closed_form(x_features: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Minimize ||Xw + b - y||^2 + alpha ||w||^2. The intercept is not penalized.

    Returns coefficients followed by the intercept, matching sklearn's split
    of coef_ and intercept_ rather than an intercept-first design.
    """
    design = np.column_stack([x_features, np.ones(len(y))])
    gram = design.T @ design
    penalty = np.zeros(x_features.shape[1] + 1)
    penalty[:-1] = alpha
    gram.flat[:: gram.shape[0] + 1] += penalty
    solved = np.linalg.solve(gram, design.T @ y)
    return np.asarray(solved, dtype=float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def pointwise_log_loss(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stable log(1 + exp(z)) - y z, the Bernoulli negative log likelihood."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0.0
    out[positive] = (1.0 - y[positive]) * z[positive] + np.log1p(np.exp(-z[positive]))
    out[~positive] = np.log1p(np.exp(z[~positive])) - y[~positive] * z[~positive]
    return out


def mean_log_loss(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pointwise_log_loss(z, y)))


def sklearn_l2_strength(n_rows: int, c_inverse_reg: float) -> float:
    """l2 strength sklearn passes to LinearModelLoss: 1 / (C * n)."""
    if not np.isfinite(c_inverse_reg):
        return 0.0
    return 1.0 / (c_inverse_reg * n_rows)


def logistic_objective(
    x_features: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    lam: float,
) -> float:
    """Average log loss plus (lam/2) ||coef||^2. Intercept is not penalized."""
    z = x_features @ coef + intercept
    penalty = 0.5 * lam * float(coef @ coef)
    return mean_log_loss(z, y) + penalty


def logistic_gradient(
    x_features: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    lam: float,
) -> tuple[np.ndarray, float]:
    resid = sigmoid(x_features @ coef + intercept) - y
    grad_coef = (x_features.T @ resid) / len(y) + lam * coef
    return grad_coef, float(np.mean(resid))


def lipschitz_logistic(x_features: np.ndarray, lam: float) -> float:
    """Global Lipschitz bound. p(1-p) never exceeds 1/4."""
    n_rows, n_features = x_features.shape
    design = np.column_stack([x_features, np.ones(n_rows)])
    gram = (0.25 / n_rows) * (design.T @ design)
    for j in range(n_features):
        gram[j, j] += lam
    return float(np.linalg.eigvalsh(gram)[-1])


def logistic_hessian(
    x_features: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    lam: float,
) -> np.ndarray:
    probability = sigmoid(x_features @ coef + intercept)
    weight = probability * (1.0 - probability)
    n_rows, n_features = x_features.shape
    weighted = x_features * weight[:, None]
    hessian = np.empty((n_features + 1, n_features + 1))
    hessian[:n_features, :n_features] = (x_features.T @ weighted) / n_rows
    feature_index = np.arange(n_features)
    hessian[feature_index, feature_index] += lam
    cross = (x_features.T @ weight) / n_rows
    hessian[:n_features, n_features] = cross
    hessian[n_features, :n_features] = cross
    hessian[n_features, n_features] = float(np.sum(weight) / n_rows)
    return hessian


def newton_logistic(
    x_features: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = NEWTON_MAX_ITER,
) -> dict[str, object]:
    """Newton on the penalized average log loss, with backtracking if needed."""
    coef = np.zeros(x_features.shape[1])
    intercept = 0.0
    damped_steps = 0
    grad_norm = None
    iterations_run = 0
    for _ in range(max_iter):
        iterations_run += 1
        grad_coef, grad_intercept = logistic_gradient(x_features, y, coef, intercept, lam)
        gradient = np.concatenate([grad_coef, [grad_intercept]])
        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm < 1e-12:
            break
        hessian = logistic_hessian(x_features, y, coef, intercept, lam)
        step = np.linalg.solve(hessian, gradient)
        loss_now = logistic_objective(x_features, y, coef, intercept, lam)
        scale = 1.0
        accepted = False
        for _ in range(30):
            trial_coef = coef - scale * step[:-1]
            trial_intercept = float(intercept - scale * step[-1])
            trial_loss = logistic_objective(x_features, y, trial_coef, trial_intercept, lam)
            if trial_loss <= loss_now + 1e-14:
                coef = trial_coef
                intercept = trial_intercept
                accepted = True
                if scale < 1.0:
                    damped_steps += 1
                break
            scale *= 0.5
        if not accepted:
            break
        if scale == 1.0 and grad_norm < 1e-10:
            break
    z = x_features @ coef + intercept
    return {
        "coef": coef,
        "intercept": intercept,
        "iterations": iterations_run,
        "damped_steps": damped_steps,
        "grad_norm": grad_norm,
        "train_log_loss": mean_log_loss(z, y),
        "objective": logistic_objective(x_features, y, coef, intercept, lam),
    }


def finite_difference_logistic(
    x_features: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    lam: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, float]:
    numeric = np.empty_like(coef)
    for j in range(len(coef)):
        step = np.zeros_like(coef)
        step[j] = eps
        plus = logistic_objective(x_features, y, coef + step, intercept, lam)
        minus = logistic_objective(x_features, y, coef - step, intercept, lam)
        numeric[j] = (plus - minus) / (2.0 * eps)
    plus_i = logistic_objective(x_features, y, coef, intercept + eps, lam)
    minus_i = logistic_objective(x_features, y, coef, intercept - eps, lam)
    return numeric, (plus_i - minus_i) / (2.0 * eps)


def batch_gd_linear(
    x: np.ndarray,
    y: np.ndarray,
    step: float,
    max_epochs: int,
    target_mse: float | None = None,
) -> dict[str, object]:
    weights = np.zeros(x.shape[1])
    history: list[float] = []
    exploded_epoch = None
    for epoch in range(1, max_epochs + 1):
        weights = weights - step * grad_mse(x, y, weights)
        loss = mse(x, y, weights)
        history.append(loss)
        if not np.isfinite(loss) or loss > EXPLODE_MSE:
            exploded_epoch = epoch
            break
        if target_mse is not None and abs(loss - target_mse) <= GD_TRAIN_TOL:
            break
    return {
        "weights": weights,
        "history": history,
        "epochs": len(history),
        "exploded_epoch": exploded_epoch,
        "final_train_mse": history[-1] if history else None,
    }


def batch_gd_logistic(
    x_features: np.ndarray,
    y: np.ndarray,
    step: float,
    lam: float,
    max_epochs: int,
    target_loss: float | None = None,
) -> dict[str, object]:
    coef = np.zeros(x_features.shape[1])
    intercept = 0.0
    history: list[float] = []
    for _ in range(max_epochs):
        grad_coef, grad_intercept = logistic_gradient(x_features, y, coef, intercept, lam)
        coef = coef - step * grad_coef
        intercept = float(intercept - step * grad_intercept)
        loss = mean_log_loss(x_features @ coef + intercept, y)
        history.append(loss)
        if not np.isfinite(loss):
            break
        if target_loss is not None and abs(loss - target_loss) <= GD_TRAIN_TOL:
            break
    return {
        "coef": coef,
        "intercept": intercept,
        "history": history,
        "epochs": len(history),
        "final_train_log_loss": history[-1] if history else None,
    }


def classification_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    positive = y_true == 1.0
    predicted = y_pred == 1.0
    tp = int(np.sum(positive & predicted))
    fn = int(np.sum(positive & ~predicted))
    fp = int(np.sum(~positive & predicted))
    tn = int(np.sum(~positive & ~predicted))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def rates_from_counts(counts: dict[str, int]) -> dict[str, float | None]:
    predicted_positive = counts["tp"] + counts["fp"]
    actual_positive = counts["tp"] + counts["fn"]
    total = actual_positive + counts["tn"] + counts["fp"]
    precision = None if predicted_positive == 0 else counts["tp"] / predicted_positive
    recall = None if actual_positive == 0 else counts["tp"] / actual_positive
    accuracy = (counts["tp"] + counts["tn"]) / total
    return {"precision_malignant": precision, "recall_malignant": recall, "accuracy": accuracy}


def clipped_log_loss(y: np.ndarray, raw: np.ndarray, eps: float = LOG_LOSS_CLIP) -> float:
    probability = np.clip(raw, eps, 1.0 - eps)
    return mean_log_loss_from_probability(y, probability)


def mean_log_loss_from_probability(y: np.ndarray, probability: np.ndarray) -> float:
    return float(-np.mean(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)))


def rmse_dollars(mse_value: float) -> float:
    return float(np.sqrt(mse_value) * 100_000.0)


def _require_close(name: str, diff: float, tol: float) -> None:
    if diff > tol:
        raise RuntimeError(f"{name} differs from scikit-learn by {diff}, above {tol}")


def fit_housing() -> dict[str, object]:
    bunch = fetch_california_housing()
    x = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target, dtype=float)
    names = [str(name) for name in bunch.feature_names]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED
    )
    x_train_s, x_test_s, mu, sigma = scale_from_train(x_train, x_test)
    design_train = add_intercept(x_train_s)
    design_test = add_intercept(x_test_s)
    ols = ols_closed_form(design_train, y_train)
    library = LinearRegression().fit(x_train_s, y_train)
    library_w = np.concatenate([[library.intercept_], library.coef_])
    ols_diff = float(np.max(np.abs(ols - library_w)))
    _require_close("housing OLS", ols_diff, 1e-8)

    ridge = ridge_closed_form(x_train_s, y_train, RIDGE_ALPHA)
    library_ridge = Ridge(alpha=RIDGE_ALPHA).fit(x_train_s, y_train)
    ridge_diff = float(
        max(
            np.max(np.abs(ridge[:-1] - library_ridge.coef_)),
            abs(ridge[-1] - library_ridge.intercept_),
        )
    )
    _require_close("housing ridge", ridge_diff, 1e-8)

    mean_prediction = float(y_train.mean())
    baseline_holdout = float(np.mean((y_test - mean_prediction) ** 2))
    ols_train = mse(design_train, y_train, ols)
    ols_pred = design_test @ ols
    ols_holdout = mse(design_test, y_test, ols)
    ridge_w = np.concatenate([[ridge[-1]], ridge[:-1]])
    ridge_holdout = mse(design_test, y_test, ridge_w)

    lipschitz_scaled = lipschitz_mse(design_train)
    step_scaled = 1.0 / lipschitz_scaled
    gd_scaled = batch_gd_linear(
        design_train, y_train, step_scaled, GD_MAX_EPOCHS, target_mse=ols_train
    )
    design_train_raw = add_intercept(x_train)
    design_test_raw = add_intercept(x_test)
    ols_raw = ols_closed_form(design_train_raw, y_train)
    ols_raw_train = mse(design_train_raw, y_train, ols_raw)
    lipschitz_raw = lipschitz_mse(design_train_raw)
    gd_raw = batch_gd_linear(
        design_train_raw,
        y_train,
        1.0 / lipschitz_raw,
        GD_MAX_EPOCHS,
        target_mse=ols_raw_train,
    )
    gd_oversize = batch_gd_linear(
        design_train_raw, y_train, step_scaled, GD_MAX_EPOCHS, target_mse=None
    )

    income = names.index("MedInc")
    rng = np.random.default_rng(COLLINEAR_SEED)
    train_copy = x_train[:, income] + rng.normal(0.0, COLLINEAR_NOISE, size=len(x_train))
    test_copy = x_test[:, income] + rng.normal(0.0, COLLINEAR_NOISE, size=len(x_test))
    x_train_c = np.column_stack([x_train, train_copy])
    x_test_c = np.column_stack([x_test, test_copy])
    x_train_cs, x_test_cs, _, _ = scale_from_train(x_train_c, x_test_c)
    ols_c = ols_closed_form(add_intercept(x_train_cs), y_train)
    collinear_holdout = mse(add_intercept(x_test_cs), y_test, ols_c)
    original_income = float(ols[income + 1])
    split_income = float(ols_c[income + 1])
    split_copy = float(ols_c[-1])

    coef_order = np.argsort(-np.abs(ols[1:]))
    largest = int(coef_order[0])
    return {
        "n_rows": int(len(y)),
        "n_features": int(x.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_train_mean": mean_prediction,
        "feature_names": names,
        "train_mean": mu.tolist(),
        "train_std_ddof1": sigma.tolist(),
        "mean_baseline_holdout_mse": baseline_holdout,
        "mean_baseline_holdout_rmse_dollars": rmse_dollars(baseline_holdout),
        "ols_train_mse": ols_train,
        "ols_holdout_mse": ols_holdout,
        "ols_holdout_prediction_min": float(ols_pred.min()),
        "ols_holdout_prediction_max": float(ols_pred.max()),
        "ols_holdout_predictions_above_target_max": int(np.sum(ols_pred > y.max())),
        "ols_holdout_predictions_below_zero": int(np.sum(ols_pred < 0.0)),
        "ols_holdout_rows_at_target_max": int(np.sum(y_test == y.max())),
        "ols_holdout_rmse_dollars": rmse_dollars(ols_holdout),
        "ols_holdout_r2_vs_mean": 1.0 - ols_holdout / baseline_holdout,
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_holdout_mse": ridge_holdout,
        "ridge_minus_ols_holdout_mse": ridge_holdout - ols_holdout,
        "sklearn_ols_max_abs_coef_diff": ols_diff,
        "sklearn_ridge_max_abs_coef_diff": ridge_diff,
        "scaled_ols_coefficients": ols[1:].tolist(),
        "scaled_ols_intercept": float(ols[0]),
        "largest_abs_scaled_coefficient": {
            "name": names[largest],
            "value": float(ols[largest + 1]),
        },
        "gd_scaled_step": step_scaled,
        "gd_scaled_lipschitz": lipschitz_scaled,
        "gd_scaled_epochs": gd_scaled["epochs"],
        "gd_scaled_final_train_mse": gd_scaled["final_train_mse"],
        "gd_scaled_train_gap_to_ols": float(gd_scaled["final_train_mse"]) - ols_train,
        "gd_scaled_holdout_mse": mse(design_test, y_test, gd_scaled["weights"]),
        "gd_unscaled_safe_step": 1.0 / lipschitz_raw,
        "gd_unscaled_lipschitz": lipschitz_raw,
        "gd_unscaled_safe_epochs": gd_raw["epochs"],
        "gd_unscaled_safe_final_train_mse": gd_raw["final_train_mse"],
        "gd_unscaled_safe_train_gap_to_ols": float(gd_raw["final_train_mse"]) - ols_raw_train,
        "gd_unscaled_safe_holdout_mse": mse(design_test_raw, y_test, gd_raw["weights"]),
        "gd_unscaled_ols_train_mse": ols_raw_train,
        "gd_unscaled_ols_holdout_mse": mse(design_test_raw, y_test, ols_raw),
        "gd_oversized_step": step_scaled,
        "gd_oversized_epochs": gd_oversize["epochs"],
        "gd_oversized_exploded_epoch": gd_oversize["exploded_epoch"],
        "gd_oversized_final_train_mse": gd_oversize["final_train_mse"],
        "collinear_noise_std": COLLINEAR_NOISE,
        "collinear_seed": COLLINEAR_SEED,
        "collinear_original_medinc_coefficient": original_income,
        "collinear_medinc_coefficient": split_income,
        "collinear_copy_coefficient": split_copy,
        "collinear_coefficient_sum": split_income + split_copy,
        "collinear_holdout_mse": collinear_holdout,
        "collinear_minus_ols_holdout_mse": collinear_holdout - ols_holdout,
        "plot": {
            "y_test": y_test,
            "ols_pred": ols_pred,
            "gd_scaled_history": gd_scaled["history"],
            "gd_raw_history": gd_raw["history"],
            "ols_train": ols_train,
            "ols_raw_train": ols_raw_train,
        },
    }


def fit_cancer() -> dict[str, object]:
    bunch = load_breast_cancer()
    x = np.asarray(bunch.data, dtype=float)
    # sklearn target 0 is malignant. The positive class here is malignant.
    y = (np.asarray(bunch.target) == 0).astype(float)
    names = [str(name) for name in bunch.feature_names]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    x_train_s, x_test_s, _, _ = scale_from_train(x_train, x_test)
    n_train = len(y_train)
    lam = sklearn_l2_strength(n_train, LOGISTIC_C)
    newton = newton_logistic(x_train_s, y_train, lam)
    library = LogisticRegression(
        C=LOGISTIC_C,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-12,
        random_state=SEED,
    ).fit(x_train_s, y_train.astype(int))
    coef_diff = float(np.max(np.abs(newton["coef"] - library.coef_[0])))
    intercept_diff = abs(float(newton["intercept"]) - float(library.intercept_[0]))
    _require_close("cancer logistic coefficients", max(coef_diff, intercept_diff), LIBRARY_COEF_TOL)

    test_probability = sigmoid(x_test_s @ newton["coef"] + newton["intercept"])
    library_test = library.predict_proba(x_test_s)[:, 1]
    prob_diff = float(np.max(np.abs(test_probability - library_test)))
    _require_close("cancer logistic probabilities", prob_diff, 1e-6)

    counts = classification_counts(y_test, (test_probability >= 0.5).astype(float))
    rate = rates_from_counts(counts)
    train_rate = float(y_train.mean())
    constant = np.full(len(y_test), train_rate)
    majority = 1.0 if train_rate >= 0.5 else 0.0
    majority_accuracy = float(np.mean(y_test == majority))

    unreg = newton_logistic(x_train_s, y_train, 0.0)
    unreg_test = sigmoid(x_test_s @ unreg["coef"] + unreg["intercept"])
    library_unreg = LogisticRegression(
        C=np.inf,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-12,
        random_state=SEED,
    ).fit(x_train_s, y_train.astype(int))
    unreg_library_test = library_unreg.predict_proba(x_test_s)[:, 1]
    unreg_prob_diff = float(np.max(np.abs(unreg_test - unreg_library_test)))
    unreg_train_z = x_train_s @ unreg["coef"] + unreg["intercept"]
    unreg_margin = unreg_train_z * (2.0 * y_train - 1.0)
    library_unreg_z_train = x_train_s @ library_unreg.coef_[0] + library_unreg.intercept_[0]
    library_unreg_z_test = x_test_s @ library_unreg.coef_[0] + library_unreg.intercept_[0]

    step = 1.0 / lipschitz_logistic(x_train_s, lam)
    gd = batch_gd_logistic(
        x_train_s,
        y_train,
        step,
        lam,
        GD_MAX_EPOCHS,
        target_loss=float(newton["train_log_loss"]),
    )
    label_fit = LinearRegression().fit(x_train_s, y_train)
    raw_test = label_fit.predict(x_test_s)
    outside = float(np.mean((raw_test < 0.0) | (raw_test > 1.0)))
    linear_loss = clipped_log_loss(y_test, raw_test)
    linear_counts = classification_counts(
        y_test, (np.clip(raw_test, 0.0, 1.0) >= 0.5).astype(float)
    )
    linear_rate = rates_from_counts(linear_counts)
    outside_count = int(np.sum((raw_test < 0.0) | (raw_test > 1.0)))

    radius = names.index("mean radius")
    perimeter = names.index("mean perimeter")
    radius_perimeter_corr = float(np.corrcoef(x_train[:, radius], x_train[:, perimeter])[0, 1])
    largest = int(np.argmax(np.abs(newton["coef"])))
    return {
        "n_rows": int(len(y)),
        "n_features": int(x.shape[1]),
        "n_malignant": int(y.sum()),
        "n_benign": int(len(y) - y.sum()),
        "positive_class": "malignant",
        "sklearn_target_zero": "malignant",
        "n_train": n_train,
        "n_test": int(len(y_test)),
        "n_train_malignant": int(y_train.sum()),
        "n_test_malignant": int(y_test.sum()),
        "train_malignant_rate": train_rate,
        "test_malignant_rate": float(y_test.mean()),
        "C": LOGISTIC_C,
        "l2_strength_on_average_loss": lam,
        "newton_iterations": newton["iterations"],
        "newton_damped_steps": newton["damped_steps"],
        "newton_grad_norm": newton["grad_norm"],
        "newton_train_log_loss": newton["train_log_loss"],
        "newton_holdout_log_loss": mean_log_loss(
            x_test_s @ newton["coef"] + newton["intercept"], y_test
        ),
        "constant_rate_holdout_log_loss": mean_log_loss_from_probability(y_test, constant),
        "majority_class": int(majority),
        "majority_holdout_accuracy": majority_accuracy,
        "holdout_accuracy": rate["accuracy"],
        "holdout_recall_malignant": rate["recall_malignant"],
        "holdout_precision_malignant": rate["precision_malignant"],
        "confusion_at_0_5": counts,
        "sklearn_max_abs_coef_diff": max(coef_diff, intercept_diff),
        "sklearn_max_abs_probability_diff": prob_diff,
        "sklearn_n_iter": int(library.n_iter_[0]),
        "unregularized_train_log_loss": unreg["train_log_loss"],
        "unregularized_train_accuracy": float(np.mean((sigmoid(unreg_train_z) >= 0.5) == y_train)),
        "unregularized_min_train_margin": float(unreg_margin.min()),
        "unregularized_holdout_log_loss": mean_log_loss(
            x_test_s @ unreg["coef"] + unreg["intercept"], y_test
        ),
        "unregularized_sklearn_train_log_loss": mean_log_loss(library_unreg_z_train, y_train),
        "unregularized_sklearn_holdout_log_loss": mean_log_loss(library_unreg_z_test, y_test),
        "unregularized_sklearn_max_abs_probability_diff": unreg_prob_diff,
        "unregularized_max_abs_coefficient": float(np.max(np.abs(unreg["coef"]))),
        "unregularized_sklearn_max_abs_coefficient": float(np.max(np.abs(library_unreg.coef_))),
        "regularized_max_abs_coefficient": float(np.max(np.abs(newton["coef"]))),
        "gd_step": step,
        "gd_epochs": gd["epochs"],
        "gd_final_train_log_loss": gd["final_train_log_loss"],
        "gd_train_gap_to_newton": (
            float(gd["final_train_log_loss"]) - float(newton["train_log_loss"])
        ),
        "gd_holdout_log_loss": mean_log_loss(x_test_s @ gd["coef"] + gd["intercept"], y_test),
        "linear_on_labels_holdout_log_loss_clipped": linear_loss,
        "linear_on_labels_clip": LOG_LOSS_CLIP,
        "linear_on_labels_outside_count": outside_count,
        "linear_on_labels_fraction_outside_unit_interval": outside,
        "linear_on_labels_holdout_accuracy": linear_rate["accuracy"],
        "linear_on_labels_min_prediction": float(raw_test.min()),
        "linear_on_labels_max_prediction": float(raw_test.max()),
        "linear_on_labels_confusion_at_0_5": linear_counts,
        "mean_radius_mean_perimeter_train_correlation": radius_perimeter_corr,
        "largest_abs_scaled_coefficient": {
            "name": names[largest],
            "value": float(newton["coef"][largest]),
        },
        "plot": {
            "y_test": y_test,
            "probability": test_probability,
            "linear_raw": raw_test,
            "gd_history": gd["history"],
            "newton_train_log_loss": float(newton["train_log_loss"]),
        },
    }


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)


def plot_housing(housing: dict[str, object]) -> None:
    plot = housing["plot"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(plot["y_test"], plot["ols_pred"], s=8, alpha=0.35, color="#1f77b4", linewidths=0)
    limits = [
        min(plot["y_test"].min(), plot["ols_pred"].min()),
        max(plot["y_test"].max(), plot["ols_pred"].max()),
    ]
    ax.plot(limits, limits, color="#333333", linewidth=1.0, label="Predicted = actual")
    ax.set_xlabel("Actual median house value ($100,000)")
    ax.set_ylabel("OLS prediction ($100,000)")
    ax.set_title("Holdout OLS predictions")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "housing_predictions.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    scaled_epochs = np.arange(1, len(plot["gd_scaled_history"]) + 1)
    axes[0].plot(scaled_epochs, plot["gd_scaled_history"], color="#1f77b4")
    axes[0].axhline(
        plot["ols_train"], color="#333333", linewidth=1.0, label="Closed-form train MSE"
    )
    axes[0].set_title("Scaled features, step 1/L")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train MSE")
    axes[0].legend(frameon=False)
    raw_epochs = np.arange(1, len(plot["gd_raw_history"]) + 1)
    axes[1].plot(raw_epochs, plot["gd_raw_history"], color="#ff7f0e")
    axes[1].axhline(
        plot["ols_raw_train"], color="#333333", linewidth=1.0, label="Closed-form train MSE"
    )
    axes[1].set_title("Raw features, their own step 1/L")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train MSE")
    axes[1].legend(frameon=False)
    over_epoch = housing["gd_oversized_exploded_epoch"]
    fig.suptitle(
        f"The scaled step, applied to raw features, stopped at epoch {over_epoch}",
        fontsize=11,
    )
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "housing_gd.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].bar(
        ["Original MedInc", "Sum of the two pieces"],
        [
            housing["collinear_original_medinc_coefficient"],
            housing["collinear_coefficient_sum"],
        ],
        color=["#1f77b4", "#9467bd"],
    )
    axes[0].set_ylabel("Scaled OLS coefficient")
    axes[0].set_title("The pieces still add up")
    axes[1].bar(
        ["MedInc", "Copy of MedInc"],
        [
            housing["collinear_medinc_coefficient"],
            housing["collinear_copy_coefficient"],
        ],
        color=["#ff7f0e", "#2ca02c"],
    )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Scaled OLS coefficient")
    axes[1].set_title("Each piece is large and opposite")
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "housing_collinear.png", dpi=120)
    plt.close(fig)


def plot_cancer(cancer: dict[str, object]) -> None:
    plot = cancer["plot"]
    y_test = plot["y_test"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=False)
    bins = np.linspace(0.0, 1.0, 21)
    for label, mask, color in (
        ("Benign", y_test == 0.0, "#1f77b4"),
        ("Malignant", y_test == 1.0, "#d62728"),
    ):
        axes[0].hist(
            plot["probability"][mask],
            bins=bins,
            alpha=0.65,
            label=label,
            color=color,
        )
    axes[0].set_xlabel("Newton P(malignant)")
    axes[0].set_ylabel("Holdout count")
    axes[0].set_title("Logistic probabilities stay in (0, 1)")
    axes[0].legend(frameon=False)
    raw_bins = np.linspace(plot["linear_raw"].min(), plot["linear_raw"].max(), 21)
    for label, mask, color in (
        ("Benign", y_test == 0.0, "#1f77b4"),
        ("Malignant", y_test == 1.0, "#d62728"),
    ):
        axes[1].hist(plot["linear_raw"][mask], bins=raw_bins, alpha=0.65, label=label, color=color)
    axes[1].axvline(0.0, color="#333333", linewidth=1.0)
    axes[1].axvline(1.0, color="#333333", linewidth=1.0, label="0 and 1")
    axes[1].set_xlabel("Linear regression score fit to 0/1 labels")
    axes[1].set_ylabel("Holdout count")
    outside_count = cancer["linear_on_labels_outside_count"]
    axes[1].set_title(f"{outside_count} of {cancer['n_test']} scores outside [0, 1]")
    axes[1].legend(frameon=False)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "cancer_scores.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    history = plot["gd_history"]
    ax.plot(
        np.arange(1, len(history) + 1),
        history,
        color="#1f77b4",
        label="Batch gradient descent",
    )
    ax.axhline(
        plot["newton_train_log_loss"],
        color="#333333",
        linewidth=1.0,
        label="Newton train log loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train mean log loss")
    ax.set_title("Same convex objective, slower algorithm")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "cancer_gd.png", dpi=120)
    plt.close(fig)


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if key != "plot"}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    housing = fit_housing()
    cancer = fit_cancer()
    plot_housing(housing)
    plot_cancer(cancer)
    payload = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "gd_max_epochs": GD_MAX_EPOCHS,
        "elapsed_seconds": time.perf_counter() - started,
        "housing": _jsonable(housing),
        "cancer": _jsonable(cancer),
    }
    text = json.dumps(payload, indent=2) + "\n"
    (RESULTS / "metrics.json").write_text(text, encoding="utf-8", newline="\n")
    print(
        "housing OLS holdout MSE",
        housing["ols_holdout_mse"],
        "cancer holdout log loss",
        cancer["newton_holdout_log_loss"],
        "elapsed",
        payload["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
