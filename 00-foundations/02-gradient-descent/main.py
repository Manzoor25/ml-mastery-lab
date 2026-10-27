"""Gradient descent variants on a visible bowl and on UCI wine alcohol.

The two-dimensional problem is simulated so the loss surface can be drawn.
The alcohol regression uses the UCI wine measurements. Optimizers are judged
by distance to the least-squares solution, not by a story about the path.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")

SEED = 42
WINE_EPOCHS = 250
SURFACE_EPOCHS = 40
MOMENTUM_BETA = 0.9
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
MINIBATCH = 16
FAIL_MULTIPLE = 2.5
ADAM_LR = 0.05
ADAM_PAPER_LR = 0.001
SGD_LR_FRACTION = 0.2
EXPLODE_MSE = 1_000_000.0
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def mse(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    resid = x @ w - y
    return float(np.mean(resid**2))


def grad_mse(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of mean squared error. One row or many rows, same formula."""
    resid = x @ w - y
    return (2.0 / len(y)) * (x.T @ resid)


def hessian_spectrum(x: np.ndarray) -> tuple[float, float]:
    """Smallest and largest eigenvalues of the MSE Hessian, (2/n) X^T X."""
    n_rows = x.shape[0]
    hessian = (2.0 / n_rows) * (x.T @ x)
    evals = np.linalg.eigvalsh(hessian)
    return float(evals[0]), float(evals[-1])


def closed_form(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    weights, *_ = np.linalg.lstsq(x, y, rcond=None)
    return np.asarray(weights, dtype=float)


def scale_from_train(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0, ddof=1)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    return (x_train - mu) / sigma, (x_test - mu) / sigma


def finite_difference_grad(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    numeric = np.empty_like(w)
    for j in range(len(w)):
        step = np.zeros_like(w)
        step[j] = eps
        numeric[j] = (mse(x, y, w + step) - mse(x, y, w - step)) / (2.0 * eps)
    return numeric


def _batches(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_rows = len(y)
    if method == "sgd":
        order = rng.permutation(n_rows)
        return [(x[i : i + 1], y[i : i + 1]) for i in order]
    if method == "minibatch":
        order = rng.permutation(n_rows)
        chunks = []
        for start in range(0, n_rows, batch_size):
            sl = order[start : start + batch_size]
            chunks.append((x[sl], y[sl]))
        return chunks
    return [(x, y)]


def run_optimizer(
    x: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    method: str,
    epochs: int,
    lr: float,
    rng: np.random.Generator,
    batch_size: int = MINIBATCH,
    beta: float = MOMENTUM_BETA,
) -> dict[str, object]:
    """One data pass per epoch. Stops if the loss explodes or a weight is non-finite."""
    w = w0.astype(float).copy()
    velocity = np.zeros_like(w)
    moment = np.zeros_like(w)
    second = np.zeros_like(w)
    step_count = 0
    history = [mse(x, y, w)]
    path = [w.copy()]
    exploded = False
    for _ in range(epochs):
        for xb, yb in _batches(x, y, method, batch_size, rng):
            grad = grad_mse(xb, yb, w)
            if method == "momentum":
                velocity = beta * velocity + grad
                w = w - lr * velocity
            elif method == "adam":
                step_count += 1
                moment = ADAM_BETA1 * moment + (1.0 - ADAM_BETA1) * grad
                second = ADAM_BETA2 * second + (1.0 - ADAM_BETA2) * (grad**2)
                mhat = moment / (1.0 - ADAM_BETA1**step_count)
                vhat = second / (1.0 - ADAM_BETA2**step_count)
                w = w - lr * mhat / (np.sqrt(vhat) + ADAM_EPS)
            else:
                w = w - lr * grad
            if not np.all(np.isfinite(w)):
                exploded = True
                break
            if mse(x, y, w) > EXPLODE_MSE:
                exploded = True
                break
        if exploded:
            break
        history.append(mse(x, y, w))
        path.append(w.copy())
    return {
        "weights": None if exploded else w,
        "history": history,
        "path": np.vstack(path),
        "exploded": exploded,
        "epochs_completed": len(history) - 1,
    }


def epochs_to_match(history: list[float], target: float, rel: float = 0.01) -> int | None:
    """First epoch whose loss is at most 1% above the closed-form loss."""
    limit = target * (1.0 + rel)
    for epoch, loss in enumerate(history):
        if loss <= limit:
            return epoch
    return None


def make_surface(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ill-conditioned two-feature least squares. Simulated; the caller owns the seed."""
    n_rows = 120
    latent = rng.normal(size=n_rows)
    x = np.column_stack(
        [
            latent + 0.05 * rng.normal(size=n_rows),
            0.25 * latent + 0.05 * rng.normal(size=n_rows),
        ]
    )
    w_true = np.array([1.5, -2.0])
    y = x @ w_true + rng.normal(scale=0.35, size=n_rows)
    return x, y, w_true


def alcohol_arrays() -> tuple[np.ndarray, np.ndarray, list[str]]:
    wine = load_wine()
    names = list(wine.feature_names)
    alcohol = names.index("alcohol")
    cols = [i for i in range(len(names)) if i != alcohol]
    x = np.asarray(wine.data[:, cols], dtype=float)
    y = np.asarray(wine.data[:, alcohol], dtype=float)
    return x, y, [names[i] for i in cols]


def split_indices(
    n_rows: int, rng: np.random.Generator, test_fraction: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n_rows)
    n_test = int(n_rows * test_fraction)
    return order[n_test:], order[:n_test]


def _fit_record(
    name: str,
    method: str,
    lr: float,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_mu: float,
    y_test_orig: np.ndarray,
    w0: np.ndarray,
    w_closed: np.ndarray,
    closed_train_mse: float,
    epochs: int,
) -> dict[str, object]:
    fitted = run_optimizer(
        x_train,
        y_train,
        w0,
        method,
        epochs,
        lr,
        np.random.default_rng(seed),
    )
    history = [float(value) for value in fitted["history"]]
    record: dict[str, object] = {
        "name": name,
        "method": method,
        "lr": float(lr),
        "seed": seed,
        "exploded": bool(fitted["exploded"]),
        "epochs_completed": int(fitted["epochs_completed"]),
        "last_completed_train_mse": history[-1],
        "history": history,
        "path": fitted["path"],
    }
    if fitted["exploded"] or fitted["weights"] is None:
        record["final_train_mse"] = None
        record["final_test_mse"] = None
        record["weight_l2_to_closed_form"] = None
        record["epochs_to_1pct"] = None
        return record
    weights = np.asarray(fitted["weights"], dtype=float)
    train_pred = x_train @ weights + y_mu
    test_pred = x_test @ weights + y_mu
    record["final_train_mse"] = float(np.mean((train_pred - (y_train + y_mu)) ** 2))
    record["final_test_mse"] = float(np.mean((test_pred - y_test_orig) ** 2))
    record["weight_l2_to_closed_form"] = float(np.linalg.norm(weights - w_closed))
    record["epochs_to_1pct"] = epochs_to_match(history, closed_train_mse)
    return record


def wine_problem() -> dict[str, object]:
    x, y, feature_names = alcohol_arrays()
    train_idx, test_idx = split_indices(len(y), np.random.default_rng(SEED))
    x_train, x_test = scale_from_train(x[train_idx], x[test_idx])
    y_mu = float(y[train_idx].mean())
    y_train = y[train_idx] - y_mu
    y_test_orig = y[test_idx]
    w_closed = closed_form(x_train, y_train)
    sklearn_coef = LinearRegression(fit_intercept=False).fit(x_train, y_train).coef_
    train_mse = float(np.mean((x_train @ w_closed - y_train) ** 2))
    test_mse = float(np.mean((x_test @ w_closed + y_mu - y_test_orig) ** 2))
    mean_test = float(np.mean((y_test_orig - y_mu) ** 2))
    mean_train = float(np.mean(y_train**2))
    smallest, smooth = hessian_spectrum(x_train)
    lr_safe = 1.0 / smooth
    specs = [
        ("batch_gd_safe", "batch", lr_safe, 0),
        ("batch_gd_too_large", "batch", FAIL_MULTIPLE / smooth, 0),
        ("minibatch_16", "minibatch", lr_safe, SEED + 1),
        ("sgd", "sgd", SGD_LR_FRACTION / smooth, SEED + 2),
        ("momentum_naive_lr", "momentum", lr_safe, 0),
        ("momentum_scaled_lr", "momentum", (1.0 - MOMENTUM_BETA) / smooth, 0),
        ("adam_lr_0.05", "adam", ADAM_LR, SEED + 3),
        ("adam_paper_default", "adam", ADAM_PAPER_LR, SEED + 4),
    ]
    w0 = np.zeros(x_train.shape[1])
    runs = [
        _fit_record(
            name,
            method,
            lr,
            seed,
            x_train,
            y_train,
            x_test,
            y_mu,
            y_test_orig,
            w0,
            w_closed,
            train_mse,
            WINE_EPOCHS,
        )
        for name, method, lr, seed in specs
    ]
    probe = np.linspace(-0.5, 0.5, x_train.shape[1])
    analytic = grad_mse(x_train, y_train, probe)
    numeric = finite_difference_grad(x_train, y_train, probe)
    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(x_train.shape[1]),
        "feature_names": feature_names,
        "y_train_mean": y_mu,
        "lipschitz": smooth,
        "smallest_hessian_eigenvalue": smallest,
        "lr_safe": lr_safe,
        "condition_number": float(smooth / smallest),
        "mean_baseline_train_mse": mean_train,
        "mean_baseline_test_mse": mean_test,
        "closed_form_train_mse": train_mse,
        "closed_form_test_mse": test_mse,
        "sklearn_closed_form_max_abs_diff": float(np.max(np.abs(sklearn_coef - w_closed))),
        "gradient_check_max_abs": float(np.max(np.abs(analytic - numeric))),
        "runs": runs,
    }


def surface_problem() -> dict[str, object]:
    x, y, w_true = make_surface(np.random.default_rng(SEED + 7))
    w_closed = closed_form(x, y)
    smallest, smooth = hessian_spectrum(x)
    lr_safe = 1.0 / smooth
    w0 = w_closed + np.array([2.5, -2.0])
    specs = [
        ("batch_gd_safe", "batch", lr_safe, 0),
        ("batch_gd_too_large", "batch", FAIL_MULTIPLE / smooth, 0),
        ("momentum_naive_lr", "momentum", lr_safe, 0),
        ("momentum_scaled_lr", "momentum", (1.0 - MOMENTUM_BETA) / smooth, 0),
        ("sgd", "sgd", SGD_LR_FRACTION / smooth, SEED + 8),
        ("adam_lr_0.05", "adam", ADAM_LR, SEED + 9),
    ]
    runs = [
        _fit_record(
            name,
            method,
            lr,
            seed,
            x,
            y,
            x,
            0.0,
            y,
            w0,
            w_closed,
            mse(x, y, w_closed),
            SURFACE_EPOCHS,
        )
        for name, method, lr, seed in specs
    ]
    return {
        "n": int(len(y)),
        "w_true": w_true,
        "w_closed": w_closed,
        "w0": w0,
        "lipschitz": smooth,
        "condition_number": float(smooth / smallest),
        "closed_form_mse": mse(x, y, w_closed),
        "start_mse": mse(x, y, w0),
        "distance_closed_form_to_true": float(np.linalg.norm(w_closed - w_true)),
        "runs": runs,
        "x": x,
        "y": y,
    }


def _public_run(run: dict[str, object]) -> dict[str, object]:
    skip = {"history", "path"}
    return {key: value for key, value in run.items() if key not in skip}


def save_contour(surface: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(surface["x"])
    y = np.asarray(surface["y"])
    w_closed = np.asarray(surface["w_closed"])
    w0 = np.asarray(surface["w0"])
    axis = np.linspace(-3.2, 3.2, 160)
    w1 = w_closed[0] + axis
    w2 = w_closed[1] + axis
    ww1, ww2 = np.meshgrid(w1, w2)
    weights = np.stack([ww1, ww2], axis=-1)
    loss = np.mean((weights @ x.T - y) ** 2, axis=-1)
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    levels = np.quantile(loss, np.linspace(0.02, 0.85, 12))
    ax.contourf(ww1, ww2, loss, levels=levels, cmap="Blues", alpha=0.9)
    contour = ax.contour(ww1, ww2, loss, levels=levels, colors="#4D4D4D", linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=7, fmt="%.2f")
    colors = {
        "batch_gd_safe": "#0072B2",
        "momentum_scaled_lr": "#009E73",
        "sgd": "#E69F00",
        "adam_lr_0.05": "#CC79A7",
    }
    labels = {
        "batch_gd_safe": "Batch GD, lr = 1/L",
        "momentum_scaled_lr": "Momentum, lr = (1-beta)/L",
        "sgd": "SGD, lr = 0.2/L",
        "adam_lr_0.05": "Adam, lr = 0.05",
    }
    ordered = sorted(surface["runs"], key=lambda run: run["name"] == "batch_gd_safe")
    for run in ordered:
        if run["name"] not in colors:
            continue
        trail = np.asarray(run["path"])
        inside = (
            (trail[:, 0] > w1[0])
            & (trail[:, 0] < w1[-1])
            & (trail[:, 1] > w2[0])
            & (trail[:, 1] < w2[-1])
        )
        drawn = trail[inside]
        if len(drawn) == 0:
            continue
        mark_every = 4 if run["name"] == "batch_gd_safe" else None
        ax.plot(
            drawn[:, 0],
            drawn[:, 1],
            color=colors[run["name"]],
            linewidth=2.0,
            marker="o" if run["name"] == "batch_gd_safe" else None,
            markersize=3.5,
            markevery=mark_every,
            label=labels[run["name"]],
            zorder=3 if run["name"] == "batch_gd_safe" else 2,
        )
    ax.scatter([w0[0]], [w0[1]], c="black", s=28, zorder=4, label="Start")
    ax.scatter(
        [w_closed[0]],
        [w_closed[1]],
        c="#D55E00",
        marker="*",
        s=90,
        zorder=4,
        label="Least squares",
    )
    ax.set_xlabel("Weight on feature 1")
    ax.set_ylabel("Weight on feature 2")
    condition = float(surface["condition_number"])
    ax.set_title(f"Simulated least squares: valley, condition number {condition:.1f}")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_curves(runs: list[dict[str, object]], closed_mse: float, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    colors = {
        "batch_gd_safe": "#0072B2",
        "minibatch_16": "#56B4E9",
        "sgd": "#E69F00",
        "momentum_scaled_lr": "#009E73",
        "adam_lr_0.05": "#CC79A7",
        "adam_paper_default": "#000000",
    }
    labels = {
        "batch_gd_safe": "Batch GD, lr = 1/L",
        "minibatch_16": "Mini-batch 16, lr = 1/L",
        "sgd": "SGD, lr = 0.2/L",
        "momentum_scaled_lr": "Momentum, lr = (1-beta)/L",
        "adam_lr_0.05": "Adam, lr = 0.05",
        "adam_paper_default": "Adam, lr = 0.001",
    }
    for run in runs:
        if run["name"] not in colors or run["exploded"]:
            continue
        history = run["history"]
        ax.plot(
            np.arange(len(history)), history, color=colors[run["name"]], label=labels[run["name"]]
        )
    ax.axhline(closed_mse, color="#D55E00", linestyle="--", label="Least-squares loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (one pass over the training rows)")
    ax.set_ylabel("Train mean squared error")
    ax.set_title("Wine alcohol: training loss on a log scale")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_test_bars(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    labels = [str(row["label"]) for row in rows]
    values = [float(row["test_mse"]) for row in rows]
    colors = ["#666666" if label == "Train-mean baseline" else "#0072B2" for label in labels]
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("Test mean squared error")
    ax.set_xlabel("Predictor")
    ax.set_title("Wine alcohol: held-out error after a fixed epoch budget")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def analyze() -> dict[str, object]:
    surface = surface_problem()
    wine = wine_problem()
    metrics: dict[str, object] = {
        "seed": SEED,
        "wine_epochs": WINE_EPOCHS,
        "surface_epochs": SURFACE_EPOCHS,
        "momentum_beta": MOMENTUM_BETA,
        "fail_lr_multiple": FAIL_MULTIPLE,
        "explode_mse": EXPLODE_MSE,
        "surface_n": surface["n"],
        "surface_condition_number": surface["condition_number"],
        "surface_lipschitz": surface["lipschitz"],
        "surface_start_mse": surface["start_mse"],
        "surface_start_l2_to_closed_form": float(
            np.linalg.norm(np.asarray(surface["w0"]) - np.asarray(surface["w_closed"]))
        ),
        "surface_closed_form_mse": surface["closed_form_mse"],
        "surface_closed_form_l2_to_generating_weights": surface["distance_closed_form_to_true"],
        "surface_runs": [_public_run(run) for run in surface["runs"]],
        "n_train": wine["n_train"],
        "n_test": wine["n_test"],
        "n_features": wine["n_features"],
        "wine_condition_number": wine["condition_number"],
        "wine_lipschitz": wine["lipschitz"],
        "lr_safe": wine["lr_safe"],
        "mean_baseline_train_mse": wine["mean_baseline_train_mse"],
        "mean_baseline_test_mse": wine["mean_baseline_test_mse"],
        "closed_form_train_mse": wine["closed_form_train_mse"],
        "closed_form_test_mse": wine["closed_form_test_mse"],
        "sklearn_closed_form_max_abs_diff": wine["sklearn_closed_form_max_abs_diff"],
        "gradient_check_max_abs": wine["gradient_check_max_abs"],
        "wine_runs": [_public_run(run) for run in wine["runs"]],
    }
    return {"metrics": metrics, "surface": surface, "wine": wine}


def write_outputs(bundle: dict[str, object]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics = bundle["metrics"]
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_contour(bundle["surface"], RESULTS / "loss_surface.png")
    wine = bundle["wine"]
    save_curves(wine["runs"], float(wine["closed_form_train_mse"]), RESULTS / "train_loss.png")
    bars = [
        {"label": "Train-mean baseline", "test_mse": wine["mean_baseline_test_mse"]},
        {"label": "Least squares", "test_mse": wine["closed_form_test_mse"]},
    ]
    nice = {
        "batch_gd_safe": "Batch GD",
        "minibatch_16": "Mini-batch",
        "sgd": "SGD",
        "momentum_scaled_lr": "Momentum",
        "adam_lr_0.05": "Adam 0.05",
        "adam_paper_default": "Adam 0.001",
    }
    for run in wine["runs"]:
        if run["name"] in nice and not run["exploded"]:
            bars.append({"label": nice[run["name"]], "test_mse": run["final_test_mse"]})
    save_test_bars(bars, RESULTS / "test_mse.png")


def main() -> None:
    bundle = analyze()
    write_outputs(bundle)
    print(json.dumps(bundle["metrics"], indent=2))


if __name__ == "__main__":
    main()
