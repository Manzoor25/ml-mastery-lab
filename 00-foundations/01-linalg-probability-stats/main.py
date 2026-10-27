"""Linear algebra and sampling uncertainty on the UCI wine data.

Descriptive analysis only. There is no train/test split because nothing here
is a predictive model. Hypothesis tests use the chemical measurements as they
are, not a scaled copy, so the mean gap stays in the original units.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats
from sklearn.datasets import load_wine

matplotlib.use("Agg")

SEED = 42
N_BOOT = 10_000
N_PERM = 10_000
N_NULL = 1_000
ALPHA = 0.05
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score columns with the sample standard deviation (ddof=1)."""
    mu = x.mean(axis=0)
    sigma = x.std(axis=0, ddof=1)
    return (x - mu) / sigma


def center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0)


def pca_eigh(x_centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen decomposition of the sample covariance of a centered matrix.

    Returns eigenvalues and eigenvectors sorted from largest eigenvalue to
    smallest. Eigenvectors are columns.
    """
    cov = np.cov(x_centered, rowvar=False, ddof=1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def variance_fractions(evals: np.ndarray) -> np.ndarray:
    total = float(evals.sum())
    if total <= 0:
        raise ValueError("eigenvalues have no positive mass")
    return evals / total


def svd_variance_fractions(x_centered: np.ndarray) -> np.ndarray:
    """Same fractions via SVD. Used only as a check on the eigen path."""
    n_rows = x_centered.shape[0]
    _u, singular, _vt = np.linalg.svd(x_centered, full_matrices=False)
    evals = (singular**2) / (n_rows - 1)
    return variance_fractions(evals)


def reconstruct(x_centered: np.ndarray, evecs: np.ndarray, k: int) -> np.ndarray:
    basis = evecs[:, :k]
    scores = x_centered @ basis
    return scores @ basis.T


def column_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(a.mean() - b.mean())


def bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Percentile bootstrap of the difference of means. Resamples each group."""
    n_a = len(a)
    n_b = len(b)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        a_s = a[rng.integers(0, n_a, n_a)]
        b_s = b[rng.integers(0, n_b, n_b)]
        diffs[i] = mean_diff(a_s, b_s)
    return diffs


def percentile_ci(samples: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def welch_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    result = stats.ttest_ind(a, b, equal_var=False)
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "df": float(result.df),
    }


def permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided permutation p-value for the difference of means.

    The +1 correction counts the observed sample, so the p-value cannot be 0.
    """
    observed = abs(mean_diff(a, b))
    pooled = np.concatenate([a, b]).copy()
    n_a = len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(mean_diff(pooled[:n_a], pooled[n_a:]))
        hits += int(diff >= observed)
    return (hits + 1) / (n_perm + 1)


def family_wise_null_rate(
    x: np.ndarray,
    y: np.ndarray,
    n_sim: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    """How often any feature looks significant when labels are shuffled.

    Each simulation permutes the labels, then runs a Welch test of class 0
    against class 1 on every column. A hit is at least one p-value below alpha.
    """
    hits = 0
    n_features = x.shape[1]
    for _ in range(n_sim):
        y_perm = rng.permutation(y)
        any_sig = False
        for j in range(n_features):
            left = x[y_perm == 0, j]
            right = x[y_perm == 1, j]
            pvalue = welch_test(left, right)["pvalue"]
            if np.isfinite(pvalue) and pvalue < alpha:
                any_sig = True
                break
        hits += int(any_sig)
    return hits / n_sim


def binomial_se(rate: float, n_sim: int) -> float:
    return float(np.sqrt(rate * (1.0 - rate) / n_sim))


def _jsonable(value: object) -> object:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [float(item) for item in value]
    return value


def save_scree(fractions: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    cumulative = np.cumsum(fractions)
    index = np.arange(1, len(fractions) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(index, fractions, color="#0072B2", label="This component")
    ax.plot(index, cumulative, color="#D55E00", marker="o", label="Cumulative")
    ax.set_xlabel("Principal component (scaled features)")
    ax.set_ylabel("Fraction of variance")
    ax.set_title("Scaled wine features: variance explained by each component")
    ax.set_xticks(index)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_bootstrap(
    diffs: np.ndarray,
    observed: float,
    ci: tuple[float, float],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(diffs, bins=40, color="#0072B2", edgecolor="white")
    ax.axvline(observed, color="#000000", linewidth=1.5, label="Observed difference")
    ax.axvline(ci[0], color="#D55E00", linestyle="--", label="95% percentile CI")
    ax.axvline(ci[1], color="#D55E00", linestyle="--")
    ax.set_xlabel("Mean alcohol difference, class_0 minus class_1 (dataset units)")
    ax.set_ylabel("Bootstrap replicates")
    ax.set_title("Bootstrap distribution of the alcohol mean difference")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_scatter(
    scores: np.ndarray,
    y: np.ndarray,
    fractions: np.ndarray,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label in (0, 1, 2):
        mask = y == label
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=28,
            alpha=0.85,
            c=colors[label],
            label=f"class_{label}",
        )
    ax.set_xlabel(f"PC1 ({fractions[0]:.1%} of scaled variance)")
    ax.set_ylabel(f"PC2 ({fractions[1]:.1%} of scaled variance)")
    ax.set_title("Wine cultivars in the first two scaled principal components")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def analyze() -> dict[str, object]:
    wine = load_wine()
    x = np.asarray(wine.data, dtype=float)
    y = np.asarray(wine.target, dtype=int)
    names = list(wine.feature_names)
    alcohol_idx = names.index("alcohol")

    centered = center(x)
    unscaled_evals, unscaled_evecs = pca_eigh(centered)
    unscaled_frac = variance_fractions(unscaled_evals)
    unscaled_recon = reconstruct(centered, unscaled_evecs, k=1)
    unscaled_alcohol_r2 = column_r2(
        centered[:, alcohol_idx],
        unscaled_recon[:, alcohol_idx],
    )
    unscaled_pc1_feature = names[int(np.argmax(np.abs(unscaled_evecs[:, 0])))]
    unscaled_pc1_loading = float(np.max(np.abs(unscaled_evecs[:, 0])))
    proline_idx = names.index("proline")

    scaled = standardize(x)
    scaled_evals, scaled_evecs = pca_eigh(scaled)
    scaled_frac = variance_fractions(scaled_evals)
    svd_frac = svd_variance_fractions(scaled)
    eigen_svd_max_abs_diff = float(np.max(np.abs(scaled_frac - svd_frac)))
    scaled_recon = reconstruct(scaled, scaled_evecs, k=1)
    scaled_alcohol_r2 = column_r2(scaled[:, alcohol_idx], scaled_recon[:, alcohol_idx])
    scores = scaled @ scaled_evecs

    a = x[y == 0, alcohol_idx]
    b = x[y == 1, alcohol_idx]
    observed = mean_diff(a, b)
    welch = welch_test(a, b)
    boot = bootstrap_mean_diff(a, b, N_BOOT, np.random.default_rng(SEED))
    ci_low, ci_high = percentile_ci(boot)
    perm_p = permutation_pvalue(a, b, N_PERM, np.random.default_rng(SEED + 1))

    feature_rows = []
    unequal_means = 0
    for j, name in enumerate(names):
        left = x[y == 0, j]
        right = x[y == 1, j]
        diff = mean_diff(left, right)
        if diff != 0:
            unequal_means += 1
        test = welch_test(left, right)
        feature_rows.append(
            {
                "feature": name,
                "mean_class_0": float(left.mean()),
                "mean_class_1": float(right.mean()),
                "mean_diff": diff,
                "welch_pvalue": test["pvalue"],
            }
        )
    pvalues = np.array([row["welch_pvalue"] for row in feature_rows])
    bonferroni_alpha = ALPHA / len(names)
    weakest = max(feature_rows, key=lambda row: row["welch_pvalue"])

    fwer = family_wise_null_rate(x, y, N_NULL, ALPHA, np.random.default_rng(SEED + 2))
    independence_fwer = float(1.0 - (1.0 - ALPHA) ** len(names))

    class0 = scaled[y == 0]
    cosine = float(
        np.dot(class0[0], class0[1]) / (np.linalg.norm(class0[0]) * np.linalg.norm(class0[1]))
    )

    metrics: dict[str, object] = {
        "seed": SEED,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "class_counts": {
            "class_0": int(np.sum(y == 0)),
            "class_1": int(np.sum(y == 1)),
            "class_2": int(np.sum(y == 2)),
        },
        "cosine_first_two_class0_scaled": cosine,
        "unscaled_pc1_variance_fraction": float(unscaled_frac[0]),
        "unscaled_pc1_dominant_feature": unscaled_pc1_feature,
        "unscaled_pc1_dominant_abs_loading": unscaled_pc1_loading,
        "unscaled_pc1_alcohol_r2": unscaled_alcohol_r2,
        "alcohol_column_std": float(x[:, alcohol_idx].std(ddof=1)),
        "proline_column_std": float(x[:, proline_idx].std(ddof=1)),
        "scaled_pc1_variance_fraction": float(scaled_frac[0]),
        "scaled_top2_variance_fraction": float(scaled_frac[:2].sum()),
        "scaled_top3_variance_fraction": float(scaled_frac[:3].sum()),
        "scaled_pc1_alcohol_r2": scaled_alcohol_r2,
        "eigen_svd_max_abs_diff": eigen_svd_max_abs_diff,
        "naive_unequal_mean_count": unequal_means,
        "alcohol_mean_class_0": float(a.mean()),
        "alcohol_mean_class_1": float(b.mean()),
        "alcohol_mean_diff": observed,
        "alcohol_welch_statistic": welch["statistic"],
        "alcohol_welch_df": welch["df"],
        "alcohol_welch_pvalue": welch["pvalue"],
        "alcohol_permutation_pvalue": perm_p,
        "alcohol_bootstrap_replicates": N_BOOT,
        "alcohol_bootstrap_ci_low": ci_low,
        "alcohol_bootstrap_ci_high": ci_high,
        "alcohol_bootstrap_ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "n_features_welch_p_below_0_05": int(np.sum(pvalues < ALPHA)),
        "n_features_welch_p_below_bonferroni": int(np.sum(pvalues < bonferroni_alpha)),
        "bonferroni_alpha": bonferroni_alpha,
        "weakest_feature": weakest["feature"],
        "weakest_feature_welch_pvalue": weakest["welch_pvalue"],
        "nonsignificant_features": [
            {"feature": row["feature"], "welch_pvalue": row["welch_pvalue"]}
            for row in feature_rows
            if not (row["welch_pvalue"] < ALPHA)
        ],
        "null_simulations": N_NULL,
        "null_family_wise_error_rate": fwer,
        "null_family_wise_error_se": binomial_se(fwer, N_NULL),
        "independence_family_wise_error_rate": independence_fwer,
        "alpha": ALPHA,
        "scaled_variance_fractions": scaled_frac,
    }
    return {
        "metrics": metrics,
        "feature_rows": feature_rows,
        "scaled_frac": scaled_frac,
        "scores": scores,
        "y": y,
        "boot": boot,
        "observed": observed,
        "ci": (ci_low, ci_high),
    }


def write_outputs(bundle: dict[str, object]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics = {key: _jsonable(value) for key, value in bundle["metrics"].items()}
    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "feature",
        "mean_class_0",
        "mean_class_1",
        "mean_diff",
        "welch_pvalue",
    ]
    with (RESULTS / "feature_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bundle["feature_rows"])

    save_scree(bundle["scaled_frac"], RESULTS / "scree.png")
    save_bootstrap(
        bundle["boot"],
        bundle["observed"],
        bundle["ci"],
        RESULTS / "bootstrap_alcohol.png",
    )
    save_scatter(
        bundle["scores"],
        bundle["y"],
        bundle["scaled_frac"],
        RESULTS / "pca_scatter.png",
    )


def main() -> None:
    bundle = analyze()
    write_outputs(bundle)
    printed = {
        key: _jsonable(value)
        for key, value in bundle["metrics"].items()
        if key != "scaled_variance_fractions"
    }
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
