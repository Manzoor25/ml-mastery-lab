# 00 Foundations

The math you use when a library call is not an explanation. Linear algebra, probability, optimization, the bias-variance tradeoff, and linear models written out before they are compared with scikit-learn.

`01-linalg-probability-stats`, `02-gradient-descent`, and `03-bias-variance` have been run. The other folders in this section are created when the work starts.
Status lives in [PROGRESS.md](../PROGRESS.md). Run a project with `python main.py`
once its folder exists. Raw data is downloaded by the script and is not committed.

| Project | What it covers |
| --- | --- |
| [`01-linalg-probability-stats`](01-linalg-probability-stats/) | Vectors, matrices, eigendecomposition, distributions, hypothesis tests, bootstrap. Unscaled PC1 fraction 0.9981; null 13-test false-alarm rate 0.387. |
| [`02-gradient-descent`](02-gradient-descent/) | Batch, stochastic, mini-batch, momentum, and Adam. Held-out MSE 0.4190 versus a mean baseline of 0.7444. |
| [`03-bias-variance`](03-bias-variance/) | Empirical bias-variance on a known curve, then red-wine quality. Holdout MSE 0.4024 versus a mean of 0.6044. |
| `04-linear-logistic-from-scratch` | Both models from scratch, then a check against scikit-learn |
