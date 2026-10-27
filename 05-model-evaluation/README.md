# 05 Model evaluation and selection

Metrics, validation splits, tuning cost, a threshold chosen from money, and whether a small gap is real.

Nothing in this section has been run. Folders are created when the work starts.
Status lives in [PROGRESS.md](../PROGRESS.md). Run a project with `python main.py`
once its folder exists. Raw data is downloaded by the script and is not committed.

| Project | What it covers |
| --- | --- |
| `01-metrics-deep-dive` | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, calibration, ranking metrics |
| `02-cross-validation` | K-fold, stratified, group, time-series split, nested CV |
| `03-hyperparameter-tuning` | Grid, random, and Bayesian search with Optuna, including the compute cost |
| `04-cost-sensitive-threshold` | Translate a cost matrix into a threshold and a business metric |
| `05-statistical-comparison` | Paired tests and an offline comparison that is not a single score |
