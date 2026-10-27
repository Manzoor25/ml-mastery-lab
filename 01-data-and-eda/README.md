# 01 Data and EDA

Real extracts are messy. This section is cleaning, looking, validating, and building features with SQL before a model is allowed to hide the data problems.

`01-messy-data-cleanup` and `02-eda-report` have measured results. The other folders are created when the work starts.
Status lives in [PROGRESS.md](../PROGRESS.md). Run a project with `python main.py`
once its folder exists. Raw data is downloaded by the script and is not committed.

| Project | What it covers |
| --- | --- |
| [`01-messy-data-cleanup`](01-messy-data-cleanup/) | Question marks, a label period, padded fields, and an IQR rule that deleted every positive capital gain. Published-test log loss 0.42049919201950064 when spaces were stripped on one side only. |
| [`02-eda-report`](02-eda-report/) | Hourly bike-share rentals. The hour shape barely moved, the level did. 2012 RMSE 158.0887211276591 versus a training-mean baseline of 227.80800899708046. A random split scored 110.56838688362778 because it trained on 2012 hours. |
| `03-data-validation` | Schema checks that fail a bad file before scoring |
| `04-sql-for-ml` | Multi-table feature extraction with DuckDB |
