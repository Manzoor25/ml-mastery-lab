# 01 Data and EDA

Real extracts are messy. This section is cleaning, looking, validating, and building features with SQL before a model is allowed to hide the data problems.

`01-messy-data-cleanup`, `02-eda-report`, `03-data-validation`, and `04-sql-for-ml` have measured results.
Status lives in [PROGRESS.md](../PROGRESS.md). Run a project with `python main.py`
once its folder exists. Raw data is downloaded by the script and is not committed.

| Project | What it covers |
| --- | --- |
| [`01-messy-data-cleanup`](01-messy-data-cleanup/) | Question marks, a label period, padded fields, and an IQR rule that deleted every positive capital gain. Published-test log loss 0.42049919201950064 when spaces were stripped on one side only. |
| [`02-eda-report`](02-eda-report/) | Hourly bike-share rentals. The hour shape barely moved, the level did. 2012 RMSE 158.0887211276591 versus a training-mean baseline of 227.80800899708046. A random split scored 110.56838688362778 because it trained on 2012 hours. |
| [`03-data-validation`](03-data-validation/) | Adult extract. A null-only schema passed the stripped test file. The reported schema failed all 16281 income labels. Padded features raised log loss from 0.31833658576426743 to 0.425486700706032. |
| [`04-sql-for-ml`](04-sql-for-ml/) | Invoice lines to customer features. Honest history log loss 0.6063879510172124 versus a training-rate baseline of 0.6772728728233448. A query through the extract end scored 0.03506553353837828 and was not selected. |
