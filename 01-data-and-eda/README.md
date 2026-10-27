# 01 Data and EDA

Real extracts are messy. This section is cleaning, looking, validating, and building features with SQL before a model is allowed to hide the data problems.

Nothing in this section has been run. Folders are created when the work starts.
Status lives in [PROGRESS.md](../PROGRESS.md). Run a project with `python main.py`
once its folder exists. Raw data is downloaded by the script and is not committed.

| Project | What it covers |
| --- | --- |
| `01-messy-data-cleanup` | MCAR, MAR, MNAR, duplicates, inconsistent categories, outliers |
| `02-eda-report` | Univariate and bivariate views, correlation versus causation, train versus test shift |
| `03-data-validation` | Schema checks that fail a bad file before scoring |
| `04-sql-for-ml` | Multi-table feature extraction with DuckDB |
