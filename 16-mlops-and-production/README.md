# 16 MLOps and production

Tracking, serving, drift, batch versus online scoring, features, CI gates, and compression with a measured tradeoff.

Nothing in this section has been run. Folders are created when the work starts.
Status lives in [PROGRESS.md](../PROGRESS.md). Run a project with `python main.py`
once its folder exists. Raw data is downloaded by the script and is not committed.

| Project | What it covers |
| --- | --- |
| `01-mlflow-tracking` | Log parameters and metrics with MLflow and find the run again |
| `02-fastapi-docker` | FastAPI, request validation, a health check, and a Docker image |
| `03-drift-monitoring` | Simulated data drift and concept drift, and which detector fires |
| `04-batch-vs-realtime` | The same model scored in batch and one request at a time |
| `05-feature-store` | Define a feature once, compute it offline, and read it the way a serving job would |
| `06-cicd-quality-gates` | Data tests, a metric gate, and a model registry concept |
| `07-compression` | Measured speed and accuracy after compression |
