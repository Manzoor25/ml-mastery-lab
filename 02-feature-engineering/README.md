# 02 Feature engineering

The work that changes tabular model quality. Transforms, encodings, leakage, time, text, location, aggregations, selection, imbalance, and the pipeline pattern that keeps the fit on the training fold.

Nothing in this section has been run. Folders are created when the work starts.
Status lives in [PROGRESS.md](../PROGRESS.md). Run a project with `python main.py`
once its folder exists. Raw data is downloaded by the script and is not committed.

| Project | What it covers |
| --- | --- |
| `01-numeric-transforms` | Scaling, log, Box-Cox, Yeo-Johnson, binning, interactions, polynomials |
| `02-categorical-encoding` | One-hot, ordinal, out-of-fold target encoding, frequency, hashing, high cardinality |
| `03-data-leakage` | Target leakage, temporal leakage, and preprocessing fit before the split |
| `04-datetime-features` | Calendar fields, cyclical encoding, lags, rolling windows |
| `05-text-features` | Bag of words, TF-IDF, n-grams, and a basic embedding comparison |
| `06-geospatial-features` | Haversine distance, location clusters, and a non-spatial baseline |
| `07-aggregation-features` | Customer-level features from a transaction log, without future rows |
| `08-feature-selection` | Filter, wrapper, embedded methods, permutation importance, SHAP-based selection |
| `09-imbalanced-data` | Class weights, resampling, SMOTE, threshold tuning, and when each helps |
| `10-sklearn-pipelines` | Pipeline and ColumnTransformer as the default way to avoid leakage |
