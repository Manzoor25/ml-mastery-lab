# 02 Feature engineering

The work that changes tabular model quality. Transforms, encodings, leakage, time, text, location, aggregations, selection, imbalance, and the pipeline pattern that keeps the fit on the training fold.

`01-numeric-transforms`, `02-categorical-encoding`, `03-data-leakage`, `04-datetime-features`, `05-text-features`, and `06-geospatial-features` have measured results. The other folders are created when the work starts.
Status lives in [PROGRESS.md](../PROGRESS.md). Run a project with `python main.py`
once its folder exists. Raw data is downloaded by the script and is not committed.

| Project | What it covers |
| --- | --- |
| [`01-numeric-transforms`](01-numeric-transforms/) | Concrete strength. Group CV selected degree-3 polynomials. Holdout RMSE 6.190033630528194 versus a training-mean baseline of 15.664861218083027. Logging age did most of the work, and the holdout gap against that log included zero. |
| [`02-categorical-encoding`](02-categorical-encoding/) | Diabetic readmission. Group CV selected smoothed out-of-fold target encoding. Holdout log loss 0.32616331460860226 versus a training-rate baseline of 0.33981780404157347. Full one-hot lost, and an in-sample target encoding was not eligible. |
| [`03-data-leakage`](03-data-leakage/) | Bank telemarketing. The reported model was fixed before fitting. Honest holdout log loss 0.3736699669516322 versus a training-rate baseline of 0.3946184390200872. The date-cluster interval included zero. Same-period rows scored 0.31782332466998575 and were not eligible. |
| [`04-datetime-features`](04-datetime-features/) | Hourly I-94 volume. Training CV selected harmonics plus lags. Holdout RMSE 493.8430276959893 versus a training-mean RMSE of 1970.206195090623. A rolling mean that includes the current hour scored 492.7845532696366 and was not eligible. Its date-cluster interval included zero. |
| [`05-text-features`](05-text-features/) | GoEmotions comments. Development log loss selected character n-grams. Official-test log loss 0.9546742014250085 versus a training-rate log loss of 1.5041699927045604. Word bigrams did not beat unigrams. A TF-IDF vectorizer fit on the test text scored 0.998050295566316 and was not eligible. Its interval included zero. |
| [`06-geospatial-features`](06-geospatial-features/) | California block groups. Development RMSE selected haversine distances to five locked cities. Spatial-holdout RMSE 0.6079330366056027 versus a training-mean RMSE of 0.9652129570335812. An honest neighbor mean lost. A neighbor mean fit before the split scored 0.39365233020049006 and was not eligible. Its interval excluded zero. |
| `07-aggregation-features` | Customer-level features from a transaction log, without future rows |
| `08-feature-selection` | Filter, wrapper, embedded methods, permutation importance, SHAP-based selection |
| `09-imbalanced-data` | Class weights, resampling, SMOTE, threshold tuning, and when each helps |
| `10-sklearn-pipelines` | Pipeline and ColumnTransformer as the default way to avoid leakage |
