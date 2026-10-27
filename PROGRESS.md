# Progress

Status values: `todo`, `in progress`, `done`, or `blocked: <reason>`.

If you are starting a new session, read this file first and resume from the Next line.
Do not skip ahead. Do not mark a project done unless `python main.py` wrote
`results/metrics.json` and the README numbers match that file.

Next: `00-foundations/03-bias-variance`

Last updated: 2026-10-27

Project count: 87

## Log

- 2026-10-27: Phase 1 complete. Public repo, MIT license, CI, curriculum plan,
  and README draft are in place. No project has been run. There are no model
  metrics to report. LinkedIn drafts: none.
- 2026-10-27: `00-foundations/01-linalg-probability-stats` done. Unscaled PC1
  variance fraction 0.9981 is a proline-unit artifact. Null 13-test family-wise
  error rate 0.387. LinkedIn draft: `linkedin-posts/01-linalg-probability-stats.md`.
- 2026-10-27: `00-foundations/02-gradient-descent` done. Wine least-squares test
  MSE 0.4190 versus mean baseline 0.7444. Step 2.5/L stopped after epoch 22.
  LinkedIn draft: `linkedin-posts/02-gradient-descent.md`.

## Phase 1 - Setup

| Item | Status |
| --- | --- |
| Public repository, MIT license, Python gitignore | done |
| Pinned requirements, ruff, pytest, GitHub Actions | done |
| Curriculum plan in this file | done |
| Root README draft | done |

<!-- projects-start -->

## Phase 2 - Foundations, data, and features

Sections 00 to 02. Build these before the algorithm tour.

### 00 Foundations

| Path | Status | Teaches |
| --- | --- | --- |
| `00-foundations/01-linalg-probability-stats` | done | Vectors, matrices, eigendecomposition, distributions, hypothesis tests, bootstrap |
| `00-foundations/02-gradient-descent` | done | Batch, stochastic, mini-batch, momentum, and Adam on a visible loss surface |
| `00-foundations/03-bias-variance` | todo | An empirical decomposition, not only the textbook picture |
| `00-foundations/04-linear-logistic-from-scratch` | todo | Both models from scratch, then a check against scikit-learn |

### 01 Data and EDA

| Path | Status | Teaches |
| --- | --- | --- |
| `01-data-and-eda/01-messy-data-cleanup` | todo | MCAR, MAR, MNAR, duplicates, inconsistent categories, outliers |
| `01-data-and-eda/02-eda-report` | todo | Univariate and bivariate views, correlation versus causation, train versus test shift |
| `01-data-and-eda/03-data-validation` | todo | Schema checks that fail a bad file before scoring |
| `01-data-and-eda/04-sql-for-ml` | todo | Multi-table feature extraction with DuckDB |

### 02 Feature engineering

| Path | Status | Teaches |
| --- | --- | --- |
| `02-feature-engineering/01-numeric-transforms` | todo | Scaling, log, Box-Cox, Yeo-Johnson, binning, interactions, polynomials |
| `02-feature-engineering/02-categorical-encoding` | todo | One-hot, ordinal, out-of-fold target encoding, frequency, hashing, high cardinality |
| `02-feature-engineering/03-data-leakage` | todo | Target leakage, temporal leakage, and preprocessing fit before the split |
| `02-feature-engineering/04-datetime-features` | todo | Calendar fields, cyclical encoding, lags, rolling windows |
| `02-feature-engineering/05-text-features` | todo | Bag of words, TF-IDF, n-grams, and a basic embedding comparison |
| `02-feature-engineering/06-geospatial-features` | todo | Haversine distance, location clusters, and a non-spatial baseline |
| `02-feature-engineering/07-aggregation-features` | todo | Customer-level features from a transaction log, without future rows |
| `02-feature-engineering/08-feature-selection` | todo | Filter, wrapper, embedded methods, permutation importance, SHAP-based selection |
| `02-feature-engineering/09-imbalanced-data` | todo | Class weights, resampling, SMOTE, threshold tuning, and when each helps |
| `02-feature-engineering/10-sklearn-pipelines` | todo | Pipeline and ColumnTransformer as the default way to avoid leakage |

## Phase 3 - Classical ML, evaluation, ensembles, time series

Sections 03 to 07.

### 03 Classical supervised learning

| Path | Status | Teaches |
| --- | --- | --- |
| `03-classical-supervised/01-knn` | todo | Distance-based classification, the curse of dimensionality, a simple baseline |
| `03-classical-supervised/02-naive-bayes` | todo | A probabilistic spam filter and the independence assumption |
| `03-classical-supervised/03-svm-kernels` | todo | Margins, kernels, and a plot of a boundary that is not a straight line |
| `03-classical-supervised/04-decision-trees` | todo | Splits, depth, and a tree a reviewer can read |
| `03-classical-supervised/05-regularized-regression` | todo | Correlated features, shrinkage, and what each penalty keeps |

### 04 Unsupervised learning

| Path | Status | Teaches |
| --- | --- | --- |
| `04-unsupervised/01-clustering` | todo | k-means, hierarchical clustering, DBSCAN, Gaussian mixtures, a segmentation case |
| `04-unsupervised/02-dimensionality-reduction` | todo | PCA from scratch, t-SNE, UMAP, and the plots that mislead |

### 05 Model evaluation and selection

| Path | Status | Teaches |
| --- | --- | --- |
| `05-model-evaluation/01-metrics-deep-dive` | todo | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, calibration, ranking metrics |
| `05-model-evaluation/02-cross-validation` | todo | K-fold, stratified, group, time-series split, nested CV |
| `05-model-evaluation/03-hyperparameter-tuning` | todo | Grid, random, and Bayesian search with Optuna, including the compute cost |
| `05-model-evaluation/04-cost-sensitive-threshold` | todo | Translate a cost matrix into a threshold and a business metric |
| `05-model-evaluation/05-statistical-comparison` | todo | Paired tests and an offline comparison that is not a single score |

### 06 Ensembles and boosting

| Path | Status | Teaches |
| --- | --- | --- |
| `06-ensembles-and-boosting/01-random-forests` | todo | Why averaging trees helps, and where a single deep tree still fails |
| `06-ensembles-and-boosting/02-gradient-boosting-from-scratch` | todo | A small boosting loop written out, so the library is not a black box |
| `06-ensembles-and-boosting/03-boosting-libraries` | todo | The same tabular problem, three libraries, score and fit time |
| `06-ensembles-and-boosting/04-stacking-blending` | todo | Out-of-fold stacking, and a stack that may lose to the best single model |

### 07 Time series

| Path | Status | Teaches |
| --- | --- | --- |
| `07-time-series/01-classical-forecasting` | todo | Decomposition, ARIMA or SARIMA, exponential smoothing, a naive baseline |
| `07-time-series/02-ml-forecasting` | todo | Gradient boosting with lag and rolling features, scored with a backtest |
| `07-time-series/03-lstm-vs-baselines` | todo | A small LSTM or temporal model against the classical and boosting baselines |
| `07-time-series/04-demand-forecasting` | todo | From a forecast error to an inventory-style decision |

## Phase 4 - Deep learning, vision, NLP, and generative models

Sections 08 to 11. Small subsets. GPU notes only where a CPU run cannot finish.

### 08 Deep learning basics

| Path | Status | Teaches |
| --- | --- | --- |
| `08-deep-learning-basics/01-nn-from-scratch` | todo | A NumPy network with backprop, then the same model in PyTorch |
| `08-deep-learning-basics/02-training-dynamics` | todo | Initialization, batch norm, dropout, learning-rate schedules, overfitting |
| `08-deep-learning-basics/03-debugging-training` | todo | Overfit-one-batch, gradient checks, and how to read a loss curve |

### 09 Computer vision

| Path | Status | Teaches |
| --- | --- | --- |
| `09-computer-vision/01-cnn-classifier` | todo | A small convolutional net against a linear baseline on pixels |
| `09-computer-vision/02-transfer-learning` | todo | A pretrained backbone and an augmentation ablation |
| `09-computer-vision/03-object-detection` | todo | A pretrained detector on a small image set, with a CPU limit written down |
| `09-computer-vision/04-segmentation` | todo | A small segmentation demo against a trivial mask baseline |
| `09-computer-vision/05-grad-cam` | todo | Where a classifier is looking, including a case that highlights the wrong region |

### 10 NLP

| Path | Status | Teaches |
| --- | --- | --- |
| `10-nlp/01-classical-vs-transformer` | todo | TF-IDF plus a linear model versus a small transformer, including cost |
| `10-nlp/02-named-entity-recognition` | todo | Token-level labels on a public corpus, with a small subset if needed |
| `10-nlp/03-semantic-search` | todo | Sentence embeddings and retrieval metrics, not only a nearest-neighbor demo |
| `10-nlp/04-topic-modeling` | todo | Topics on a corpus short enough to read, checked against the documents |

### 11 LLMs and generative models

| Path | Status | Teaches |
| --- | --- | --- |
| `11-llms-and-genai/01-tokenization` | todo | Vocabulary, merges, and how tokenization changes sequence length |
| `11-llms-and-genai/02-tiny-transformer` | todo | A small transformer trained on a small corpus, against a simpler next-token baseline |
| `11-llms-and-genai/03-rag-pipeline` | todo | Retrieval metrics plus a check on the answer, not a demo screenshot |
| `11-llms-and-genai/04-lora-finetune` | todo | A small open model on a tiny set, or a documented Colab run if CPU cannot finish |
| `11-llms-and-genai/05-llm-evaluation` | todo | An eval set you can read, and a concrete LLM-as-judge failure |
| `11-llms-and-genai/06-prompt-engineering` | todo | Prompt variants with a measured outcome, not a vibe |
| `11-llms-and-genai/07-generative-models` | todo | Small generative models on simple images, with a boring baseline |

## Phase 5 - Recommenders, RL, causal ML, anomaly detection, MLOps, responsible AI

Sections 12 to 17.

### 12 Recommender systems

| Path | Status | Teaches |
| --- | --- | --- |
| `12-recommender-systems/01-movielens` | todo | Popularity, collaborative filtering, matrix factorization, content features, ranking metrics |

### 13 Reinforcement learning

| Path | Status | Teaches |
| --- | --- | --- |
| `13-reinforcement-learning/01-multi-armed-bandits` | todo | Explore-exploit policies compared with a fixed A/B test |
| `13-reinforcement-learning/02-q-learning` | todo | A tabular agent on a small grid or FrozenLake, against a random policy |
| `13-reinforcement-learning/03-dqn` | todo | DQN on a small Gymnasium environment such as CartPole |
| `13-reinforcement-learning/04-policy-gradient` | todo | A simple policy gradient on a small environment, compared with the value-based run |

### 14 Probabilistic and causal ML

| Path | Status | Teaches |
| --- | --- | --- |
| `14-probabilistic-and-causal/01-bayesian-ab-test` | todo | A posterior over a conversion rate, compared with a classical test |
| `14-probabilistic-and-causal/02-confounding-propensity` | todo | A naive comparison, then propensity score matching |
| `14-probabilistic-and-causal/03-uplift-modeling` | todo | Who responds because of the treatment, not who would buy anyway |
| `14-probabilistic-and-causal/04-difference-in-differences` | todo | A before-after comparison with a control group, and the parallel-trends assumption |

### 15 Anomaly detection

| Path | Status | Teaches |
| --- | --- | --- |
| `15-anomaly-detection/01-classical-detectors` | todo | Isolation forest, one-class SVM, and a statistical threshold |
| `15-anomaly-detection/02-fraud-detection` | todo | Extreme imbalance, an autoencoder, and a metric that does not hide the rare class |

### 16 MLOps and production

| Path | Status | Teaches |
| --- | --- | --- |
| `16-mlops-and-production/01-mlflow-tracking` | todo | Log parameters and metrics with MLflow and find the run again |
| `16-mlops-and-production/02-fastapi-docker` | todo | FastAPI, request validation, a health check, and a Docker image |
| `16-mlops-and-production/03-drift-monitoring` | todo | Simulated data drift and concept drift, and which detector fires |
| `16-mlops-and-production/04-batch-vs-realtime` | todo | The same model scored in batch and one request at a time |
| `16-mlops-and-production/05-feature-store` | todo | Define a feature once, compute it offline, and read it the way a serving job would |
| `16-mlops-and-production/06-cicd-quality-gates` | todo | Data tests, a metric gate, and a model registry concept |
| `16-mlops-and-production/07-compression` | todo | Measured speed and accuracy after compression |

### 17 Responsible AI

| Path | Status | Teaches |
| --- | --- | --- |
| `17-responsible-ai/01-fairness` | todo | Group metrics and a mitigation attempt, including a gap that remains |
| `17-responsible-ai/02-explainability` | todo | SHAP, LIME, partial dependence, and a case where the explanation misleads |
| `17-responsible-ai/03-privacy` | todo | Differential privacy intuition and a PII pass on synthetic data |
| `17-responsible-ai/04-model-cards` | todo | A model card for a model this repo actually trained |

## Phase 6 - Real-world case studies

Section 18. Each capstone goes from a vague ask to a monitoring plan.

### 18 Real-world case studies

| Path | Status | Teaches |
| --- | --- | --- |
| `18-real-world-case-studies/01-customer-churn` | todo | Framing, features, model, threshold, deployment sketch, monitoring, stakeholder note |
| `18-real-world-case-studies/02-credit-default` | todo | A risk model with an explanation a reviewer can ask for |
| `18-real-world-case-studies/03-fraud-cost-sensitive` | todo | A fraud model whose threshold comes from a cost matrix |
| `18-real-world-case-studies/04-demand-forecasting` | todo | A forecast tied to an inventory decision and a stockout cost |
| `18-real-world-case-studies/05-search-ranking` | todo | Ranking features, a click model, and a ranking metric that is not click rate alone |
| `18-real-world-case-studies/06-support-ticket-routing` | todo | Text classification into queues, with the cost of a wrong queue |
| `18-real-world-case-studies/07-document-qa` | todo | A RAG case study that separates retrieval from answer quality |
| `18-real-world-case-studies/08-uplift-marketing` | todo | Uplift or a dynamic-pricing style decision, against targeting likely buyers |

<!-- projects-end -->

<!-- guides-start -->

## Phase 7 - Guides

These are documents, not training runs. A `done` guide must exist as a file.
It does not need `results/metrics.json`. Write them after the projects they cite.

| Path | Status | What it is |
| --- | --- | --- |
| `interview-prep/ml-theory.md` | todo | 100 or more theory questions with concise answers, grouped by topic, linked to projects. |
| `interview-prep/coding.md` | todo | ML coding tasks with solutions in the repo: k-means, logistic regression, a tree split, softmax, attention, AUC, a leak-free split. |
| `interview-prep/ml-system-design.md` | todo | 10 or more worked designs: ranking, fraud, search, ads, moderation, ETA, RAG, and related systems. |
| `interview-prep/case-study-questions.md` | todo | Product and metric questions, including a rising AUC that does not raise revenue. |
| `interview-prep/sql-and-stats.md` | todo | SQL and statistics questions commonly asked in ML and data roles. |
| `interview-prep/behavioral.md` | todo | STAR-format guidance for ML work: failed experiments, metric disagreements, shipping under ambiguity. |
| `interview-prep/rapid-fire.md` | todo | One-line answers for last-minute review, each pointing at a project when one exists. |
| `on-the-job/first-90-days.md` | todo | What onboarding as an ML engineer or data scientist usually looks like, and how to get a first win. |
| `on-the-job/typical-problems.md` | todo | Data fires, degradation, stakeholder requests, retraining, latency, and labeling. |
| `on-the-job/problem-framing.md` | todo | Turn a vague business ask into an ML problem, or say when ML is the wrong tool. |
| `on-the-job/working-with-stakeholders.md` | todo | How to explain results, plus a one-page experiment summary template. |
| `on-the-job/code-and-collaboration.md` | todo | Review norms, notebooks versus production code, and a design-doc template. |
| `on-the-job/debugging-models-in-production.md` | todo | A runbook for the model got worse. |
| `on-the-job/roles-explained.md` | todo | Data scientist, ML engineer, applied scientist, research scientist, MLOps engineer. |
| `cheatsheets/algorithm-selection.md` | todo | One-page guide to choosing a model family. |
| `cheatsheets/metric-selection.md` | todo | One-page guide to choosing a metric. |
| `cheatsheets/feature-engineering.md` | todo | Feature transforms by data type. |
| `cheatsheets/pandas-sklearn-pytorch.md` | todo | Pandas, scikit-learn, and PyTorch calls used in this repo. |
| `cheatsheets/hyperparameter-starting-points.md` | todo | Starting hyperparameters by algorithm, to be filled from projects that measured them. |
| `cheatsheets/common-errors.md` | todo | Common error messages and the fix that actually worked. |

<!-- guides-end -->
