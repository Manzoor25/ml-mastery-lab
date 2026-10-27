# ml-mastery-lab

[![CI](https://github.com/Manzoor25/ml-mastery-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Manzoor25/ml-mastery-lab/actions/workflows/ci.yml)

Runnable machine learning projects, from the linear algebra you need on a whiteboard to the production questions a team actually asks. Each project is a small piece of work an ML engineer would recognize: a problem, a baseline, an improvement, a failure, and numbers produced by the code in that folder.

This is not a reading list. If a number is in a project README, it came from `results/metrics.json` after `python main.py`. Rows that still say `not run` have not been executed.

> Status: Phase 2 in progress. Foundations and the data section have measured results. Feature engineering is next.

## Who it is for

- You are learning ML and want to build, not only watch.
- You have an interview and need to explain leakage, metrics, boosting, and system design from something you ran.
- You already work in the field and want a map of the gaps: features, evaluation, production, and how to talk about the work.

## How to use it

1. Pick a path below, or open [PROGRESS.md](PROGRESS.md) and start at the `Next:` line.
2. Create a virtual environment and install the shared dependencies.
3. Enter a project folder and run `python main.py`.
4. Read the project README next to the plots in `results/`.

Finished project folders are linked from the table. Until a project is built, the path is the planned location, not a link to a missing file.

## Learning paths

Content lands in curriculum order. The paths below are the intended order. If a folder is not in the repo yet, check [PROGRESS.md](PROGRESS.md) and take the closest finished project in that section. Do not invent the missing result.

### Complete beginner

Go in section order. Do not start with a transformer.

1. [00 Foundations](00-foundations/) - algebra, probability, gradient descent, bias-variance, linear and logistic regression from scratch.
2. [01 Data and EDA](01-data-and-eda/) - messy data, a real EDA, schema checks, SQL features.
3. [02 Feature engineering](02-feature-engineering/) - slow down here.
4. [03 Classical supervised learning](03-classical-supervised/) through [07 Time series](07-time-series/).
5. [08 Deep learning basics](08-deep-learning-basics/) before vision, NLP, or LLMs.
6. [09 Computer vision](09-computer-vision/) through [18 Real-world case studies](18-real-world-case-studies/).
7. [Interview prep](interview-prep/), [on the job](on-the-job/), and [cheatsheets](cheatsheets/) after the related projects exist. Those guides are Phase 7.

Deep learning, detection, and LLM fine-tuning stay on a small subset so they fit a laptop CPU. Where a GPU is actually required, the project README will say so and point at a free Colab or Kaggle path. That note is written when the project is built, after the runtime is known.

### Interview in 4 weeks

Assumes you can already write Python and read a confusion matrix. This is a focus path, not a claim that four weeks covers the field.

Week 1 - models you can derive:

- `00-foundations/03-bias-variance`
- `00-foundations/04-linear-logistic-from-scratch`
- `03-classical-supervised/04-decision-trees`
- `03-classical-supervised/05-regularized-regression`
- `06-ensembles-and-boosting/01-random-forests`

Week 2 - what changes the score:

- `02-feature-engineering/03-data-leakage`
- `02-feature-engineering/09-imbalanced-data`
- `02-feature-engineering/10-sklearn-pipelines`
- `05-model-evaluation/01-metrics-deep-dive`
- `05-model-evaluation/02-cross-validation`
- `05-model-evaluation/04-cost-sensitive-threshold`
- `06-ensembles-and-boosting/03-boosting-libraries`

Week 3 - one sequence problem, one text problem, how you would ship it:

- `07-time-series/02-ml-forecasting`
- `10-nlp/01-classical-vs-transformer`
- `16-mlops-and-production/02-fastapi-docker`
- `17-responsible-ai/02-explainability`

Week 4 - talk about the work:

- `18-real-world-case-studies/01-customer-churn`
- [interview-prep](interview-prep/) once those files exist
- `on-the-job/problem-framing.md` once that file exists

### Already working, fill the gaps

You can train a model. The misses are usually leakage, the wrong metric, a threshold nobody chose, and a model nobody can operate.

- All of [02 Feature engineering](02-feature-engineering/)
- All of [05 Model evaluation](05-model-evaluation/)
- [06 Ensembles](06-ensembles-and-boosting/), especially the library comparison and stacking
- [16 MLOps](16-mlops-and-production/)
- [17 Responsible AI](17-responsible-ai/)
- [On the job](on-the-job/)
- `interview-prep/ml-system-design.md` when it exists

## Setup

Requires Python 3.11 or newer. This repo is developed on Python 3.12. `uv` and the standard-library `venv` module both work.

```bash
git clone https://github.com/Manzoor25/ml-mastery-lab.git
cd ml-mastery-lab
```

With venv, Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
```

With venv, Unix:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
```

With uv:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-dev.txt
```

Check the install:

```bash
ruff check .
ruff format --check .
pytest
```

GNU make is optional:

```bash
make ci
```

`python main.py` is the command that matters inside a project. You do not need `make` to run one.

## Run a project

From the project directory:

```bash
python main.py
pytest tests/test_smoke.py
```

`main.py` writes `results/metrics.json` and the plots. The README is updated from those files. The smoke test uses a tiny sample and should finish in under 30 seconds. It does not replace the full run.

If the project has its own `requirements.txt`, install that too. Shared libraries are pinned in the root [requirements.txt](requirements.txt).

## Rules for numbers

- The default random seed is 42 unless a project README says why it is not.
- Fit preprocessing on training data only.
- Scripts download raw data. Do not commit it. `data/` is gitignored.
- Every dataset is public, permissively licensed, and cited in the project README, including the license.
- A baseline is required. The change from that baseline is reported when it is small or negative.
- At least one experiment in each project should do worse than the best run. Say why.
- Do not paste a metric from memory. If `results/metrics.json` changes, the README changes with it.

## Repository map

| Path | What it is |
| --- | --- |
| [00-foundations](00-foundations/) | NumPy math, optimization, bias-variance, linear models from scratch |
| [01-data-and-eda](01-data-and-eda/) | Cleaning, EDA, validation, SQL features |
| [02-feature-engineering](02-feature-engineering/) | Transforms, encodings, leakage, selection, imbalance, pipelines |
| [03-classical-supervised](03-classical-supervised/) | k-NN, Naive Bayes, SVM, trees, regularized regression |
| [04-unsupervised](04-unsupervised/) | Clustering and dimensionality reduction |
| [05-model-evaluation](05-model-evaluation/) | Metrics, cross-validation, tuning, costs, statistical comparison |
| [06-ensembles-and-boosting](06-ensembles-and-boosting/) | Forests, boosting, library comparison, stacking |
| [07-time-series](07-time-series/) | Classical forecasts, lag models, a neural baseline, demand |
| [08-deep-learning-basics](08-deep-learning-basics/) | Backprop, training dynamics, debugging a training run |
| [09-computer-vision](09-computer-vision/) | CNNs, transfer learning, detection, segmentation, Grad-CAM |
| [10-nlp](10-nlp/) | Classical text models, transformers, NER, search, topics |
| [11-llms-and-genai](11-llms-and-genai/) | Tokenizers, a tiny transformer, RAG, LoRA, evals, prompts, generative models |
| [12-recommender-systems](12-recommender-systems/) | MovieLens ranking, from popularity to matrix factorization |
| [13-reinforcement-learning](13-reinforcement-learning/) | Bandits, Q-learning, DQN, policy gradients |
| [14-probabilistic-and-causal](14-probabilistic-and-causal/) | Bayesian tests and causal inference |
| [15-anomaly-detection](15-anomaly-detection/) | Rare events and a fraud-style case |
| [16-mlops-and-production](16-mlops-and-production/) | Tracking, serving, drift, features, CI, compression |
| [17-responsible-ai](17-responsible-ai/) | Fairness, explanations, privacy, model cards |
| [18-real-world-case-studies](18-real-world-case-studies/) | End-to-end capstones |
| [interview-prep](interview-prep/) | Questions and worked system designs (Phase 7) |
| [on-the-job](on-the-job/) | What the job is after the offer (Phase 7) |
| [cheatsheets](cheatsheets/) | One-page references (Phase 7) |
| [linkedin-posts](linkedin-posts/) | Draft posts copied from each project. Not published. |
| [PROGRESS.md](PROGRESS.md) | Checklist. Resume here. |

## Projects

87 planned projects. Topic and use case are the plan. The key result stays
`not run` until that project's `python main.py` writes `results/metrics.json`.
A `not run` cell is not a result.

### 00 Foundations

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| [`00-foundations/01-linalg-probability-stats`](00-foundations/01-linalg-probability-stats/) | Linear algebra, probability, statistics | Decide whether a measured gap is signal or sampling noise before anyone fits a model. | Unscaled PC1 holds 0.9981 of variance; null 13-test false-alarm rate is 0.387 |
| [`00-foundations/02-gradient-descent`](00-foundations/02-gradient-descent/) | Gradient descent | A training loss stalls or blows up, and you need to see why the step moved that way. | Held-out MSE 0.4190 versus mean baseline 0.7444; step 2.5/L stopped after epoch 22 |
| [`00-foundations/03-bias-variance`](00-foundations/03-bias-variance/) | Bias-variance tradeoff | A flexible model wins on the training customers and loses on the next month. | Degree-3 fresh MSE 0.3384; wine holdout 0.4024 versus mean 0.6044 |
| [`00-foundations/04-linear-logistic-from-scratch`](00-foundations/04-linear-logistic-from-scratch/) | Linear and logistic regression | A yes-or-no score whose coefficients you can derive, then verify against a library. | OLS holdout MSE 0.5558915986952442 versus mean 1.3106960720039365; penalized log loss 0.07716713894283675 |

### 01 Data and EDA

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| [`01-data-and-eda/01-messy-data-cleanup`](01-data-and-eda/01-messy-data-cleanup/) | Missing data and cleanup | A census extract whose test labels and padded fields do not match the training file. | Padded categories raised published-test log loss from 0.3207506692781671 to 0.42049919201950064 |
| [`01-data-and-eda/02-eda-report`](01-data-and-eda/02-eda-report/) | Exploratory data analysis | A stakeholder asks whether last year's bike-share hours describe next year. | 2012 RMSE 158.0887211276591 versus training-mean 227.80800899708046; a random split scored 110.56838688362778 |
| [`01-data-and-eda/03-data-validation`](01-data-and-eda/03-data-validation/) | Data validation | A new census extract should fail a schema check instead of being scored silently. | Null-only schema passed; reported schema failed all 16281 income labels. Padded features raised log loss from 0.31833658576426743 to 0.425486700706032 |
| [`01-data-and-eda/04-sql-for-ml`](01-data-and-eda/04-sql-for-ml/) | SQL for ML | Build customer features from invoice lines without letting the label window into the query. | Honest history log loss 0.6063879510172124 versus training-rate 0.6772728728233448. A query through the extract end scored 0.03506553353837828 and was not selected |

### 02 Feature engineering

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `02-feature-engineering/01-numeric-transforms` | Numeric transforms | Skewed prices and incomes. Measure which transform changes a linear model, and which does not. | not run |
| `02-feature-engineering/02-categorical-encoding` | Categorical encoding | City, plan tier, and user id do not want the same encoding. | not run |
| `02-feature-engineering/03-data-leakage` | Data leakage | A model looks excellent because the label leaked, or because the full file was scaled before the split. | not run |
| `02-feature-engineering/04-datetime-features` | Date and time features | Hour and day of week are cycles. Treating them as ordinary numbers points the wrong way at midnight. | not run |
| `02-feature-engineering/05-text-features` | Text features | Turn reviews or tickets into numeric features a linear model can use. | not run |
| `02-feature-engineering/06-geospatial-features` | Geospatial features | How far is this order from the warehouse, and do nearby orders behave alike? | not run |
| `02-feature-engineering/07-aggregation-features` | Aggregation features | Collapse a purchase log to one row per customer without using purchases that had not happened yet. | not run |
| `02-feature-engineering/08-feature-selection` | Feature selection | Hundreds of columns, most of them noise. Decide what to drop, and where importance lies. | not run |
| `02-feature-engineering/09-imbalanced-data` | Imbalanced data | A rare positive class. Accuracy is high and useless. The fixes do not agree with each other. | not run |
| `02-feature-engineering/10-sklearn-pipelines` | sklearn pipelines | The standard way to fit transforms on the training fold only, including a deliberate leak to compare. | not run |

### 03 Classical supervised learning

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `03-classical-supervised/01-knn` | k-nearest neighbors | Label a case from similar past cases, and see where that falls apart as the feature count grows. | not run |
| `03-classical-supervised/02-naive-bayes` | Naive Bayes | Filter spam with a simple probabilistic model, and compare it with a linear baseline. | not run |
| `03-classical-supervised/03-svm-kernels` | Support vector machines | Separate two classes that are not linearly separable, and show what the kernel changed. | not run |
| `03-classical-supervised/04-decision-trees` | Decision trees | An approve-or-review decision a risk partner can read as a tree, not only as a score. | not run |
| `03-classical-supervised/05-regularized-regression` | Ridge, lasso, elastic net | Predict a price when features move together. The three penalties do not tell the same story. | not run |

### 04 Unsupervised learning

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `04-unsupervised/01-clustering` | Clustering | Group customers when there is no churn label, and compare methods that assume different shapes. | not run |
| `04-unsupervised/02-dimensionality-reduction` | Dimensionality reduction | Compress and plot a high-dimensional set without treating a pretty picture as a cluster proof. | not run |

### 05 Model evaluation and selection

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `05-model-evaluation/01-metrics-deep-dive` | Metrics | Pick a metric when the positive class is rare and a pretty accuracy number would hide the failures. | not run |
| `05-model-evaluation/02-cross-validation` | Cross-validation | Random k-fold is wrong when rows share a user, or when the rows are a timeline. | not run |
| `05-model-evaluation/03-hyperparameter-tuning` | Hyperparameter tuning | Compare search methods on the same model. Count the fits, not only the best score. | not run |
| `05-model-evaluation/04-cost-sensitive-threshold` | Decision thresholds | False alarms and missed events have different dollar costs. Do not leave the threshold at 0.5. | not run |
| `05-model-evaluation/05-statistical-comparison` | Model comparison | Two models are close. Decide whether the gap is larger than noise. | not run |

### 06 Ensembles and boosting

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `06-ensembles-and-boosting/01-random-forests` | Bagging and random forests | A tabular baseline that is hard to beat, compared with one tree on the same split. | not run |
| `06-ensembles-and-boosting/02-gradient-boosting-from-scratch` | Gradient boosting | Fit a tiny booster by hand on a problem you can plot, including a learning rate that fails. | not run |
| `06-ensembles-and-boosting/03-boosting-libraries` | XGBoost, LightGBM, CatBoost | One table, one split, three boosting libraries. Report which one won and whether it mattered. | not run |
| `06-ensembles-and-boosting/04-stacking-blending` | Stacking and blending | Combine models the way competition notebooks do. Keep the result if the stack loses. | not run |

### 07 Time series

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `07-time-series/01-classical-forecasting` | Classical forecasting | Seasonal demand. A seasonal naive forecast is the bar the fancier model has to clear. | not run |
| `07-time-series/02-ml-forecasting` | ML forecasting | The same series as a table of lags. The split must not peek at later weeks. | not run |
| `07-time-series/03-lstm-vs-baselines` | Neural forecasting | A neural forecast has to beat the simpler models on a fair backtest, or you say it did not. | not run |
| `07-time-series/04-demand-forecasting` | Demand forecasting | Turn a forecast into an ordering decision, with a stockout cost, not only an error number. | not run |

### 08 Deep learning basics

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `08-deep-learning-basics/01-nn-from-scratch` | Neural nets from scratch | Write the gradient yourself, then check that a framework agrees on a tiny problem. | not run |
| `08-deep-learning-basics/02-training-dynamics` | Training dynamics | Watch a small net memorize, then see which regularizer actually changes the holdout loss. | not run |
| `08-deep-learning-basics/03-debugging-training` | Debugging training | A loss curve that looks fine while the model has not learned the batch you gave it. | not run |

### 09 Computer vision

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `09-computer-vision/01-cnn-classifier` | CNN classifier | Classify images from a small public set. A linear model on pixels is the baseline. | not run |
| `09-computer-vision/02-transfer-learning` | Transfer learning | Start from a pretrained model and measure what augmentation changed, including a change that hurt. | not run |
| `09-computer-vision/03-object-detection` | Object detection | Find objects in a few images. If the CPU cannot finish a fair run, document a free GPU path. | not run |
| `09-computer-vision/04-segmentation` | Image segmentation | Label pixels, not just the image. Compare with a baseline that predicts the majority class per pixel. | not run |
| `09-computer-vision/05-grad-cam` | Grad-CAM | Ask which pixels drove a prediction, and show one example where the highlight is not the object. | not run |

### 10 NLP

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `10-nlp/01-classical-vs-transformer` | Text classification | Classify text. Report accuracy and the cost of the neural model, not accuracy alone. | not run |
| `10-nlp/02-named-entity-recognition` | Named entity recognition | Pull people, places, and organizations out of text, and measure token-level errors. | not run |
| `10-nlp/03-semantic-search` | Semantic search | Search a document collection by meaning. Measure whether the right passage was retrieved. | not run |
| `10-nlp/04-topic-modeling` | Topic modeling | Summarize a corpus into themes, then read the documents and say which themes are artifacts. | not run |

### 11 LLMs and generative models

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `11-llms-and-genai/01-tokenization` | Tokenization | Tokenize a small corpus and measure how the tokenizer choice changes the sequence the model sees. | not run |
| `11-llms-and-genai/02-tiny-transformer` | Tiny transformer | Train a tiny language model. A bigram or n-gram baseline has to be on the same table. | not run |
| `11-llms-and-genai/03-rag-pipeline` | Retrieval-augmented generation | Answer questions from a document set. Separate a retrieved passage from a correct answer. | not run |
| `11-llms-and-genai/04-lora-finetune` | LoRA fine-tuning | Adapt a small open model to a narrow task and compare it with the untuned model. | not run |
| `11-llms-and-genai/05-llm-evaluation` | LLM evaluation | Score model answers. Show one case where an automatic judge rewards a worse answer. | not run |
| `11-llms-and-genai/06-prompt-engineering` | Prompt experiments | Change the prompt on a fixed question set and report which change moved the score. | not run |
| `11-llms-and-genai/07-generative-models` | Autoencoders, VAEs, GAN or diffusion | Rebuild simple images. Compare an autoencoder, a VAE, and a small GAN or diffusion run. | not run |

### 12 Recommender systems

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `12-recommender-systems/01-movielens` | Recommender systems | Recommend movies. A popularity list is the baseline a personalized model has to beat. | not run |

### 13 Reinforcement learning

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `13-reinforcement-learning/01-multi-armed-bandits` | Multi-armed bandits | Allocate traffic across options. Compare a bandit with a test that waits for a fixed sample. | not run |
| `13-reinforcement-learning/02-q-learning` | Q-learning | Learn a small policy from rewards. Random action is the baseline. | not run |
| `13-reinforcement-learning/03-dqn` | Deep Q-learning | Use a neural Q function on a small control task, and record a run that does not learn. | not run |
| `13-reinforcement-learning/04-policy-gradient` | Policy gradients | Optimize the policy directly. Compare it with Q-learning where both can run. | not run |

### 14 Probabilistic and causal ML

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `14-probabilistic-and-causal/01-bayesian-ab-test` | Bayesian A/B testing | Stop or continue a test. Show a case where the Bayesian decision and the p-value disagree. | not run |
| `14-probabilistic-and-causal/02-confounding-propensity` | Confounding and propensity scores | A treatment looks effective because healthier users opted in. Adjust, and show the naive number too. | not run |
| `14-probabilistic-and-causal/03-uplift-modeling` | Uplift modeling | Target a coupon at people it changes. Targeting likely buyers is the wrong objective. | not run |
| `14-probabilistic-and-causal/04-difference-in-differences` | Difference-in-differences | Estimate the effect of a change that was not a randomized trial. State the assumption in the README. | not run |

### 15 Anomaly detection

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `15-anomaly-detection/01-classical-detectors` | Classical anomaly detection | Flag unusual rows when labels are scarce. Compare two detectors with a simple cutoff. | not run |
| `15-anomaly-detection/02-fraud-detection` | Fraud detection | Catch rare fraud. A high ROC-AUC can still miss the events that cost money. | not run |

### 16 MLOps and production

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `16-mlops-and-production/01-mlflow-tracking` | Experiment tracking | A week later, answer which parameters produced the number in the README. | not run |
| `16-mlops-and-production/02-fastapi-docker` | Model serving | Serve a model that rejects a bad payload and reports whether it is up. | not run |
| `16-mlops-and-production/03-drift-monitoring` | Drift monitoring | The input distribution moves, then the relationship to the label moves. Detect each one. | not run |
| `16-mlops-and-production/04-batch-vs-realtime` | Batch versus real-time inference | Choose freshness or throughput. Measure latency and say what fails in each mode. | not run |
| `16-mlops-and-production/05-feature-store` | Feature store concept | Stop recomputing the same feature differently in training and in the request path. | not run |
| `16-mlops-and-production/06-cicd-quality-gates` | CI/CD for ML | A pipeline should fail when the data or the holdout metric breaks, not only when the unit tests pass. | not run |
| `16-mlops-and-production/07-compression` | Quantization and distillation | Make a small model faster. Report the accuracy you gave up, including a compression that was not worth it. | not run |

### 17 Responsible AI

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `17-responsible-ai/01-fairness` | Fairness metrics | A model treats groups differently. Measure it, try a mitigation, and do not hide a remaining gap. | not run |
| `17-responsible-ai/02-explainability` | Explainability | Explain a prediction to a reviewer, and show one plot that would send them to the wrong feature. | not run |
| `17-responsible-ai/03-privacy` | Privacy basics | A training extract contains identifiers. Handle them on synthetic data. Do not use private data. | not run |
| `17-responsible-ai/04-model-cards` | Model cards | Write the card from measured results in this repo, not from a generic template with blank metrics. | not run |

### 18 Real-world case studies

| Project | Topic | Use case | Key result |
| --- | --- | --- | --- |
| `18-real-world-case-studies/01-customer-churn` | Customer churn | A subscription business wants to know who will cancel and what they will do about it. | not run |
| `18-real-world-case-studies/02-credit-default` | Credit default risk | Estimate default risk and answer why an applicant was scored that way. | not run |
| `18-real-world-case-studies/03-fraud-cost-sensitive` | Cost-sensitive fraud | Catch fraud without burying the review team in false alarms. | not run |
| `18-real-world-case-studies/04-demand-forecasting` | Inventory demand | Decide how much to stock, not only how low the forecast error looks. | not run |
| `18-real-world-case-studies/05-search-ranking` | Search ranking | Rank products or documents. More clicks is not automatically a better ranking. | not run |
| `18-real-world-case-studies/06-support-ticket-routing` | Ticket routing | Route a support ticket to the right team, and measure what a misroute costs. | not run |
| `18-real-world-case-studies/07-document-qa` | Document question answering | Answer questions from internal-style documents. Say whether you found the passage and whether the answer was right. | not run |
| `18-real-world-case-studies/08-uplift-marketing` | Targeted marketing | Spend the promotion on people it changes, not on people who would have bought anyway. | not run |

## How to contribute

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: one project per commit, the smoke test passes, ruff passes, and every README number matches `results/metrics.json`.

## License

[MIT](LICENSE). Datasets keep their own licenses. Cite them in the project that uses them.
