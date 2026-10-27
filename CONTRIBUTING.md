# Contributing

This repo teaches by running code. A change that looks finished but was not run is not finished.

## Add or update a project

Each project lives in its own folder:

```text
section/project-name/
├── README.md
├── main.py
├── requirements.txt    # only if it needs more than the root file
├── results/            # metrics.json and plots produced by main.py
├── tests/test_smoke.py
└── LINKEDIN.md
```

`notebook.ipynb` is optional. If you add one, execute it and save the outputs. `python main.py` remains the command that produces `results/metrics.json`.

Copy the headings in [templates/README_TEMPLATE.md](templates/README_TEMPLATE.md) and [templates/LINKEDIN_TEMPLATE.md](templates/LINKEDIN_TEMPLATE.md). Do not leave the template instructions in the project files.

## Rules

- Use a public dataset with a permissive license. Download it in code. Do not commit raw data. `data/` is gitignored.
- Cite the dataset and its license in the project README.
- Set random seeds. The repo default is 42 unless the README explains a different seed.
- Fit preprocessing on training data only. Cross-validation folds must not see their own labels through encoders, scalers, or feature selection.
- Pin any extra dependency in the project `requirements.txt`.
- Include a baseline. Report the change from that baseline even when the change is small or negative.
- Include at least one experiment that did worse, and explain why.
- Every number in the README must match `results/metrics.json`. Do not type a metric from memory.
- Plots need a title and axis labels.
- Smoke tests run on a tiny sample and finish in under 30 seconds.
- Keep a full project under about 15 minutes on a laptop CPU, unless the project is about scale. If it is, run a small subset and write how you would scale it.
- Copy `LINKEDIN.md` to `linkedin-posts/` using the next number in curriculum order. Drafts only. Do not publish them from this repo.

## Before you mark a project done

- [ ] `python main.py` runs from a fresh environment.
- [ ] `pytest tests/test_smoke.py` passes.
- [ ] README numbers match `results/metrics.json`.
- [ ] A baseline is reported, including a failed or weaker experiment.
- [ ] No leakage: split first, fit on train only.
- [ ] Plots are labeled.
- [ ] `LINKEDIN.md` quotes a number that is in `results/metrics.json`.
- [ ] `ruff check .` and `ruff format --check .` pass from the repo root.
- [ ] [PROGRESS.md](PROGRESS.md) status is `done`, and `Next:` points at the first project that is not done.

Then commit that project on its own. Do not batch several projects into one commit.

```text
feat(02-feature-engineering): show target leakage and the fix
```

## Status values

Use only `todo`, `in progress`, `done`, or `blocked: <reason>` in PROGRESS.md.

If you are blocked, say what failed and what you tried. Then move to the next project. Do not invent a result so the checklist looks finished.
