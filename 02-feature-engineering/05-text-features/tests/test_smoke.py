"""Text-feature checks. No comment download and no model fit."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location(
        "goemotions_text_features",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_experiment_list_is_locked_and_the_vectorizer_leak_is_ineligible():
    main = load_main()
    names = [spec["name"] for spec in main.experiment_specs()]
    assert names == list(main.EXPERIMENT_ORDER)
    eligible = {spec["name"] for spec in main.experiment_specs() if spec["eligible"]}
    assert main.LEAKS.isdisjoint(eligible)
    assert "leaky_tfidf_fit_on_all" in main.LEAKS


def test_selection_uses_development_log_loss_not_the_test_score():
    main = load_main()
    chosen = main.select_model(
        [
            {
                "name": "bow_unigram",
                "eligible": True,
                "dev_log_loss": 0.4,
                "test_log_loss": 0.1,
            },
            {
                "name": "tfidf_unigram",
                "eligible": True,
                "dev_log_loss": 0.5,
                "test_log_loss": 0.01,
            },
            {
                "name": "leaky_tfidf_fit_on_all",
                "eligible": False,
                "dev_log_loss": 0.01,
                "test_log_loss": 0.01,
            },
        ]
    )
    assert chosen == "bow_unigram"


def test_ekman_map_collapses_one_group_and_drops_a_cross_group_comment():
    main = load_main()
    names = main.EMOTION_NAMES
    anger = names.index("anger")
    annoyance = names.index("annoyance")
    joy = names.index("joy")
    neutral = names.index("neutral")
    assert main.coarse_label(f"{anger},{annoyance}", names) == "anger"
    assert main.coarse_label(f"{anger},{joy}", names) is None
    assert main.coarse_label(str(neutral), names) == "neutral"
    covered = [fine for fines in main.EKMAN_GROUPS.values() for fine in fines]
    assert set(covered) == set(names) - {"neutral"}
    assert len(covered) == len(set(covered))


def test_leaky_fit_sees_the_test_text_and_the_honest_fit_does_not():
    main = load_main()
    train, dev, test = ["train comment"], ["dev comment"], ["test comment"]
    assert main.fit_corpus(train, dev, test, leaky=False) == ["train comment"]
    assert main.fit_corpus(train, dev, test, leaky=True) == [
        "train comment",
        "dev comment",
        "test comment",
    ]


def test_probabilities_are_renormalized_to_one():
    main = load_main()
    raw = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1]], dtype=np.float32)
    aligned = main.align_proba(np.array(main.LABELS), raw)
    assert aligned.shape == (1, 7)
    assert aligned.sum() == pytest.approx(1.0)


def test_overlap_mask_is_exact_and_does_not_choose_a_model():
    main = load_main()
    keep = main.unseen_text_mask(["same text", "new text"], {"same text", "other"})
    assert keep.tolist() == [False, True]


def test_readme_quotes_committed_metrics():
    metrics_path = PROJECT / "results" / "metrics.json"
    readme_path = PROJECT / "README.md"
    linkedin_path = PROJECT / "LINKEDIN.md"
    if not metrics_path.exists() or not readme_path.exists() or not linkedin_path.exists():
        pytest.skip("metrics and writeups are written after main.py")
    metrics = metrics_path.read_text(encoding="utf-8")
    payload = json.loads(metrics)
    readme = readme_path.read_text(encoding="utf-8")
    linkedin = linkedin_path.read_text(encoding="utf-8")
    missing = [number for number in re.findall(r"`([0-9eE.+-]+)`", readme) if number not in metrics]
    assert missing == []
    reported = payload["protocol"]["reported_model"]
    holdout = str(payload["experiments"][reported]["test"]["log_loss"])
    baseline = str(payload["experiments"]["training_rate"]["test"]["log_loss"])
    leaked = str(payload["experiments"]["leaky_tfidf_fit_on_all"]["test"]["log_loss"])
    assert holdout in readme
    assert holdout in linkedin
    assert baseline in readme
    assert leaked in readme
    assert leaked in linkedin
    assert payload["protocol"]["selection_uses_test_scores"] is False
    assert payload["experiments"]["leaky_tfidf_fit_on_all"]["eligible"] is False
    words = re.findall(r"[A-Za-z0-9']+", linkedin)
    assert 150 <= len(words) <= 250
