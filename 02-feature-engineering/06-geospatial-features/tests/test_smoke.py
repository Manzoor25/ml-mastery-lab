"""Geospatial-feature checks. No housing download and no model fit."""

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
        "california_geospatial_features",
        PROJECT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_experiment_list_is_locked_and_the_leaks_are_ineligible():
    main = load_main()
    names = [spec["name"] for spec in main.experiment_specs()]
    assert names == list(main.EXPERIMENT_ORDER)
    eligible = {spec["name"] for spec in main.experiment_specs() if spec["eligible"]}
    assert main.LEAKS.isdisjoint(eligible)
    assert "clusters_fit_on_all" in main.LEAKS
    assert "neighbor_fit_before_split" in main.LEAKS


def test_selection_uses_development_rmse_not_the_test_score():
    main = load_main()
    chosen = main.select_model(
        [
            {
                "name": "nonspatial",
                "eligible": True,
                "dev_rmse": 0.4,
                "n_features": 6,
                "test_rmse": 0.1,
            },
            {
                "name": "haversine_cities",
                "eligible": True,
                "dev_rmse": 0.5,
                "n_features": 12,
                "test_rmse": 0.01,
            },
            {
                "name": "clusters_fit_on_all",
                "eligible": False,
                "dev_rmse": 0.01,
                "n_features": 14,
                "test_rmse": 0.01,
            },
        ]
    )
    assert chosen == "nonspatial"


def test_selection_breaks_a_tie_by_fewer_features_then_name():
    main = load_main()
    chosen = main.select_model(
        [
            {"name": "spatial_block", "eligible": True, "dev_rmse": 0.4, "n_features": 22},
            {"name": "haversine_cities", "eligible": True, "dev_rmse": 0.4, "n_features": 12},
            {"name": "raw_coordinates", "eligible": True, "dev_rmse": 0.4, "n_features": 12},
        ]
    )
    assert chosen == "haversine_cities"


def test_cell_split_keeps_a_cell_on_one_side():
    main = load_main()
    latitude = np.array([32.1, 32.2, 33.1, 33.2, 34.1, 34.6, 35.1, 35.6, 36.1, 36.6])
    longitude = np.array(
        [-117.1, -117.2, -118.1, -118.2, -119.1, -119.4, -120.1, -120.4, -121.1, -121.4]
    )
    keys = main.cell_keys(latitude, longitude)
    train, dev, test, _unique = main.split_masks(keys)
    assert np.sum([train, dev, test], axis=0).tolist() == [1] * len(keys)
    train_cells = {keys[index] for index in np.flatnonzero(train)}
    test_cells = {keys[index] for index in np.flatnonzero(test)}
    assert train_cells.isdisjoint(test_cells)


def test_haversine_is_zero_at_a_point_and_about_111_km_per_degree():
    main = load_main()
    same = main.haversine_km(
        np.array([34.0]),
        np.array([-118.0]),
        np.array([34.0]),
        np.array([-118.0]),
    )
    degree = main.haversine_km(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
    reverse = main.haversine_km(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert same[0] == pytest.approx(0.0, abs=1e-8)
    assert 110.0 < degree[0] < 112.0
    assert degree[0] == pytest.approx(reverse[0])


def test_training_neighbor_mean_excludes_the_row_itself():
    main = load_main()
    latitude = np.linspace(32.0, 33.9, 20)
    longitude = np.full(20, -118.0)
    target = np.arange(20, dtype=float)
    columns = main.neighbor_columns(
        latitude,
        longitude,
        latitude,
        longitude,
        target,
        exclude_self=np.arange(20),
    )
    assert columns[0, 0] == pytest.approx(np.arange(1, 16).mean())
    assert columns[0, 1] > 0.0


def test_reference_cities_are_frozen_in_population_order():
    main = load_main()
    assert [city["name"] for city in main.CITIES] == [
        "Los Angeles",
        "San Diego",
        "San Jose",
        "San Francisco",
        "Fresno",
    ]
    assert [city["geonameid"] for city in main.CITIES] == [
        "5368361",
        "5391811",
        "5392171",
        "5391959",
        "5350937",
    ]


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
    holdout = str(payload["experiments"][reported]["test"]["rmse"])
    baseline = str(payload["experiments"]["training_mean"]["test"]["rmse"])
    leaked = str(payload["experiments"]["neighbor_fit_before_split"]["test"]["rmse"])
    assert holdout in readme
    assert holdout in linkedin
    assert baseline in readme
    assert leaked in readme
    assert leaked in linkedin
    assert payload["protocol"]["selection_uses_test_scores"] is False
    assert payload["experiments"]["neighbor_fit_before_split"]["eligible"] is False
    words = re.findall(r"[A-Za-z0-9']+", linkedin)
    assert 150 <= len(words) <= 250
