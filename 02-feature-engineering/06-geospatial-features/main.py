"""Haversine distances, location clusters, and a non-spatial baseline.

The cell split, the five reference cities, and the experiment list are fixed
before any ridge model is fit. Development RMSE on the held-out cells chooses
the eligible recipe. The holdout cells are not used to add, drop, or replace
a feature. A cluster fit on the holdout coordinates, and a neighbor mean built
before the split, are scored and are not eligible.

This is a teaching model on a 1990 census extract. It is not an appraisal.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
BOOTSTRAP_SEED = 7
BOOTSTRAP_RESAMPLES = 1000
CELL_DEGREES = 0.5
HOLDOUT_MODULO = 4
DEV_MODULO = 3
SPLIT_MODULO = 5
N_CLUSTERS = 8
N_NEIGHBORS = 15
RIDGE_ALPHA = 1.0
EARTH_RADIUS_KM = 6371.0
CENSOR_AT = 5.0
HOUSING_URL = "https://ndownloader.figshare.com/files/5976036"
HOUSING_SHA256 = "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"
NONSPATIAL = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
)
FEATURE_NAMES = (
    *NONSPATIAL,
    "Latitude",
    "Longitude",
)
# Frozen from the GeoNames cities15000 dump read on 2026-10-27, before any fit.
# Rule: five largest populated places with country US, admin1 CA, feature class P.
# A later dump does not change this list.
CITIES = (
    {
        "name": "Los Angeles",
        "geonameid": "5368361",
        "latitude": 34.05223,
        "longitude": -118.24368,
        "population": 3820914,
    },
    {
        "name": "San Diego",
        "geonameid": "5391811",
        "latitude": 32.71571,
        "longitude": -117.16472,
        "population": 1404452,
    },
    {
        "name": "San Jose",
        "geonameid": "5392171",
        "latitude": 37.33939,
        "longitude": -121.89496,
        "population": 997368,
    },
    {
        "name": "San Francisco",
        "geonameid": "5391959",
        "latitude": 37.77493,
        "longitude": -122.41942,
        "population": 827526,
    },
    {
        "name": "Fresno",
        "geonameid": "5350937",
        "latitude": 36.74773,
        "longitude": -119.77237,
        "population": 542107,
    },
)
EXPERIMENT_ORDER = (
    "training_mean",
    "nonspatial",
    "raw_coordinates",
    "haversine_cities",
    "euclidean_degrees",
    "train_clusters",
    "train_neighbor_mean",
    "spatial_block",
    "clusters_fit_on_all",
    "neighbor_fit_before_split",
)
LEAKS = frozenset({"clusters_fit_on_all", "neighbor_fit_before_split"})
DISTANCE_BINS_KM = (0.0, 25.0, 50.0, 100.0, 200.0, 400.0)

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"


def experiment_specs() -> list[dict[str, object]]:
    return [{"name": name, "eligible": name not in LEAKS} for name in EXPERIMENT_ORDER]


def select_model(rows: list[dict[str, object]]) -> str:
    """Lowest development RMSE. Test scores are not an argument to the sort."""
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no eligible experiment")
    order = {name: index for index, name in enumerate(EXPERIMENT_ORDER)}
    eligible.sort(
        key=lambda row: (
            row["dev_rmse"],
            row["n_features"],
            str(row["name"]),
            order[str(row["name"])],
        )
    )
    return str(eligible[0]["name"])


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lon2) - np.radians(lon1)
    half = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(half), np.sqrt(np.maximum(0.0, 1.0 - half)))


def cell_keys(latitude: np.ndarray, longitude: np.ndarray) -> list[tuple[float, float]]:
    cell_lat = np.floor(latitude / CELL_DEGREES) * CELL_DEGREES
    cell_lon = np.floor(longitude / CELL_DEGREES) * CELL_DEGREES
    return list(zip(cell_lat.tolist(), cell_lon.tolist(), strict=True))


def split_masks(
    keys: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float]]]:
    unique = sorted(set(keys))
    holdout = {cell for index, cell in enumerate(unique) if index % SPLIT_MODULO == HOLDOUT_MODULO}
    development = {cell for index, cell in enumerate(unique) if index % SPLIT_MODULO == DEV_MODULO}
    train = set(unique) - holdout - development
    train_mask = np.fromiter((key in train for key in keys), dtype=bool, count=len(keys))
    dev_mask = np.fromiter((key in development for key in keys), dtype=bool, count=len(keys))
    test_mask = np.fromiter((key in holdout for key in keys), dtype=bool, count=len(keys))
    return train_mask, dev_mask, test_mask, unique


def edge_adjacent_cells(
    source: set[tuple[float, float]],
    other: set[tuple[float, float]],
) -> int:
    count = 0
    for lat, lon in source:
        neighbors = (
            (lat + CELL_DEGREES, lon),
            (lat - CELL_DEGREES, lon),
            (lat, lon + CELL_DEGREES),
            (lat, lon - CELL_DEGREES),
        )
        if any(cell in other for cell in neighbors):
            count += 1
    return count


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ml-mastery-lab/geospatial"})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        handle.write(response.read())


def load_housing() -> tuple[np.ndarray, np.ndarray]:
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "cal_housing.tgz"
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != HOUSING_SHA256:
        if archive.exists():
            archive.unlink()
        _download(HOUSING_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != HOUSING_SHA256:
        raise RuntimeError(f"cal_housing.tgz hash {digest} does not match the pinned file")
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.extractfile("CaliforniaHousing/cal_housing.data")
        if member is None:
            raise RuntimeError("cal_housing.data is missing from the archive")
        raw = np.loadtxt(member, delimiter=",")
    ordered = raw[:, [8, 7, 2, 3, 4, 5, 6, 1, 0]]
    target = ordered[:, 0] / 100000.0
    data = ordered[:, 1:].copy()
    data[:, 2] /= data[:, 5]
    data[:, 3] /= data[:, 5]
    data[:, 5] = data[:, 4] / data[:, 5]
    return data, target


def city_distances(latitude: np.ndarray, longitude: np.ndarray, metric: str) -> np.ndarray:
    columns = []
    for city in CITIES:
        city_lat = np.full(latitude.shape, city["latitude"])
        city_lon = np.full(longitude.shape, city["longitude"])
        if metric == "haversine":
            columns.append(haversine_km(latitude, longitude, city_lat, city_lon))
        elif metric == "euclidean":
            columns.append(np.hypot(latitude - city_lat, longitude - city_lon))
        else:
            raise KeyError(metric)
    stacked = np.column_stack(columns)
    return np.column_stack([stacked, stacked.min(axis=1)])


def cluster_columns(train_coords: np.ndarray, query_coords: np.ndarray) -> np.ndarray:
    model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
    model.fit(train_coords)
    labels = model.predict(query_coords)
    return np.eye(N_CLUSTERS, dtype=np.float64)[labels]


def neighbor_columns(
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    pool_lat: np.ndarray,
    pool_lon: np.ndarray,
    pool_y: np.ndarray,
    exclude_self: np.ndarray | None,
) -> np.ndarray:
    ask = N_NEIGHBORS + (1 if exclude_self is not None else 0)
    if len(pool_lat) < ask:
        raise RuntimeError("neighbor pool is smaller than the locked neighbor count")
    index = NearestNeighbors(n_neighbors=ask, metric="haversine", algorithm="ball_tree")
    index.fit(np.radians(np.column_stack([pool_lat, pool_lon])))
    _distances, neighbors = index.kneighbors(np.radians(np.column_stack([query_lat, query_lon])))
    means = np.empty(len(query_lat), dtype=np.float64)
    nearest = np.empty(len(query_lat), dtype=np.float64)
    for row in range(len(query_lat)):
        kept = []
        for neighbor in neighbors[row]:
            if exclude_self is not None and int(neighbor) == int(exclude_self[row]):
                continue
            kept.append(int(neighbor))
            if len(kept) == N_NEIGHBORS:
                break
        if len(kept) != N_NEIGHBORS:
            raise RuntimeError("could not find the locked number of other neighbors")
        means[row] = float(pool_y[kept].mean())
        nearest[row] = float(
            haversine_km(
                np.array([query_lat[row]]),
                np.array([query_lon[row]]),
                np.array([pool_lat[kept[0]]]),
                np.array([pool_lon[kept[0]]]),
            )[0]
        )
    return np.column_stack([means, nearest])


def design_matrix(
    name: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
    nonspatial: np.ndarray,
    train_mask: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    coords = np.column_stack([latitude, longitude])
    if name == "training_mean":
        return np.zeros((len(latitude), 0), dtype=np.float64)
    if name == "nonspatial":
        return nonspatial.copy()
    if name == "raw_coordinates":
        return np.column_stack([nonspatial, latitude, longitude])
    if name == "haversine_cities":
        return np.column_stack([nonspatial, city_distances(latitude, longitude, "haversine")])
    if name == "euclidean_degrees":
        return np.column_stack([nonspatial, city_distances(latitude, longitude, "euclidean")])
    if name == "train_clusters":
        clusters = cluster_columns(coords[train_mask], coords)
        return np.column_stack([nonspatial, clusters])
    if name == "clusters_fit_on_all":
        clusters = cluster_columns(coords, coords)
        return np.column_stack([nonspatial, clusters])
    if name == "train_neighbor_mean":
        neighbors = np.zeros((len(latitude), 2), dtype=np.float64)
        neighbors[train_mask] = neighbor_columns(
            latitude[train_mask],
            longitude[train_mask],
            latitude[train_mask],
            longitude[train_mask],
            target[train_mask],
            exclude_self=np.arange(int(train_mask.sum())),
        )
        outside = ~train_mask
        neighbors[outside] = neighbor_columns(
            latitude[outside],
            longitude[outside],
            latitude[train_mask],
            longitude[train_mask],
            target[train_mask],
            exclude_self=None,
        )
        return np.column_stack([nonspatial, neighbors])
    if name == "neighbor_fit_before_split":
        neighbors = neighbor_columns(
            latitude,
            longitude,
            latitude,
            longitude,
            target,
            exclude_self=np.arange(len(latitude)),
        )
        return np.column_stack([nonspatial, neighbors])
    if name == "spatial_block":
        cities = city_distances(latitude, longitude, "haversine")
        clusters = cluster_columns(coords[train_mask], coords)
        neighbors = design_matrix(
            "train_neighbor_mean",
            latitude,
            longitude,
            nonspatial,
            train_mask,
            target,
        )[:, -2:]
        return np.column_stack([nonspatial, cities, clusters, neighbors])
    raise KeyError(name)


def score_regression(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predicted))),
        "mae": float(mean_absolute_error(y_true, predicted)),
        "r2": float(r2_score(y_true, predicted)),
    }


def fit_predict(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    dev_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.shape[1] == 0:
        mean = float(target[train_mask].mean())
        predicted = np.full(len(target), mean, dtype=np.float64)
        return predicted[dev_mask], predicted[test_mask], predicted
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    model = Ridge(alpha=RIDGE_ALPHA, solver="cholesky")
    train_x = scaler.fit_transform(imputer.fit_transform(features[train_mask]))
    model.fit(train_x, target[train_mask])
    transformed = scaler.transform(imputer.transform(features))
    predicted = model.predict(transformed)
    return predicted[dev_mask], predicted[test_mask], predicted


def bootstrap_gap(
    y_true: np.ndarray,
    cells: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | bool]:
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    gaps = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for draw in range(BOOTSTRAP_RESAMPLES):
        chosen = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[index] for index in chosen])
        ref = float(np.sqrt(mean_squared_error(y_true[rows], reference[rows])))
        cand = float(np.sqrt(mean_squared_error(y_true[rows], candidate[rows])))
        gaps[draw] = ref - cand
    low, high = np.quantile(gaps, [0.025, 0.975])
    point = float(
        np.sqrt(mean_squared_error(y_true, reference))
        - np.sqrt(mean_squared_error(y_true, candidate))
    )
    return {
        "point_gap": point,
        "interval_low": float(low),
        "interval_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def column_audit(values: np.ndarray) -> dict[str, float | int]:
    rows = {}
    for index, name in enumerate(FEATURE_NAMES):
        column = values[:, index]
        rows[name] = {
            "min": float(column.min()),
            "max": float(column.max()),
            "mean": float(column.mean()),
            "missing": int(np.isnan(column).sum()),
        }
    return rows


def plot_split(
    path: Path, longitude: np.ndarray, latitude: np.ndarray, masks: dict[str, np.ndarray]
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    for name, color, mask in (
        ("train", "#4C78A8", masks["train"]),
        ("dev", "#F58518", masks["dev"]),
        ("test", "#54A24B", masks["test"]),
    ):
        ax.scatter(
            longitude[mask],
            latitude[mask],
            s=6,
            alpha=0.45,
            c=color,
            label=name,
            linewidths=0,
        )
    ax.set_title("0.5-degree cells. Every fifth sorted cell is held out.")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Split", markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_rmse(
    path: Path,
    experiments: dict[str, dict[str, object]],
    reported: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    names = [name for name in EXPERIMENT_ORDER]
    x = np.arange(len(names))
    heights = [experiments[name]["test"]["rmse"] for name in names]
    bars = ax.bar(x, heights, color="#4C78A8")
    for bar, name in zip(bars, names, strict=True):
        if name in LEAKS:
            bar.set_hatch("//")
            bar.set_facecolor("#F58518")
    bars[names.index(reported)].set_facecolor("#54A24B")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("Holdout RMSE, median house value / 100000")
    ax.set_xlabel("Feature recipe")
    ax.set_title("Same held-out cells. Hatched bars are not eligible.")
    handles = [
        Patch(facecolor="#54A24B", label="Selected on development RMSE"),
        Patch(facecolor="#4C78A8", label="Eligible, not selected"),
        Patch(facecolor="#F58518", hatch="//", edgecolor="black", label="Not eligible"),
    ]
    ax.legend(handles=handles)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    path: Path,
    longitude: np.ndarray,
    latitude: np.ndarray,
    residual: np.ndarray,
    model_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    limit = float(np.quantile(np.abs(residual), 0.98))
    image = ax.scatter(
        longitude,
        latitude,
        c=residual,
        s=8,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        linewidths=0,
    )
    fig.colorbar(image, ax=ax, label="Actual minus predicted")
    ax.set_title(f"Holdout residual, {model_name}. Not used to choose the model.")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def distance_profile(
    distance_km: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, float | int | None]]:
    edges = list(DISTANCE_BINS_KM) + [float(np.inf)]
    rows: list[dict[str, float | int | None]] = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (distance_km >= left) & (distance_km < right)
        rows.append(
            {
                "left_km": float(left),
                "right_km": None if np.isinf(right) else float(right),
                "rows": int(mask.sum()),
                "actual_mean": None if not np.any(mask) else float(actual[mask].mean()),
                "predicted_mean": None if not np.any(mask) else float(predicted[mask].mean()),
            }
        )
    return rows


def plot_distance(
    path: Path,
    profile: list[dict[str, float | int | None]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = []
    actual_means = []
    predicted_means = []
    counts = []
    for row in profile:
        if not row["rows"]:
            continue
        right = row["right_km"]
        right_label = "plus" if right is None else f"{right:g}"
        labels.append(f"{row['left_km']:g} to {right_label}")
        actual_means.append(float(row["actual_mean"]))
        predicted_means.append(float(row["predicted_mean"]))
        counts.append(int(row["rows"]))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, actual_means, width, label="Actual", color="#4C78A8")
    ax.bar(x + width / 2, predicted_means, width, label="Predicted", color="#F58518")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Mean median house value / 100000")
    ax.set_xlabel("Kilometers to the nearest locked city")
    ax.set_title("Holdout means by distance. Bins were locked before the fit.")
    ax.legend()
    for index, count in enumerate(counts):
        top = max(actual_means[index], predicted_means[index])
        ax.text(index, top, f"n={count}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    features, target = load_housing()
    if features.shape != (len(target), len(FEATURE_NAMES)):
        raise RuntimeError("parsed housing matrix does not match the locked column list")
    latitude = features[:, FEATURE_NAMES.index("Latitude")]
    longitude = features[:, FEATURE_NAMES.index("Longitude")]
    nonspatial = features[:, : len(NONSPATIAL)]
    keys = cell_keys(latitude, longitude)
    train_mask, dev_mask, test_mask, unique_cells = split_masks(keys)
    if not (np.sum([train_mask, dev_mask, test_mask], axis=0) == 1).all():
        raise RuntimeError("a block group landed in more than one split")
    train_cells = {keys[index] for index in np.flatnonzero(train_mask)}
    dev_cells = {keys[index] for index in np.flatnonzero(dev_mask)}
    test_cells = {keys[index] for index in np.flatnonzero(test_mask)}
    if train_cells & dev_cells or train_cells & test_cells or dev_cells & test_cells:
        raise RuntimeError("a cell is on both sides of the split")
    if (len(target), int((target >= CENSOR_AT).sum())) != (20640, 992):
        raise RuntimeError("housing row count or censored-target count does not match the lock")
    if not np.isclose(float(target.max()), 5.00001):
        raise RuntimeError(f"target max {float(target.max())} does not match the lock")
    if (len(train_cells), len(dev_cells), len(test_cells)) != (106, 35, 35):
        raise RuntimeError(
            "cell split "
            f"{len(train_cells), len(dev_cells), len(test_cells)} does not match the lock"
        )
    if (int(train_mask.sum()), int(dev_mask.sum()), int(test_mask.sum())) != (13700, 5056, 1884):
        raise RuntimeError(
            "row split "
            f"{int(train_mask.sum()), int(dev_mask.sum()), int(test_mask.sum())} "
            "does not match the lock"
        )

    fitted = []
    holdout_predictions = {}
    full_predictions = {}
    honest_nearest_km = None
    for name in EXPERIMENT_ORDER:
        print(f"fitting {name}", flush=True)
        matrix = design_matrix(name, latitude, longitude, nonspatial, train_mask, target)
        dev_pred, test_pred, full_pred = fit_predict(
            matrix, target, train_mask, dev_mask, test_mask
        )
        dev_score = score_regression(target[dev_mask], dev_pred)
        test_score = score_regression(target[test_mask], test_pred)
        fitted.append(
            {
                "name": name,
                "eligible": name not in LEAKS,
                "dev_rmse": dev_score["rmse"],
                "n_features": int(matrix.shape[1]),
                "dev": dev_score,
                "test": test_score,
            }
        )
        holdout_predictions[name] = test_pred
        full_predictions[name] = full_pred
        if name == "train_neighbor_mean":
            honest_nearest_km = matrix[:, -1].copy()
    if honest_nearest_km is None:
        raise RuntimeError("honest neighbor distance was not built")
    reported = select_model(fitted)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y_test = target[test_mask]
    test_cells_array = np.array([keys[index] for index in np.flatnonzero(test_mask)], dtype=float)
    pairs = (
        ("training_mean", reported, "selected_vs_training_mean"),
        ("nonspatial", reported, "selected_vs_nonspatial"),
        ("raw_coordinates", "haversine_cities", "haversine_vs_raw_coordinates"),
        ("euclidean_degrees", "haversine_cities", "haversine_vs_euclidean_degrees"),
        ("nonspatial", "train_neighbor_mean", "neighbor_vs_nonspatial"),
        ("nonspatial", "train_clusters", "clusters_vs_nonspatial"),
        ("haversine_cities", "spatial_block", "spatial_block_vs_haversine"),
        ("train_clusters", "clusters_fit_on_all", "leaky_clusters_vs_train_clusters"),
        (
            "train_neighbor_mean",
            "neighbor_fit_before_split",
            "leaky_neighbors_vs_train_neighbors",
        ),
    )
    comparisons = {
        key: {
            "reference": reference,
            "candidate": candidate,
            "gap_definition": "reference_rmse - candidate_rmse",
            **bootstrap_gap(
                y_test,
                test_cells_array,
                holdout_predictions[reference],
                holdout_predictions[candidate],
                rng,
            ),
        }
        for reference, candidate, key in pairs
    }

    random_rng = np.random.default_rng(SEED)
    permutation = random_rng.permutation(len(target))
    random_test = np.zeros(len(target), dtype=bool)
    random_test[permutation[: int(test_mask.sum())]] = True
    random_train = ~random_test
    random_matrix = design_matrix(
        "spatial_block", latitude, longitude, nonspatial, random_train, target
    )
    _random_dev, random_pred, _random_full = fit_predict(
        random_matrix,
        target,
        random_train,
        random_test,
        random_test,
    )
    random_score = score_regression(target[random_test], random_pred)
    random_keys = [keys[index] for index in np.flatnonzero(random_test)]
    random_train_cells = {keys[index] for index in np.flatnonzero(random_train)}
    random_overlap_rows = sum(key in random_train_cells for key in random_keys)

    reported_test = full_predictions[reported][test_mask]
    uncapped = y_test < CENSOR_AT
    uncapped_score = score_regression(y_test[uncapped], reported_test[uncapped])
    city_km = city_distances(latitude, longitude, "haversine")
    min_city_km = city_km[:, -1]

    experiments = {
        row["name"]: {
            "eligible": row["eligible"],
            "n_features": row["n_features"],
            "dev": row["dev"],
            "test": row["test"],
        }
        for row in fitted
    }
    payload = {
        "dataset": {
            "name": "California housing",
            "source": "StatLib houses extract redistributed by scikit-learn from figshare",
            "url": HOUSING_URL,
            "sha256": HOUSING_SHA256,
            "citation": "Pace and Barry, Statistics and Probability Letters, 33:291-297, 1997",
            "target": "MedHouseVal",
            "target_unit": "median house value divided by 100000",
            "cities_source": "GeoNames cities15000, CC BY 4.0, frozen 2026-10-27",
        },
        "protocol": {
            "reported_model": reported,
            "selection_metric": "dev_rmse",
            "selection_uses_test_scores": False,
            "split": (
                "every fifth sorted 0.5-degree cell is the holdout; "
                "the previous remainder is development"
            ),
            "seed_used_for_split": False,
            "seed": SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "cell_degrees": CELL_DEGREES,
            "n_clusters": N_CLUSTERS,
            "n_neighbors": N_NEIGHBORS,
            "ridge_alpha": RIDGE_ALPHA,
            "earth_radius_km": EARTH_RADIUS_KM,
            "gap_definition": "reference_rmse - candidate_rmse",
            "positive_gap_means": "candidate has lower RMSE",
            "adjacent_cells_can_leak": True,
            "fixed_alpha_means_wider_matrix_is_not_equal_footing": True,
        },
        "cities": list(CITIES),
        "audit": {
            "rows": int(len(target)),
            "missing_feature_values": int(np.isnan(features).sum()),
            "censored_at_or_above_5": int((target >= CENSOR_AT).sum()),
            "target_max": float(target.max()),
            "columns": column_audit(features),
            "cells": int(len(unique_cells)),
            "split_cells": {
                "train": int(len(train_cells)),
                "dev": int(len(dev_cells)),
                "test": int(len(test_cells)),
            },
            "split_rows": {
                "train": int(train_mask.sum()),
                "dev": int(dev_mask.sum()),
                "test": int(test_mask.sum()),
            },
            "target_mean": {
                "train": float(target[train_mask].mean()),
                "dev": float(target[dev_mask].mean()),
                "test": float(target[test_mask].mean()),
            },
            "censored_rows": {
                "train": int((target[train_mask] >= CENSOR_AT).sum()),
                "dev": int((target[dev_mask] >= CENSOR_AT).sum()),
                "test": int((target[test_mask] >= CENSOR_AT).sum()),
            },
            "holdout_cells_sharing_an_edge_with_train": edge_adjacent_cells(
                test_cells, train_cells
            ),
            "holdout_rows_in_an_edge_adjacent_cell": int(
                sum(
                    edge_adjacent_cells({keys[index]}, train_cells)
                    for index in np.flatnonzero(test_mask)
                )
            ),
            "holdout_nearest_training_km": {
                "min": float(honest_nearest_km[test_mask].min()),
                "median": float(np.median(honest_nearest_km[test_mask])),
                "mean": float(honest_nearest_km[test_mask].mean()),
                "p90": float(np.quantile(honest_nearest_km[test_mask], 0.9)),
                "max": float(honest_nearest_km[test_mask].max()),
            },
            "cell_overlap": {
                "train_dev": int(len(train_cells & dev_cells)),
                "train_test": int(len(train_cells & test_cells)),
                "dev_test": int(len(dev_cells & test_cells)),
            },
            "random_split": {
                "eligible": False,
                "recipe": "spatial_block",
                "seed": SEED,
                "test_rows": int(random_test.sum()),
                "test_rows_whose_cell_is_also_in_fit": int(random_overlap_rows),
                "test": random_score,
            },
        },
        "experiments": experiments,
        "comparisons": comparisons,
        "uncapped_holdout_sensitivity": {
            "does_not_reselect": True,
            "rule": "target < 5",
            "excluded_test_rows": int((~uncapped).sum()),
            "kept_test_rows": int(uncapped.sum()),
            "test": uncapped_score,
        },
        "distance_bins_km": list(DISTANCE_BINS_KM),
        "distance_profile": distance_profile(min_city_km[test_mask], y_test, reported_test),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(
        json.dumps(jsonable(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    plot_split(
        RESULTS / "split_map.png",
        longitude,
        latitude,
        {"train": train_mask, "dev": dev_mask, "test": test_mask},
    )
    plot_rmse(RESULTS / "rmse.png", experiments, reported)
    plot_residuals(
        RESULTS / "residual_map.png",
        longitude[test_mask],
        latitude[test_mask],
        y_test - reported_test,
        reported,
    )
    plot_distance(RESULTS / "distance_profile.png", payload["distance_profile"])
    print(f"reported {reported}", flush=True)
    print(f"test rmse {experiments[reported]['test']['rmse']}", flush=True)


if __name__ == "__main__":
    main()
