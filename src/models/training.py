"""
Reproducible training of the clustering model and the cluster behavior profiles.

Everything is computed from data up to `as_of`: the same date and the same data always give
the same model. Artifacts are saved in a versioned folder with a metadata.json that records
the library versions, the training windows and a fingerprint of the training data.
"""
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.contracts import CLUSTER_PROFILES, validate, validate_usd_rates
from src.features.currency import country_currency_map, static_usd_rates
from src.features.customer_features import build_customer_features
from src.features.sources import load_customers_with_products, load_transactions
from src.features.transaction_behavior import (MONTHLY_RATE_FEATURES, RECENT_DAYS,
                                               profile_feature_names, profile_window,
                                               window_behavior)
from src.models.behavior_scoring import build_cluster_profiles
from src.models.customer_clustering import (N_CLUSTERS, N_PCA_COMPONENTS, RANDOM_STATE,
                                            assign_clusters, fit_customer_clustering,
                                            validate_bundle)

CUSTOMER_MODEL_FILE = "customer_cluster_model.joblib"
PROFILES_FILE = "cluster_behavior_profiles.joblib"
METADATA_FILE = "metadata.json"


class ArtifactVersionError(RuntimeError):
    """The artifacts were saved with a scikit-learn version that cannot be loaded here."""


@dataclass
class TrainedArtifacts:
    customer_model: dict
    profiles: dict
    metadata: dict


def model_version(as_of) -> str:
    return f"clusters_{pd.Timestamp(as_of):%Y%m%d}"


def fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame's content (row order included)."""
    hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()[:16]


def cluster_activity(clusters: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    """Customers per cluster and how many had activity in the profile window (notebook §16)."""
    active = clusters["customer_id"].isin(behavior["customer_id"])
    activity = (clusters.assign(active=active).groupby("cluster")
                        .agg(total_customers=("customer_id", "nunique"),
                             active_customers=("active", "sum"))
                        .reset_index())
    activity["active_customer_ratio"] = activity["active_customers"] / activity["total_customers"]
    return activity


def train(con, as_of, n_components: int = N_PCA_COMPONENTS, n_clusters: int = N_CLUSTERS,
          random_state: int = RANDOM_STATE, usd_rates: dict | None = None) -> TrainedArtifacts:
    """
    Fit the customer model and the cluster profiles with data up to `as_of`.
    USD rates are the fixed ones of currency_config.json unless `usd_rates` is given.
    """
    as_of = pd.Timestamp(as_of).normalize()
    rates = usd_rates if usd_rates is not None else static_usd_rates()
    validate_usd_rates(rates)
    features = build_customer_features(load_customers_with_products(con), as_of,
                                       country_currency_map(), rates)
    customer_model = fit_customer_clustering(features, n_components, n_clusters, random_state)
    clusters = assign_clusters(customer_model, features)

    window = profile_window(as_of)
    transactions = load_transactions(con, window.start, window.end,
                                     customer_ids=clusters["customer_id"].tolist())
    behavior = window_behavior(transactions, window)
    features_used = profile_feature_names(behavior.columns)
    statistics = build_cluster_profiles(behavior, clusters, features_used)

    version = model_version(as_of)
    profiles = {
        "profile_statistics": statistics,
        "cluster_activity": cluster_activity(clusters, behavior),
        "profile_features": features_used,
        "baseline_window": {"start": window.start, "end": window.end - pd.Timedelta(days=1),
                            "months": 12},
        "recent_window_days": RECENT_DAYS,
        "monthly_rate_features": [f"{f}_monthly" for f in MONTHLY_RATE_FEATURES
                                  if f"{f}_monthly" in features_used],
        "model_version": version,
    }
    customer_model["model_version"] = version
    sizes = clusters["cluster"].value_counts().sort_index()
    metadata = {
        "model_version": version,
        "as_of": as_of.date().isoformat(),
        "profile_window": {"start": window.start.date().isoformat(),
                           "end_inclusive": (window.end - pd.Timedelta(days=1)).date().isoformat()},
        "recent_window_days": RECENT_DAYS,
        "n_customers": int(len(features)),
        "n_customers_with_activity": int(behavior["customer_id"].isin(clusters["customer_id"]).sum()),
        "n_transactions": int(len(transactions)),
        "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        "n_clusters": n_clusters, "n_pca_components": n_components,
        "random_state": random_state,
        "pca_explained_variance": round(float(customer_model["pca"].explained_variance_ratio_.sum()), 4),
        "profile_features": features_used,
        "usd_rates": rates,
        "usd_rates_source": "currency_config.json" if usd_rates is None else "argumento usd_rates",
        "data_fingerprint": {"customer_features": fingerprint(features),
                             "behavior": fingerprint(behavior)},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    return TrainedArtifacts(customer_model, profiles, metadata)


def save_artifacts(artifacts: TrainedArtifacts, models_dir) -> Path:
    """Write the artifacts to <models_dir>/<model_version>/ and return that folder."""
    folder = Path(models_dir) / artifacts.metadata["model_version"]
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.customer_model, folder / CUSTOMER_MODEL_FILE)
    joblib.dump(artifacts.profiles, folder / PROFILES_FILE)
    (folder / METADATA_FILE).write_text(json.dumps(artifacts.metadata, indent=2,
                                                   ensure_ascii=False), encoding="utf-8")
    return folder


def _same_minor(a: str, b: str) -> bool:
    return a.split(".")[:2] == b.split(".")[:2]


def load_artifacts(folder) -> TrainedArtifacts:
    """Load a versioned folder, refusing artifacts from another scikit-learn version."""
    folder = Path(folder)
    metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    trained_with = metadata["versions"]["scikit-learn"]
    if not _same_minor(trained_with, sklearn.__version__):
        raise ArtifactVersionError(
            f"{folder.name} se entrenó con scikit-learn {trained_with} y aquí está instalado "
            f"{sklearn.__version__}. Reentrena con train_clusters.py o instala esa versión.")
    customer_model = joblib.load(folder / CUSTOMER_MODEL_FILE)
    validate_bundle(customer_model)
    profiles = joblib.load(folder / PROFILES_FILE)
    validate(profiles["profile_statistics"], CLUSTER_PROFILES)
    return TrainedArtifacts(customer_model, profiles, metadata)
