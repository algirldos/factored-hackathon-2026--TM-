"""Reproducible training: determinism, no future data, versioned artifacts and the CLI."""
import json

import pandas as pd
import pytest

import train_clusters
from factories import AS_OF, bronze_connection, load_bronze, raw_users, transactions
from src.features.currency import country_currency_map, static_usd_rates
from src.features.customer_features import build_customer_features
from src.features.sources import load_customers_with_products, load_usd_rates
from src.features.transaction_behavior import profile_window
from src.models.customer_clustering import assign_clusters
from src.models.training import (ArtifactVersionError, load_artifacts, model_version,
                                 save_artifacts, train)

N_CUSTOMERS = 45


def bronze(extra_tx: pd.DataFrame | None = None):
    users = raw_users(N_CUSTOMERS)
    ids = sorted(users["customer_id"].unique())
    window = profile_window(AS_OF)
    tx = transactions(ids, window.start, window.days, per_day=0.1,
                      amount={cid: 20.0 + 3.0 * i for i, cid in enumerate(ids)})
    if extra_tx is not None:
        tx = pd.concat([tx, extra_tx])
    con = bronze_connection()
    load_bronze(con, users, tx)
    return con


def fit(con):
    return train(con, AS_OF, n_components=4, n_clusters=3)


@pytest.fixture(scope="module")
def artifacts():
    return fit(bronze())


def test_metadata_records_what_was_trained(artifacts):
    meta = artifacts.metadata
    assert meta["model_version"] == model_version(AS_OF) == "clusters_20260617"
    assert meta["as_of"] == "2026-06-17"
    assert meta["profile_window"] == {"start": "2025-06-18", "end_inclusive": "2026-06-17"}
    assert meta["n_customers"] == N_CUSTOMERS
    assert sum(meta["cluster_sizes"].values()) == N_CUSTOMERS
    assert set(meta["versions"]) == {"python", "scikit-learn", "pandas", "numpy"}
    assert meta["usd_rates"] == static_usd_rates()           # fixed rates of currency_config.json
    assert meta["usd_rates_source"] == "currency_config.json"
    assert artifacts.customer_model["model_version"] == artifacts.profiles["model_version"]


def test_profiles_keep_the_notebook_format(artifacts):
    profiles = artifacts.profiles
    assert {"profile_statistics", "cluster_activity", "profile_features", "baseline_window",
            "recent_window_days", "monthly_rate_features"} <= set(profiles)
    activity = profiles["cluster_activity"]
    assert activity["total_customers"].sum() == N_CUSTOMERS
    assert (activity["active_customer_ratio"] <= 1).all()


def test_same_date_and_data_give_the_same_model(artifacts):
    again = fit(bronze())
    assert again.metadata["data_fingerprint"] == artifacts.metadata["data_fingerprint"]
    pd.testing.assert_frame_equal(again.profiles["profile_statistics"],
                                  artifacts.profiles["profile_statistics"])
    assert again.metadata["cluster_sizes"] == artifacts.metadata["cluster_sizes"]


def test_data_after_as_of_is_never_used(artifacts):
    future = transactions(["CLI-0001", "CLI-0002"], AS_OF + pd.Timedelta(days=1), 20,
                          per_day=1.0, amount=50_000.0, seed=9)
    with_future = fit(bronze(extra_tx=future))
    pd.testing.assert_frame_equal(with_future.profiles["profile_statistics"],
                                  artifacts.profiles["profile_statistics"])
    assert with_future.metadata["n_transactions"] == artifacts.metadata["n_transactions"]


def test_saved_artifacts_load_and_predict_the_same(tmp_path, artifacts):
    folder = save_artifacts(artifacts, tmp_path)
    assert folder.name == "clusters_20260617"
    assert {p.name for p in folder.iterdir()} == {"customer_cluster_model.joblib",
                                                  "cluster_behavior_profiles.joblib",
                                                  "metadata.json"}
    loaded = load_artifacts(folder)
    assert loaded.metadata == json.loads(json.dumps(artifacts.metadata))
    con = bronze()
    features = build_customer_features(load_customers_with_products(con), AS_OF,
                                       country_currency_map(), load_usd_rates(con, AS_OF))
    pd.testing.assert_frame_equal(assign_clusters(loaded.customer_model, features),
                                  assign_clusters(artifacts.customer_model, features))


def test_artifacts_from_another_sklearn_version_are_refused(tmp_path, artifacts):
    folder = save_artifacts(artifacts, tmp_path)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    meta["versions"]["scikit-learn"] = "1.1.3"
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ArtifactVersionError, match="1.1.3"):
        load_artifacts(folder)


def test_cli_trains_and_saves(tmp_path, capsys):
    exit_code = train_clusters.main(["--as-of", "2026-06-17", "--models-dir", str(tmp_path)],
                                    connect_fn=bronze)
    assert exit_code == 0
    assert (tmp_path / "clusters_20260617" / "metadata.json").exists()
    assert "clusters_20260617" in capsys.readouterr().out
