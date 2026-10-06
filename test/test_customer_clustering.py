"""Customer clustering: fit, persistence and assignment (03_clustering sections 2-10, 21-23)."""
import joblib
import pandas as pd
import pytest

from factories import AS_OF, COUNTRY_CURRENCY, USD_RATES, raw_users
from src.features.customer_features import build_customer_features
from src.models.customer_clustering import (assign_clusters, fit_customer_clustering,
                                            load_bundle)


@pytest.fixture(scope="module")
def features():
    return build_customer_features(raw_users(90), AS_OF, COUNTRY_CURRENCY, USD_RATES)


@pytest.fixture(scope="module")
def bundle(features):
    return fit_customer_clustering(features, n_components=4, n_clusters=3)


def test_every_customer_gets_one_valid_cluster(bundle, features):
    clusters = assign_clusters(bundle, features)
    assert list(clusters["customer_id"]) == list(features["customer_id"])
    assert set(clusters["cluster"]) <= {0, 1, 2}


def test_fit_is_deterministic(features, bundle):
    again = fit_customer_clustering(features, n_components=4, n_clusters=3)
    pd.testing.assert_frame_equal(assign_clusters(bundle, features),
                                  assign_clusters(again, features))


def test_saved_bundle_predicts_the_same(tmp_path, bundle, features):
    path = tmp_path / "customer_cluster_model.joblib"
    joblib.dump(bundle, path)
    pd.testing.assert_frame_equal(assign_clusters(load_bundle(path), features),
                                  assign_clusters(bundle, features))


def test_single_customer_and_unseen_category(bundle, features):
    one = features.head(1).copy()
    one["education_level"] = "Doctorado"                        # not seen in training
    assert len(assign_clusters(bundle, one)) == 1


def test_missing_input_column_raises(bundle, features):
    with pytest.raises(KeyError, match="income_usd"):
        assign_clusters(bundle, features.drop(columns=["income_usd"]))


def test_incomplete_bundle_is_rejected(tmp_path):
    path = tmp_path / "broken.joblib"
    joblib.dump({"pca": None}, path)
    with pytest.raises(KeyError):
        load_bundle(path)


def test_empty_input_returns_empty_assignment(bundle, features):
    assert assign_clusters(bundle, features.head(0)).empty
