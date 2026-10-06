"""Cluster profiles and deviation scoring (03_clustering sections 19-26)."""
import numpy as np
import pandas as pd
import pytest

from src.models.behavior_scoring import (build_cluster_profiles, robust_deviation_score,
                                         score_behavior_against_cluster, score_customers,
                                         summarize_customer)


@pytest.mark.parametrize("value, mad, p25, p75, expected", [
    (13.0, 2.0, 0.0, 0.0, 3 / (1.4826 * 2)),  # MAD
    (14.0, 0.0, 9.0, 11.0, 4 / (2 / 1.349)),  # IQR when the MAD is 0
    (10.0, 0.0, 10.0, 10.0, 0.0),             # no spread, same value
    (11.0, 0.0, 10.0, 10.0, np.inf),          # no spread, different value
])
def test_robust_deviation_score(value, mad, p25, p75, expected):
    assert robust_deviation_score(value, 10.0, mad, p25, p75) == pytest.approx(expected)


def test_missing_value_has_no_score():
    assert np.isnan(robust_deviation_score(np.nan, 10.0, 1.0, 9.0, 11.0))


@pytest.fixture
def profiles():
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame({"customer_id": [f"C{i}" for i in range(40)],
                             "monto_promedio_usd": rng.normal(50, 5, 40),
                             "tasa_aprobacion": rng.uniform(0.95, 1.0, 40)})
    clusters = pd.DataFrame({"customer_id": baseline["customer_id"], "cluster": [0, 1] * 20})
    return build_cluster_profiles(baseline, clusters, ["monto_promedio_usd", "tasa_aprobacion"])


def test_profiles_have_one_row_per_cluster_and_feature(profiles):
    assert len(profiles) == 4
    row = profiles.query("cluster == 0 and feature == 'monto_promedio_usd'").iloc[0]
    assert 40 < row["median"] < 60 and row["mad"] > 0 and row["p25"] <= row["p75"]


def test_only_customers_with_baseline_activity_count():
    baseline = pd.DataFrame({"customer_id": ["A", "B"], "x": [1.0, 3.0]})
    clusters = pd.DataFrame({"customer_id": ["A", "B", "SIN-ACTIVIDAD"], "cluster": [0, 0, 0]})
    assert build_cluster_profiles(baseline, clusters, ["x"]).loc[0, "count"] == 2


def test_big_deviation_is_suspicious_and_normal_is_not(profiles):
    features = ["monto_promedio_usd", "tasa_aprobacion"]
    normal = score_behavior_against_cluster(
        pd.Series({"monto_promedio_usd": 51.0, "tasa_aprobacion": 0.98}), 0, profiles, features)
    unusual = score_behavior_against_cluster(
        pd.Series({"monto_promedio_usd": 7000.0, "tasa_aprobacion": 0.98}), 0, profiles, features)
    assert not normal["is_suspicious"].any()
    assert unusual.set_index("feature").loc["monto_promedio_usd", "is_suspicious"]
    assert summarize_customer(unusual)["suspicious_features"] == ["monto_promedio_usd"]


def test_summary_ignores_infinite_scores_in_the_mean():
    scores = pd.DataFrame({"feature": ["a", "b"], "deviation_score": [1.0, np.inf],
                           "is_suspicious": [False, True]})
    summary = summarize_customer(scores)
    assert summary["anomaly_score"] == 1.0
    assert summary["n_suspicious"] == 1


def test_score_customers_skips_customers_without_cluster_or_activity(profiles):
    recent = pd.DataFrame({"customer_id": ["C0", "SIN-CLUSTER"],
                           "monto_promedio_usd": [7000.0, 50.0], "tasa_aprobacion": [1.0, 1.0]})
    clusters = pd.DataFrame({"customer_id": ["C0", "SIN-ACTIVIDAD"], "cluster": [0, 1]})
    detail, summary = score_customers(recent, clusters, profiles,
                                      ["monto_promedio_usd", "tasa_aprobacion"])
    assert list(summary["customer_id"]) == ["C0"]
    assert set(detail["customer_id"]) == {"C0"}
    assert summary.loc[0, "n_suspicious"] == 1
