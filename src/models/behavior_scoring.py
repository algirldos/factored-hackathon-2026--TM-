"""
Behavior profiles per cluster and deviation scoring
(notebooks/03_clustering_refactored.ipynb, sections 19-20 and 22-26).

A metric is suspicious when it is at least SUSPICIOUS_DEVIATION robust standard
deviations away from the median of the customer's cluster.
"""
import numpy as np
import pandas as pd

from src.contracts import BEHAVIOR_SCORES, CLUSTER_PROFILES, CUSTOMER_SCORES, validate

SUSPICIOUS_DEVIATION = 3.0
MAD_TO_STD = 1.4826   # MAD -> standard deviation for normal data
IQR_TO_STD = 1.349    # IQR -> standard deviation for normal data


def build_cluster_profiles(baseline_behavior: pd.DataFrame, clusters: pd.DataFrame,
                           features: list[str]) -> pd.DataFrame:
    """
    Robust statistics (median, MAD, percentiles) of each feature per cluster.
    Only customers with activity in the baseline window count.
    """
    data = clusters.merge(baseline_behavior[["customer_id", *features]], on="customer_id",
                          how="inner")
    rows = []
    for cluster_id, group in data.groupby("cluster"):
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if values.empty:
                continue
            median = values.median()
            rows.append({"cluster": cluster_id, "feature": feature, "count": len(values),
                         "mean": values.mean(), "median": median,
                         "p25": values.quantile(0.25), "p75": values.quantile(0.75),
                         "p95": values.quantile(0.95),
                         "mad": float(np.median(np.abs(values - median)))})
    profiles = pd.DataFrame(rows, columns=["cluster", "feature", "count", "mean", "median",
                                           "p25", "p75", "p95", "mad"])
    validate(profiles, CLUSTER_PROFILES)
    return profiles


def robust_deviation_score(value: float, median: float, mad: float, p25: float,
                           p75: float) -> float:
    """
    |value - median| in robust standard deviations: MAD first, IQR if the MAD is 0.
    If both are 0, any difference from the median is infinitely unusual.
    """
    if pd.isna(value):
        return np.nan
    if mad > 0:
        return abs(value - median) / (MAD_TO_STD * mad)
    if p75 - p25 > 0:
        return abs(value - median) / ((p75 - p25) / IQR_TO_STD)
    return 0.0 if value == median else np.inf


def score_behavior_against_cluster(behavior: pd.Series, cluster_id: int,
                                   profiles: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """One row per feature of ONE customer: value, cluster reference, deviation and suspicion."""
    cluster_profile = profiles[profiles["cluster"] == cluster_id].set_index("feature")
    rows = []
    for feature in features:
        if feature not in behavior.index or feature not in cluster_profile.index:
            continue
        ref = cluster_profile.loc[feature]
        value = pd.to_numeric(pd.Series([behavior[feature]]), errors="coerce").iloc[0]
        rows.append({"feature": feature, "value": value, "cluster_median": ref["median"],
                     "cluster_p25": ref["p25"], "cluster_p75": ref["p75"],
                     "deviation_score": robust_deviation_score(value, ref["median"], ref["mad"],
                                                               ref["p25"], ref["p75"])})
    result = pd.DataFrame(rows, columns=["feature", "value", "cluster_median", "cluster_p25",
                                         "cluster_p75", "deviation_score"])
    result["is_suspicious"] = result["deviation_score"] >= SUSPICIOUS_DEVIATION
    return result


def summarize_customer(scores: pd.DataFrame) -> dict:
    """Customer-level summary: mean finite deviation and the suspicious metrics."""
    finite = scores["deviation_score"].replace([np.inf, -np.inf], np.nan).dropna()
    suspicious = scores[scores["is_suspicious"]]
    return {
        "anomaly_score": float(finite.mean()) if not finite.empty else np.nan,
        "n_features": int(len(scores)),
        "n_suspicious": int(len(suspicious)),
        "suspicious_features": suspicious["feature"].tolist(),
    }


def score_customers(recent_behavior: pd.DataFrame, clusters: pd.DataFrame,
                    profiles: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score every customer that has a cluster and recent activity.
    Returns (detail: one row per customer and feature, summary: one row per customer).
    """
    validate(profiles, CLUSTER_PROFILES)   # profiles may come from a saved file
    data = clusters.merge(recent_behavior, on="customer_id", how="inner")
    details, summaries = [], []
    for _, row in data.iterrows():
        scores = score_behavior_against_cluster(row, int(row["cluster"]), profiles, features)
        if scores.empty:
            continue
        details.append(scores.assign(customer_id=row["customer_id"], cluster=int(row["cluster"])))
        summaries.append({"customer_id": row["customer_id"], "cluster": int(row["cluster"]),
                          **summarize_customer(scores)})
    detail = (pd.concat(details, ignore_index=True) if details
              else pd.DataFrame(columns=["customer_id", "cluster", "feature", "value",
                                         "cluster_median", "cluster_p25", "cluster_p75",
                                         "deviation_score", "is_suspicious"]))
    summary = pd.DataFrame(summaries, columns=["customer_id", "cluster", "anomaly_score",
                                               "n_features", "n_suspicious",
                                               "suspicious_features"])
    validate(detail, BEHAVIOR_SCORES)
    validate(summary, CUSTOMER_SCORES)
    return detail, summary
