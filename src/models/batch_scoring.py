"""
Batch scoring: compares every customer's last 30 days with the profile of their cluster and
writes the results to the fraud_ops schema of MotherDuck.

Tables written (all keyed by as_of + model_version, so a run can be repeated safely):
- customer_cluster     cluster of every eligible customer
- behavior_anomalies   one row per customer and scored metric
- customer_anomaly     one row per scored customer: suspicious metrics and a 0-1 score
- scoring_runs         one row per run, with its counts
- anomaly_queue        customers with at least `min_suspicious` suspicious metrics; the queue
                       that `fraud_flow.py --process-queue` turns into OTP-verified cases
"""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.features.currency import country_currency_map, static_usd_rates
from src.features.customer_features import build_customer_features
from src.features.sources import load_customers_with_products, load_transactions
from src.features.transaction_behavior import behavior_windows, window_behavior
from src.models.behavior_scoring import SUSPICIOUS_FEATURES, score_customers
from src.models.customer_clustering import assign_clusters
from src.models.training import TrainedArtifacts

log = logging.getLogger(__name__)

OPS_SCHEMA = "fraud_ops"
DEFAULT_MIN_SUSPICIOUS = 4   # provisional: ~0.7% of active customers on the real data


@dataclass
class ScoringResult:
    as_of: pd.Timestamp
    model_version: str
    clusters: pd.DataFrame
    detail: pd.DataFrame
    summary: pd.DataFrame
    features: list[str]
    started_at: datetime = field(default_factory=datetime.now)


def score_as_of(con, artifacts: TrainedArtifacts, as_of,
                features: list[str] = SUSPICIOUS_FEATURES) -> ScoringResult:
    """Score the 30 days that end on `as_of`. Reads only; nothing is written."""
    started_at = datetime.now()
    as_of = pd.Timestamp(as_of).normalize()
    trained_on = pd.Timestamp(artifacts.metadata["as_of"])
    if as_of <= trained_on:
        log.warning("Se puntúa %s con un modelo entrenado hasta %s: la ventana reciente está "
                    "dentro de los datos de entrenamiento", as_of.date(), trained_on.date())
    profiles = artifacts.profiles["profile_statistics"]
    missing = sorted(set(features) - set(profiles["feature"]))
    if missing:
        raise KeyError(f"El modelo {artifacts.metadata['model_version']} no tiene perfil para {missing}")

    customers = build_customer_features(load_customers_with_products(con), as_of,
                                        country_currency_map(), static_usd_rates())
    clusters = assign_clusters(artifacts.customer_model, customers)
    _, recent = behavior_windows(as_of)
    transactions = load_transactions(con, recent.start, recent.end,
                                     customer_ids=clusters["customer_id"].tolist())
    detail, summary = score_customers(window_behavior(transactions, recent), clusters, profiles,
                                      features)
    return ScoringResult(as_of, artifacts.metadata["model_version"], clusters, detail, summary,
                         list(features), started_at)


def queue_score(summary: pd.DataFrame) -> pd.Series:
    """Share of scored metrics that are suspicious: the 0-1 anomaly_score of the queue."""
    return summary["n_suspicious"] / summary["n_features"]


def flagged_customers(result: ScoringResult, min_suspicious: int) -> pd.DataFrame:
    flagged = result.summary[result.summary["n_suspicious"] >= min_suspicious]
    return flagged.assign(anomaly_score=queue_score(flagged))


# ---------------------------------------------------------------------------
# Writing to fraud_ops
# ---------------------------------------------------------------------------
def setup_tables(con, schema: str = OPS_SCHEMA) -> None:
    """Create the scoring tables (idempotent). anomaly_queue matches fraud_flow.setup_tables."""
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    con.execute(f"""CREATE TABLE IF NOT EXISTS {schema}.customer_cluster (
        as_of DATE NOT NULL, model_version VARCHAR NOT NULL, customer_id VARCHAR NOT NULL,
        cluster INTEGER NOT NULL)""")
    con.execute(f"""CREATE TABLE IF NOT EXISTS {schema}.behavior_anomalies (
        as_of DATE NOT NULL, model_version VARCHAR NOT NULL, customer_id VARCHAR NOT NULL,
        cluster INTEGER NOT NULL, feature VARCHAR NOT NULL, value DOUBLE,
        cluster_median DOUBLE, cluster_p25 DOUBLE, cluster_p75 DOUBLE,
        deviation_score DOUBLE, is_suspicious BOOLEAN NOT NULL)""")
    con.execute(f"""CREATE TABLE IF NOT EXISTS {schema}.customer_anomaly (
        as_of DATE NOT NULL, model_version VARCHAR NOT NULL, customer_id VARCHAR NOT NULL,
        cluster INTEGER NOT NULL, anomaly_score DOUBLE, n_features INTEGER NOT NULL,
        n_suspicious INTEGER NOT NULL, suspicious_features VARCHAR)""")
    con.execute(f"""CREATE TABLE IF NOT EXISTS {schema}.scoring_runs (
        run_id VARCHAR NOT NULL, as_of DATE NOT NULL, model_version VARCHAR NOT NULL,
        started_at TIMESTAMP NOT NULL, finished_at TIMESTAMP NOT NULL,
        n_customers INTEGER, n_scored INTEGER, n_flagged INTEGER, n_enqueued INTEGER,
        min_suspicious INTEGER, features VARCHAR, status VARCHAR NOT NULL)""")
    con.execute(f"""CREATE TABLE IF NOT EXISTS {schema}.anomaly_queue (
        customer_id VARCHAR NOT NULL, detected_at TIMESTAMP NOT NULL, model_name VARCHAR,
        anomaly_score DOUBLE, processed_at TIMESTAMP, case_id VARCHAR, result VARCHAR)""")


def _replace(con, schema: str, table: str, frame: pd.DataFrame, as_of, version: str) -> None:
    """Delete this run's previous rows and insert the new ones (same as_of + model_version)."""
    con.execute(f"DELETE FROM {schema}.{table} WHERE as_of = ? AND model_version = ?",
                [as_of.date(), version])
    if frame.empty:
        return
    name = f"_new_{table}"
    con.register(name, frame)
    try:
        columns = ", ".join(frame.columns)
        con.execute(f"INSERT INTO {schema}.{table} ({columns}) SELECT {columns} FROM {name}")
    finally:
        con.unregister(name)


def write_results(con, result: ScoringResult, min_suspicious: int = DEFAULT_MIN_SUSPICIOUS,
                  schema: str = OPS_SCHEMA) -> dict:
    """
    Write a scoring run in one transaction and enqueue the flagged customers.
    Repeating a run replaces its rows, and a customer is enqueued at most once per
    as_of + model_version, even if the queue row was already processed.
    """
    setup_tables(con, schema)
    as_of, version = result.as_of, result.model_version
    keys = {"as_of": as_of.date(), "model_version": version}
    clusters = result.clusters.assign(**keys)[["as_of", "model_version", "customer_id", "cluster"]]
    detail = result.detail.assign(**keys)
    summary = result.summary.assign(
        **keys, suspicious_features=result.summary["suspicious_features"].map(",".join))
    flagged = flagged_customers(result, min_suspicious)
    queue = pd.DataFrame({"customer_id": flagged["customer_id"],
                          "detected_at": as_of, "model_name": version,
                          "anomaly_score": flagged["anomaly_score"]})

    con.execute("BEGIN TRANSACTION")
    try:
        _replace(con, schema, "customer_cluster", clusters, as_of, version)
        _replace(con, schema, "behavior_anomalies", detail, as_of, version)
        _replace(con, schema, "customer_anomaly", summary, as_of, version)
        enqueued = 0
        if not queue.empty:
            con.register("_new_queue", queue)
            try:
                enqueued = con.execute(f"""
                    INSERT INTO {schema}.anomaly_queue
                        (customer_id, detected_at, model_name, anomaly_score)
                    SELECT q.customer_id, q.detected_at, q.model_name, q.anomaly_score
                    FROM _new_queue q
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {schema}.anomaly_queue a
                        WHERE a.customer_id = q.customer_id AND a.model_name = q.model_name
                          AND a.detected_at = q.detected_at)""").fetchone()[0]
            finally:
                con.unregister("_new_queue")
        run = {"run_id": uuid.uuid4().hex[:12], "as_of": as_of.date(), "model_version": version,
               "started_at": result.started_at, "finished_at": datetime.now(),
               "n_customers": len(result.clusters), "n_scored": len(result.summary),
               "n_flagged": len(flagged), "n_enqueued": int(enqueued or 0),
               "min_suspicious": min_suspicious, "features": ",".join(result.features),
               "status": "ok"}
        con.register("_new_run", pd.DataFrame([run]))
        try:
            con.execute(f"INSERT INTO {schema}.scoring_runs SELECT * FROM _new_run")
        finally:
            con.unregister("_new_run")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return run


def run_summary(result: ScoringResult, min_suspicious: int) -> dict:
    """Counts of a scoring run, for the console and dry runs."""
    counts = result.summary["n_suspicious"].value_counts().sort_index()
    return {"as_of": result.as_of.date().isoformat(), "model_version": result.model_version,
            "n_customers": len(result.clusters), "n_scored": len(result.summary),
            "n_suspicious_distribution": {str(k): int(v) for k, v in counts.items()},
            "min_suspicious": min_suspicious,
            "n_flagged": len(flagged_customers(result, min_suspicious)),
            "features": result.features}
