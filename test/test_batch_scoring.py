"""Batch scoring: what is written to fraud_ops, idempotency, the queue and the CLI."""
import duckdb
import pandas as pd
import pytest

import score_customers
from factories import AS_OF, bronze_connection, load_bronze, raw_users, transactions
from src.features.transaction_behavior import profile_window
from src.models.batch_scoring import (flagged_customers, run_summary, score_as_of,
                                      write_results)
from src.models.behavior_scoring import SUSPICIOUS_FEATURES
from src.models.training import save_artifacts, train

SCORE_AS_OF = AS_OF + pd.Timedelta(days=30)        # the 30 days right after training
CHANGED = "CLI-0007"                                # spends much more in the scored window


def bronze(database: str = ":memory:"):
    users = raw_users(45)
    ids = sorted(users["customer_id"].unique())
    typical = {cid: 20.0 + 3.0 * i for i, cid in enumerate(ids)}
    history = profile_window(AS_OF)
    recent_start = AS_OF + pd.Timedelta(days=1)
    recent = transactions(ids, recent_start, 30, per_day=0.3, amount=typical, seed=2)
    recent.loc[recent["customer_id"] == CHANGED, ["amount", "amount_usd"]] = 9_000.0
    tx = pd.concat([transactions(ids, history.start, history.days, per_day=0.3, amount=typical,
                                 seed=1), recent])
    con = bronze_connection(database)
    load_bronze(con, users, tx)
    return con


@pytest.fixture(scope="module")
def artifacts():
    return train(bronze(), AS_OF, n_components=4, n_clusters=3)


@pytest.fixture
def con():
    con = bronze()
    yield con
    con.close()


def count(con, table, where="TRUE"):
    return con.execute(f"SELECT COUNT(*) FROM fraud_ops.{table} WHERE {where}").fetchone()[0]


def test_only_the_chosen_metrics_are_scored(con, artifacts):
    result = score_as_of(con, artifacts, SCORE_AS_OF)
    assert set(result.detail["feature"]) == set(SUSPICIOUS_FEATURES)
    assert (result.summary["n_features"] == len(SUSPICIOUS_FEATURES)).all()


def test_the_customer_that_changed_is_flagged(con, artifacts):
    result = score_as_of(con, artifacts, SCORE_AS_OF)
    flagged = flagged_customers(result, min_suspicious=4)
    assert CHANGED in set(flagged["customer_id"])
    assert len(flagged) <= 3                                   # the rest behave as usual
    assert flagged["anomaly_score"].between(0, 1).all()


def test_scoring_reads_only(con, artifacts):
    score_as_of(con, artifacts, SCORE_AS_OF)
    schemas = con.execute("SELECT schema_name FROM information_schema.schemata").df()
    assert "fraud_ops" not in set(schemas["schema_name"])


def test_write_creates_every_table_with_consistent_counts(con, artifacts):
    result = score_as_of(con, artifacts, SCORE_AS_OF)
    run = write_results(con, result, min_suspicious=4)

    assert count(con, "customer_cluster") == len(result.clusters)
    assert count(con, "customer_anomaly") == len(result.summary)
    assert count(con, "behavior_anomalies") == len(result.summary) * len(SUSPICIOUS_FEATURES)
    assert count(con, "anomaly_queue") == run["n_enqueued"] == run["n_flagged"]
    queued = con.execute("SELECT customer_id, model_name, processed_at FROM fraud_ops.anomaly_queue").df()
    assert CHANGED in set(queued["customer_id"])
    assert queued["processed_at"].isna().all()                # pending for fraud_flow
    assert set(queued["model_name"]) == {artifacts.metadata["model_version"]}
    assert count(con, "scoring_runs", "status = 'ok'") == 1


def test_repeating_a_run_replaces_rows_and_never_requeues(con, artifacts):
    result = score_as_of(con, artifacts, SCORE_AS_OF)
    write_results(con, result, min_suspicious=4)
    con.execute("UPDATE fraud_ops.anomaly_queue SET processed_at = now(), result = 'case_opened'")
    again = write_results(con, result, min_suspicious=4)

    assert count(con, "customer_anomaly") == len(result.summary)
    assert count(con, "anomaly_queue") == again["n_flagged"]   # no second row per customer
    assert again["n_enqueued"] == 0
    assert count(con, "scoring_runs") == 2


def test_lower_threshold_enqueues_more(con, artifacts):
    result = score_as_of(con, artifacts, SCORE_AS_OF)
    assert len(flagged_customers(result, 1)) >= len(flagged_customers(result, 4))


def test_missing_profile_for_a_metric_is_an_error(con, artifacts):
    with pytest.raises(KeyError, match="no tiene perfil"):
        score_as_of(con, artifacts, SCORE_AS_OF, features=["metrica_inexistente"])


def test_summary_counts(con, artifacts):
    result = score_as_of(con, artifacts, SCORE_AS_OF)
    summary = run_summary(result, 4)
    assert sum(summary["n_suspicious_distribution"].values()) == summary["n_scored"]


def test_cli_dry_run_writes_nothing_and_uses_a_read_only_connection(tmp_path, artifacts, capsys):
    folder = save_artifacts(artifacts, tmp_path)
    opened = []

    def connect(read_only):
        opened.append(read_only)
        return bronze()

    assert score_customers.main(["--as-of", str(SCORE_AS_OF.date()), "--model-dir", str(folder),
                                 "--dry-run"], connect_fn=connect) == 0
    assert opened == [True]
    assert "Simulación" in capsys.readouterr().out


def test_cli_writes_with_a_write_connection(tmp_path, artifacts, capsys):
    folder = save_artifacts(artifacts, tmp_path)
    database = str(tmp_path / "latam_bank.duckdb")
    bronze(database).close()
    opened = []

    def connect(read_only):
        opened.append(read_only)
        return duckdb.connect(database)

    assert score_customers.main(["--as-of", str(SCORE_AS_OF.date()), "--model-dir", str(folder)],
                                connect_fn=connect) == 0
    assert opened == [False]
    with duckdb.connect(database) as con:                      # reopen: the CLI closed it
        assert count(con, "anomaly_queue") >= 1
    assert "fraud_flow.py --process-queue" in capsys.readouterr().out


def test_latest_model_dir_picks_the_newest(tmp_path):
    for name in ["clusters_20260101", "clusters_20260617", "otra"]:
        (tmp_path / name).mkdir()
    assert score_customers.latest_model_dir(str(tmp_path)).name == "clusters_20260617"
