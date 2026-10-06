"""Held-out evaluation: metrics, the no-leakage guard, baselines, the report and the CLI."""
import json

import numpy as np
import pandas as pd
import pytest

import evaluate_clusters
import fraud_agent as fa
from factories import AS_OF, bronze_connection, load_bronze, raw_users, transactions
from src.features.transaction_behavior import profile_window
from src.models.evaluation import (evaluate, evaluate_window, flag_metrics, precision_at_k,
                                   ranking_metrics, to_markdown)
from src.models.training import train

EVAL_DATES = [AS_OF + pd.Timedelta(days=30), AS_OF + pd.Timedelta(days=60)]
FRAUD = {EVAL_DATES[0]: ["CLI-0003", "CLI-0011"], EVAL_DATES[1]: ["CLI-0020", "CLI-0031"]}


def with_labels(tx: pd.DataFrame, rng) -> pd.DataFrame:
    return tx.assign(is_fraud=False, fraud_score=rng.uniform(0, 100, len(tx)).round(2))


def bronze(database: str = ":memory:"):
    """History before AS_OF, then two 30-day windows where a few customers' spending jumps and
    those transactions are labelled as fraud. fraud_score is noise, unrelated to the label."""
    rng = np.random.default_rng(0)
    users = raw_users(45)
    ids = sorted(users["customer_id"].unique())
    typical = {cid: 20.0 + 3.0 * i for i, cid in enumerate(ids)}
    history = profile_window(AS_OF)
    parts = [with_labels(transactions(ids, history.start, history.days, per_day=0.3,
                                      amount=typical, seed=1), rng)]
    for n, end in enumerate(EVAL_DATES, start=2):
        start = end - pd.Timedelta(days=29)
        window = with_labels(transactions(ids, start, 30, per_day=0.3, amount=typical, seed=n), rng)
        window["transaction_id"] = window["transaction_id"] + f"-w{n}"
        hit = window["customer_id"].isin(FRAUD[end])
        window.loc[hit, ["amount", "amount_usd"]] = 9_000.0
        window.loc[hit, "is_fraud"] = True
        parts.append(window)
    con = bronze_connection(database)
    load_bronze(con, users, pd.concat(parts, ignore_index=True))
    return con


@pytest.fixture(scope="module")
def con():
    con = bronze()
    yield con
    con.close()


@pytest.fixture(scope="module")
def artifacts(con):
    return train(con, AS_OF, n_components=4, n_clusters=3)


# --- metrics -------------------------------------------------------------------------
def test_flag_metrics():
    y = pd.Series([True, True, False, False, False])
    flagged = pd.Series([True, False, True, False, False])
    m = flag_metrics(y, flagged)
    assert (m["flagged"], m["true_positives"], m["precision"], m["recall"]) == (2, 1, 0.5, 0.5)
    assert m["lift"] == pytest.approx(0.5 / 0.4, abs=0.01)


def test_weighted_flag_metrics_estimate_the_population():
    # One sampled negative stands for 10: precision drops accordingly
    y = pd.Series([True, False])
    flagged = pd.Series([True, True])
    m = flag_metrics(y, flagged, weight=pd.Series([1.0, 10.0]))
    assert m["precision"] == pytest.approx(1 / 11, abs=1e-4)


def test_precision_at_k_and_single_class():
    y = pd.Series([False, True, False, True])
    assert precision_at_k(y, pd.Series([0.1, 0.9, 0.2, 0.8]), 2) == 1.0
    assert precision_at_k(y, pd.Series([0.1, 0.9, 0.2, 0.8]), 0) is None
    assert ranking_metrics(pd.Series([False, False]), pd.Series([0.1, 0.2])) == \
        {"average_precision": None, "roc_auc": None}


# --- protocol ----------------------------------------------------------------------------
def test_a_window_inside_the_training_data_is_refused(con, artifacts):
    with pytest.raises(ValueError, match="fuga de datos"):
        evaluate_window(con, artifacts, AS_OF)


def test_model_beats_random_on_held_out_windows(con, artifacts):
    report = evaluate(con, artifacts, EVAL_DATES)
    for end, window in zip(EVAL_DATES, report["windows"].values()):
        assert window["positives"] == len(FRAUD[end])
        ranking = window["ranking"]
        assert ranking["cluster_model"]["average_precision"] > ranking["random"]["average_precision"]
        assert ranking["cluster_model"]["average_precision"] > ranking["fraud_score"]["average_precision"]
        top = window["cluster_thresholds"]["4"]
        assert top["recall"] == 1.0 and top["true_positives"] == len(FRAUD[end])
    assert report["pooled"]["positives"] == sum(len(v) for v in FRAUD.values())


def test_label_comes_only_from_the_window(con, artifacts):
    window = evaluate_window(con, artifacts, EVAL_DATES[0])
    flagged = set(window.data.loc[window.data["has_fraud"], "customer_id"])
    assert flagged == set(FRAUD[EVAL_DATES[0]])          # the other window's fraud is not here


def test_rules_baseline_on_a_weighted_sample(con, artifacts, monkeypatch):
    monkeypatch.setattr(fa, "con", None)
    fa.init(con)
    window = evaluate_window(con, artifacts, EVAL_DATES[0], rules_negatives=10)
    rules = window.report["rules"]
    assert rules["sample_customers"] == 10 + len(FRAUD[EVAL_DATES[0]])
    assert set(rules["thresholds"]) == {"30", "60"}
    assert rules["thresholds"]["30"]["recall"] == 1.0     # big jumps break the amount rule


def test_markdown_report_has_every_section(con, artifacts):
    text = to_markdown(evaluate(con, artifacts, EVAL_DATES))
    assert text.count("## Ventana") == 2
    assert "Modelo de clústeres" in text and "fraud_score del banco" in text and "Aleatorio" in text
    assert "## Todas las ventanas juntas" in text


def test_cli_writes_json_and_markdown(tmp_path, capsys):
    database = str(tmp_path / "latam_bank.duckdb")
    bronze(database).close()
    import duckdb
    code = evaluate_clusters.main(
        ["--train-as-of", str(AS_OF.date()), "--eval-as-of", *[str(d.date()) for d in EVAL_DATES],
         "--out", str(tmp_path / "reports")],
        connect_fn=lambda: duckdb.connect(database))
    assert code == 0
    report = json.loads((tmp_path / "reports" / "evaluation_20260617.json").read_text(encoding="utf-8"))
    assert list(report["windows"]) == [str(d.date()) for d in EVAL_DATES]
    assert (tmp_path / "reports" / "evaluation_20260617.md").exists()
