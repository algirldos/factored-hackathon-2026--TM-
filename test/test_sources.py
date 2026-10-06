"""
SQL readers against an in-memory DuckDB with the bronze schema, and the full flow
from the database to the suspicious-metric flags.
"""
import duckdb
import pandas as pd
import pytest

from factories import AS_OF, COUNTRY_CURRENCY, raw_users, transactions
from src.features.customer_features import build_customer_features
from src.features.sources import load_customers_with_products, load_transactions, load_usd_rates
from src.features.transaction_behavior import (behavior_windows, profile_feature_names,
                                               window_behavior)
from src.models.behavior_scoring import build_cluster_profiles, score_customers
from src.models.customer_clustering import assign_clusters, fit_customer_clustering


@pytest.fixture
def con():
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA bronze")
    con.execute("""CREATE TABLE bronze.daily_exchange_rates (date DATE, source_currency VARCHAR,
                   target_currency VARCHAR, exchange_rate DOUBLE)""")
    con.execute("""INSERT INTO bronze.daily_exchange_rates VALUES
                   ('2026-06-16', 'COP', 'USD', 0.00020), ('2026-06-17', 'COP', 'USD', 0.00025),
                   ('2026-06-17', 'USD', 'MXN', 20.0),    ('2026-06-17', 'ARS', 'USD', 0.001),
                   ('2026-06-18', 'COP', 'USD', 0.00099)""")
    yield con
    con.close()


def load_bronze(con, users: pd.DataFrame, tx: pd.DataFrame) -> None:
    customers = (users.drop(columns=["product_type", "currency", "current_balance"])
                      .drop_duplicates("customer_id")
                      .rename(columns={"income": "estimated_monthly_income"})
                      .assign(last_updated=pd.Timestamp("2026-06-01")))
    products = users[["customer_id", "product_type", "currency", "current_balance"]].copy()
    products["product_id"] = [f"PRD-{i}" for i in range(len(products))]
    products["last_updated"] = pd.Timestamp("2026-06-01")
    con.register("customers_df", customers)
    con.register("products_df", products)
    con.register("tx_df", tx)
    con.execute("CREATE TABLE bronze.customers AS SELECT * FROM customers_df")
    con.execute("CREATE TABLE bronze.products AS SELECT * FROM products_df")
    con.execute("CREATE TABLE bronze.transactions AS SELECT * FROM tx_df")


def test_rates_use_the_latest_day_on_or_before_as_of(con):
    rates = load_usd_rates(con, AS_OF)
    assert rates["COP"] == 0.00025                     # not the 2026-06-18 rate
    assert rates["MXN"] == pytest.approx(0.05)


def test_missing_rates_raise(con):
    with pytest.raises(LookupError):
        load_usd_rates(con, "2020-01-01")


def test_duplicates_are_removed_in_sql(con):
    users = raw_users(6)
    _, recent = behavior_windows(AS_OF)
    tx = transactions(["CLI-0001"], recent.start, recent.days, per_day=0.5)
    load_bronze(con, users, tx)
    con.execute("INSERT INTO bronze.customers SELECT * FROM bronze.customers")   # duplicates
    con.execute("INSERT INTO bronze.transactions SELECT * FROM bronze.transactions")

    rows = load_customers_with_products(con)
    assert len(rows) == len(users)
    assert len(load_transactions(con, recent.start, recent.end)) == len(tx)


def test_transactions_are_filtered_by_window_and_customer(con):
    users = raw_users(6)
    baseline, recent = behavior_windows(AS_OF)
    tx = pd.concat([transactions(["CLI-0001", "CLI-0002"], baseline.start, baseline.days, seed=1),
                    transactions(["CLI-0001", "CLI-0002"], recent.start, recent.days, seed=2)])
    load_bronze(con, users, tx)

    window = load_transactions(con, recent.start, recent.end, customer_ids=["CLI-0001"])
    assert set(window["customer_id"]) == {"CLI-0001"}
    assert window["transaction_date"].min() >= recent.start
    assert load_transactions(con, recent.start, recent.end, customer_ids=[]).empty


def test_end_to_end_flags_only_the_customer_that_changed(con):
    """Database -> features -> clusters -> profiles -> recent behavior -> suspicious metrics."""
    users = raw_users(60)
    ids = sorted(users["customer_id"].unique())
    # Each customer keeps a stable spending level, different across customers
    typical = {cid: 20.0 + 5.0 * i for i, cid in enumerate(ids)}
    baseline, recent = behavior_windows(AS_OF)
    recent_tx = transactions(ids, recent.start, recent.days, amount=typical, seed=2)
    changed = recent_tx["customer_id"] == "CLI-0007"
    recent_tx.loc[changed, ["amount", "amount_usd"]] = 9_000.0       # sudden big spending
    load_bronze(con, users, pd.concat([
        transactions(ids, baseline.start, baseline.days, amount=typical, seed=1), recent_tx]))

    features = build_customer_features(load_customers_with_products(con), AS_OF,
                                       COUNTRY_CURRENCY, load_usd_rates(con, AS_OF))
    clusters = assign_clusters(fit_customer_clustering(features, n_components=4, n_clusters=2),
                               features)
    baseline_behavior = window_behavior(load_transactions(con, baseline.start, baseline.end),
                                        baseline)
    profile_features = profile_feature_names(baseline_behavior.columns)
    profiles = build_cluster_profiles(baseline_behavior, clusters, profile_features)
    recent_behavior = window_behavior(load_transactions(con, recent.start, recent.end), recent)

    detail, summary = score_customers(recent_behavior, clusters, profiles, profile_features)

    flagged = summary.set_index("customer_id")
    assert "monto_promedio_usd" in flagged.loc["CLI-0007", "suspicious_features"]
    others = flagged.drop(index="CLI-0007")
    assert not others["suspicious_features"].map(lambda f: "monto_promedio_usd" in f).any()
