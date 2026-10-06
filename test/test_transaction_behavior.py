"""Transaction behavior windows and metrics (03_clustering sections 12-18)."""
import pandas as pd
import pytest

from factories import AS_OF, transactions
from src.features.transaction_behavior import (MONTHLY_RATE_FEATURES, behavior_windows,
                                               build_transaction_behavior,
                                               clean_transaction_amounts,
                                               normalize_behavior_period,
                                               profile_feature_names, select_window,
                                               window_behavior)


def test_windows_end_on_as_of_and_do_not_overlap():
    baseline, recent = behavior_windows(AS_OF)
    assert recent.end == pd.Timestamp("2026-06-18")           # as_of is included
    assert recent.days == 30
    assert baseline.end == recent.start
    assert baseline.start == recent.start - pd.DateOffset(months=12)
    assert recent.start.time() == baseline.start.time() == pd.Timestamp("00:00").time()


def test_select_window_is_half_open():
    _, recent = behavior_windows(AS_OF)
    tx = pd.DataFrame({"transaction_date": [recent.start - pd.Timedelta(seconds=1), recent.start,
                                            recent.end - pd.Timedelta(seconds=1), recent.end]})
    assert len(select_window(tx, recent)) == 2


def test_clean_drops_duplicate_ids_and_fills_usd():
    tx = pd.DataFrame({"transaction_id": ["A", "A", "B", "C"], "amount": [10.0, 10.0, 20.0, 30.0],
                       "currency": ["USD", "USD", "USD", "COP"],
                       "amount_usd": [None, None, None, None]})
    clean = clean_transaction_amounts(tx).set_index("transaction_id")
    assert list(clean.index) == ["A", "B", "C"]
    assert clean.loc["B", "amount_usd"] == 20.0
    assert pd.isna(clean.loc["C", "amount_usd"])              # non-USD stays missing


def test_behavior_has_one_row_per_customer_without_the_fraud_label():
    _, recent = behavior_windows(AS_OF)
    tx = transactions(["CLI-1", "CLI-2"], recent.start, recent.days, per_day=0.5)
    assert "is_fraud" not in tx.columns
    behavior = build_transaction_behavior(tx, recent)
    assert sorted(behavior["customer_id"]) == ["CLI-1", "CLI-2"]
    assert behavior.set_index("customer_id").loc["CLI-1", "num_transacciones"] == \
        (tx["customer_id"] == "CLI-1").sum()


def test_duplicates_do_not_inflate_amounts():
    _, recent = behavior_windows(AS_OF)
    tx = transactions(["CLI-1"], recent.start, recent.days, per_day=0.5)
    doubled = pd.concat([tx, tx])
    pd.testing.assert_frame_equal(build_transaction_behavior(tx, recent),
                                  build_transaction_behavior(doubled, recent))


def test_rates_are_normalized_to_30_days():
    behavior = pd.DataFrame({"customer_id": ["A"], "num_transacciones": [36]})
    assert normalize_behavior_period(behavior, 360).loc[0, "num_transacciones_monthly"] == \
        pytest.approx(3.0)


def test_profile_features_keep_a_fixed_order():
    columns = ["tasa_aprobacion", "num_transacciones_monthly", "monto_promedio_usd", "otra"]
    assert profile_feature_names(columns) == ["num_transacciones_monthly", "monto_promedio_usd",
                                              "tasa_aprobacion"]


def test_window_behavior_ignores_transactions_outside_the_window():
    baseline, recent = behavior_windows(AS_OF)
    tx = pd.concat([transactions(["CLI-1"], baseline.start, baseline.days, seed=1),
                    transactions(["CLI-2"], recent.start, recent.days, seed=2)])
    assert list(window_behavior(tx, recent)["customer_id"]) == ["CLI-2"]
    assert {f"{f}_monthly" for f in MONTHLY_RATE_FEATURES} <= set(window_behavior(tx, recent).columns)


def test_empty_window_returns_empty_behavior():
    _, recent = behavior_windows(AS_OF)
    assert build_transaction_behavior(transactions([], recent.start, 1), recent).empty
