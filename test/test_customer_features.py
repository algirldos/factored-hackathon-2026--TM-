"""Customer features (01_eda sections 1 and 2)."""
import pandas as pd
import pytest

from factories import AS_OF, COUNTRY_CURRENCY, USD_RATES, raw_users
from src.features.customer_features import (DEBT_PRODUCTS, FEATURE_COLUMNS, PRODUCT_COLUMNS,
                                            age_at, build_customer_features)


def build(users, as_of=AS_OF):
    return build_customer_features(users, as_of, COUNTRY_CURRENCY, USD_RATES)


@pytest.mark.parametrize("dob, as_of, age", [
    ("1990-06-17", "2026-06-17", 36),   # birthday today
    ("1990-06-18", "2026-06-17", 35),   # birthday tomorrow
    ("1990-12-31", "2026-01-01", 35),
])
def test_age_counts_the_birthday(dob, as_of, age):
    assert age_at(pd.Series([dob]), as_of).iloc[0] == age


def test_output_has_the_model_columns_and_one_row_per_customer():
    features = build(raw_users())
    assert list(features.columns) == FEATURE_COLUMNS
    assert features["customer_id"].is_unique


def test_result_depends_on_as_of_not_on_today():
    users = raw_users()
    pd.testing.assert_frame_equal(build(users), build(users))
    older = build(users, as_of=AS_OF + pd.DateOffset(years=1))
    assert (older["Edad"] - build(users)["Edad"]).eq(1).all()


def test_every_product_column_exists_for_a_single_customer():
    users = raw_users()
    one = users[users["customer_id"] == "CLI-0000"]          # only has "Cuenta Ahorro"
    features = build(one)
    assert set(PRODUCT_COLUMNS) <= set(features.columns)
    assert features.loc[0, "Tarjeta Crédito"] == 0
    assert features.loc[0, "num_products"] == 1


def test_amounts_are_converted_to_usd_and_aggregated():
    users = raw_users()
    cid = "CLI-0003"                                          # Colombia, 4 products
    rows = users[users["customer_id"] == cid]
    features = build(users).set_index("customer_id").loc[cid]

    assert features["income_usd"] == pytest.approx(rows["income"].iloc[0] * USD_RATES["COP"])
    debt = rows[rows["product_type"].isin(DEBT_PRODUCTS)]["current_balance"].sum()
    assert features["Deuda_usd"] == pytest.approx(debt * USD_RATES["COP"])


def test_ineligible_customers_are_excluded():
    users = raw_users(n_customers=6)
    users.loc[users["customer_id"] == "CLI-0000", "customer_status"] = "Closed"
    users.loc[users["customer_id"] == "CLI-0001", "credit_score"] = None
    users.loc[users["customer_id"] == "CLI-0002", "current_balance"] = 0.0     # no balance
    users.loc[users["customer_id"] == "CLI-0003", "country"] = "Atlantis"     # no currency
    users = pd.concat([users, pd.DataFrame([{**users.iloc[0].to_dict(), "customer_id": "CLI-NOPROD",
                                             "customer_status": "Active", "product_type": None,
                                             "currency": None, "current_balance": None}])])

    kept = set(build(users)["customer_id"])
    assert kept == {"CLI-0004", "CLI-0005"}


def test_missing_raw_column_raises():
    with pytest.raises(KeyError, match="income"):
        build(raw_users().drop(columns=["income"]))
