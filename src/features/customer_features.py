"""
Customer features for clustering (notebooks/01_eda.ipynb, sections 1 and 2).

Input: one row per customer-product, as returned by sources.load_customers_with_products.
Output: one row per eligible customer with the columns the clustering model expects.

Changes from the notebook, so the result is reproducible and safe for production:
- Age is computed at an explicit `as_of` date, not "today".
- USD rates are passed in (daily rates of the scoring date), not read from a fixed file.
- Every product column always exists, even when a batch has no product of that type
  (the notebook's pivot only created the columns present in the data).
- Customers whose income cannot be converted to USD are dropped and counted, instead of
  reaching the model with NaN.
"""
import logging

import pandas as pd

from src.contracts import CUSTOMER_FEATURES, RAW_CUSTOMER_PRODUCTS, validate
from src.features.currency import normalize_key

log = logging.getLogger(__name__)

DEBT_PRODUCTS = ["Préstamo Hipotecario", "Préstamo Personal", "Tarjeta Crédito"]
ASSET_PRODUCTS = ["Cuenta Ahorro", "Cuenta Corriente", "Seguro", "Inversión", "Tarjeta Débito"]
PRODUCT_COLUMNS = ["Cuenta Ahorro", "Cuenta Corriente", "Inversión", "Préstamo Hipotecario",
                   "Préstamo Personal", "Seguro", "Tarjeta Crédito", "Tarjeta Débito"]

CUSTOMER_INFO_COLUMNS = [
    "customer_id", "date_of_birth", "country", "segment", "credit_score", "income_usd",
    "occupation", "education_level", "customer_status",
]
FEATURE_COLUMNS = [*CUSTOMER_INFO_COLUMNS, *PRODUCT_COLUMNS, "num_products", "Deuda_usd",
                   "Activos_financieros_usd", "Edad"]


def age_at(date_of_birth: pd.Series, as_of) -> pd.Series:
    """Age in whole years at `as_of`, counting whether the birthday already happened."""
    as_of = pd.Timestamp(as_of)
    dob = pd.to_datetime(date_of_birth, errors="coerce")
    before_birthday = (dob.dt.month > as_of.month) | (
        (dob.dt.month == as_of.month) & (dob.dt.day > as_of.day))
    return as_of.year - dob.dt.year - before_birthday.astype("Int64")


def filter_active_customers(users: pd.DataFrame) -> pd.DataFrame:
    """Section 1: active customers with credit score, income and occupation."""
    return (users.dropna(subset=["credit_score", "income", "occupation"])
                 .query("customer_status == 'Active'")
                 .copy())


def add_usd_amounts(users: pd.DataFrame, country_currency: dict[str, str],
                    usd_rates: dict[str, float]) -> pd.DataFrame:
    """Income is in the country's currency; product balances in the product's currency."""
    users = users.copy()
    income_currency = users["country"].map(lambda c: country_currency.get(normalize_key(c)))
    users["income_usd"] = users["income"] * income_currency.map(usd_rates)
    users["current_balance_usd"] = users["current_balance"] * users["currency"].map(usd_rates)
    return users


def build_customer_features(users: pd.DataFrame, as_of, country_currency: dict[str, str],
                            usd_rates: dict[str, float]) -> pd.DataFrame:
    """Sections 1 and 2 of 01_eda: one row per eligible customer."""
    users, _ = validate(users, RAW_CUSTOMER_PRODUCTS, mode="drop")
    active = add_usd_amounts(filter_active_customers(users), country_currency, usd_rates)

    no_income = active["income_usd"].isna()
    if no_income.any():
        log.warning("%d clientes sin conversión de ingreso a USD (país o moneda sin tasa); se excluyen",
                    active.loc[no_income, "customer_id"].nunique())
        active = active[~no_income]

    with_products = active.dropna(subset=["product_type", "current_balance_usd"])
    unknown = set(with_products["product_type"]) - set(PRODUCT_COLUMNS)
    if unknown:
        log.warning("Tipos de producto sin columna en el modelo, se ignoran: %s", sorted(unknown))

    balances = (with_products.pivot_table(index="customer_id", columns="product_type",
                                          values="current_balance_usd", aggfunc="sum",
                                          fill_value=0)
                             .reindex(columns=PRODUCT_COLUMNS, fill_value=0)
                             .reset_index())
    balances.columns.name = None

    info = active[CUSTOMER_INFO_COLUMNS].drop_duplicates("customer_id")
    features = info.merge(balances, on="customer_id", how="inner")
    features["num_products"] = features[PRODUCT_COLUMNS].gt(0).sum(axis=1)
    features["Deuda_usd"] = features[DEBT_PRODUCTS].sum(axis=1)
    features["Activos_financieros_usd"] = features[ASSET_PRODUCTS].sum(axis=1)
    features["Edad"] = age_at(features["date_of_birth"], as_of)

    # Section 2: customers whose products show no balance add nothing to the segmentation
    features = features[features["num_products"] > 0]
    features = features[FEATURE_COLUMNS].sort_values("customer_id").reset_index(drop=True)
    validate(features, CUSTOMER_FEATURES)
    return features
