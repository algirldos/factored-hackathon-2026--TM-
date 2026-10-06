"""
Reads the inputs of the feature pipeline from MotherDuck (or any DuckDB connection).

The filters and deduplication run in SQL, so only the needed rows leave the database.
The bronze layer has ~2% duplicate rows: customers and products keep their latest
version (last_updated), transactions keep one row per transaction_id.
"""
import pandas as pd

from src.features.currency import usd_rates_from_rows

CUSTOMERS_TABLE = "bronze.customers"
PRODUCTS_TABLE = "bronze.products"
TRANSACTIONS_TABLE = "bronze.transactions"
RATES_TABLE = "bronze.daily_exchange_rates"

TRANSACTION_COLUMNS = [
    "transaction_id", "transaction_date", "product_id", "customer_id", "transaction_type",
    "amount", "currency", "amount_usd", "transaction_country", "transaction_city",
    "transaction_status",
]


def load_customers_with_products(con) -> pd.DataFrame:
    """One row per customer-product (customers without products appear once, with NULL product)."""
    sql = f"""
        WITH customers AS (
            SELECT customer_id, date_of_birth, country, segment, credit_score,
                   estimated_monthly_income AS income, occupation, education_level,
                   customer_status
            FROM {CUSTOMERS_TABLE}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY last_updated DESC) = 1
        ), products AS (
            SELECT product_id, customer_id, product_type, currency, current_balance
            FROM {PRODUCTS_TABLE}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY last_updated DESC) = 1
        )
        SELECT c.*, p.product_type, p.currency, p.current_balance
        FROM customers c LEFT JOIN products p USING (customer_id)
        ORDER BY c.customer_id, p.product_id
    """
    return con.execute(sql).df()


def load_transactions(con, start, end, customer_ids: list[str] | None = None) -> pd.DataFrame:
    """Transactions with transaction_date in [start, end), one row per transaction_id."""
    params = [pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()]
    customer_filter = ""
    if customer_ids is not None:
        if not customer_ids:
            return pd.DataFrame(columns=TRANSACTION_COLUMNS)
        customer_filter = "AND customer_id IN (SELECT UNNEST(?))"
        params.append(list(customer_ids))
    sql = f"""
        SELECT {", ".join(TRANSACTION_COLUMNS)}
        FROM {TRANSACTIONS_TABLE}
        WHERE transaction_date >= ? AND transaction_date < ? {customer_filter}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY transaction_date DESC) = 1
        ORDER BY customer_id, transaction_date
    """
    return con.execute(sql, params).df()


def load_usd_rates(con, as_of) -> dict[str, float]:
    """USD rates of the latest day with rates on or before `as_of`."""
    sql = f"""
        SELECT source_currency, target_currency, exchange_rate
        FROM {RATES_TABLE}
        WHERE date = (SELECT MAX(date) FROM {RATES_TABLE} WHERE date <= ?)
          AND 'USD' IN (source_currency, target_currency)
    """
    rows = con.execute(sql, [pd.Timestamp(as_of).date()]).fetchall()
    if not rows:
        raise LookupError(f"No hay tasas de cambio en {RATES_TABLE} en o antes de {as_of}")
    return usd_rates_from_rows(rows)
