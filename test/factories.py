"""Small synthetic datasets shaped like the LATAM Bank bronze tables."""
import duckdb
import numpy as np
import pandas as pd

AS_OF = pd.Timestamp("2026-06-17")
COUNTRY_CURRENCY = {"colombia": "COP", "mexico": "MXN", "argentina": "ARS"}
USD_RATES = {"USD": 1.0, "COP": 0.00025, "MXN": 0.05, "ARS": 0.001}
PRODUCTS = ["Cuenta Ahorro", "Tarjeta Crédito", "Préstamo Personal", "Tarjeta Débito"]


def raw_users(n_customers: int = 60, seed: int = 0) -> pd.DataFrame:
    """One row per customer-product, as sources.load_customers_with_products returns it."""
    rng = np.random.default_rng(seed)
    countries = ["Colombia", "México", "Argentina"]
    currency = {"Colombia": "COP", "México": "MXN", "Argentina": "ARS"}
    rows = []
    for i in range(n_customers):
        country = countries[i % 3]
        base = {
            "customer_id": f"CLI-{i:04d}", "date_of_birth": f"{1960 + i % 40}-0{1 + i % 9}-15",
            "country": country, "segment": ["Basic", "Plus", "Premium"][i % 3],
            "credit_score": float(rng.integers(400, 850)),
            "income": float(rng.integers(1_000, 20_000)) / USD_RATES[currency[country]] / 1000,
            "occupation": "Engineer", "education_level": ["University", "High School"][i % 2],
            "customer_status": "Active",
        }
        for product in PRODUCTS[: 1 + i % len(PRODUCTS)]:
            rows.append({**base, "product_type": product, "currency": currency[country],
                         "current_balance": float(rng.integers(100, 50_000))})
    return pd.DataFrame(rows)


def transactions(customer_ids, start, days: int, per_day: float = 0.3,
                 amount: float | dict = 50.0, seed: int = 0) -> pd.DataFrame:
    """Regular activity: a few transactions per customer, amounts around `amount` USD
    (a number for everyone, or {customer_id: typical amount})."""
    rng = np.random.default_rng(seed)
    rows, n = [], 0
    for cid in customer_ids:
        for day in range(days):
            if rng.random() > per_day:
                continue
            n += 1
            when = pd.Timestamp(start) + pd.Timedelta(days=day, hours=int(rng.integers(8, 20)))
            typical = amount[cid] if isinstance(amount, dict) else amount
            value = float(rng.normal(typical, typical * 0.05))
            rows.append({"transaction_id": f"TRX-{seed}-{n}", "transaction_date": when,
                         "product_id": f"PRD-{cid}", "customer_id": cid,
                         "transaction_type": "Purchase", "amount": value, "currency": "USD",
                         "amount_usd": value, "transaction_country": "Colombia",
                         "transaction_city": "Bogotá", "transaction_status": "Approved"})
    return pd.DataFrame(rows)


def bronze_connection(database: str = ":memory:"):
    """DuckDB (in memory by default) with the bronze schema and exchange rates around AS_OF."""
    con = duckdb.connect(database)
    con.execute("CREATE SCHEMA bronze")
    con.execute("""CREATE TABLE bronze.daily_exchange_rates (date DATE, source_currency VARCHAR,
                   target_currency VARCHAR, exchange_rate DOUBLE)""")
    con.execute("""INSERT INTO bronze.daily_exchange_rates VALUES
                   ('2026-06-16', 'COP', 'USD', 0.00020), ('2026-06-17', 'COP', 'USD', 0.00025),
                   ('2026-06-17', 'USD', 'MXN', 20.0),    ('2026-06-17', 'ARS', 'USD', 0.001),
                   ('2026-06-18', 'COP', 'USD', 0.00099)""")
    return con


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
