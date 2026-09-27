import pandas as pd


class BankDBReader:

    def __init__(self, connection):
        self.conn = connection

    # ==========================================================
    # CLIENTES + PRODUCTOS
    # ==========================================================

    def consulta_usuarios(self, limit=None):

        sql = """
        SELECT
            u.customer_id,
            u.date_of_birth,
            u.country,
            u.segment,
            u.credit_score,
            u.estimated_monthly_income AS income,
            u.occupation,
            u.customer_status,

            p.product_id,
            p.product_type,
            p.currency,
            p.current_balance,
            p.credit_limit

        FROM bronze.customers AS u

        LEFT JOIN bronze.products AS p
            ON u.customer_id = p.customer_id
        """

        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        return pd.read_sql(sql, self.conn)