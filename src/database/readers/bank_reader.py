import pandas as pd


class BankDBReader:

    def __init__(self, connection):
        self.conn = connection

    # ==========================================================
    # CLIENTES
    # ==========================================================

    def consulta_clientes(self, customer_ids=None, limit=None):

        sql = """
        SELECT
            customer_id,
            date_of_birth,
            country,
            segment,
            credit_score,
            estimated_monthly_income AS income,
            occupation,
            education_level,
            customer_status
        FROM bronze.customers
        """

        params = []

        if customer_ids:

            placeholders = ", ".join(["?"] * len(customer_ids))

            sql += f"""
                WHERE customer_id IN ({placeholders})
            """

            params.extend(customer_ids)

        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        return pd.read_sql(
            sql,
            self.conn,
            params=params
        )

    # ==========================================================
    # PRODUCTOS POR CLIENTES
    # ==========================================================

    def consulta_productos(self, customer_ids):

        if not customer_ids:
            return pd.DataFrame()

        placeholders = ", ".join(["?"] * len(customer_ids))

        sql = f"""
            SELECT
                customer_id,
                product_id,
                product_type,
                currency,
                current_balance,
                credit_limit
            FROM bronze.products
            WHERE customer_id IN ({placeholders})
        """

        return pd.read_sql(
            sql,
            self.conn,
            params=customer_ids
        )

    # ==========================================================
    # CLIENTES + PRODUCTOS POR BATCHES
    # ==========================================================

    def consulta_usuarios(
        self,
        customer_ids=None,
        limit=None,
        batch_size=5000
    ):

        clientes = self.consulta_clientes(
            customer_ids=customer_ids,
            limit=limit
        )

        if clientes.empty:
            return clientes

        resultados = []

        for inicio in range(0, len(clientes), batch_size):

            fin = inicio + batch_size

            batch_clientes = clientes.iloc[inicio:fin]

            ids = batch_clientes["customer_id"].tolist()

            productos = self.consulta_productos(ids)

            usuarios_batch = batch_clientes.merge(
                productos,
                on="customer_id",
                how="left"
            )

            resultados.append(usuarios_batch)

            print(
                f"Procesados {min(fin, len(clientes)):,} "
                f"de {len(clientes):,} clientes"
            )

        usuarios = pd.concat(
            resultados,
            ignore_index=True
        )

        return usuarios
    

    # ==========================================================
    # TRANSACCIONES
    # ==========================================================

    def consulta_transacciones(
        self,
        customer_ids=None,
        fecha_inicio=None,
        fecha_fin=None,
        limit=None
    ):

        sql = """
        SELECT
            transaction_id,
            transaction_date,
            process_date,
            product_id,
            customer_id,
            transaction_type,
            transaction_category,
            amount,
            currency,
            amount_usd,
            channel,
            branch_id,
            merchant_name,
            merchant_category,
            transaction_country,
            transaction_city,
            transaction_status,
            response_code,
            is_fraud,
            fraud_score,
            latitude,
            longitude,
            filename,
            "day",
            "month",
            "year"
        FROM bronze.transactions
        """

        conditions = []
        params = []

        # ------------------------------------------------------
        # FILTRO POR CLIENTES
        # ------------------------------------------------------

        if customer_ids is not None:

            if not customer_ids:
                return pd.DataFrame()

            placeholders = ", ".join(["?"] * len(customer_ids))

            conditions.append(
                f"customer_id IN ({placeholders})"
            )

            params.extend(customer_ids)

        # ------------------------------------------------------
        # FILTRO POR FECHA INICIAL
        # ------------------------------------------------------

        if fecha_inicio is not None:

            conditions.append(
                "transaction_date >= CAST(? AS DATE)"
            )

            params.append(fecha_inicio)

        # ------------------------------------------------------
        # FILTRO POR FECHA FINAL
        # ------------------------------------------------------
        # La fecha final es inclusiva.
        # Ejemplo:
        # fecha_fin = '2026-09-30'
        # incluye todo el 30 de septiembre.

        if fecha_fin is not None:

            conditions.append(
                """
                transaction_date <
                CAST(? AS DATE) + INTERVAL 1 DAY
                """
            )

            params.append(fecha_fin)

        # ------------------------------------------------------
        # WHERE
        # ------------------------------------------------------

        if conditions:

            sql += "\nWHERE " + "\nAND ".join(conditions)

        # ------------------------------------------------------
        # ORDEN
        # ------------------------------------------------------

        sql += """
        ORDER BY
            transaction_date,
            transaction_id
        """

        # ------------------------------------------------------
        # LIMIT
        # ------------------------------------------------------

        if limit is not None:

            sql += f"LIMIT {int(limit)}"

        return pd.read_sql(
            sql,
            self.conn,
            params=params
        )

    # ==========================================================
    # TRANSACCIONES POR BATCHES DE CLIENTES
    # ==========================================================
    def consulta_transacciones_batches(
        self,
        customer_ids=None,
        fecha_inicio=None,
        fecha_fin=None,
        batch_size=10000,
        limit=None
    ):

        resultados = []

        # ======================================================
        # CASO 1:
        # Se proporcionan customer_ids
        # ======================================================

        if customer_ids is not None:

            if not customer_ids:
                return pd.DataFrame()

            total_clientes = len(customer_ids)
            transacciones_restantes = limit

            for inicio in range(
                0,
                total_clientes,
                batch_size
            ):

                fin = inicio + batch_size

                batch_ids = customer_ids[inicio:fin]

                if transacciones_restantes is not None:

                    if transacciones_restantes <= 0:
                        break

                    batch_limit = transacciones_restantes

                else:
                    batch_limit = None

                transacciones_batch = self.consulta_transacciones(
                    customer_ids=batch_ids,
                    fecha_inicio=fecha_inicio,
                    fecha_fin=fecha_fin,
                    limit=batch_limit
                )

                if not transacciones_batch.empty:

                    resultados.append(
                        transacciones_batch
                    )

                    if transacciones_restantes is not None:

                        transacciones_restantes -= len(
                            transacciones_batch
                        )

                print(
                    f"Procesados {min(fin, total_clientes):,} "
                    f"de {total_clientes:,} clientes"
                )

        # ======================================================
        # CASO 2:
        # No se proporcionan customer_ids
        #
        # Se recorre directamente la tabla de transacciones
        # ======================================================

        else:

            offset = 0
            registros_procesados = 0

            while True:

                # ----------------------------------------------
                # Determinar tamaño del batch
                # ----------------------------------------------

                current_batch_size = batch_size

                if limit is not None:

                    registros_restantes = (
                        limit - registros_procesados
                    )

                    if registros_restantes <= 0:
                        break

                    current_batch_size = min(
                        batch_size,
                        registros_restantes
                    )

                # ----------------------------------------------
                # Consultar batch
                # ----------------------------------------------

                sql = """
                    SELECT
                        transaction_id,
                        transaction_date,
                        product_id,
                        customer_id,
                        transaction_type,
                        amount,
                        currency,
                        amount_usd,
                        transaction_country,
                        transaction_city,
                        transaction_status,
                        is_fraud
                    FROM bronze.transactions
                """

                conditions = []
                params = []

                # ----------------------------------------------
                # Filtro de fecha inicial
                # ----------------------------------------------

                if fecha_inicio is not None:

                    conditions.append(
                        "transaction_date >= CAST(? AS DATE)"
                    )

                    params.append(fecha_inicio)

                # ----------------------------------------------
                # Filtro de fecha final
                # ----------------------------------------------

                if fecha_fin is not None:

                    conditions.append(
                        """
                        transaction_date <
                        CAST(? AS DATE) + INTERVAL 1 DAY
                        """
                    )

                    params.append(fecha_fin)

                # ----------------------------------------------
                # WHERE
                # ----------------------------------------------

                if conditions:

                    sql += "\nWHERE " + "\nAND ".join(
                        conditions
                    )

                # ----------------------------------------------
                # Orden estable
                # ----------------------------------------------

                sql += """
                    ORDER BY
                        transaction_date,
                        transaction_id
                """

                # ----------------------------------------------
                # Batch
                # ----------------------------------------------

                sql += f"""
                    LIMIT {current_batch_size}
                    OFFSET {offset}
                """

                transacciones_batch = pd.read_sql(
                    sql,
                    self.conn,
                    params=params
                )

                # ----------------------------------------------
                # No hay más registros
                # ----------------------------------------------

                if transacciones_batch.empty:
                    break

                resultados.append(
                    transacciones_batch
                )

                registros_batch = len(
                    transacciones_batch
                )

                registros_procesados += registros_batch

                offset += registros_batch

                print(
                    f"Procesadas {registros_procesados:,}"
                    + (
                        f" de {limit:,}"
                        if limit is not None
                        else ""
                    )
                    + " transacciones"
                )

                # ----------------------------------------------
                # Último batch
                # ----------------------------------------------

                if registros_batch < current_batch_size:
                    break

        # ======================================================
        # RESULTADO FINAL
        # ======================================================

        if not resultados:
            return pd.DataFrame()

        transacciones = pd.concat(
            resultados,
            ignore_index=True
        )

        transacciones = (
            transacciones
            .sort_values(
                ["transaction_date", "transaction_id"]
            )
            .reset_index(drop=True)
        )

        return transacciones    