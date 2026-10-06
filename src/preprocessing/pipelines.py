import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler
)


class CustomerPreprocessor:
    """
    Reusable preprocessing pipeline for customer datasets.

    The preprocessor is responsible only for transforming raw features.
    It does not perform PCA, clustering, or supervised modeling.

    Processing steps:
        - Logarithmic transformation of selected numerical variables.
        - Scaling of numerical variables.
        - One-hot encoding of categorical variables.

    The fitted preprocessor can be reused to transform new observations
    using the same transformations learned from the training data.
    """

    def __init__(
        self,
        log_columns=None,
        numerical_columns=None,
        categorical_columns=None
    ):
        """
        Initialize the customer preprocessor.

        Parameters
        ----------
        log_columns : list[str], optional
            Numerical variables that require a log1p transformation
            before scaling.

        numerical_columns : list[str], optional
            Numerical variables that only require scaling.

        categorical_columns : list[str], optional
            Categorical variables that will be one-hot encoded.
        """

        self.log_columns = log_columns or []
        self.numerical_columns = numerical_columns or []
        self.categorical_columns = categorical_columns or []

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """
        Build the underlying sklearn preprocessing pipeline.
        """

        log_pipeline = Pipeline(
            steps=[
                (
                    "log_transform",
                    FunctionTransformer(
                        np.log1p,
                        feature_names_out="one-to-one"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "one_hot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse=False
                    )
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "log_numeric",
                    log_pipeline,
                    self.log_columns
                ),
                (
                    "numeric",
                    numerical_pipeline,
                    self.numerical_columns
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_columns
                )
            ],
            remainder="drop"
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor)
            ]
        )

    def fit(self, X):
        """
        Learn preprocessing parameters from the dataset.

        Parameters
        ----------
        X : pandas.DataFrame
            Input dataset.

        Returns
        -------
        CustomerPreprocessor
            Fitted preprocessor.
        """

        self.pipeline.fit(X)

        return self

    def transform(self, X):
        """
        Transform a dataset using the parameters learned during fitting.

        Parameters
        ----------
        X : pandas.DataFrame
            Input dataset.

        Returns
        -------
        numpy.ndarray
            Transformed feature matrix.
        """

        return self.pipeline.transform(X)

    def fit_transform(self, X):
        """
        Fit the preprocessor and transform the same dataset.

        Parameters
        ----------
        X : pandas.DataFrame
            Input dataset.

        Returns
        -------
        numpy.ndarray
            Transformed feature matrix.
        """

        return self.pipeline.fit_transform(X)

    def get_feature_names(self):
        """
        Return the names of the transformed features.

        Returns
        -------
        numpy.ndarray
            Names of the generated features.
        """

        preprocessor = self.pipeline.named_steps["preprocessor"]

        return preprocessor.get_feature_names_out()
    


class TransactionBehaviorPipeline:

    REQUIRED_COLUMNS = [
        "transaction_id",
        "transaction_date",
        "product_id",
        "customer_id",
        "transaction_type",
        "amount",
        "currency",
        "amount_usd",
        "transaction_country",
        "transaction_city",
        "transaction_status",
        "is_fraud",
    ]

    def __init__(
        self,
        period_start=None,
        period_end=None
    ):
        """
        Parameters
        ----------
        period_start : str or datetime, optional
            Start of the observation period.

        period_end : str or datetime, optional
            End of the observation period.

        If not provided, the period is inferred from the
        transaction dataframe during fit().
        """

        self.period_start = (
            pd.to_datetime(period_start)
            if period_start is not None
            else None
        )

        self.period_end = (
            pd.to_datetime(period_end)
            if period_end is not None
            else None
        )

        self.period_months_ = None

    # ==========================================================
    # VALIDATION
    # ==========================================================

    def _validate_columns(self, df):

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    # ==========================================================
    # PREPARATION
    # ==========================================================

    def _prepare_data(self, df):

        df = df.copy()

        self._validate_columns(df)

        # ------------------------------------------------------
        # Dates
        # ------------------------------------------------------

        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["customer_id", "transaction_date"]
        )

        # ------------------------------------------------------
        # Amount in USD
        #
        # Use amount_usd whenever available.
        # If it is missing and currency is USD, use amount.
        # ------------------------------------------------------

        df["amount_usd_clean"] = df["amount_usd"]

        mask_usd = (
            df["amount_usd_clean"].isna()
            & df["currency"].eq("USD")
        )

        df.loc[
            mask_usd,
            "amount_usd_clean"
        ] = df.loc[
            mask_usd,
            "amount"
        ]

        # ------------------------------------------------------
        # Temporal variables
        # ------------------------------------------------------

        df["transaction_hour"] = (
            df["transaction_date"].dt.hour
        )

        df["transaction_weekday"] = (
            df["transaction_date"].dt.dayofweek
        )

        df["is_weekend"] = (
            df["transaction_weekday"] >= 5
        )

        df["is_night"] = (
            (df["transaction_hour"] < 6)
            | (df["transaction_hour"] >= 22)
        )

        df["transaction_month"] = (
            df["transaction_date"].dt.to_period("M")
        )

        # ------------------------------------------------------
        # Status
        # ------------------------------------------------------

        df["is_approved"] = (
            df["transaction_status"]
            .eq("Approved")
        )

        return df

    # ==========================================================
    # FIT
    # ==========================================================

    def fit(self, df):

        df = self._prepare_data(df)

        if self.period_start is None:
            self.period_start = (
                df["transaction_date"].min()
                .to_period("M")
                .start_time
            )

        if self.period_end is None:
            self.period_end = (
                df["transaction_date"].max()
                .to_period("M")
                .start_time
            )

        self.period_months_ = (
            (
                self.period_end.year
                - self.period_start.year
            ) * 12
            + (
                self.period_end.month
                - self.period_start.month
            )
            + 1
        )

        return self

    # ==========================================================
    # TRANSFORM
    # ==========================================================

    def transform(self, df):

        if self.period_months_ is None:
            raise RuntimeError(
                "Pipeline must be fitted before transform()."
            )

        df = self._prepare_data(df)

        # ------------------------------------------------------
        # Basic aggregations
        # ------------------------------------------------------

        behavior = (
            df.groupby("customer_id")
            .agg(
                num_transacciones=(
                    "transaction_id",
                    "nunique"
                ),

                num_dias_activos=(
                    "transaction_date",
                    lambda x: x.dt.date.nunique()
                ),

                num_meses_activos=(
                    "transaction_month",
                    "nunique"
                ),

                monto_total_usd=(
                    "amount_usd_clean",
                    "sum"
                ),

                monto_promedio_usd=(
                    "amount_usd_clean",
                    "mean"
                ),

                monto_mediano_usd=(
                    "amount_usd_clean",
                    "median"
                ),

                monto_maximo_usd=(
                    "amount_usd_clean",
                    "max"
                ),

                monto_p95_usd=(
                    "amount_usd_clean",
                    lambda x: x.quantile(0.95)
                ),

                monto_std_usd=(
                    "amount_usd_clean",
                    "std"
                ),

                num_productos=(
                    "product_id",
                    "nunique"
                ),

                num_tipos_transaccion=(
                    "transaction_type",
                    "nunique"
                ),

                num_monedas=(
                    "currency",
                    "nunique"
                ),

                num_paises_transaccion=(
                    "transaction_country",
                    "nunique"
                ),

                num_ciudades_transaccion=(
                    "transaction_city",
                    "nunique"
                ),

                porcentaje_fin_semana=(
                    "is_weekend",
                    "mean"
                ),

                porcentaje_nocturnas=(
                    "is_night",
                    "mean"
                ),

                tasa_aprobacion=(
                    "is_approved",
                    "mean"
                ),

                porcentaje_amount_usd_faltante=(
                    "amount_usd_clean",
                    lambda x: x.isna().mean()
                ),
            )
            .reset_index()
        )

        # ------------------------------------------------------
        # Monthly frequency
        # ------------------------------------------------------

        behavior["transacciones_promedio_mes"] = (
            behavior["num_transacciones"]
            / self.period_months_
        )

        behavior["transacciones_promedio_mes_activo"] = (
            behavior["num_transacciones"]
            / behavior["num_meses_activos"]
        )

        # ------------------------------------------------------
        # Daily frequency
        # ------------------------------------------------------

        behavior["transacciones_por_dia_activo"] = (
            behavior["num_transacciones"]
            / behavior["num_dias_activos"]
        )

        # ------------------------------------------------------
        # Dominant transaction type
        # ------------------------------------------------------

        type_counts = (
            df.groupby(
                [
                    "customer_id",
                    "transaction_type"
                ]
            )
            .size()
            .reset_index(name="count")
        )

        dominant_type = (
            type_counts
            .sort_values(
                ["customer_id", "count"],
                ascending=[True, False]
            )
            .drop_duplicates("customer_id")
            .rename(
                columns={
                    "transaction_type":
                        "tipo_transaccion_dominante",
                    "count":
                        "count_dominante"
                }
            )
        )

        dominant_type = dominant_type[
            [
                "customer_id",
                "tipo_transaccion_dominante",
                "count_dominante"
            ]
        ]

        behavior = behavior.merge(
            dominant_type,
            on="customer_id",
            how="left"
        )

        behavior["participacion_tipo_dominante"] = (
            behavior["count_dominante"]
            / behavior["num_transacciones"]
        )

        behavior = behavior.drop(
            columns=["count_dominante"]
        )

        # ------------------------------------------------------
        # Final ordering
        # ------------------------------------------------------

        behavior = behavior.sort_values(
            "customer_id"
        ).reset_index(drop=True)

        return behavior

    # ==========================================================
    # FIT + TRANSFORM
    # ==========================================================

    def fit_transform(self, df):

        return self.fit(df).transform(df)