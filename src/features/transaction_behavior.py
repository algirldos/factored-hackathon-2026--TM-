"""
Transaction behavior per customer (notebooks/03_clustering_refactored.ipynb, sections 12-18).

Changes from the notebook:
- Windows are built from an explicit `as_of` date, with half-open day boundaries
  [start, end), instead of the maximum timestamp found in the data.
- Duplicate transaction IDs (~2% in the bronze layer) are removed before aggregating,
  so amounts are not counted twice.
- The behavior period is passed to TransactionBehaviorPipeline, not inferred from the data.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing.pipelines import TransactionBehaviorPipeline

BASELINE_MONTHS = 12
RECENT_DAYS = 30

MONTHLY_RATE_FEATURES = [
    "num_transacciones", "monto_total_usd", "num_dias_activos", "num_productos",
    "num_tipos_transaccion", "num_monedas", "num_paises_transaccion", "num_ciudades_transaccion",
]
SCALE_INDEPENDENT_FEATURES = [
    "monto_promedio_usd", "monto_mediano_usd", "monto_maximo_usd", "monto_p95_usd",
    "monto_std_usd", "transacciones_por_dia_activo", "porcentaje_fin_semana",
    "porcentaje_nocturnas", "tasa_aprobacion", "participacion_tipo_dominante",
]


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp          # first day included (00:00)
    end: pd.Timestamp            # first day NOT included (00:00)

    @property
    def days(self) -> int:
        return (self.end - self.start).days


def behavior_windows(as_of, baseline_months: int = BASELINE_MONTHS,
                     recent_days: int = RECENT_DAYS) -> tuple[Window, Window]:
    """
    Baseline and recent windows that end on `as_of` (included).
    The recent window covers the last `recent_days` days; the baseline covers the
    `baseline_months` months right before it, so the two never overlap.
    """
    recent_end = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
    recent_start = recent_end - pd.Timedelta(days=recent_days)
    baseline_start = recent_start - pd.DateOffset(months=baseline_months)
    return Window(baseline_start, recent_start), Window(recent_start, recent_end)


def select_window(transactions: pd.DataFrame, window: Window) -> pd.DataFrame:
    dates = pd.to_datetime(transactions["transaction_date"], errors="coerce")
    return transactions[(dates >= window.start) & (dates < window.end)].copy()


def clean_transaction_amounts(transactions: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate IDs and fill amount_usd with amount when the currency is already USD."""
    transactions = transactions.drop_duplicates("transaction_id", keep="last").copy()
    if "amount_usd" not in transactions.columns:
        transactions["amount_usd"] = np.nan
    usd_mask = transactions["amount_usd"].isna() & transactions["currency"].eq("USD")
    transactions.loc[usd_mask, "amount_usd"] = transactions.loc[usd_mask, "amount"]
    return transactions


def build_transaction_behavior(transactions: pd.DataFrame, window: Window) -> pd.DataFrame:
    """One row per customer with the behavior metrics of the window."""
    if transactions.empty:
        return pd.DataFrame(columns=["customer_id"])
    pipeline = TransactionBehaviorPipeline(period_start=window.start,
                                           period_end=window.end - pd.Timedelta(days=1))
    behavior = pipeline.fit_transform(clean_transaction_amounts(transactions))
    if not behavior["customer_id"].is_unique:
        raise ValueError("TransactionBehaviorPipeline debe devolver una fila por cliente.")
    return behavior


def normalize_behavior_period(behavior: pd.DataFrame, period_days: int) -> pd.DataFrame:
    """Rate features become per-30-days values (`<feature>_monthly`), comparable across windows."""
    result = behavior.copy()
    factor = 30.0 / period_days
    for column in MONTHLY_RATE_FEATURES:
        if column in result.columns:
            result[f"{column}_monthly"] = pd.to_numeric(result[column], errors="coerce") * factor
    return result


def profile_feature_names(columns) -> list[str]:
    """Features compared against the cluster profile, in a fixed order."""
    columns = set(columns)
    return ([f"{f}_monthly" for f in MONTHLY_RATE_FEATURES if f"{f}_monthly" in columns]
            + [f for f in SCALE_INDEPENDENT_FEATURES if f in columns])


def window_behavior(transactions: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Select the window, aggregate and normalize: the scaled behavior used for profiles and scoring."""
    behavior = build_transaction_behavior(select_window(transactions, window), window)
    return normalize_behavior_period(behavior, window.days)
