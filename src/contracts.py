"""
Data contracts: the schema every step of the clustering pipeline accepts and produces.

Two modes:
- "drop" for raw inputs from the bronze layer, which has known quality issues
  (~5% nulls, ~2% duplicates): invalid rows are removed and counted, but if they exceed
  the contract's `max_invalid_fraction` the whole step stops, because the data is broken.
- "raise" for what our own code produces: any violation is a bug and stops the step.

Schema errors (missing column, wrong type) always stop the step, in both modes.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from src.features.currency import normalize_key

log = logging.getLogger(__name__)

KINDS = {"string", "number", "datetime", "bool"}


class ContractError(ValueError):
    """The data does not satisfy its contract."""


@dataclass(frozen=True)
class Column:
    name: str
    kind: str
    nullable: bool = True
    min: float | str | None = None          # inclusive; a date string for datetime columns
    max: float | str | None = None
    allowed: frozenset | None = None        # compared after `normalize`, if given
    normalize: Callable | None = None
    finite: bool = True                     # numbers: reject +/-inf

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Tipo de columna desconocido: {self.kind}")


@dataclass(frozen=True)
class Contract:
    name: str
    columns: tuple[Column, ...]
    unique: tuple[str, ...] = ()
    row_checks: tuple[tuple[str, Callable[[pd.DataFrame], pd.Series]], ...] = ()
    max_invalid_fraction: float = 0.0

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class ValidationReport:
    contract: str
    rows_in: int
    rows_out: int
    violations: dict[str, int] = field(default_factory=dict)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def _check_schema(df: pd.DataFrame, contract: Contract) -> None:
    missing = [c for c in contract.column_names if c not in df.columns]
    if missing:
        raise ContractError(f"{contract.name}: faltan columnas {missing}")
    wrong = []
    for col in contract.columns:
        series = df[col.name]
        if series.isna().all():          # an all-null column has no reliable dtype
            continue
        ok = {"string": ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series),
              "number": ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series),
              "datetime": True,      # parsed row by row: DATE columns may arrive as text
              "bool": ptypes.is_bool_dtype(series) or ptypes.is_object_dtype(series)}[col.kind]
        if not ok:
            wrong.append(f"{col.name} ({series.dtype}, se esperaba {col.kind})")
    if wrong:
        raise ContractError(f"{contract.name}: tipos incorrectos: {wrong}")


def _column_violations(df: pd.DataFrame, col: Column) -> dict[str, pd.Series]:
    series = df[col.name]
    present = series.notna()
    found = {}
    if not col.nullable:
        found[f"{col.name}: nulo"] = ~present
    values = series
    if col.kind == "datetime":
        values = pd.to_datetime(series, errors="coerce")
        found[f"{col.name}: fecha inválida"] = present & values.isna()
    if col.kind == "number":
        values = pd.to_numeric(series, errors="coerce").astype("float64")
        if col.finite:
            found[f"{col.name}: infinito"] = present & np.isinf(values)
    if col.min is not None:
        bound = pd.Timestamp(col.min) if col.kind == "datetime" else col.min
        found[f"{col.name}: < {col.min}"] = present & (values < bound).fillna(False)
    if col.max is not None:
        bound = pd.Timestamp(col.max) if col.kind == "datetime" else col.max
        found[f"{col.name}: > {col.max}"] = present & (values > bound).fillna(False)
    if col.allowed is not None:
        normalized = series.map(col.normalize) if col.normalize else series
        found[f"{col.name}: valor no permitido"] = present & ~normalized.isin(col.allowed)
    return {rule: mask.astype(bool) for rule, mask in found.items() if mask.any()}


def validate(df: pd.DataFrame, contract: Contract, mode: str = "raise"
             ) -> tuple[pd.DataFrame, ValidationReport]:
    """Check `df` against `contract`. Returns the valid rows and a report."""
    if mode not in {"raise", "drop"}:
        raise ValueError("mode debe ser 'raise' o 'drop'")
    _check_schema(df, contract)

    masks: dict[str, pd.Series] = {}
    for col in contract.columns:
        masks.update(_column_violations(df, col))
    for rule, check in contract.row_checks:
        mask = check(df).fillna(False).astype(bool)
        if mask.any():
            masks[rule] = mask
    invalid = pd.Series(False, index=df.index)
    for mask in masks.values():
        invalid |= mask
    if contract.unique:
        dup = df.duplicated(subset=list(contract.unique), keep="first") & ~invalid
        if dup.any():
            masks[f"duplicado en {list(contract.unique)}"] = dup
            invalid |= dup

    report = ValidationReport(contract.name, len(df), int(len(df) - invalid.sum()),
                              {rule: int(m.sum()) for rule, m in masks.items()})
    if not invalid.any():
        log.info("%s: %d filas válidas", contract.name, len(df))
        return df, report

    detail = ", ".join(f"{rule} ({n})" for rule, n in report.violations.items())
    if mode == "raise":
        raise ContractError(f"{contract.name}: {detail}")
    fraction = report.rows_dropped / max(len(df), 1)
    if fraction > contract.max_invalid_fraction:
        raise ContractError(f"{contract.name}: {fraction:.1%} de filas inválidas supera el "
                            f"límite de {contract.max_invalid_fraction:.0%}: {detail}")
    log.warning("%s: se descartan %d de %d filas: %s", contract.name, report.rows_dropped,
                len(df), detail)
    return df[~invalid], report


def validate_usd_rates(rates: dict[str, float]) -> None:
    """Rates must include USD = 1 and be positive and finite."""
    bad = {c: r for c, r in rates.items() if r is None or not np.isfinite(r) or r <= 0}
    if bad:
        raise ContractError(f"Tasas de cambio inválidas: {bad}")
    if rates.get("USD") != 1.0:
        raise ContractError("La tasa de USD debe ser 1.0")


# ---------------------------------------------------------------------------
# Contracts of the clustering pipeline
# ---------------------------------------------------------------------------
COUNTRIES = frozenset({"colombia", "mexico", "argentina"})
CUSTOMER_STATUSES = frozenset({"Active", "Inactive", "Suspended", "Closed"})
TRANSACTION_STATUSES = frozenset({"Approved", "Declined", "Pending", "Reversed"})

# Customers joined with their products, as read from bronze (one row per customer-product)
RAW_CUSTOMER_PRODUCTS = Contract(
    name="clientes+productos (bronze)",
    columns=(
        Column("customer_id", "string", nullable=False),
        Column("date_of_birth", "datetime", nullable=False, min="1900-01-01"),
        Column("country", "string", nullable=False),
        Column("segment", "string"),
        Column("credit_score", "number", min=300, max=850),
        Column("income", "number", min=0),
        Column("occupation", "string"),
        Column("education_level", "string"),
        Column("customer_status", "string", nullable=False, allowed=CUSTOMER_STATUSES),
        Column("product_type", "string"),
        Column("currency", "string"),
        Column("current_balance", "number"),
    ),
    max_invalid_fraction=0.05,
)

PRODUCT_BALANCE_COLUMNS = ["Cuenta Ahorro", "Cuenta Corriente", "Inversión",
                           "Préstamo Hipotecario", "Préstamo Personal", "Seguro",
                           "Tarjeta Crédito", "Tarjeta Débito"]

# Input of the clustering model: one row per eligible customer
CUSTOMER_FEATURES = Contract(
    name="variables de cliente",
    columns=(
        Column("customer_id", "string", nullable=False),
        Column("country", "string", nullable=False, allowed=COUNTRIES, normalize=normalize_key),
        Column("education_level", "string"),
        Column("credit_score", "number", nullable=False, min=300, max=850),
        Column("income_usd", "number", nullable=False, min=0),
        Column("Edad", "number", nullable=False, min=0, max=120),
        *(Column(name, "number", nullable=False) for name in PRODUCT_BALANCE_COLUMNS),
        Column("num_products", "number", nullable=False, min=1, max=len(PRODUCT_BALANCE_COLUMNS)),
        Column("Deuda_usd", "number", nullable=False),
        Column("Activos_financieros_usd", "number", nullable=False),
    ),
    unique=("customer_id",),
)

# Transactions read from bronze, after deduplication
RAW_TRANSACTIONS = Contract(
    name="transacciones (bronze)",
    columns=(
        Column("transaction_id", "string", nullable=False),
        Column("transaction_date", "datetime", nullable=False),
        Column("customer_id", "string", nullable=False),
        Column("product_id", "string"),
        Column("transaction_type", "string"),
        Column("amount", "number", nullable=False),
        Column("currency", "string", nullable=False),
        Column("amount_usd", "number"),
        Column("transaction_status", "string", nullable=False, allowed=TRANSACTION_STATUSES),
    ),
    unique=("transaction_id",),
    max_invalid_fraction=0.05,
)

_SHARE_COLUMNS = ["porcentaje_fin_semana", "porcentaje_nocturnas", "tasa_aprobacion",
                  "participacion_tipo_dominante"]

# Behavior metrics per customer and window
TRANSACTION_BEHAVIOR = Contract(
    name="comportamiento transaccional",
    columns=(
        Column("customer_id", "string", nullable=False),
        Column("num_transacciones", "number", nullable=False, min=1),
        Column("num_dias_activos", "number", nullable=False, min=1),
        Column("monto_total_usd", "number"),
        Column("monto_promedio_usd", "number"),
        *(Column(name, "number", min=0, max=1) for name in _SHARE_COLUMNS),
    ),
    unique=("customer_id",),
)

# Robust statistics per cluster and feature
CLUSTER_PROFILES = Contract(
    name="perfiles por clúster",
    columns=(
        Column("cluster", "number", nullable=False, min=0),
        Column("feature", "string", nullable=False),
        Column("count", "number", nullable=False, min=1),
        Column("median", "number", nullable=False),
        Column("p25", "number", nullable=False),
        Column("p75", "number", nullable=False),
        Column("mad", "number", nullable=False, min=0),
    ),
    unique=("cluster", "feature"),
    row_checks=(("p25 > p75", lambda df: df["p25"] > df["p75"]),),
)

# Scores per customer and feature
BEHAVIOR_SCORES = Contract(
    name="desviaciones por métrica",
    columns=(
        Column("customer_id", "string", nullable=False),
        Column("cluster", "number", nullable=False, min=0),
        Column("feature", "string", nullable=False),
        # inf is valid: no spread in the cluster and a value different from the median
        Column("deviation_score", "number", min=0, finite=False),
    ),
    unique=("customer_id", "feature"),
)

# Summary per customer
CUSTOMER_SCORES = Contract(
    name="resumen por cliente",
    columns=(
        Column("customer_id", "string", nullable=False),
        Column("cluster", "number", nullable=False, min=0),
        Column("anomaly_score", "number", min=0),
        Column("n_features", "number", nullable=False, min=1),
        Column("n_suspicious", "number", nullable=False, min=0),
    ),
    unique=("customer_id",),
    row_checks=(("n_suspicious > n_features", lambda df: df["n_suspicious"] > df["n_features"]),),
)
