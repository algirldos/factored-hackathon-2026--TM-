"""Data contracts: the engine, and how each pipeline step enforces them."""
import numpy as np
import pandas as pd
import pytest

from factories import AS_OF, COUNTRY_CURRENCY, USD_RATES, raw_users, transactions
from src.contracts import (CLUSTER_PROFILES, CUSTOMER_FEATURES, Column, Contract, ContractError,
                           validate, validate_usd_rates)
from src.features.currency import normalize_key
from src.features.customer_features import build_customer_features
from src.features.transaction_behavior import behavior_windows, build_transaction_behavior
from src.models.behavior_scoring import score_customers

CONTRACT = Contract(
    name="prueba",
    columns=(Column("id", "string", nullable=False),
             Column("score", "number", min=0, max=100),
             Column("when", "datetime", min="2020-01-01"),
             Column("country", "string", allowed=frozenset({"mexico"}), normalize=normalize_key)),
    unique=("id",),
    row_checks=(("score impar", lambda df: df["score"] % 2 == 1),),
    max_invalid_fraction=0.5,
)


def frame(**overrides):
    data = {"id": ["a", "b", "c", "d"], "score": [10.0, 20.0, 30.0, 40.0],
            "when": ["2024-01-01"] * 4, "country": ["México"] * 4}
    data.update(overrides)
    return pd.DataFrame(data)


# --- engine -----------------------------------------------------------------
def test_valid_data_passes_unchanged():
    df, report = validate(frame(), CONTRACT)
    assert len(df) == 4 and report.rows_dropped == 0 and report.violations == {}


def test_missing_column_always_raises():
    with pytest.raises(ContractError, match="faltan columnas"):
        validate(frame().drop(columns=["score"]), CONTRACT, mode="drop")


def test_wrong_type_always_raises():
    with pytest.raises(ContractError, match="tipos incorrectos"):
        validate(frame(score=["10", "20", "30", "40"]), CONTRACT, mode="drop")


def test_all_null_column_has_no_type_to_check():
    df, _ = validate(frame(score=[None] * 4), CONTRACT)
    assert len(df) == 4


@pytest.mark.parametrize("overrides, rule", [
    ({"id": ["a", None, "c", "d"]}, "id: nulo"),
    ({"score": [10.0, 200.0, 30.0, 40.0]}, "score: > 100"),
    ({"score": [10.0, -2.0, 30.0, 40.0]}, "score: < 0"),
    ({"score": [10.0, np.inf, 30.0, 40.0]}, "score: infinito"),
    ({"when": ["2024-01-01", "no es fecha", "2024-01-01", "2024-01-01"]}, "when: fecha inválida"),
    ({"when": ["2024-01-01", "2019-12-31", "2024-01-01", "2024-01-01"]}, "when: < 2020-01-01"),
    ({"country": ["México", "Peru", "Mexico", "MEXICO"]}, "country: valor no permitido"),
    ({"id": ["a", "a", "c", "d"]}, "duplicado en ['id']"),
    ({"score": [10.0, 21.0, 30.0, 40.0]}, "score impar"),
])
def test_each_rule_is_detected_and_named(overrides, rule):
    with pytest.raises(ContractError, match=rule.replace("[", r"\[").replace("]", r"\]")):
        validate(frame(**overrides), CONTRACT)
    df, report = validate(frame(**overrides), CONTRACT, mode="drop")
    assert len(df) == 3 and report.violations[rule] == 1 and report.rows_dropped == 1


def test_drop_mode_stops_when_too_many_rows_are_invalid():
    with pytest.raises(ContractError, match="supera el límite"):
        validate(frame(score=[-1.0, -1.0, -1.0, 40.0]), CONTRACT, mode="drop")


def test_usd_rates_contract():
    validate_usd_rates({"USD": 1.0, "COP": 0.00025})
    for bad in ({"USD": 1.0, "COP": 0.0}, {"USD": 1.0, "MXN": float("nan")}, {"COP": 0.1}):
        with pytest.raises(ContractError):
            validate_usd_rates(bad)


# --- pipeline steps -----------------------------------------------------------
def test_customer_step_drops_a_few_bad_raw_rows():
    users = raw_users(60)
    users.loc[users["customer_id"] == "CLI-0005", "credit_score"] = 9_999.0     # out of range
    features = build_customer_features(users, AS_OF, COUNTRY_CURRENCY, USD_RATES)
    assert "CLI-0005" not in set(features["customer_id"])
    assert len(features) == 59


def test_customer_step_stops_when_raw_data_is_broken():
    users = raw_users(20)
    users["customer_status"] = "Activo"                    # not a valid status for any row
    with pytest.raises(ContractError, match="customer_status"):
        build_customer_features(users, AS_OF, COUNTRY_CURRENCY, USD_RATES)


def test_customer_features_contract_rejects_bad_model_input():
    features = build_customer_features(raw_users(10), AS_OF, COUNTRY_CURRENCY, USD_RATES)
    features.loc[0, "income_usd"] = np.nan
    with pytest.raises(ContractError, match="income_usd: nulo"):
        validate(features, CUSTOMER_FEATURES)


def test_transaction_step_drops_invalid_and_duplicate_rows():
    _, recent = behavior_windows(AS_OF)
    tx = transactions(["CLI-1", "CLI-2"], recent.start, recent.days, per_day=0.9)
    tx.loc[tx.index[0], "transaction_status"] = "Desconocido"
    behavior = build_transaction_behavior(tx, recent).set_index("customer_id")
    assert behavior["num_transacciones"].sum() == len(tx) - 1


def test_corrupted_profiles_are_rejected_before_scoring():
    profiles = pd.DataFrame({"cluster": [0], "feature": ["x"], "count": [10], "mean": [1.0],
                             "median": [1.0], "p25": [2.0], "p75": [1.0], "p95": [3.0],
                             "mad": [0.5]})
    with pytest.raises(ContractError, match="p25 > p75"):
        validate(profiles, CLUSTER_PROFILES)
    with pytest.raises(ContractError):
        score_customers(pd.DataFrame({"customer_id": ["A"], "x": [1.0]}),
                        pd.DataFrame({"customer_id": ["A"], "cluster": [0]}), profiles, ["x"])
