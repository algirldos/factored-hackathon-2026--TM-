"""
Behavior alerts in the agent: the customer gets a neutral message and nothing about how they
are profiled reaches the LLM; bank staff get the internal detail by email.
"""
import json

import duckdb
import pytest

import fraud_agent as fa
import fraud_flow as ff
import policy
from conftest import DEMO_OTP
from src.models.batch_scoring import setup_tables
from src.models.behavior_alerts import (CUSTOMER_MESSAGES, METRIC_LABELS, customer_facing_alert,
                                        internal_note, latest_behavior_anomaly)
from src.models.behavior_scoring import SUSPICIOUS_FEATURES

FLAGGED = "CLI-TEST"          # customer of the agent fixtures in conftest
QUIET = "CLI-QUIET"
# Anything that would tell the customer they are profiled, or leak internal values
FORBIDDEN = ["clusters_2026", "clúster", "cluster", "segmento", "perfil", "mediana", "median",
             *SUSPICIOUS_FEATURES, *METRIC_LABELS.values()]


@pytest.fixture
def ops(monkeypatch):
    """In-memory database with two scoring runs and the agent's connection pointing to it."""
    con = duckdb.connect(":memory:")
    setup_tables(con)
    con.execute("""INSERT INTO fraud_ops.customer_anomaly VALUES
        ('2026-05-18', 'clusters_20260517', 'CLI-TEST', 1, 0.5, 5, 1, 'monto_total_usd_monthly'),
        ('2026-06-17', 'clusters_20260617', 'CLI-TEST', 2, 9.1, 5, 4,
         'num_transacciones_monthly,monto_total_usd_monthly,monto_promedio_usd,monto_maximo_usd'),
        ('2026-06-17', 'clusters_20260617', 'CLI-QUIET', 0, 0.4, 5, 0, '')""")
    monkeypatch.setattr(fa, "con", con)
    yield con
    con.close()


def leaked(payload) -> list[str]:
    text = json.dumps(payload, ensure_ascii=False, default=str).lower()
    return [word for word in FORBIDDEN if word.lower() in text]


# --- reading and building the alert -------------------------------------------
def test_latest_run_is_used(ops):
    anomaly = latest_behavior_anomaly(ops, FLAGGED)
    assert anomaly["model_version"] == "clusters_20260617"
    assert anomaly["n_suspicious"] == 4


def test_missing_tables_or_customer_mean_no_alert(ops):
    assert latest_behavior_anomaly(ops, "CLI-NO-EXISTE") is None
    assert latest_behavior_anomaly(duckdb.connect(":memory:"), FLAGGED) is None


def test_alert_only_above_the_threshold(ops):
    assert customer_facing_alert(latest_behavior_anomaly(ops, QUIET)) is None
    alert = customer_facing_alert(latest_behavior_anomaly(ops, FLAGGED))
    assert alert["unusual_activity_detected"] is True
    assert alert["message"]["es"] == "Hemos detectado un comportamiento sospechoso en su actividad bancaria."
    assert set(alert["message"]) == set(CUSTOMER_MESSAGES)


def test_alert_is_built_from_an_allow_list(ops):
    alert = customer_facing_alert(latest_behavior_anomaly(ops, FLAGGED))
    assert set(alert) == {"unusual_activity_detected", "message", "reviewed_period_end", "instruction"}
    payload = {k: v for k, v in alert.items() if k != "instruction"}
    assert leaked(payload) == []


@pytest.mark.parametrize("language", sorted(CUSTOMER_MESSAGES))
def test_customer_messages_reveal_nothing(language):
    assert leaked(CUSTOMER_MESSAGES[language]) == []


# --- what the LLM receives ------------------------------------------------------
def test_console_agent_assessment_carries_the_neutral_alert(ops, agent_tools):
    result = agent_tools["assess_fraud_risk"](180)
    assert result["behavior_alert"]["message"]["es"] == CUSTOMER_MESSAGES["es"]
    tool_output = {k: v for k, v in result.items() if k != "behavior_alert"}
    tool_output["behavior_alert"] = {k: v for k, v in result["behavior_alert"].items()
                                     if k != "instruction"}
    assert leaked(tool_output) == []


def test_no_alert_for_a_customer_without_suspicious_activity(ops, offline):
    tools = {t.__name__: t for t in fa.build_tools(QUIET, {"full_name": "Quieto"})}
    assert "behavior_alert" not in tools["assess_fraud_risk"](180)


def test_agent_works_when_scoring_tables_do_not_exist(monkeypatch, agent_tools):
    monkeypatch.setattr(fa, "con", duckdb.connect(":memory:"))
    assert "behavior_alert" not in agent_tools["assess_fraud_risk"](180)


def test_verified_flow_case_details_carry_the_alert(ops, flow_tools):
    assert flow_tools["get_case_details"]() == policy.NOT_VERIFIED   # nothing before the OTP
    flow_tools["verify_otp"](DEMO_OTP)
    details = flow_tools["get_case_details"]()
    assert details["behavior_alert"]["message"]["pt"] == CUSTOMER_MESSAGES["pt"]


def test_prompts_forbid_explaining_the_detection():
    for prompt in (fa.SYSTEM_INSTRUCTIONS, ff.FLOW_INSTRUCTIONS):
        assert "behavior_alert" in prompt
        assert "Never explain how" in prompt


# --- what bank staff receive -----------------------------------------------------
def test_internal_note_names_the_metrics_for_staff_only(ops):
    note = internal_note(latest_behavior_anomaly(ops, FLAGGED))
    assert "monto de la transacción más alta" in note and "4 de 5" in note
    assert internal_note(latest_behavior_anomaly(ops, QUIET)) == ""


def test_escalation_emails_staff_the_detail_but_returns_nothing_of_it(ops, monkeypatch, offline):
    ff.setup_tables()
    agent = {"agent_id": "AGT-1", "first_name": "Luz", "last_name": "Rojas", "email": "luz@banco",
             "specialty": "Fraudes", "routing_score": 100.0, "routing_reasons": ["habla español"]}
    monkeypatch.setattr(ff, "rank_agents", lambda language, profile: [agent])
    sent = {}
    monkeypatch.setattr(fa, "send_email", lambda to, subject, body: sent.update(body=body) or "sent")

    result = ff.escalate({"case_id": "CASE-1", "customer_id": FLAGGED}, {}, "es",
                         "no reconoce cobros", "resumen")

    assert "monto de la transacción más alta" in sent["body"]
    assert result["escalated"] is True and leaked(result) == []
